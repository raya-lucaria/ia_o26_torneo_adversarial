"""Expectiminimax con poda alfa-beta (Star1 en los nodos de azar), compilado con numba.

Representación (la misma del motor oficial, `sistema/motor/nucleo.py`):
- `tab`: arreglo int8[81], casilla = fila*9 + col (fila 0 = fila «1»).
  0 = vacía, 1 = peón blanco, 2 = peón negro.
- `color`: 0 = Blancas, 1 = Negras. `pts`: int64[2] con los puntos de cada color.

Valores en negamax: siempre desde el punto de vista del jugador en turno, dentro de [-1, 1].
Ganar vale ~1, perder ~-1, una carrera ya decidida ~±0.89 y la evaluación heurística se queda
en (-0.85, 0.85), así que una victoria segura siempre pesa más que cualquier posición «buena».

El tablero se modifica en sitio (hacer/deshacer) y el hash Zobrist se actualiza de forma
incremental; usamos las mismas tablas Zobrist que el motor, así que nuestro hash coincide con
`estado.clave`.
"""

import math
from pathlib import Path

import numpy as np
from numba import njit

from motor import nucleo

LADO = 9
CASILLAS = 81
MAX_JUGADAS = 54
MAX_PLY = 64
PUNTOS_PARA_GANAR = 3

GANAR = 1.0  # valor de una victoria (se le resta un poco por ply: preferimos ganar antes)
INF = 2.0  # cota de ventana «sin límite» (mayor que cualquier valor posible)
COTA = 1.0  # todos los valores están en [-COTA, COTA]: lo necesita Star1

Z_PIEZAS = nucleo.Z_PIEZAS
Z_TURNO = nucleo.Z_TURNO
Z_PUNTOS = nucleo.Z_PUNTOS

# --- Búsqueda de variante principal (PVS) ---------------------------------------------------
PVS_ACTIVO = True
VENTANA_MIN = 1e-6  # ancho de la ventana mínima (los valores son reales en [-1, 1])

# --- Búsqueda de quietud -------------------------------------------------------------------
# Al llegar a profundidad 0 no evaluamos de inmediato si hay ataques o anotaciones pendientes:
# seguimos solo con esas jugadas (hasta QS_PROF plies más) y con la opción de «quedarse» con la
# evaluación estática. Evita el efecto horizonte: evaluar justo antes de un intercambio.
# 0 = apagada. QS_PMIN: solo ataques con al menos esta probabilidad de ganar.
QS_PROF = 0
QS_PMIN = 0.0

# --- Tabla de transposición ---------------------------------------------------------------
TT_BITS = 22  # 4 M entradas, ~84 MB: con 2^20 se llenaba (11 % más nodos a prof 7)
TT_TAM = 1 << TT_BITS
EXACTO, COTA_INF, COTA_SUP = 0, 1, 2  # el valor guardado es exacto, ≥ o ≤ el real

# --- Estadísticas compartidas con Python (arreglo int64) ----------------------------------
# MEJOR_RAIZ: al entrar, la mejor jugada de la iteración anterior (va primero); al salir, la
# mejor de esta. COMPLETAS: cuántas jugadas de la raíz se evaluaron completas.
NODOS, ABORTAR, LIMITE_NODOS, MEJOR_RAIZ, COMPLETAS = 0, 1, 2, 3, 4

# --- Evaluación: rasgos × pesos -------------------------------------------------------------
# Amenaza segura: un punto garantizado, pero vale menos que uno anotado (1.5) porque cobrarla
# cuesta una jugada. Los pesos de «libre» son chicos por lo mismo: un peón libre en avance 6
# vale 0.65 + 0.3 + 6·0.05 = 1.25 < 1.3 < 1.5, así que avanzar y anotar siempre suma.
W_SEGURA = 1.3
VALOR_CARRERA = 0.89  # valor de una carrera ya decidida (menos que ganar de verdad, ~1)
ESCALA_Z = 1.0  # multiplica el logit antes de tanh: < 1 = evaluación menos confiada
RASGOS_BANDO = 14
NUM_RASGOS = 17
NOMBRES_RASGOS = (
    ["puntos"] + [f"peones_avance_{k}" for k in range(8)]
    + ["con_apoyo", "libres", "avance_libres", "ataques_favorables_propios",
       "ataques_favorables_rival", "amenazas_seguras_propias", "amenazas_seguras_rival",
       "sesgo_turno"]
)
# Pesos a mano (en unidades de logit). Si existe data/pesos.npy (ajustado con
# herramientas/ajustar_pesos.py), se usan esos.
# El peón en la fila de casa vale 0.8 (no 0.5): es el defensor de la meta, y valorarlo más
# ganó 63 % ± 5 contra v4 en la arena (400 partidas).
PESOS_A_MANO = np.array(
    [1.5, 0.8, 0.5, 0.5125, 0.525, 0.55, 0.5875, 0.65, 0.75, 0.05, 0.3, 0.05, 0.0, 0.0,
     W_SEGURA, -W_SEGURA, 0.0]
)
_archivo_pesos = Path(__file__).resolve().parent / "data" / "pesos.npy"
PESOS = np.load(_archivo_pesos) if _archivo_pesos.exists() else PESOS_A_MANO
assert PESOS.shape == (NUM_RASGOS,)


def nuevos_buferes():
    """Búferes de la búsqueda: jugadas por ply, sus puntajes de orden, killers, historia y
    espacio de trabajo de la evaluación (se reserva una vez: crear arreglos en cada hoja es caro).
    """
    return (
        np.zeros((MAX_PLY, MAX_JUGADAS), np.int64),  # origenes[ply, i]
        np.zeros((MAX_PLY, MAX_JUGADAS), np.int64),  # destinos[ply, i]
        np.zeros((MAX_PLY, MAX_JUGADAS), np.float64),  # puntajes de orden
        np.full((MAX_PLY, 2), -1, np.int64),  # killers: 2 jugadas tranquilas que cortaron por ply
        np.zeros((2, CASILLAS * CASILLAS), np.float64),  # historia[color, o*81+d]
        np.zeros(NUM_RASGOS),  # rasgos de la hoja que se evalúa
        np.zeros((2, RASGOS_BANDO)),  # rasgos de cada bando (propio, rival)
        np.zeros((2, LADO), np.int64),  # por columna: fila más alta negra, más baja blanca
    )


def nueva_tabla():
    """Arreglos de la tabla de transposición (uno por campo)."""
    return (
        np.zeros(TT_TAM, np.uint64),  # clave
        np.zeros(TT_TAM, np.float64),  # valor
        np.full(TT_TAM, -1, np.int8),  # profundidad (-1 = vacía)
        np.zeros(TT_TAM, np.int8),  # tipo de cota
        np.full(TT_TAM, -1, np.int16),  # mejor jugada (origen*81 + destino)
    )


@njit(cache=False)
def hash_completo(tab, color, pts):
    h = np.uint64(0)
    for c in range(CASILLAS):
        if tab[c] != 0:
            h ^= Z_PIEZAS[tab[c] - 1, c]
    if color == 1:
        h ^= Z_TURNO
    h ^= Z_PUNTOS[0, pts[0]]
    h ^= Z_PUNTOS[1, pts[1]]
    return h


@njit(cache=False)
def apoyo(tab, casilla, color):
    """Peones propios en las dos diagonales de atrás de `casilla` (0, 1 o 2)."""
    fila = casilla // LADO - (1 if color == 0 else -1)
    if fila < 0 or fila >= LADO:
        return 0
    col = casilla % LADO
    pieza = color + 1
    total = 0
    if col > 0 and tab[fila * LADO + col - 1] == pieza:
        total += 1
    if col < LADO - 1 and tab[fila * LADO + col + 1] == pieza:
        total += 1
    return total


@njit(cache=False)
def prob_ataque(tab, color, origen, destino):
    p = 0.5 + 0.15 * (apoyo(tab, origen, color) - apoyo(tab, destino, 1 - color))
    return min(max(p, 0.2), 0.9)


@njit(cache=False)
def rasgos_bando(tab, pts, bando, max_negra, min_blanca, r):
    """Suma en r[0:RASGOS_BANDO] los rasgos de `bando` (ver NOMBRES_RASGOS).

    Cada peón cuenta en UNA de dos formas: si es amenaza segura, solo como tal; si no, por su
    avance, apoyo, si es libre, etc. Así el valor de un peón crece a lo largo de su camino y
    nunca supera al del punto que va a anotar (si lo superara, el bot no querría anotar).
    """
    r[0] += pts[bando]
    pieza = bando + 1
    rival = 2 - bando
    for c in range(CASILLAS):
        if tab[c] != pieza:
            continue
        fila = c // LADO
        col = c % LADO
        avance = fila if bando == 0 else LADO - 1 - fila
        frente = c + (LADO if bando == 0 else -LADO)
        # Amenaza segura: a un paso de la meta y sin rivales en las dos casillas que podrían
        # atacarlo. Esas casillas están en la fila inicial del rival, a la que ya no puede
        # entrar ningún peón (solo se avanza), así que es un punto garantizado.
        if avance == 7:
            atacable = (col > 0 and tab[frente - 1] == rival) or (
                col < LADO - 1 and tab[frente + 1] == rival
            )
            puede_anotar = (
                tab[frente] == 0
                or (col > 0 and tab[frente - 1] == 0)
                or (col < LADO - 1 and tab[frente + 1] == 0)
            )
            if not atacable and puede_anotar:
                r[13] += 1.0
                continue
        r[1 + avance] += 1.0
        mi_apoyo = apoyo(tab, c, bando)
        if mi_apoyo > 0:
            r[9] += 1.0
        # Libre: ningún rival en el cono por delante (se abre una columna por fila, más una de
        # holgura por los ataques en diagonal), así que nadie lo alcanza antes de la meta.
        libre = True
        for cc in range(LADO):
            if bando == 0:
                if max_negra[cc] > fila and max_negra[cc] >= fila - 1 + abs(cc - col):
                    libre = False
                    break
            elif min_blanca[cc] < fila and min_blanca[cc] <= fila + 1 - abs(cc - col):
                libre = False
                break
        if libre:
            r[10] += 1.0
            r[11] += avance
        # Ataques favorables (p ≥ 0.65) que este peón tiene disponibles ahora.
        for dc in (-1, 1):
            if 0 <= col + dc < LADO and tab[frente + dc] == rival:
                if mi_apoyo - apoyo(tab, frente + dc, 1 - bando) >= 1:
                    r[12] += 1.0


@njit(cache=False)
def rasgos(tab, pts, color, x, bandos, columnas):
    """Llena `x` (NUM_RASGOS) con los rasgos de la posición vistos desde `color` (en turno).

    x[0:12] = rasgos propios − rasgos del rival; x[12], x[13] = ataques favorables propios
    y del rival, x[14], x[15] = amenazas seguras propias y del rival (no son simétricos:
    quien está en turno actúa primero); x[16] = 1 (sesgo).
    """
    max_negra = columnas[0]  # fila más alta con peón negro, por columna
    min_blanca = columnas[1]  # fila más baja con peón blanco, por columna
    max_negra[:] = -1
    min_blanca[:] = LADO
    for c in range(CASILLAS):
        if tab[c] == 1:
            min_blanca[c % LADO] = min(min_blanca[c % LADO], c // LADO)
        elif tab[c] == 2:
            max_negra[c % LADO] = max(max_negra[c % LADO], c // LADO)
    propio = bandos[0]
    rival = bandos[1]
    propio[:] = 0.0
    rival[:] = 0.0
    rasgos_bando(tab, pts, color, max_negra, min_blanca, propio)
    rasgos_bando(tab, pts, 1 - color, max_negra, min_blanca, rival)
    for i in range(12):
        x[i] = propio[i] - rival[i]
    x[12] = propio[12]
    x[13] = rival[12]
    x[14] = propio[13]
    x[15] = rival[13]
    x[16] = 1.0


@njit(cache=False)
def evaluar(tab, pts, color, buf):
    """Valor heurístico en (-0.89, 0.89) desde el punto de vista de `color` (en turno).

    PESOS viene de una regresión logística: z = PESOS·rasgos estima log(P(ganar)/P(perder)),
    y tanh(z/2) = 2·P(ganar) − 1. Lo acotamos a 0.85 para que ganar de verdad (~1) y una
    carrera decidida (~0.89) pesen más.
    """
    x = buf[5]
    rasgos(tab, pts, color, x, buf[6], buf[7])

    # Carrera decidida: cada amenaza segura anota una por turno y nadie la puede detener.
    # Si a alguien le alcanzan para llegar a 3 antes que el otro, la partida ya está resuelta
    # (quien está en turno anota primero, así que gana los empates de turnos).
    faltan_yo = PUNTOS_PARA_GANAR - pts[color]
    faltan_rival = PUNTOS_PARA_GANAR - pts[1 - color]
    gano = x[14] >= faltan_yo
    pierdo = x[15] >= faltan_rival
    if gano and (not pierdo or faltan_yo <= faltan_rival):
        return VALOR_CARRERA - 0.01 * faltan_yo
    if pierdo and (not gano or faltan_rival < faltan_yo):
        return -(VALOR_CARRERA - 0.01 * faltan_rival)

    z = 0.0
    for i in range(NUM_RASGOS):
        z += PESOS[i] * x[i]
    return 0.85 * math.tanh(0.5 * ESCALA_Z * z)


@njit(cache=False)
def generar(tab, color, ply, origenes, destinos):
    """Jugadas legales de `color` en la fila `ply` de los búferes. Regresa cuántas."""
    pieza = color + 1
    avance = LADO if color == 0 else -LADO
    n = 0
    for c in range(CASILLAS):
        if tab[c] != pieza:
            continue
        col = c % LADO
        frente = c + avance
        if col > 0 and tab[frente - 1] != pieza:
            origenes[ply, n] = c
            destinos[ply, n] = frente - 1
            n += 1
        if tab[frente] == 0:
            origenes[ply, n] = c
            destinos[ply, n] = frente
            n += 1
        if col < LADO - 1 and tab[frente + 1] != pieza:
            origenes[ply, n] = c
            destinos[ply, n] = frente + 1
            n += 1
    return n


@njit(cache=False)
def ordenar(tab, color, ply, n, buf, jugada_tt):
    """Ordena las jugadas de la fila `ply` de mayor a menor puntaje.

    Orden: la de la tabla, anotar, ataques probables (p ≥ 0.5), killers, el resto de las
    tranquilas por historia (cuántos cortes han causado) y, entre ellos, los ataques malos.
    """
    origenes, destinos, puntajes, killers, historia = buf[0], buf[1], buf[2], buf[3], buf[4]
    meta = LADO - 1 if color == 0 else 0
    for i in range(n):
        o = origenes[ply, i]
        d = destinos[ply, i]
        fila_d = d // LADO
        if ply == 0 and fila_d == meta and tab[d] == 0:
            # En la raíz, anotar caminando va antes que todo: si empata con otra jugada,
            # cobramos ya. Si no, la búsqueda pospone el punto «para después» una y otra vez
            # (siempre cabe dentro del horizonte) mientras el rival anota.
            s = 2e9
        elif o * CASILLAS + d == jugada_tt:
            s = 1e9
        elif tab[d] != 0 and ply == 0:
            s = 1000.0 + 1000.0 * prob_ataque(tab, color, o, d)
            if fila_d == meta:
                s += 5000.0
        elif tab[d] != 0:  # ataque: los más probables primero (p=0.5 → 1000)
            s = 2000.0 * prob_ataque(tab, color, o, d)
            if fila_d == meta:
                s += 5000.0
        elif fila_d == meta:
            s = 1e6  # anotar caminando
        elif o * CASILLAS + d == killers[ply, 0]:
            s = 900.0
        elif o * CASILLAS + d == killers[ply, 1]:
            s = 850.0
        else:
            avance = fila_d if color == 0 else LADO - 1 - fila_d
            if ply == 0:
                s = avance
            else:
                s = min(historia[color, o * CASILLAS + d], 700.0) + avance
        puntajes[ply, i] = s
    # Inserción (n ≤ 54): de mayor a menor puntaje.
    for i in range(1, n):
        so = origenes[ply, i]
        sd = destinos[ply, i]
        sp = puntajes[ply, i]
        j = i - 1
        while j >= 0 and puntajes[ply, j] < sp:
            origenes[ply, j + 1] = origenes[ply, j]
            destinos[ply, j + 1] = destinos[ply, j]
            puntajes[ply, j + 1] = puntajes[ply, j]
            j -= 1
        origenes[ply, j + 1] = so
        destinos[ply, j + 1] = sd
        puntajes[ply, j + 1] = sp


@njit(cache=False)
def hacer(tab, pts, color, o, d, gana, h):
    """Aplica la jugada en sitio. Regresa (pieza que había en d, anotó, hash nuevo)."""
    pieza = color + 1
    previa = tab[d]
    anoto = False
    tab[o] = 0
    h ^= Z_PIEZAS[color, o]
    if previa != 0:
        if not gana:  # pierde el ataque: solo sale el atacante
            return previa, anoto, h ^ Z_TURNO
        tab[d] = 0
        h ^= Z_PIEZAS[1 - color, d]
    meta = LADO - 1 if color == 0 else 0
    if d // LADO == meta:
        h ^= Z_PUNTOS[color, pts[color]]
        pts[color] += 1
        h ^= Z_PUNTOS[color, pts[color]]
        anoto = True
    else:
        tab[d] = pieza
        h ^= Z_PIEZAS[color, d]
    return previa, anoto, h ^ Z_TURNO


@njit(cache=False)
def deshacer(tab, pts, color, o, d, previa, anoto):
    tab[o] = color + 1
    tab[d] = previa
    if anoto:
        pts[color] -= 1


@njit(cache=False)
def negamax(tab, pts, color, prof, ply, alfa, beta, h, buf, tt, stats):
    """Valor de la posición para `color` (en turno), con ventana (alfa, beta). Fail-soft.

    En la raíz (ply 0) además deja la mejor jugada en stats[MEJOR_RAIZ] y cuenta en
    stats[COMPLETAS] las jugadas evaluadas completas (sirve si la iteración se aborta).
    Es la única función recursiva: los ataques (nodos de azar) se resuelven aquí mismo, lo
    que además hace que numba compile bastante más rápido que con recursión mutua.
    """
    stats[NODOS] += 1
    if stats[NODOS] >= stats[LIMITE_NODOS]:
        stats[ABORTAR] = 1
    if stats[ABORTAR]:
        return 0.0
    raiz = ply == 0

    # Terminales: el rival llegó a 3 puntos, o no tenemos jugadas.
    if pts[1 - color] >= PUNTOS_PARA_GANAR:
        return -(GANAR - 0.0005 * ply)
    if prof <= -QS_PROF or ply >= MAX_PLY - 1:
        if generar(tab, color, ply, buf[0], buf[1]) == 0:
            return -(GANAR - 0.0005 * ply)
        return evaluar(tab, pts, color, buf)
    quietud = prof <= 0
    parado = -INF
    if quietud:  # podemos «quedarnos» con la evaluación estática (no hay zugzwang que temer)
        parado = evaluar(tab, pts, color, buf)
        if parado >= beta:
            return parado
        if parado > alfa:
            alfa = parado

    tt_clave, tt_valor, tt_prof, tt_tipo, tt_jugada = tt
    i_tt = int(h & np.uint64(TT_TAM - 1))
    jugada_tt = -1
    if raiz:
        jugada_tt = stats[MEJOR_RAIZ]
    elif tt_clave[i_tt] == h:
        jugada_tt = tt_jugada[i_tt]
        if tt_prof[i_tt] >= prof:
            v = tt_valor[i_tt]
            t = tt_tipo[i_tt]
            if t == EXACTO:
                return v
            if t == COTA_INF and v >= beta:
                return v
            if t == COTA_SUP and v <= alfa:
                return v

    origenes, destinos, killers, historia = buf[0], buf[1], buf[3], buf[4]
    n = generar(tab, color, ply, origenes, destinos)
    if n == 0:
        return -(GANAR - 0.0005 * ply)
    ordenar(tab, color, ply, n, buf, jugada_tt)

    alfa0 = alfa
    mejor = parado
    mejor_jugada = origenes[ply, 0] * CASILLAS + destinos[ply, 0]
    meta = LADO - 1 if color == 0 else 0
    for i in range(n):
        o = origenes[ply, i]
        d = destinos[ply, i]
        if quietud:  # en quietud solo ataques (probables) y anotaciones
            if tab[d] == 0:
                if d // LADO != meta:
                    continue
            elif prob_ataque(tab, color, o, d) < QS_PMIN:
                continue
        # PVS: tras la primera jugada, preguntamos con ventana mínima (alfa, alfa+ε) si la
        # jugada supera a la mejor. Casi nunca pasa y esa pregunta es barata; si pasa, la
        # buscamos otra vez con la ventana completa.
        nula = PVS_ACTIVO and i > 0 and beta - alfa > 2 * VENTANA_MIN
        tope = alfa + VENTANA_MIN if nula else beta
        for _intento in range(2):
            if tab[d] == 0:
                previa, anoto, h2 = hacer(tab, pts, color, o, d, True, h)
                v = -negamax(
                    tab, pts, 1 - color, prof - 1, ply + 1, -tope, -alfa, h2, buf, tt, stats
                )
                deshacer(tab, pts, color, o, d, previa, anoto)
            else:
                # Nodo de azar: p·v_gana + (1−p)·v_pierde, con poda Star1. Como todo valor
                # está en [-COTA, COTA], a veces el primer hijo basta para saber que el
                # promedio cae fuera de (alfa, tope) y nos ahorramos el segundo.
                p = prob_ataque(tab, color, o, d)
                q = 1.0 - p
                lo = max((alfa - q * COTA) / p, -COTA)
                hi = min((tope + q * COTA) / p, COTA)
                previa, anoto, h2 = hacer(tab, pts, color, o, d, True, h)
                vg = -negamax(tab, pts, 1 - color, prof - 1, ply + 1, -hi, -lo, h2, buf, tt, stats)
                deshacer(tab, pts, color, o, d, previa, anoto)
                if vg <= lo:
                    v = p * vg + q * COTA  # cota superior, ≤ alfa
                elif vg >= hi:
                    v = p * vg - q * COTA  # cota inferior, ≥ tope
                else:
                    lo2 = max((alfa - p * vg) / q, -COTA)
                    hi2 = min((tope - p * vg) / q, COTA)
                    previa, anoto, h2 = hacer(tab, pts, color, o, d, False, h)
                    vp = -negamax(
                        tab, pts, 1 - color, prof - 1, ply + 1, -hi2, -lo2, h2, buf, tt, stats
                    )
                    deshacer(tab, pts, color, o, d, previa, anoto)
                    v = p * vg + q * vp
            if not nula or v <= alfa or v >= beta or stats[ABORTAR]:
                break
            nula = False  # la jugada superó a la mejor: la buscamos con la ventana completa
            tope = beta
        if stats[ABORTAR]:
            break
        if raiz:
            stats[COMPLETAS] += 1
        if v > mejor:
            mejor = v
            mejor_jugada = o * CASILLAS + d
            if raiz:
                stats[MEJOR_RAIZ] = mejor_jugada
            if v > alfa:
                alfa = v
                if alfa >= beta:
                    if tab[d] == 0 and not quietud:  # tranquila que cortó: killer e historia
                        if killers[ply, 0] != mejor_jugada:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = mejor_jugada
                        historia[color, mejor_jugada] += prof * prof
                    break
    if stats[ABORTAR]:
        return 0.0

    if mejor <= alfa0:
        tipo = COTA_SUP
    elif mejor >= beta:
        tipo = COTA_INF
    else:
        tipo = EXACTO
    if quietud:  # lo de quietud no va a la tabla: no es una búsqueda de profundidad `prof`
        return mejor
    if prof >= tt_prof[i_tt] or tt_clave[i_tt] != h:  # reemplazo: preferimos lo más profundo
        tt_clave[i_tt] = h
        tt_valor[i_tt] = mejor
        tt_prof[i_tt] = prof
        tt_tipo[i_tt] = tipo
        tt_jugada[i_tt] = mejor_jugada
    return mejor


# --- Libro de aperturas (data/libro.npz, generado con herramientas/generar_libro.py) ---------
_archivo_libro = Path(__file__).resolve().parent / "data" / "libro.npz"
if _archivo_libro.exists():
    with np.load(_archivo_libro) as _datos:
        LIBRO = dict(zip(_datos["claves"].tolist(), _datos["jugadas"].tolist(), strict=True))
else:
    LIBRO = {}


def jugada_de_libro(tab, pts, color):
    """La jugada del libro para esta posición (o*81+d), o -1 si no está."""
    if not LIBRO:
        return -1
    return LIBRO.get(int(hash_completo(tab, color, pts)), -1)


def nuevo_contexto():
    """Todo lo que la búsqueda guarda entre jugadas de una misma partida."""
    return nuevos_buferes(), nueva_tabla(), np.zeros(5, np.int64)


def buscar(ctx, tab, pts, color, prof, jugada_previa, limite_nodos):
    """Una iteración de profundidad `prof`. Regresa (jugada, valor, completas, nodos, abortó)."""
    buf, tt, stats = ctx
    stats[:] = 0
    stats[LIMITE_NODOS] = limite_nodos
    stats[MEJOR_RAIZ] = jugada_previa
    h = np.uint64(hash_completo(tab, color, pts))  # siempre uint64: numba compila un solo tipo
    v = negamax(tab, pts, color, prof, 0, -INF, INF, h, buf, tt, stats)
    jugada = int(stats[MEJOR_RAIZ])
    return jugada, v, int(stats[COMPLETAS]), int(stats[NODOS]), bool(stats[ABORTAR])


def nueva_jugada(ctx):
    """Antes de cada jugada: la historia vieja pesa la mitad y se olvidan los killers."""
    buf = ctx[0]
    buf[3][:] = -1
    buf[4][:] *= 0.5


@njit(cache=False)
def expectimax_puro(tab, pts, color, prof, ply, buf):
    """Expectiminimax sin poda ni tabla: solo para verificar `negamax` en las pruebas."""
    if pts[1 - color] >= PUNTOS_PARA_GANAR:
        return -(GANAR - 0.0005 * ply)
    if prof <= 0:
        if generar(tab, color, 0, buf[0], buf[1]) == 0:
            return -(GANAR - 0.0005 * ply)
        return evaluar(tab, pts, color, buf)
    origenes = np.empty((1, MAX_JUGADAS), np.int64)
    destinos = np.empty((1, MAX_JUGADAS), np.int64)
    n = generar(tab, color, 0, origenes, destinos)
    if n == 0:
        return -(GANAR - 0.0005 * ply)
    mejor = -INF
    h = np.uint64(0)
    for i in range(n):
        o = origenes[0, i]
        d = destinos[0, i]
        if tab[d] == 0:
            previa, anoto, _ = hacer(tab, pts, color, o, d, True, h)
            v = -expectimax_puro(tab, pts, 1 - color, prof - 1, ply + 1, buf)
            deshacer(tab, pts, color, o, d, previa, anoto)
        else:
            p = prob_ataque(tab, color, o, d)
            previa, anoto, _ = hacer(tab, pts, color, o, d, True, h)
            vg = -expectimax_puro(tab, pts, 1 - color, prof - 1, ply + 1, buf)
            deshacer(tab, pts, color, o, d, previa, anoto)
            previa, anoto, _ = hacer(tab, pts, color, o, d, False, h)
            vp = -expectimax_puro(tab, pts, 1 - color, prof - 1, ply + 1, buf)
            deshacer(tab, pts, color, o, d, previa, anoto)
            v = p * vg + (1.0 - p) * vp
        mejor = max(mejor, v)
    return mejor
