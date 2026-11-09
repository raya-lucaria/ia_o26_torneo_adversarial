"""Motor rápido del bot «pamplinotas»: reglas, evaluación y búsqueda compiladas con numba.

Es una reescritura de `sistema/motor/nucleo.py` (lo permite docs/avanzado.md, «Modificar el
motor») pensada para buscar millones de nodos por segundo. El árbitro sigue revisando cada
jugada con el motor oficial.

Representación (la misma que el motor oficial):
- `tab`: int8[81], casilla = fila*9 + col (fila 0 = fila «1»). 0 vacía, 1 blanca, 2 negra.
- `color`: 0 Blancas, 1 Negras. `pts`: int64[2].
- Una jugada se codifica como un entero `origen*81 + destino`.

Algoritmo (clase 3 de la unidad de juegos):
- Expectiminimax con corte: MAX y MIN eligen; cada ataque es un nodo de AZAR que promedia sus
  dos resultados con `p` y `1-p`. Se escribe en forma negamax: el valor siempre es desde el
  jugador que mueve, y al subir se cambia el signo.
- Alfa-beta en los nodos de decisión y poda *-minimax (Star1, Ballard 1983) en los de azar:
  cada resultado del dado recibe una ventana propia a partir de la cota de los valores del
  subárbol (M si cabe un final; si no, la cota de la EVAL, que es la que de verdad poda).
- Tabla de transposición con valor / piso / techo, jugada de la tabla primero, killers e
  historial para ordenar jugadas; búsqueda de quietud con stand pat que sigue las jugadas
  que anotan seguro y los ataques favorables.
- Los finales valen ±M (M = 1 000 000), muy por encima de cualquier EVAL, y un final más
  cercano vale un poco más (M − ply) para preferir ganar pronto.
"""

import time
import warnings

import numpy as np
from numba import NumbaWarning, njit, objmode

warnings.filterwarnings("ignore", category=NumbaWarning, message=".*object mode.*")  # el objmode del reloj

LADO = 9
CASILLAS = 81
PEONES = 18
MAX_JUGADAS = PEONES * 3
PUNTOS_PARA_GANAR = 3
MAX_PLY = 64
M = 1_000_000.0  # escala de los finales: |EVAL| < M siempre
VENTANA_NULA = 1e-3  # ancho de la ventana «nula» de PVS; los valores son reales, no enteros
TT_BITS = 20
TT_TAM = 1 << TT_BITS
TT_EXACTO, TT_PISO, TT_TECHO = 0, 1, 2

# Rasgos de la evaluación. EVAL(s) = ESCALA_V·tanh(k·Σ w[i]·f[i](s)/2), con f medido desde quien mueve
# (rasgo propio menos rasgo rival). Los pesos se ajustan offline (herramientas/ajustar_pesos.py).
F_TABLA = 0  # 7 filas de avance × 5 columnas (espejo): cuántos peones hay en cada casilla «tipo»
F_APOYO1 = 35  # peones con apoyo 1
F_APOYO2 = 36  # peones con apoyo 2
F_AMENAZA = 37  # suma de la mejor p con que el rival puede atacar a cada peón
F_AMENAZA_GRAVE = 38  # peones atacables con p ≥ 0.65
F_CORREDOR = 39  # 7 entradas: corredores (nadie puede detenerlos) por fila de avance
F_PUNTOS = 46  # puntos propios menos rivales
F_DOS_PUNTOS = 47  # tengo 2 puntos (a uno de ganar)
F_RIVAL_DOS = 48  # el rival tiene 2 puntos
F_MOVILIDAD = 49  # jugadas legales propias menos rivales
F_ATAQUES = 50  # Σ (p − 0.5) de mis ataques posibles; los del rival son los mismos pares con 1 − p
F_TEMPO = 51  # siempre 1: el intercepto del modelo (todo rasgo ya es desde quien mueve)
F_COLGADO = 52  # peones avanzados (fila de avance ≥ 4) sin apoyo y atacables con p ≥ 0.5
F_CARRERA_GANO = 53  # mi corredor más cercano corona antes que el suyo (o igual: muevo yo)
F_CARRERA_PIERDO = 54  # su corredor más cercano corona antes que el mío
F_CARRERA_MARGEN = 55  # (su distancia − la mía) / 3, recortado a ±1, si los dos tienen corredor
F_FALANGE = 56  # pares de peones propios lado a lado
F_CADENA = 57  # peones con un apoyo que a su vez tiene apoyo
F_MATERIAL = 58  # peones propios menos rivales
F_SEGURO = 59  # 3 entradas: peones en la mitad rival (filas de avance 4, 5, 6) que nadie puede atacar ahora
F_MITAD_RIVAL = 62  # peones en la mitad rival
F_CASA_SEGUNDA = 63  # peones en la segunda fila de casa (la tabla junta las dos filas de casa)
F_CASI_CORREDOR = 64  # 3 entradas: peones (filas de avance 3, 4, 5) con un solo rival que los puede frenar
F_DOS_Y_CERCA = 67  # tengo 2 puntos y un peón a 3 filas o menos de anotar, menos lo mismo del rival
N_BASE = 68  # rasgos del tablero
F_FASE = N_BASE  # cada rasgo base se repite multiplicado por la fase: 0 al inicio, 1 con el tablero vacío
N_RASGOS = 2 * N_BASE
SIN_CORREDOR = 99
I_ESCALA_LOGIT = N_RASGOS  # w[I_ESCALA_LOGIT]: logit por punto de la suma lineal (0 = EVAL lineal)
ESCALA_V = 1000.0  # EVAL = ESCALA_V · V, con V = 2·P(ganar) − 1 en [−1, 1]

# Estadísticas de la búsqueda (arreglo `stats`).
ST_NODOS, ST_ABORTADO, ST_SIGUIENTE_RELOJ = 0, 1, 2
ST_QUIETUD_ATAQUES, ST_PROF_QUIETUD = 3, 4  # opciones de la quietud (se fijan desde Buscador)
ST_LMR = 5  # reducción de jugadas tardías: desde qué jugada del orden se reduce (0 = apagada)
LMR_DESDE_JUGADA = 3  # las tres primeras jugadas del orden nunca se reducen
LMR_PROF_MIN = 3


def pesos_iniciales() -> np.ndarray:
    """Pesos de partida (a mano, en centésimas de peón); los buenos salen de data/pesos.json."""
    w = np.zeros(N_RASGOS + 1, np.float64)
    avance = [0.0, 4.0, 10.0, 18.0, 30.0, 48.0, 80.0]
    centro = [0.0, 2.0, 4.0, 6.0, 7.0]
    for fila in range(7):
        for col in range(5):
            w[F_TABLA + fila * 5 + col] = 100.0 + avance[fila] + centro[col]
    w[F_APOYO1] = 14.0
    w[F_APOYO2] = 28.0
    w[F_AMENAZA] = -55.0
    w[F_AMENAZA_GRAVE] = -20.0
    for fila in range(7):
        w[F_CORREDOR + fila] = 240.0 * (fila + 1) / 7.0
    w[F_PUNTOS] = 320.0
    w[F_DOS_PUNTOS] = 220.0
    w[F_RIVAL_DOS] = -220.0
    w[F_MOVILIDAD] = 2.0
    w[F_ATAQUES] = 30.0
    w[F_TEMPO] = 12.0
    return w


# --- Zobrist (mismos números que el motor oficial: `estado.clave` coincide) -----------------


def _splitmix64(semilla: int, cuantos: int) -> np.ndarray:
    mascara = (1 << 64) - 1
    x = semilla
    salida = []
    for _ in range(cuantos):
        x = (x + 0x9E3779B97F4A7C15) & mascara
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mascara
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mascara
        salida.append(z ^ (z >> 31))
    return np.array(salida, dtype=np.uint64)


_Z = _splitmix64(20261018, 2 * CASILLAS + 1 + 2 * (PUNTOS_PARA_GANAR + 1))
Z_PIEZAS = _Z[: 2 * CASILLAS].reshape(2, CASILLAS)
Z_TURNO = _Z[2 * CASILLAS]
Z_PUNTOS = _Z[2 * CASILLAS + 1 :].reshape(2, PUNTOS_PARA_GANAR + 1)


# --- Reglas ---------------------------------------------------------------------------------


@njit(cache=False, nogil=True)
def paso(color):
    return 1 if color == 0 else -1


@njit(cache=False, nogil=True)
def apoyo(tab, casilla, color):
    """Peones propios en las dos diagonales de atrás (0, 1 o 2)."""
    fila = casilla // LADO - paso(color)
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


@njit(cache=False, nogil=True)
def prob_combate(apoyo_atacante, apoyo_defensor):
    p = 0.5 + 0.15 * (apoyo_atacante - apoyo_defensor)
    return min(max(p, 0.2), 0.9)


@njit(cache=False, nogil=True)
def generar(tab, color, jugadas):
    """Llena `jugadas` (origen*81+destino) con las legales y regresa cuántas son."""
    pieza = color + 1
    avance = paso(color) * LADO
    n = 0
    for c in range(CASILLAS):
        if tab[c] != pieza:
            continue
        col = c % LADO
        frente = c + avance
        if col > 0 and tab[frente - 1] != pieza:
            jugadas[n] = c * CASILLAS + frente - 1
            n += 1
        if tab[frente] == 0:
            jugadas[n] = c * CASILLAS + frente
            n += 1
        if col < LADO - 1 and tab[frente + 1] != pieza:
            jugadas[n] = c * CASILLAS + frente + 1
            n += 1
    return n


@njit(cache=False, nogil=True)
def clave_zobrist(tab, color, pts):
    h = np.uint64(0)
    for c in range(CASILLAS):
        if tab[c] != 0:
            h ^= Z_PIEZAS[tab[c] - 1, c]
    if color == 1:
        h ^= Z_TURNO
    h ^= Z_PUNTOS[0, pts[0]]
    h ^= Z_PUNTOS[1, pts[1]]
    return h


# --- Evaluación -----------------------------------------------------------------------------


@njit(cache=False, nogil=True)
def tiene_jugada(tab, casilla, color):
    """True si el peón de `color` en `casilla` puede mover (recto a vacía o diagonal a no propia)."""
    pieza = color + 1
    frente = casilla + paso(color) * LADO
    if frente < 0 or frente >= CASILLAS:
        return False
    col = casilla % LADO
    if tab[frente] == 0:
        return True
    if col > 0 and tab[frente - 1] != pieza:
        return True
    return col < LADO - 1 and tab[frente + 1] != pieza


@njit(cache=False, nogil=True)
def tiene_jugadas(tab, color):
    """True si `color` tiene al menos una jugada legal (si no, pierde: es un final)."""
    pieza = color + 1
    for c in range(CASILLAS):
        if tab[c] == pieza and tiene_jugada(tab, c, color):
            return True
    return False


@njit(cache=False, nogil=True)
def rasgos(tab, color, pts, f):
    """Llena `f` con los rasgos desde el punto de vista de `color` (propio − rival).

    Es la función más cara de la búsqueda (la llama cada hoja), así que recorre el tablero sólo
    dos veces: la primera guarda el apoyo de cada peón y, por columna, hasta qué fila llega cada
    bando; la segunda saca todos los rasgos de cada peón con esas tablas."""
    for i in range(N_RASGOS):
        f[i] = 0.0
    apoyos = np.zeros(CASILLAS, np.int8)
    for c in range(CASILLAS):
        v = tab[c]
        if v == 0:
            continue
        apoyos[c] = apoyo(tab, c, v - 1)
    # Rivales por delante en cada columna: acumulado de peones de cada color por fila, por columna.
    acum = np.zeros((2, LADO, LADO + 1), np.int64)  # acum[k, col, fila + 1] = peones de k en esa columna con fila ≤ fila
    for col in range(LADO):
        for fila in range(LADO):
            v = tab[fila * LADO + col]
            acum[0, col, fila + 1] = acum[0, col, fila] + (1 if v == 1 else 0)
            acum[1, col, fila + 1] = acum[1, col, fila] + (1 if v == 2 else 0)
    llegada = np.full(2, SIN_CORREDOR, np.int64)  # filas que le faltan al corredor más cercano de cada bando
    mas_avanzado = np.full(2, -1, np.int64)
    for c in range(CASILLAS):
        v = tab[c]
        if v == 0:
            continue
        k = v - 1
        pieza_rival = 2 - k
        signo = 1.0 if k == color else -1.0
        fila = c // LADO
        col = c % LADO
        col_espejo = col if col <= 4 else 8 - col
        # Fila de avance 0..6: las dos filas de casa comparten la entrada 0 (un peón que aún no
        # salió de casa vale lo mismo esté en la primera o en la segunda fila).
        avanzadas = fila - 1 if k == 0 else (LADO - 2) - fila
        if avanzadas < 0:
            avanzadas = 0
        if (k == 0 and fila == 1) or (k == 1 and fila == LADO - 2):
            f[F_CASA_SEGUNDA] += signo
        if avanzadas > mas_avanzado[k]:
            mas_avanzado[k] = avanzadas
        f[F_TABLA + avanzadas * 5 + col_espejo] += signo
        f[F_MATERIAL] += signo
        if avanzadas >= 4:
            f[F_MITAD_RIVAL] += signo
        if col < LADO - 1 and tab[c + 1] == v:
            f[F_FALANGE] += signo
        a = apoyos[c]
        if a > 0:
            atras = c - paso(k) * LADO
            if (col > 0 and tab[atras - 1] == v and apoyos[atras - 1] > 0) or (
                col < LADO - 1 and tab[atras + 1] == v and apoyos[atras + 1] > 0
            ):
                f[F_CADENA] += signo
        if a == 1:
            f[F_APOYO1] += signo
        elif a == 2:
            f[F_APOYO2] += signo
        # Amenaza: la mayor p con que un rival de la fila de enfrente podría atacar a este peón.
        frente = c + paso(k) * LADO
        if 0 <= frente < CASILLAS:
            p = 0.0
            n_jugadas = 0
            if col > 0:
                if tab[frente - 1] == pieza_rival:
                    p = max(p, prob_combate(apoyos[frente - 1], a))
                    if k == color:
                        f[F_ATAQUES] += prob_combate(a, apoyos[frente - 1]) - 0.5
                if tab[frente - 1] != v:
                    n_jugadas += 1
            if col < LADO - 1:
                if tab[frente + 1] == pieza_rival:
                    p = max(p, prob_combate(apoyos[frente + 1], a))
                    if k == color:
                        f[F_ATAQUES] += prob_combate(a, apoyos[frente + 1]) - 0.5
                if tab[frente + 1] != v:
                    n_jugadas += 1
            if tab[frente] == 0:
                n_jugadas += 1
            f[F_AMENAZA] += signo * p
            if p >= 0.65:
                f[F_AMENAZA_GRAVE] += signo
            if p >= 0.5 and a == 0 and avanzadas >= 4:
                f[F_COLGADO] += signo
            if p == 0.0 and avanzadas >= 4:
                f[F_SEGURO + avanzadas - 4] += signo
            f[F_MOVILIDAD] += signo * n_jugadas
        # Corredor: ningún rival más adelante en su columna ni en las vecinas; casi corredor: uno solo.
        frenos = 0
        for cc in range(max(0, col - 1), min(LADO, col + 2)):
            if k == 0:
                frenos += acum[1, cc, LADO] - acum[1, cc, fila + 1]
            else:
                frenos += acum[0, cc, fila]
        if frenos == 1 and 3 <= avanzadas <= 5:
            f[F_CASI_CORREDOR + avanzadas - 3] += signo
        if frenos == 0:
            f[F_CORREDOR + avanzadas] += signo
            faltan = (LADO - 1 - fila) if k == 0 else fila
            if faltan < llegada[k]:
                llegada[k] = faltan
    # Carrera: quién corona primero con lo que ya no se puede frenar (en un empate corono yo, que muevo).
    mia = llegada[color]
    suya = llegada[1 - color]
    if mia != SIN_CORREDOR and mia <= suya:
        f[F_CARRERA_GANO] = 1.0
    if suya != SIN_CORREDOR and suya < mia:
        f[F_CARRERA_PIERDO] = 1.0
    if mia != SIN_CORREDOR and suya != SIN_CORREDOR:
        f[F_CARRERA_MARGEN] = max(-1.0, min(1.0, (suya - mia) / 3.0))
    if pts[color] == 2 and mas_avanzado[color] >= 4:
        f[F_DOS_Y_CERCA] += 1.0
    if pts[1 - color] == 2 and mas_avanzado[1 - color] >= 4:
        f[F_DOS_Y_CERCA] -= 1.0
    f[F_PUNTOS] = pts[color] - pts[1 - color]
    f[F_DOS_PUNTOS] = 1.0 if pts[color] == 2 else 0.0
    f[F_RIVAL_DOS] = 1.0 if pts[1 - color] == 2 else 0.0
    f[F_TEMPO] = 1.0
    # Fase de la partida: el mismo rasgo puede valer distinto en la apertura que en el final
    # (una carrera de corredores casi no importa con el tablero lleno y lo decide todo al final).
    piezas = 0
    for c in range(CASILLAS):
        if tab[c] != 0:
            piezas += 1
    fase = 1.0 - piezas / (2.0 * PEONES)
    for i in range(N_BASE):
        f[F_FASE + i] = f[i] * fase


@njit(cache=False, nogil=True)
def evaluar(tab, color, pts, w):
    """EVAL desde quien mueve: suma ponderada de rasgos, llevada a la escala de V.

    La regresión logística da P(ganar) = σ(k·Σ wᵢfᵢ); EVAL = ESCALA_V·(2P − 1) = ESCALA_V·tanh(k·Σ/2).
    Así los nodos de AZAR promedian probabilidades de ganar (como TD-Gammon) y no centésimas de
    peón: el bot no arriesga de más cuando va ganando ni de menos cuando va perdiendo."""
    f = np.empty(N_RASGOS, np.float64)
    rasgos(tab, color, pts, f)
    total = 0.0
    for i in range(N_RASGOS):
        total += w[i] * f[i]
    k = w[I_ESCALA_LOGIT]
    if k == 0.0:
        return total
    return ESCALA_V * np.tanh(0.5 * k * total)


# --- Búsqueda -------------------------------------------------------------------------------


@njit(cache=False, nogil=True)
def reloj_vencido(stats, limite_s):
    """Revisa el reloj cada 4096 nodos (objmode: numba no sabe leer el reloj)."""
    stats[ST_NODOS] += 1
    if stats[ST_ABORTADO] != 0:
        return True
    if stats[ST_NODOS] >= stats[ST_SIGUIENTE_RELOJ]:
        stats[ST_SIGUIENTE_RELOJ] = stats[ST_NODOS] + 4096
        with objmode(ahora="f8"):
            ahora = time.perf_counter()
        if ahora >= limite_s:
            stats[ST_ABORTADO] = 1
            return True
    return False


@njit(cache=False, nogil=True)
def puntuar_jugadas(tab, color, jugadas, n, puntajes, jugada_tt, killers, ply, historial):
    """Orden de exploración: jugada de la tabla, anotar, ataques por p, killers, historial."""
    meta = LADO - 1 if color == 0 else 0
    for i in range(n):
        j = jugadas[i]
        o = j // CASILLAS
        d = j % CASILLAS
        s = 0.0
        if j == jugada_tt:
            s = 1e9
        elif tab[d] != 0:
            p = prob_combate(apoyo(tab, o, color), apoyo(tab, d, 1 - color))
            s = 1e6 + p * 1e5
            if d // LADO == meta:
                s += 5e5
        elif d // LADO == meta:
            s = 2e6
        elif j == killers[ply, 0] or j == killers[ply, 1]:
            s = 5e5
        else:
            s = historial[color, j]
        puntajes[i] = s


@njit(cache=False, nogil=True)
def cota_valor(tab, pts, plies):
    """Cota U del subárbol de `plies` plies: M si algún final puede aparecer, ESCALA_V si no.

    Star1 poda con la cota de los valores que faltan. Con U = M (hay finales) la ventana estirada
    es casi infinita y no poda; pero un final sólo ocurre si alguien llega a 3 puntos o se queda
    sin peones (el peón más adelantado siempre tiene jugada: sin jugadas ⇔ sin peones). Un
    jugador con `pts` puntos necesita anotar 3 − pts veces, y cada anotación cuesta al menos
    tantos turnos propios como filas le faltan al peón: si sus 3 − pts peones más cercanos no
    suman menos filas que los turnos que caben en `plies`, no puede ganar en el subárbol. Si nada
    de esto cabe, todo valor del subárbol es una EVAL, |EVAL| < ESCALA_V."""
    peones = np.zeros(2, np.int64)
    cercanos = np.full((2, PUNTOS_PARA_GANAR), LADO, np.int64)  # distancias menores por color
    for c in range(CASILLAS):
        v = tab[c]
        if v == 0:
            continue
        k = v - 1
        peones[k] += 1
        fila = c // LADO
        d = (LADO - 1 - fila) if k == 0 else fila
        for i in range(PUNTOS_PARA_GANAR):  # inserción ordenada entre las tres menores
            if d < cercanos[k, i]:
                for j in range(PUNTOS_PARA_GANAR - 1, i, -1):
                    cercanos[k, j] = cercanos[k, j - 1]
                cercanos[k, i] = d
                break
    turnos = (plies + 1) // 2
    for k in range(2):
        if peones[k] <= plies:
            return M
        faltan = PUNTOS_PARA_GANAR - pts[k]
        if peones[k] >= faltan:
            suma = 0
            for i in range(faltan):
                suma += cercanos[k, i]
            if suma <= turnos:
                return M
    return ESCALA_V


@njit(cache=False, nogil=True)
def ataque_en_quietud(tab, color, pts, o, d, anota, gana, alfa, beta, ply, w, stats, limite_s, prof_q):
    """Aplica un resultado del dado dentro de la quietud, sigue buscando y deshace."""
    pieza = color + 1
    rival = 1 - color
    tab[o] = 0
    if gana:
        tab[d] = 0
        if anota:
            pts[color] += 1
        else:
            tab[d] = pieza
    if gana and anota and pts[color] >= PUNTOS_PARA_GANAR:
        v = -(M - ply)  # desde el rival: perdió
    else:
        v = quietud(tab, rival, pts, alfa, beta, ply, w, stats, limite_s, prof_q)
    if gana:
        if anota:
            pts[color] -= 1
        tab[d] = rival + 1
    tab[o] = pieza
    return v


@njit(cache=False, nogil=True)
def quietud(tab, color, pts, alfa, beta, ply, w, stats, limite_s, prof_q):
    """Búsqueda de quietud (Definición 7.3.2.2): en el nodo de corte, quien mueve elige entre
    «quedarse» con su EVAL (stand pat) o seguir una jugada que cambia EVAL de golpe: anotar un
    punto caminando o atacar (un nodo de azar, promediado con p y 1−p, con poda Star1).
    Sólo se siguen ataques con p ≥ 0.5 (los demás pierden material en esperanza)."""
    if reloj_vencido(stats, limite_s):
        return 0.0
    if pts[1 - color] >= PUNTOS_PARA_GANAR or not tiene_jugadas(tab, color):
        return -(M - ply)
    valor = evaluar(tab, color, pts, w)
    if prof_q <= 0 or valor >= beta:
        return valor
    if valor > alfa:
        alfa = valor
    meta = LADO - 1 if color == 0 else 0
    pieza = color + 1
    rival = 1 - color
    avance = paso(color) * LADO
    con_ataques = stats[ST_QUIETUD_ATAQUES] != 0
    for o in range(CASILLAS):
        if tab[o] != pieza:
            continue
        col = o % LADO
        frente = o + avance
        if frente < 0 or frente >= CASILLAS:
            continue
        anota = frente // LADO == meta
        for giro in range(-1, 2):
            if (giro < 0 and col == 0) or (giro > 0 and col == LADO - 1):
                continue
            d = frente + giro
            if giro == 0:
                if tab[d] != 0 or not anota:
                    continue  # recto sólo interesa si anota seguro
                tab[o] = 0
                pts[color] += 1
                if pts[color] >= PUNTOS_PARA_GANAR:
                    v = M - ply
                else:
                    v = -quietud(tab, rival, pts, -beta, -alfa, ply + 1, w, stats, limite_s, prof_q - 1)
                pts[color] -= 1
                tab[o] = pieza
            else:
                if tab[d] != rival + 1 or not con_ataques:
                    continue
                p = prob_combate(apoyo(tab, o, color), apoyo(tab, d, rival))
                if p < 0.5 and not anota:
                    continue  # un ataque desfavorable que no anota pierde material en esperanza
                gana_primero = p >= 0.5
                p1 = p if gana_primero else 1.0 - p
                p2 = 1.0 - p1
                U = cota_valor(tab, pts, prof_q)
                a1 = (alfa - p2 * U) / p1
                b1 = (beta + p2 * U) / p1
                w1 = -ataque_en_quietud(tab, color, pts, o, d, anota, gana_primero, -b1, -a1, ply + 1, w, stats, limite_s, prof_q - 1)
                if w1 <= a1:
                    v = p1 * w1 + p2 * U
                elif w1 >= b1:
                    v = p1 * w1 - p2 * U
                else:
                    a2 = (alfa - p1 * w1) / p2
                    b2 = (beta - p1 * w1) / p2
                    w2 = -ataque_en_quietud(tab, color, pts, o, d, anota, not gana_primero, -b2, -a2, ply + 1, w, stats, limite_s, prof_q - 1)
                    v = p1 * w1 + p2 * w2
            if stats[ST_ABORTADO] != 0:
                return 0.0
            if v > valor:
                valor = v
                if v > alfa:
                    alfa = v
                    if alfa >= beta:
                        return valor
    return valor


@njit(cache=False, nogil=True)
def ataque_resuelto(
    tab, color, pts, h, o, d, anota, gana, prof, alfa, beta, ply, w, jugadas_buf, puntajes_buf,
    tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
):
    """Aplica un resultado del dado (gana o pierde el atacante), busca el hijo y deshace."""
    pieza = color + 1
    rival = 1 - color
    tab[o] = 0
    h2 = h ^ Z_PIEZAS[color, o] ^ Z_TURNO
    if gana:
        tab[d] = 0
        h2 ^= Z_PIEZAS[rival, d]
        if anota:
            h2 ^= Z_PUNTOS[color, pts[color]]
            pts[color] += 1
            h2 ^= Z_PUNTOS[color, pts[color]]
        else:
            tab[d] = pieza
            h2 ^= Z_PIEZAS[color, d]
    v = negamax(
        tab, rival, pts, h2, prof, alfa, beta, ply, w, jugadas_buf, puntajes_buf,
        tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
    )
    if gana:
        if anota:
            pts[color] -= 1
        tab[d] = rival + 1
    tab[o] = pieza
    return v


@njit(cache=False, nogil=True)
def valor_jugada(
    tab, color, pts, h, j, prof, alfa, beta, ply, w, jugadas_buf, puntajes_buf,
    tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
):
    """Valor de una jugada desde quien mueve: un hijo si avanza, un nodo de AZAR si ataca."""
    pieza = color + 1
    rival = 1 - color
    meta = LADO - 1 if color == 0 else 0
    o = j // CASILLAS
    d = j % CASILLAS
    anota = d // LADO == meta
    if tab[d] == 0:
        # --- jugada sin dado: un solo hijo ---
        tab[o] = 0
        h2 = h ^ Z_PIEZAS[color, o] ^ Z_TURNO
        if anota:
            h2 ^= Z_PUNTOS[color, pts[color]]
            pts[color] += 1
            h2 ^= Z_PUNTOS[color, pts[color]]
        else:
            tab[d] = pieza
            h2 ^= Z_PIEZAS[color, d]
        v = -negamax(
            tab, rival, pts, h2, prof - 1, -beta, -alfa, ply + 1, w, jugadas_buf, puntajes_buf,
            tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
        )
        if anota:
            pts[color] -= 1
        else:
            tab[d] = 0
        tab[o] = pieza
    else:
        # --- ataque: nodo de AZAR con dos resultados, p (gana) y 1-p (pierde) ---
        p = prob_combate(apoyo(tab, o, color), apoyo(tab, d, rival))
        q = 1.0 - p
        # Star1 (Ballard): el resultado que falta está en [-U, U], así que el primero se busca
        # con una ventana estirada. Se mira primero el resultado más probable: pesa más en el
        # promedio y por eso decide antes si el nodo entero queda fuera de [alfa, beta].
        gana_primero = p >= 0.5
        p1 = p if gana_primero else q
        p2 = 1.0 - p1
        U = cota_valor(tab, pts, prof + stats[ST_PROF_QUIETUD])
        a1 = (alfa - p2 * U) / p1
        b1 = (beta + p2 * U) / p1
        w1 = -ataque_resuelto(
            tab, color, pts, h, o, d, anota, gana_primero, prof - 1, -b1, -a1, ply + 1, w,
            jugadas_buf, puntajes_buf, tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada,
            killers, historial, stats, limite_s, mejor_raiz,
        )
        if stats[ST_ABORTADO] != 0:
            return 0.0
        if w1 <= a1:
            v = p1 * w1 + p2 * U  # cota superior: ya no supera alfa
        elif w1 >= b1:
            v = p1 * w1 - p2 * U  # cota inferior: ya supera beta
        else:
            a2 = (alfa - p1 * w1) / p2
            b2 = (beta - p1 * w1) / p2
            w2 = -ataque_resuelto(
                tab, color, pts, h, o, d, anota, not gana_primero, prof - 1, -b2, -a2, ply + 1, w,
                jugadas_buf, puntajes_buf, tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada,
                killers, historial, stats, limite_s, mejor_raiz,
            )
            v = p1 * w1 + p2 * w2
    return v


@njit(cache=False, nogil=True)
def negamax(
    tab, color, pts, h, prof, alfa, beta, ply, w, jugadas_buf, puntajes_buf,
    tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
):
    """Expectiminimax con corte, alfa-beta y Star1, en forma negamax (valor desde quien mueve)."""
    if reloj_vencido(stats, limite_s):
        return 0.0
    if pts[1 - color] >= PUNTOS_PARA_GANAR:
        return -(M - ply)
    alfa_original = alfa
    idx = np.int64(h & np.uint64(TT_TAM - 1))
    jugada_tt = -1
    if tt_clave[idx] == h:
        jugada_tt = tt_jugada[idx]
        v = tt_valor[idx]
        # Una entrada más profunda sólo sirve si es una EVAL: un final visto más allá del horizonte
        # rompería la cota con que Star1 poda en el nodo de arriba.
        if ply > 0 and (tt_prof[idx] == prof or (tt_prof[idx] > prof and abs(v) <= ESCALA_V)):
            f = tt_flag[idx]
            if f == TT_EXACTO:
                return v
            if f == TT_PISO and v > alfa:
                alfa = v
            elif f == TT_TECHO and v < beta:
                beta = v
            if alfa >= beta:
                return v
    if prof <= 0:
        return quietud(tab, color, pts, alfa, beta, ply, w, stats, limite_s, stats[ST_PROF_QUIETUD])
    jugadas = jugadas_buf[ply]
    n = generar(tab, color, jugadas)
    if n == 0:
        return -(M - ply)  # sin jugadas: pierde quien mueve
    puntajes = puntajes_buf[ply]
    puntuar_jugadas(tab, color, jugadas, n, puntajes, jugada_tt, killers, ply, historial)
    pieza = color + 1
    rival = 1 - color
    meta = LADO - 1 if color == 0 else 0
    mejor = -np.inf
    mejor_jugada = -1
    for i in range(n):
        # Selección: la jugada con mayor puntaje que falta (ordenamiento perezoso).
        k = i
        for t in range(i + 1, n):
            if puntajes[t] > puntajes[k]:
                k = t
        if k != i:
            jugadas[i], jugadas[k] = jugadas[k], jugadas[i]
            puntajes[i], puntajes[k] = puntajes[k], puntajes[i]
        j = jugadas[i]
        d = j % CASILLAS
        if i == 0:
            v = valor_jugada(
                tab, color, pts, h, j, prof, alfa, beta, ply, w, jugadas_buf, puntajes_buf,
                tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
            )
        else:
            # Reducción de jugadas tardías: una jugada tranquila, lejos de la meta y mal ordenada
            # se mira primero con un nivel menos; sólo si supera alfa se busca completa.
            fila_d = d // LADO
            avance = fila_d if color == 0 else LADO - 1 - fila_d
            reducir = (
                stats[ST_LMR] != 0 and prof >= LMR_PROF_MIN and i >= stats[ST_LMR]
                and tab[d] == 0 and avance < LADO - 3
            )
            v = alfa + 1.0
            if reducir:
                v = valor_jugada(
                    tab, color, pts, h, j, prof - 1, alfa, alfa + VENTANA_NULA, ply, w, jugadas_buf, puntajes_buf,
                    tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
                )
                if stats[ST_ABORTADO] != 0:
                    return 0.0
            # Variación principal: las demás jugadas solo tienen que demostrar que no superan alfa
            # (ventana nula); si lo logran, se repite con la ventana completa para saber su valor.
            if v > alfa:
                v = valor_jugada(
                    tab, color, pts, h, j, prof, alfa, alfa + VENTANA_NULA, ply, w, jugadas_buf, puntajes_buf,
                    tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
                )
            if stats[ST_ABORTADO] == 0 and v > alfa and v < beta:
                v = valor_jugada(
                    tab, color, pts, h, j, prof, alfa, beta, ply, w, jugadas_buf, puntajes_buf,
                    tt_clave, tt_valor, tt_prof, tt_flag, tt_jugada, killers, historial, stats, limite_s, mejor_raiz,
                )
        if stats[ST_ABORTADO] != 0:
            return 0.0
        if v > mejor:
            mejor = v
            mejor_jugada = j
            if ply == 0:
                mejor_raiz[0] = j
            if v > alfa:
                alfa = v
                if alfa >= beta:
                    if tab[d] == 0:
                        if killers[ply, 0] != j:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = j
                        historial[color, j] += prof * prof
                    break
    # Guardar en la tabla de transposición (reemplazo simple).
    if mejor <= alfa_original:
        flag = TT_TECHO
    elif mejor >= beta:
        flag = TT_PISO
    else:
        flag = TT_EXACTO
    tt_clave[idx] = h
    tt_valor[idx] = mejor
    tt_prof[idx] = prof
    tt_flag[idx] = flag
    tt_jugada[idx] = mejor_jugada
    return mejor


class Buscador:
    """Guarda las tablas entre turnos y corre la profundización iterativa."""

    def __init__(self, pesos: np.ndarray | None = None, quietud_ataques: bool = True, prof_quietud: int = 4, lmr: int | None = None) -> None:
        self.w = pesos_iniciales() if pesos is None else np.asarray(pesos, np.float64)
        if len(self.w) != N_RASGOS + 1:
            raise ValueError(f"se esperaban {N_RASGOS} pesos y la escala logística; llegaron {len(self.w)} valores")
        self.quietud_ataques = quietud_ataques
        self.prof_quietud = prof_quietud
        self.lmr = LMR_DESDE_JUGADA if lmr is None else lmr
        self.tt_clave = np.zeros(TT_TAM, np.uint64)
        self.tt_valor = np.zeros(TT_TAM, np.float64)
        self.tt_prof = np.full(TT_TAM, -1, np.int8)
        self.tt_flag = np.zeros(TT_TAM, np.int8)
        self.tt_jugada = np.full(TT_TAM, -1, np.int32)
        self.killers = np.full((MAX_PLY, 2), -1, np.int32)
        self.historial = np.zeros((2, CASILLAS * CASILLAS), np.float64)
        self.jugadas_buf = np.zeros((MAX_PLY, MAX_JUGADAS), np.int32)
        self.puntajes_buf = np.zeros((MAX_PLY, MAX_JUGADAS), np.float64)
        self.stats = np.zeros(6, np.int64)
        self.mejor_raiz = np.full(1, -1, np.int64)

    def buscar(self, tab, color, pts, prof, limite_s):
        """Una búsqueda a profundidad fija. Regresa (jugada, valor, abortada)."""
        self.stats[:] = 0
        self.stats[ST_QUIETUD_ATAQUES] = 1 if self.quietud_ataques else 0
        self.stats[ST_PROF_QUIETUD] = self.prof_quietud
        self.stats[ST_LMR] = self.lmr
        self.stats[ST_SIGUIENTE_RELOJ] = 4096
        self.mejor_raiz[:] = -1
        h = np.uint64(clave_zobrist(tab, color, pts))  # numba regresa int de Python
        v = negamax(
            tab, color, pts, h, prof, -np.inf, np.inf, 0, self.w, self.jugadas_buf, self.puntajes_buf,
            self.tt_clave, self.tt_valor, self.tt_prof, self.tt_flag, self.tt_jugada,
            self.killers, self.historial, self.stats, limite_s, self.mejor_raiz,
        )
        abortada = self.stats[ST_ABORTADO] != 0
        return int(self.mejor_raiz[0]), float(v), abortada

    def nodos(self) -> int:
        return int(self.stats[ST_NODOS])

    def limpiar_tablas(self) -> None:
        self.tt_prof[:] = -1
        self.tt_clave[:] = 0
        self.killers[:] = -1
        self.historial[:] = 0


def pesos_desde_json(obj) -> np.ndarray:
    """`data/pesos.json` = {"pesos": [...N_RASGOS...], "logit_por_punto": k} → arreglo de N_RASGOS + 1."""
    w = np.asarray(obj["pesos"], np.float64)
    if len(w) != N_RASGOS:
        raise ValueError(f"pesos.json trae {len(w)} pesos y el motor usa {N_RASGOS} rasgos: hay que reajustarlos")
    return np.append(w, float(obj["logit_por_punto"]))


def jugada_a_texto(j: int) -> str:
    if j < 0:
        raise ValueError("no hay jugada: la búsqueda no terminó ninguna profundidad")
    cols = "abcdefghi"
    o, d = divmod(j, CASILLAS)
    return f"{cols[o % LADO]}{o // LADO + 1}-{cols[d % LADO]}{d // LADO + 1}"


def tablero_desde_estado(estado) -> tuple[np.ndarray, int, np.ndarray]:
    """Pasa un `Estado` del motor oficial a nuestros arreglos."""
    tab = np.zeros(CASILLAS, np.int8)
    for color, nombre in enumerate(("blancas", "negras")):
        for casilla in estado.piezas(nombre):
            tab[(int(casilla[1]) - 1) * LADO + "abcdefghi".index(casilla[0])] = color + 1
    pts = np.array([estado.puntos["blancas"], estado.puntos["negras"]], np.int64)
    return tab, 0 if estado.turno == "blancas" else 1, pts
