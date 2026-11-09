"""Motor rápido y búsqueda en numba: reglas (copiadas del motor oficial), evaluación y expectiminimax.

Tablero: int8[81], casilla = fila*9 + col; 0 vacía, 1 blanca, 2 negra. turno: 0 blancas, 1 negras.
Los valores van de -1 (pierde quien mueve) a +1 (gana quien mueve): son 2*P(ganar) - 1, así que
promediarlos en un nodo de azar es correcto.
"""

import numpy as np
from numba import njit

LADO = 9
CASILLAS = 81
PUNTOS_PARA_GANAR = 3
MAX_JUGADAS = 54
MAX_PLY = 96
PROF_QUIESCENCIA = 4  # plies extra que solo miran jugadas forzantes
VALOR_MAX_EVAL = 0.98  # solo una posición terminal vale ±1
LMR_PROF_MIN = 3  # reducir jugadas tranquilas tardías solo con esta profundidad o más
LMR_JUGADA_MIN = 3  # las primeras jugadas ordenadas nunca se reducen
LMR_FUERTE_PROF = 5  # con esta profundidad y esta posición en el orden, se reduce 2 plies
LMR_FUERTE_JUGADA = 12
VENTANA_NULA = 1e-4

# Índices de los rasgos de la evaluación. Todos son blancas menos negras.
R_FILA = 0  # 0..7: peones por fila relativa (0 = su casa, 7 = a un paso de anotar)
R_PUNTO1 = 8
R_PUNTO2 = 9
R_APOYO1 = 10
R_APOYO2 = 11
R_BORDE = 12
R_CONTACTO = 13  # suma de (apoyo propio - apoyo rival) en cada contacto diagonal
R_AMENAZADO = 14  # peones con un rival en contacto que tiene más apoyo
R_LIBRE_LEJOS = 15  # peón sin rivales en su cono de avance, fila relativa <= 4
R_LIBRE_MEDIO = 16  # fila relativa 5 o 6
R_LIBRE_CERCA = 17  # fila relativa 7
R_BLOQUEADO = 18
R_SEGURAS = 19  # jugadas a casilla vacía que ningún rival puede atacar
R_TURNO = 20
R_MEJOR_ATAQUE = 21  # ventaja de apoyo del mejor ataque de quien mueve
R_COLUMNA_VACIA = 22
R_ANOTA_YA = 23  # quien mueve tiene un peón en fila 7 con paso libre
R_MATCH_POINT = 24  # con 2 puntos: peones en filas 6 y 7
R_RUTA_MP = 25  # R_RUTA, solo si el dueño del corredor ya tiene 2 puntos
R_INFILTRADO = 26  # peón en fila relativa >= 4, con salida y sin rival en contacto
R_PELIGRO = 27  # peligro del corredor más amenazante: (1/d) * prod(1 - p_i) de los defensores en su cono
R_PELIGRO_MP = 28  # el mismo peligro, solo si el dueño del corredor ya tiene 2 puntos
R_RUTA = 29  # mejor corredor a 3 avances o menos: P(llega por su mejor camino) / avances
N_RASGOS = 30

PESOS_INICIALES = np.array(
    [0.30, 0.30, 0.30, 0.32, 0.36, 0.45, 0.70, 1.20]
    + [3.5, 4.5, 0.05, 0.10, -0.05, 0.02, -0.10, 0.20, 0.60, 1.00]
    + [-0.02, 0.01, 0.05, 0.05, -0.05, 0.60, 0.40, 3.00, 0.20, 0.50, 0.50, 0.50],
    dtype=np.float64,
)


def crear_zobrist() -> np.ndarray:
    """Claves de hash: [color, casilla] en 0..161; turno en 162; puntos en 163..170."""
    gen = np.random.Generator(np.random.PCG64(20261108))
    return gen.integers(-(2**62), 2**62, size=171, dtype=np.int64)


@njit(cache=True)
def apoyo(tab, casilla, color):
    """Peones propios en las dos diagonales de atrás de `casilla`."""
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


@njit(cache=True)
def prob_ataque(tab, color, origen, destino):
    return 0.5 + 0.15 * (apoyo(tab, origen, color) - apoyo(tab, destino, 1 - color))


@njit(cache=True)
def generar(tab, color, mo, md, ply):
    """Llena mo[ply], md[ply] con las jugadas legales y regresa cuántas son."""
    pieza = color + 1
    avance = LADO if color == 0 else -LADO
    n = 0
    for c in range(CASILLAS):
        if tab[c] != pieza:
            continue
        col = c % LADO
        frente = c + avance
        if col > 0 and tab[frente - 1] != pieza:
            mo[ply, n] = c
            md[ply, n] = frente - 1
            n += 1
        if tab[frente] == 0:
            mo[ply, n] = c
            md[ply, n] = frente
            n += 1
        if col < LADO - 1 and tab[frente + 1] != pieza:
            mo[ply, n] = c
            md[ply, n] = frente + 1
            n += 1
    return n


@njit(cache=True)
def clave_de(tab, pts, turno, Z):
    h = 0
    for c in range(CASILLAS):
        if tab[c] != 0:
            h ^= Z[(tab[c] - 1) * CASILLAS + c]
    if turno == 1:
        h ^= Z[162]
    h ^= Z[163 + pts[0]]
    h ^= Z[167 + pts[1]]
    return h


@njit(cache=True)
def hacer(tab, pts, turno, o, d, gana, h, Z):
    """Aplica la jugada en sitio y regresa la clave nueva. Deshacer es responsabilidad de quien llama."""
    h ^= Z[162] ^ Z[turno * CASILLAS + o]
    tab[o] = 0
    if tab[d] != 0:
        if not gana:
            return h
        h ^= Z[(1 - turno) * CASILLAS + d]
        tab[d] = 0
    meta = LADO - 1 if turno == 0 else 0
    if d // LADO == meta:
        base = 163 if turno == 0 else 167
        h ^= Z[base + pts[turno]]
        pts[turno] += 1
        h ^= Z[base + pts[turno]]
    else:
        tab[d] = turno + 1
        h ^= Z[turno * CASILLAS + d]
    return h


@njit(cache=True)
def deshacer(tab, pts, turno, o, d, antes_en_d):
    """Revierte una jugada que se aplicó con éxito (`gana=True` o sin ataque)."""
    tab[o] = turno + 1
    meta = LADO - 1 if turno == 0 else 0
    if d // LADO == meta:
        pts[turno] -= 1
    tab[d] = antes_en_d


@njit(cache=True)
def _llega(tab, c, color, restantes):
    """P(el peón en `c` anota por su mejor camino en `restantes` avances), con el resto del tablero quieto.

    Cuenta casillas bloqueadas, los ataques que necesita y el riesgo de que lo ataquen donde se detiene.
    """
    if restantes <= 0:
        return 1.0
    pieza = color + 1
    rival = 3 - pieza
    avance = LADO if color == 0 else -LADO
    col = c % LADO
    frente = c + avance
    mejor = 0.0
    for dc in (-1, 0, 1):
        nc = col + dc
        if nc < 0 or nc >= LADO:
            continue
        d = frente + dc
        v = tab[d]
        if v == pieza or (dc == 0 and v != 0):
            continue
        valor = prob_ataque(tab, color, c, d) if v != 0 else 1.0
        if restantes > 1:
            # El propio peón deja su casilla: no se apoya a sí mismo al llegar en diagonal.
            propio = apoyo(tab, d, color) - (1 if dc != 0 else 0)
            siguiente = d + avance
            for lado in (-1, 1):
                nl = nc + lado
                if 0 <= nl < LADO and tab[siguiente + lado] == rival:
                    p_rival = 0.5 + 0.15 * (apoyo(tab, siguiente + lado, 1 - color) - propio)
                    valor *= 1.0 - p_rival
            valor *= _llega(tab, d, color, restantes - 1)
        if valor > mejor:
            mejor = valor
    return mejor


@njit(cache=True)
def rasgos(tab, pts, turno, f, mascaras):
    """Llena `f` con los rasgos de la posición. `mascaras` es un búfer int64[18] (filas por color)."""
    for i in range(N_RASGOS):
        f[i] = 0.0
    for i in range(18):
        mascaras[i] = 0
    columnas_b = 0
    columnas_n = 0
    for c in range(CASILLAS):
        v = tab[c]
        if v != 0:
            mascaras[(v - 1) * LADO + c // LADO] |= 1 << (c % LADO)
            if v == 1:
                columnas_b |= 1 << (c % LADO)
            else:
                columnas_n |= 1 << (c % LADO)
    mejor_ataque = -9
    ruta_b = 0.0
    ruta_n = 0.0
    peligro_b = 0.0
    peligro_n = 0.0
    anota_ya = 0
    for c in range(CASILLAS):
        v = tab[c]
        if v == 0:
            continue
        color = v - 1
        signo = 1.0 if color == 0 else -1.0
        paso = 1 if color == 0 else -1
        fila = c // LADO
        col = c % LADO
        rel = fila if color == 0 else LADO - 1 - fila
        rival = 3 - v
        frente = c + paso * LADO
        f[R_FILA + rel] += signo
        if rel >= 4:
            contacto = (col > 0 and tab[frente - 1] == rival) or (
                col < LADO - 1 and tab[frente + 1] == rival
            )
            salida = (
                tab[frente] == 0
                or (col > 0 and tab[frente - 1] == 0)
                or (col < LADO - 1 and tab[frente + 1] == 0)
            )
            if salida and not contacto:
                f[R_INFILTRADO] += signo * (rel - 3) / 4.0
        a = apoyo(tab, c, color)
        if a == 1:
            f[R_APOYO1] += signo
        elif a == 2:
            f[R_APOYO2] += signo
        if col == 0 or col == LADO - 1:
            f[R_BORDE] += signo
        if tab[frente] != 0:
            f[R_BLOQUEADO] += signo
        amenazado = False
        for dc in (-1, 1):
            nc = col + dc
            if nc < 0 or nc >= LADO:
                continue
            casilla = frente + dc
            if tab[casilla] == rival:
                b = apoyo(tab, casilla, 1 - color)
                f[R_CONTACTO] += signo * (a - b)
                if b > a:
                    amenazado = True
                if color == turno and a - b > mejor_ataque:
                    mejor_ataque = a - b
        if amenazado:
            f[R_AMENAZADO] += signo
        # Jugadas seguras: destino vacío al que ningún rival puede atacar. La fila de meta es segura.
        if rel < LADO - 2:
            siguiente = frente + paso * LADO
            for dc in (-1, 0, 1):
                nc = col + dc
                if nc < 0 or nc >= LADO or tab[frente + dc] != 0:
                    continue
                izq = nc > 0 and tab[siguiente + dc - 1] == rival
                der = nc < LADO - 1 and tab[siguiente + dc + 1] == rival
                if not izq and not der:
                    f[R_SEGURAS] += signo
        else:
            f[R_SEGURAS] += signo
            if color == turno:
                if tab[frente] == 0 or (col > 0 and tab[frente - 1] == 0) or (
                    col < LADO - 1 and tab[frente + 1] == 0
                ):
                    anota_ya = 1
        # Peón libre: ningún rival en el cono de avance.
        libre = True
        fila_cono = fila + paso
        k = 1
        while 0 <= fila_cono < LADO:
            cono = ((1 << min(LADO, col + k + 1)) - 1) & ~((1 << max(0, col - k)) - 1)
            if mascaras[(rival - 1) * LADO + fila_cono] & cono:
                libre = False
                break
            fila_cono += paso
            k += 1
        if libre:
            if rel <= 4:
                f[R_LIBRE_LEJOS] += signo
            elif rel <= 6:
                f[R_LIBRE_MEDIO] += signo
            else:
                f[R_LIBRE_CERCA] += signo
        if rel >= 5:
            ruta = _llega(tab, c, color, 8 - rel) / (8 - rel)
            if color == 0:
                ruta_b = max(ruta_b, ruta)
            else:
                ruta_n = max(ruta_n, ruta)
        # Peligro: probabilidad de que el corredor sobreviva a cada defensor de su cono, entre su distancia.
        if rel >= 4:
            sobrevive = 1.0
            fila_cono = fila + paso
            k = 1
            while 0 <= fila_cono < LADO:
                defensores = mascaras[(rival - 1) * LADO + fila_cono]
                for cc in range(max(0, col - k), min(LADO - 1, col + k) + 1):
                    if (defensores >> cc) & 1:
                        p_defensa = 0.5 + 0.15 * (apoyo(tab, fila_cono * LADO + cc, 1 - color) - a)
                        sobrevive *= 1.0 - p_defensa
                fila_cono += paso
                k += 1
            peligro = sobrevive / (8 - rel)
            if color == 0:
                peligro_b = max(peligro_b, peligro)
            else:
                peligro_n = max(peligro_n, peligro)
        if pts[color] == 2 and rel >= 6:
            f[R_MATCH_POINT] += signo
    f[R_RUTA] = ruta_b - ruta_n
    f[R_RUTA_MP] = (ruta_b if pts[0] == 2 else 0.0) - (ruta_n if pts[1] == 2 else 0.0)
    f[R_PELIGRO] = peligro_b - peligro_n
    f[R_PELIGRO_MP] = (peligro_b if pts[0] == 2 else 0.0) - (peligro_n if pts[1] == 2 else 0.0)
    signo_turno = 1.0 if turno == 0 else -1.0
    f[R_PUNTO1] = (1.0 if pts[0] >= 1 else 0.0) - (1.0 if pts[1] >= 1 else 0.0)
    f[R_PUNTO2] = (1.0 if pts[0] >= 2 else 0.0) - (1.0 if pts[1] >= 2 else 0.0)
    f[R_TURNO] = signo_turno
    if mejor_ataque > 0:
        f[R_MEJOR_ATAQUE] = signo_turno * mejor_ataque
    f[R_ANOTA_YA] = signo_turno * anota_ya
    for col in range(LADO):
        if not (columnas_b >> col) & 1:
            f[R_COLUMNA_VACIA] += 1.0
        if not (columnas_n >> col) & 1:
            f[R_COLUMNA_VACIA] -= 1.0


@njit(cache=True)
def evaluar(tab, pts, turno, f, mascaras, w):
    """Valor para quien mueve: tanh(s/2) es 2*P(ganar) - 1 si s es el logit de que ganen blancas."""
    rasgos(tab, pts, turno, f, mascaras)
    s = 0.0
    for i in range(N_RASGOS):
        s += w[i] * f[i]
    v = np.tanh(s / 2.0) * VALOR_MAX_EVAL
    return v if turno == 0 else -v


@njit(cache=True)
def _buscar(tab, pts, mo, md, sc, ttk, ttv, tti, hist, ctx, f, mascaras, w, Z, ttmask, turno, h, prof, ply, alfa, beta):
    """Expectiminimax negamax con alfa-beta, poda Star1 en los ataques, TT y quiescencia.

    `prof` <= 0 es quiescencia: se puede plantar y solo mira anotar y ataques con p >= 0.65.
    ctx: [nodos, límite, abortó, mejor jugada en la raíz, jugadas de la raíz terminadas, sin selectividad].
    ctx[5] = 1 apaga LMR y extensión táctica: lo usa la prueba contra fuerza bruta.
    """
    ctx[0] += 1
    if ctx[0] > ctx[1]:
        ctx[2] = 1
        return 0.0
    n = generar(tab, turno, mo, md, ply)
    if n == 0:
        return -1.0
    alfa_0 = alfa
    tt_jugada = -1
    idx = h & ttmask
    if prof > 0 and ttk[idx] == h:
        info = tti[idx]
        tt_jugada = info >> 8
        if ply > 0 and (info & 63) >= prof:
            bandera = (info >> 6) & 3
            v = ttv[idx]
            if bandera == 0 or (bandera == 1 and v >= beta) or (bandera == 2 and v <= alfa):
                return v
    mejor = -2.0
    if prof <= 0:
        mejor = evaluar(tab, pts, turno, f, mascaras, w)
        if prof <= -PROF_QUIESCENCIA or mejor >= beta:
            return mejor
        if mejor > alfa:
            alfa = mejor
    meta = LADO - 1 if turno == 0 else 0
    # Orden: jugada de la TT, anotar, buenos ataques, historia; los malos ataques al final.
    m = 0
    for i in range(n):
        o = mo[ply, i]
        d = md[ply, i]
        ataque = tab[d] != 0
        anota = d // LADO == meta
        p = prob_ataque(tab, turno, o, d) if ataque else 1.0
        if prof <= 0 and not (anota or (ataque and p >= 0.64)):
            continue
        if o * CASILLAS + d == tt_jugada:
            s = 100000000
        elif anota:
            s = 50000000
        elif ataque:
            s = 10000000 if p >= 0.64 else (1000000 if p >= 0.49 else -1000000)
        else:
            s = hist[turno, o, d - o + 10]
        mo[ply, m] = o
        md[ply, m] = d
        sc[ply, m] = s
        m += 1
    mejor_jugada = -1
    for i in range(m):
        k = i
        for j in range(i + 1, m):
            if sc[ply, j] > sc[ply, k]:
                k = j
        o = mo[ply, k]
        d = md[ply, k]
        mo[ply, k] = mo[ply, i]
        md[ply, k] = md[ply, i]
        sc[ply, k] = sc[ply, i]
        antes = tab[d]
        # Extensión táctica: con 2 puntos, el corredor que entra a las dos últimas filas no gasta profundidad.
        fila_rel = d // LADO if turno == 0 else LADO - 1 - d // LADO
        extiende = ctx[5] == 0 and prof > 0 and pts[turno] == 2 and fila_rel >= 6
        prof_hijo = prof if extiende else prof - 1
        if antes == 0:
            h2 = hacer(tab, pts, turno, o, d, True, h, Z)
            if pts[turno] >= PUNTOS_PARA_GANAR:
                val = 1.0
            else:
                # LMR: jugada tranquila tardía, se busca menos y solo se repite si sorprende.
                reduccion = 0
                if (
                    ctx[5] == 0
                    and ply > 0
                    and prof >= LMR_PROF_MIN
                    and i >= LMR_JUGADA_MIN
                    and fila_rel < 6
                    and pts[0] < 2
                    and pts[1] < 2
                ):
                    reduccion = 2 if prof >= LMR_FUERTE_PROF and i >= LMR_FUERTE_JUGADA else 1
                val = alfa + 1.0
                if reduccion > 0:
                    val = -_buscar(tab, pts, mo, md, sc, ttk, ttv, tti, hist, ctx, f, mascaras, w, Z, ttmask, 1 - turno, h2, prof_hijo - reduccion, ply + 1, -alfa - VENTANA_NULA, -alfa)
                if val > alfa and ctx[2] == 0:
                    val = -_buscar(tab, pts, mo, md, sc, ttk, ttv, tti, hist, ctx, f, mascaras, w, Z, ttmask, 1 - turno, h2, prof_hijo, ply + 1, -beta, -alfa)
            deshacer(tab, pts, turno, o, d, antes)
        else:
            p = prob_ataque(tab, turno, o, d)
            q = 1.0 - p
            # Star1: ventana para el resultado «gana», suponiendo lo extremo en «pierde».
            # El corte usa las cotas sin recortar: si «gana» vale ±1, recortadas cortarían sin buscar «pierde».
            corte_bajo = (alfa - q) / p
            corte_alto = (beta + q) / p
            a1 = max(-1.0, corte_bajo)
            b1 = min(1.0, corte_alto)
            h2 = hacer(tab, pts, turno, o, d, True, h, Z)
            if pts[turno] >= PUNTOS_PARA_GANAR:
                v1 = 1.0
            else:
                v1 = -_buscar(tab, pts, mo, md, sc, ttk, ttv, tti, hist, ctx, f, mascaras, w, Z, ttmask, 1 - turno, h2, prof_hijo, ply + 1, -b1, -a1)
            deshacer(tab, pts, turno, o, d, antes)
            if ctx[2] != 0:
                return 0.0
            if v1 >= corte_alto:
                val = p * v1 - q
            elif v1 <= corte_bajo:
                val = p * v1 + q
            else:
                a2 = max(-1.0, (alfa - p * v1) / q)
                b2 = min(1.0, (beta - p * v1) / q)
                h2 = hacer(tab, pts, turno, o, d, False, h, Z)
                v2 = -_buscar(tab, pts, mo, md, sc, ttk, ttv, tti, hist, ctx, f, mascaras, w, Z, ttmask, 1 - turno, h2, prof_hijo, ply + 1, -b2, -a2)
                tab[o] = turno + 1
                val = p * v1 + q * v2
        if ctx[2] != 0:
            return 0.0
        if val > mejor:
            mejor = val
            mejor_jugada = o * CASILLAS + d
            if ply == 0:
                ctx[3] = mejor_jugada
        if ply == 0:
            ctx[4] += 1
        if val > alfa:
            alfa = val
        if alfa >= beta:
            if antes == 0:
                hist[turno, o, d - o + 10] += prof * prof if prof > 0 else 1
            break
    if prof > 0:
        bandera = 0
        if mejor <= alfa_0:
            bandera = 2
        elif mejor >= beta:
            bandera = 1
        ttk[idx] = h
        ttv[idx] = mejor
        tti[idx] = min(prof, 63) | (bandera << 6) | (max(mejor_jugada, 0) << 8)
    return mejor


@njit(cache=True)
def buscar_raiz(B, turno, h, prof):
    """Una iteración de profundidad `prof` desde la raíz. Los contadores los pone quien llama."""
    return _buscar(*B, turno, h, prof, 0, -1.0, 1.0)


def crear_buffers(pesos, bits_tt=21):
    """Reserva todo una vez: en numba no se crean arreglos en el camino caliente."""
    tam = 1 << bits_tt
    return (
        np.zeros(CASILLAS, np.int8),  # tablero
        np.zeros(2, np.int64),  # puntos
        np.zeros((MAX_PLY, MAX_JUGADAS), np.int64),  # orígenes por ply
        np.zeros((MAX_PLY, MAX_JUGADAS), np.int64),  # destinos por ply
        np.zeros((MAX_PLY, MAX_JUGADAS), np.int64),  # puntajes de orden por ply
        np.zeros(tam, np.int64),  # TT: clave
        np.zeros(tam, np.float64),  # TT: valor
        np.zeros(tam, np.int64),  # TT: prof | bandera | jugada
        np.zeros((2, CASILLAS, 21), np.int64),  # historia [turno, origen, delta+10]
        np.zeros(6, np.int64),  # ctx
        np.zeros(N_RASGOS, np.float64),  # búfer de rasgos
        np.zeros(18, np.int64),  # búfer de máscaras
        np.asarray(pesos, dtype=np.float64),
        crear_zobrist(),
        np.int64(tam - 1),
    )
