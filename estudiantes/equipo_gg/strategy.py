"""Bot «gg» para Breakthrough de combate.

Idea: expectiminimax con poda alfa-beta y nodos de azar (Star1), profundización iterativa,
tabla de transposición, PVS con reducciones de jugadas tardías y una búsqueda de quietud corta.
La evaluación es una regresión logística sobre 55 rasgos de la posición, ajustada con partidas
de autojuego y corregida con partidas reales contra montecarlo (ver herramientas/), y devuelve
2·P(ganar) - 1: así los promedios de los nodos de azar son promedios de probabilidad de ganar.

Encima de esa evaluación sumamos una «ventaja asumida» de VENTAJA log-momios a nuestro favor.
Con eso la función tanh queda cóncava alrededor de las posiciones normales y la búsqueda evita
tiradas de dado innecesarias (las nuestras y las que dejaríamos al rival): contra un rival más
débil, el azar es lo único que puede quitarnos la partida.

En data/ van los pesos vigentes (pesos.json, corregidos con 848 partidas contra montecarlo) y un
libro de aperturas (aperturas.json: nuestras primeras 2-3 jugadas contra cualquier respuesta del
rival, buscadas offline mucho más hondo). Si data/ falta o está mal, se usan los pesos de este
archivo y se juega sin libro.

Todo el árbol corre en numba sobre un arreglo int8[81] (casilla = fila*9 + col; 0 vacía,
1 blanca, 2 negra), igual que el motor del curso. Del motor oficial solo usamos la API pública
(`estado.piezas`, `estado.puntos`, `estado.turno`, `estado.jugadas`) y siempre regresamos una
jugada de `estado.jugadas()`.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
from numba import njit

from motor import reportar

# ----------------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------------
GANA = 1.0          # valor de una victoria segura; la evaluación vive en (-1, 1)
MAXPLY = 64         # profundidad máxima del árbol (con quietud)
MAXMOV = 64         # a lo más 18 peones × 3 jugadas = 54
TT_BITS = 20        # tabla de transposición de 2^20 entradas (~27 MB)
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
NFEAT = 56          # 55 rasgos ajustados + 1 para la ventaja asumida
EXACTO, COTA_INF, COTA_SUP = 0, 1, 2

# Ventaja asumida sobre el rival, en log-momios (1.0 ≈ «ganamos 73 % de lo parejo»).
# Contra montecarlo subió de 75 % a 82.5 % las victorias con los mismos dados; el ajuste con
# partidas reales estima una ventaja de +1.1, así que 1.0 es conservador.
VENTAJA = 1.0


def _zobrist():
    """Números aleatorios fijos para el hash de posiciones (tabla de transposición)."""
    rng = np.random.Generator(np.random.PCG64(20261107))
    zp = rng.integers(1, 2**63 - 1, size=(2, 81), dtype=np.int64).astype(np.uint64)
    zt = np.uint64(rng.integers(1, 2**63 - 1, dtype=np.int64))
    zs = rng.integers(1, 2**63 - 1, size=(2, 4), dtype=np.int64).astype(np.uint64)
    return zp, zt, zs


# ----------------------------------------------------------------------------------------
# Reglas (mismas que sistema/motor/nucleo.py; verificadas contra el motor oficial)
# ----------------------------------------------------------------------------------------
@njit(cache=False)
def apoyo(tab, sq, lado):
    """Peones propios en las dos diagonales de atrás de `sq` (0, 1 o 2)."""
    r = sq // 9
    c = sq - r * 9
    br = r - 1 if lado == 0 else r + 1
    if br < 0 or br > 8:
        return 0
    pieza = lado + 1
    t = 0
    if c > 0 and tab[br * 9 + c - 1] == pieza:
        t += 1
    if c < 8 and tab[br * 9 + c + 1] == pieza:
        t += 1
    return t


@njit(cache=False)
def prob_ataque(tab, lado, o, d):
    """p = 0.5 + 0.15·(apoyo atacante − apoyo defensor), acotada a [0.2, 0.9]."""
    p = 0.5 + 0.15 * (apoyo(tab, o, lado) - apoyo(tab, d, 1 - lado))
    if p < 0.2:
        p = 0.2
    if p > 0.9:
        p = 0.9
    return p


@njit(cache=False)
def generar(tab, lado, mo, md, ply):
    """Jugadas legales de `lado` en mo[ply], md[ply]; regresa cuántas son."""
    pieza = lado + 1
    paso = 9 if lado == 0 else -9
    n = 0
    for c in range(81):
        if tab[c] != pieza:
            continue
        col = c % 9
        f = c + paso            # nunca sale del tablero: al llegar a la meta el peón sale
        if col > 0 and tab[f - 1] != pieza:
            mo[ply, n] = c
            md[ply, n] = f - 1
            n += 1
        if tab[f] == 0:
            mo[ply, n] = c
            md[ply, n] = f
            n += 1
        if col < 8 and tab[f + 1] != pieza:
            mo[ply, n] = c
            md[ply, n] = f + 1
            n += 1
    return n


@njit(cache=False)
def hacer(tab, pts, lado, o, d, gana, h, zp, zt, zs):
    """Aplica la jugada en sitio (con el resultado `gana` si es ataque) y regresa el hash nuevo."""
    pieza = lado + 1
    antes = tab[d]
    tab[o] = 0
    h ^= zp[lado, o]
    if antes != 0:              # ataque
        if not gana:            # pierde: sale el atacante
            return h ^ zt
        tab[d] = 0
        h ^= zp[1 - lado, d]
    meta = 8 if lado == 0 else 0
    if d // 9 == meta:          # anota y sale del tablero
        h ^= zs[lado, pts[lado]]
        pts[lado] += 1
        h ^= zs[lado, pts[lado]]
    else:
        tab[d] = pieza
        h ^= zp[lado, d]
    return h ^ zt


@njit(cache=False)
def deshacer(tab, pts, lado, o, d, antes, pts_antes):
    tab[o] = lado + 1
    tab[d] = antes
    pts[lado] = pts_antes


# ----------------------------------------------------------------------------------------
# Evaluación: rasgos -> z = w·rasgos (log-momios) -> tanh(z/2) = 2·P(ganar) - 1
# ----------------------------------------------------------------------------------------
# Búfer entero B preasignado (numba no debe crear arreglos en el camino caliente).
B_MASK = 0      # 2×9  máscara de columnas ocupadas por fila
B_CNT = 18      # 2×8  peones por fila de progreso (0 = casa, 7 = a un paso de anotar)
B_N = 34        # 2    peones
B_DST = 36      # 2×18 distancias de los corredores
B_NCR = 72      # 2    corredores
B_SUPZ = 74     # 2×3  apoyos por zona (progreso 0-2, 3-4, 5-7)
B_SIN = 80      # 2    peones avanzados (progreso ≥ 4) sin apoyo
B_MOV = 82      # 2    jugadas disponibles
B_A65 = 84      # 2    ataques disponibles con p ≥ 0.65
B_A50 = 86      # 2    ataques disponibles con p ≥ 0.5
B_EFF = 96      # 2×18 distancias efectivas (distancia + 3·interceptores)
B_TAM = 140

# «yo» = quien mueve en esa posición, «el» = su rival.
NOMBRES_RASGOS = [
    "sesgo", "pts_yo", "pts_el", "yo_2pts", "el_2pts",
    "fila0", "fila1", "fila2", "fila3", "fila4",
    "yo_f5", "el_f5", "yo_f6", "el_f6", "yo_f7", "el_f7",
    "apoyo_z0", "apoyo_z1", "apoyo_z2", "yo_sin_apoyo", "el_sin_apoyo",
    "yo_ataques65", "yo_ataques50", "el_ataques65", "el_ataques50",
    "yo_mejor_ataque_intruso", "el_mejor_ataque_intruso",
    "yo_corredores", "el_corredores", "yo_gana_carrera", "el_gana_carrera",
    "yo_corr_cerca", "el_corr_cerca",
    "yo_anota_libre", "yo_anota_ataque", "el_amenaza", "el_amenaza_imparable",
    "movilidad", "yo_f6_libres", "el_f6_libres",
    "el_pts_virtual", "el_2v", "el_3v", "yo_gana_ya", "yo_huecos_casa", "el_huecos_casa",
    "yo_f7_apoyado", "el_f7_apoyado",
    "yo_semi", "el_semi", "carrera_eff", "yo_t_eff", "el_t_eff", "yo_icp_avanz", "el_icp_avanz",
]
NFEAT_USADOS = len(NOMBRES_RASGOS)

# Pesos: regresión logística con autojuego (herramientas/autojuego.py + ajustar.py) y luego
# corregida con 228 partidas reales contra montecarlo (herramientas/ajustar_vs_rival.py).
PESOS_AJUSTADOS = [
    +0.27124, +0.24490, -0.16865, +0.81016, -0.49766,
    +0.06374, +0.13810, +0.11777, +0.11633, +0.28759,
    +0.27771, -0.32363, +0.18854, -0.34216, +0.15205,
    -0.36169, +0.10031, +0.06950, +0.01127, +0.07369,
    -0.09178, +0.03557, +0.00250, -0.02262, -0.05568,
    +0.10466, +0.05874, -0.66991, +0.55269, +1.98085,
    -0.81415, +0.56423, -0.31616, +0.31313, -0.18167,
    +0.01190, +0.04893, +0.53230, +0.31457, -0.14045,
    -0.07904, -0.34544, -3.39672, +3.88759, -0.15363,
    +0.13617, +0.21990, -0.36710, +0.04390, +0.04868,
    +0.04739, -0.77199, +0.70750, -0.37914, +0.44555,
]
PESOS = np.zeros(NFEAT, np.float64)
PESOS[:NFEAT_USADOS] = PESOS_AJUSTADOS
VENTAJA_ARCHIVO = VENTAJA


@njit(cache=False)
def bits(x):
    """Número de bits encendidos."""
    t = 0
    while x:
        x &= x - 1
        t += 1
    return t


@njit(cache=False)
def rasgos(tab, pts, lado, f, B):
    """Llena f con los rasgos de la posición desde la perspectiva de `lado` (quien mueve)."""
    for i in range(NFEAT):
        f[i] = 0.0
    for i in range(B_TAM):
        B[i] = 0
    yo = lado
    el = 1 - lado
    best0 = 0.0     # mejor p de ataque de blancas contra un peón negro avanzado
    best1 = 0.0
    # --- recorrido 1: conteos, apoyos, jugadas y ataques disponibles ---
    for sq in range(81):
        v = tab[sq]
        if v == 0:
            continue
        s = v - 1
        r = sq // 9
        c = sq - r * 9
        B[B_MASK + s * 9 + r] |= 1 << c
        prog = r if s == 0 else 8 - r
        B[B_CNT + s * 8 + prog] += 1
        B[B_N + s] += 1
        a = apoyo(tab, sq, s)
        z = 0 if prog <= 2 else (1 if prog <= 4 else 2)
        B[B_SUPZ + s * 3 + z] += a
        if prog >= 4 and a == 0:
            B[B_SIN + s] += 1
        paso = 9 if s == 0 else -9
        fr = sq + paso
        pieza = s + 1
        if tab[fr] == 0:
            B[B_MOV + s] += 1
        for dc in (-1, 1):
            cc = c + dc
            if cc < 0 or cc > 8:
                continue
            t = fr + dc
            vt = tab[t]
            if vt == pieza:
                continue
            B[B_MOV + s] += 1
            if vt != 0:
                p = 0.5 + 0.15 * (a - apoyo(tab, t, 1 - s))
                if p >= 0.64:
                    B[B_A65 + s] += 1
                if p >= 0.49:
                    B[B_A50 + s] += 1
                rt = t // 9
                prog_t = (8 - rt) if s == 0 else rt
                if prog_t >= 4:
                    if s == 0 and p > best0:
                        best0 = p
                    elif s == 1 and p > best1:
                        best1 = p
    # --- recorrido 2: interceptores de cada peón ---
    # Un rival a d filas adelante puede interceptar si su distancia de columna es ≤ d//2
    # (si mueve el dueño del peón) o ≤ (d+1)//2 (si mueve el rival). Sin interceptores el
    # peón es un «corredor»: nadie lo puede detener.
    semi0 = 0
    semi1 = 0
    icpa0 = 0
    icpa1 = 0
    ne0 = 0
    ne1 = 0
    for sq in range(81):
        v = tab[sq]
        if v == 0:
            continue
        s = v - 1
        r = sq // 9
        c = sq - r * 9
        mueve_s = s == yo
        icp = 0
        if s == 0:
            for r2 in range(r + 1, 9):
                dd = r2 - r
                k = dd // 2 if mueve_s else (dd + 1) // 2
                lo = c - k
                hi = c + k
                if lo < 0:
                    lo = 0
                if hi > 8:
                    hi = 8
                m = ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)
                icp += bits(B[B_MASK + 9 + r2] & m)
            d2 = 8 - r
        else:
            for r2 in range(r - 1, -1, -1):
                dd = r - r2
                k = dd // 2 if mueve_s else (dd + 1) // 2
                lo = c - k
                hi = c + k
                if lo < 0:
                    lo = 0
                if hi > 8:
                    hi = 8
                m = ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)
                icp += bits(B[B_MASK + r2] & m)
            d2 = r
        prog = 8 - d2
        if icp == 0:
            B[B_DST + s * 18 + B[B_NCR + s]] = d2
            B[B_NCR + s] += 1
        if icp == 1 and prog >= 3:
            if s == 0:
                semi0 += 1
            else:
                semi1 += 1
        if prog >= 4:
            if s == 0:
                icpa0 += min(icp, 4)
            else:
                icpa1 += min(icp, 4)
        ic = icp if icp < 3 else 3
        if s == 0:
            B[B_EFF + ne0] = d2 + 3 * ic
            ne0 += 1
        else:
            B[B_EFF + 18 + ne1] = d2 + 3 * ic
            ne1 += 1
    # --- carrera efectiva: suma de las (3 - puntos) distancias efectivas más cortas ---
    teff0 = 0
    teff1 = 0
    for s in range(2):
        ne = ne0 if s == 0 else ne1
        base = B_EFF + 18 * s
        for i in range(1, ne):
            x = B[base + i]
            j = i - 1
            while j >= 0 and B[base + j] > x:
                B[base + j + 1] = B[base + j]
                j -= 1
            B[base + j + 1] = x
        need = 3 - pts[s]
        t = 0
        for i in range(need):
            if i < ne:
                t += B[base + i]
            else:
                t += 20
        if s == 0:
            teff0 = t
        else:
            teff1 = t
    # --- carrera de corredores: cada jugada mueve un solo peón ---
    t_yo = 999
    t_el = 999
    dmin_yo = 99
    dmin_el = 99
    for s in range(2):
        nc = B[B_NCR + s]
        need = 3 - pts[s]
        for i in range(1, nc):
            x = B[B_DST + s * 18 + i]
            j = i - 1
            while j >= 0 and B[B_DST + s * 18 + j] > x:
                B[B_DST + s * 18 + j + 1] = B[B_DST + s * 18 + j]
                j -= 1
            B[B_DST + s * 18 + j + 1] = x
        t = 999
        if nc >= need:
            t = 0
            for i in range(need):
                t += B[B_DST + s * 18 + i]
        dm = B[B_DST + s * 18] if nc > 0 else 99
        if s == yo:
            t_yo = t
            dmin_yo = dm
        else:
            t_el = t
            dmin_el = dm
    # --- amenazas inmediatas (peones a 1 o 2 pasos de anotar) ---
    anota_libre = 0     # mis peones que anotan ya sin dado
    anota_ataque = 0    # mis peones que solo anotan atacando
    amenaza = 0         # peones rivales que anotan en su turno con casilla libre
    imparable = 0       # ...y que yo no puedo atacar ahora
    f6_yo = 0
    f6_el = 0
    f7ap_yo = 0
    f7ap_el = 0
    for sq in range(81):
        v = tab[sq]
        if v == 0:
            continue
        s = v - 1
        r = sq // 9
        c = sq - r * 9
        prog = r if s == 0 else 8 - r
        if prog < 6:
            continue
        paso = 9 if s == 0 else -9
        fr = sq + paso
        rival = 2 - s
        libre = tab[fr] == 0
        hay_ataque = False
        for dc in (-1, 1):
            cc = c + dc
            if cc < 0 or cc > 8:
                continue
            vt = tab[fr + dc]
            if vt == 0:
                libre = True
            elif vt == rival:
                hay_ataque = True
        if prog == 6:
            atacable = False
            for dc in (-1, 1):
                cc = c + dc
                if cc < 0 or cc > 8:
                    continue
                if tab[fr + dc] == rival:
                    atacable = True
            if not atacable:
                if s == yo:
                    f6_yo += 1
                else:
                    f6_el += 1
            continue
        if apoyo(tab, sq, s) > 0:
            if s == yo:
                f7ap_yo += 1
            else:
                f7ap_el += 1
        if s == yo:
            if libre:
                anota_libre += 1
            elif hay_ataque:
                anota_ataque += 1
        else:
            if libre:
                amenaza += 1
                puedo = False
                for dc in (-1, 1):
                    cc = c + dc
                    if cc < 0 or cc > 8:
                        continue
                    if tab[fr + dc] == rival:
                        puedo = True
                if not puedo:
                    imparable += 1
    # --- vector de rasgos ---
    f[0] = 1.0
    f[1] = pts[yo]
    f[2] = pts[el]
    f[3] = 1.0 if pts[yo] == 2 else 0.0
    f[4] = 1.0 if pts[el] == 2 else 0.0
    for k in range(5):
        f[5 + k] = B[B_CNT + yo * 8 + k] - B[B_CNT + el * 8 + k]
    f[10] = B[B_CNT + yo * 8 + 5]
    f[11] = B[B_CNT + el * 8 + 5]
    f[12] = B[B_CNT + yo * 8 + 6]
    f[13] = B[B_CNT + el * 8 + 6]
    f[14] = B[B_CNT + yo * 8 + 7]
    f[15] = B[B_CNT + el * 8 + 7]
    f[16] = B[B_SUPZ + yo * 3] - B[B_SUPZ + el * 3]
    f[17] = B[B_SUPZ + yo * 3 + 1] - B[B_SUPZ + el * 3 + 1]
    f[18] = B[B_SUPZ + yo * 3 + 2] - B[B_SUPZ + el * 3 + 2]
    f[19] = B[B_SIN + yo]
    f[20] = B[B_SIN + el]
    f[21] = B[B_A65 + yo]
    f[22] = B[B_A50 + yo]
    f[23] = B[B_A65 + el]
    f[24] = B[B_A50 + el]
    f[25] = best0 if yo == 0 else best1
    f[26] = best1 if yo == 0 else best0
    f[27] = min(B[B_NCR + yo], 3)
    f[28] = min(B[B_NCR + el], 3)
    f[29] = 1.0 if (t_yo < 999 and t_yo <= t_el) else 0.0
    f[30] = 1.0 if (t_el < 999 and t_el < t_yo) else 0.0
    f[31] = (8 - dmin_yo) / 8.0 if dmin_yo < 99 else 0.0
    f[32] = (8 - dmin_el) / 8.0 if dmin_el < 99 else 0.0
    f[33] = min(anota_libre, 3)
    f[34] = min(anota_ataque, 3)
    f[35] = min(amenaza, 3)
    f[36] = min(imparable, 3)
    f[37] = (B[B_MOV + yo] - B[B_MOV + el]) / 10.0
    f[38] = f6_yo
    f[39] = f6_el
    # puntos «virtuales» del rival: los suyos más las anotaciones que ya no puedo impedir
    pv = pts[el] + min(imparable, 3)
    if pv > 3:
        pv = 3
    f[40] = pv
    f[41] = 1.0 if pv >= 2 else 0.0
    f[42] = 1.0 if pv >= 3 else 0.0
    f[43] = 1.0 if (pts[yo] == 2 and anota_libre > 0) else 0.0
    fila_yo = 0 if yo == 0 else 8
    fila_el = 0 if el == 0 else 8
    hy = 0
    he = 0
    for c in range(9):
        if tab[fila_yo * 9 + c] == 0:
            hy += 1
        if tab[fila_el * 9 + c] == 0:
            he += 1
    f[44] = hy
    f[45] = he
    f[46] = f7ap_yo
    f[47] = f7ap_el
    f[48] = semi0 if yo == 0 else semi1
    f[49] = semi1 if yo == 0 else semi0
    ty = teff0 if yo == 0 else teff1
    te = teff1 if yo == 0 else teff0
    dif = te - ty
    if dif > 12:
        dif = 12
    if dif < -12:
        dif = -12
    f[50] = dif / 6.0
    f[51] = min(ty, 40) / 10.0
    f[52] = min(te, 40) / 10.0
    f[53] = (icpa0 if yo == 0 else icpa1) / 10.0
    f[54] = (icpa1 if yo == 0 else icpa0) / 10.0
    # ventaja asumida: w[55] = +VENTAJA si somos blancas, -VENTAJA si somos negras
    f[55] = 1.0 if yo == 0 else -1.0
    return 0


@njit(cache=False)
def evaluar(tab, pts, lado, w, f, B):
    """Valor para quien mueve en (-1, 1): tanh(z/2) = 2·P(ganar) - 1."""
    rasgos(tab, pts, lado, f, B)
    z = 0.0
    for i in range(NFEAT):
        z += w[i] * f[i]
    return math.tanh(0.5 * z)


# ----------------------------------------------------------------------------------------
# Búsqueda
# ----------------------------------------------------------------------------------------
@njit(cache=False)
def ordenar(tab, lado, mo, md, ms, n, ply, tt_mov, killers, hist):
    """Orden: jugada de la tabla, anotaciones, ataques (p y víctima avanzada), killers, historia."""
    meta = 8 if lado == 0 else 0
    for i in range(n):
        o = mo[ply, i]
        d = md[ply, i]
        cod = o * 81 + d
        s = 0.0
        if cod == tt_mov:
            s = 1e9
        elif d // 9 == meta:
            s = 1e8
        elif tab[d] != 0:
            p = prob_ataque(tab, lado, o, d)
            rv = d // 9
            prog_v = (8 - rv) if lado == 0 else rv
            s = 1e7 + p * 1000.0 + prog_v * 100.0
        elif cod == killers[ply, 0]:
            s = 5e6
        elif cod == killers[ply, 1]:
            s = 4e6
        else:
            s = hist[lado, cod]
        ms[ply, i] = s
    for i in range(1, n):           # inserción: n ≤ 54
        so = ms[ply, i]
        oo = mo[ply, i]
        dd = md[ply, i]
        j = i - 1
        while j >= 0 and ms[ply, j] < so:
            ms[ply, j + 1] = ms[ply, j]
            mo[ply, j + 1] = mo[ply, j]
            md[ply, j + 1] = md[ply, j]
            j -= 1
        ms[ply, j + 1] = so
        mo[ply, j + 1] = oo
        md[ply, j + 1] = dd


@njit(cache=False)
def quiescencia(tab, pts, lado, alpha, beta, ply, qd, h, ctx_i, w, f, B, mo, md, ms, zp, zt, zs):
    """Búsqueda de quietud: evaluación estática o, si conviene, anotar o atacar a un intruso a
    ≤ 2 pasos de anotar. Basta una anotación sin dado (son equivalentes); los ataques son nodos
    de azar con ventanas Star1, como en negamax. Se exploran de mayor a menor prioridad."""
    ctx_i[0] += 1
    est = evaluar(tab, pts, lado, w, f, B)
    if qd <= 0 or ply >= MAXPLY - 1:
        return est
    if est >= beta:
        return est
    if est > alpha:
        alpha = est
    mejor = est
    meta = 8 if lado == 0 else 0
    n = generar(tab, lado, mo, md, ply)
    if n == 0:
        return -GANA + ply * 1e-4
    hay_libre = False
    for i in range(n):
        o = mo[ply, i]
        d = md[ply, i]
        rd = d // 9
        if rd == meta:
            if tab[d] == 0:
                ms[ply, i] = -1.0 if hay_libre else 4.0
                hay_libre = True
            else:
                ms[ply, i] = 2.0 + prob_ataque(tab, lado, o, d)
        elif tab[d] != 0:
            prog_v = (8 - rd) if lado == 0 else rd
            ms[ply, i] = (1.0 + prob_ataque(tab, lado, o, d)) if prog_v >= 6 else -1.0
        else:
            ms[ply, i] = -1.0
    while True:
        # siguiente candidata (selección del máximo)
        bi = -1
        bs = 0.0
        for i in range(n):
            if ms[ply, i] > bs:
                bs = ms[ply, i]
                bi = i
        if bi < 0:
            break
        ms[ply, bi] = -1.0
        o = mo[ply, bi]
        d = md[ply, bi]
        antes = tab[d]
        pa = pts[lado]
        if antes == 0:
            hacer(tab, pts, lado, o, d, True, h, zp, zt, zs)
            if pts[lado] >= 3:
                v = GANA - ply * 1e-4
            else:
                v = -quiescencia(tab, pts, 1 - lado, -beta, -alpha, ply + 1, qd - 1, h, ctx_i, w, f, B,
                                 mo, md, ms, zp, zt, zs)
            deshacer(tab, pts, lado, o, d, antes, pa)
        else:
            p = prob_ataque(tab, lado, o, d)
            q = 1.0 - p
            lo1 = (alpha - q * GANA) / p
            hi1 = (beta + q * GANA) / p
            wlo = lo1 if lo1 > -GANA else -GANA
            whi = hi1 if hi1 < GANA else GANA
            hacer(tab, pts, lado, o, d, True, h, zp, zt, zs)
            if pts[lado] >= 3:
                v1 = GANA - ply * 1e-4
            else:
                v1 = -quiescencia(tab, pts, 1 - lado, -whi, -wlo, ply + 1, qd - 1, h, ctx_i, w, f, B,
                                  mo, md, ms, zp, zt, zs)
            deshacer(tab, pts, lado, o, d, antes, pa)
            if v1 <= lo1:
                v = p * v1 + q * GANA
            elif v1 >= hi1:
                v = p * v1 - q * GANA
            else:
                lo2 = (alpha - p * v1) / q
                hi2 = (beta - p * v1) / q
                if lo2 < -GANA:
                    lo2 = -GANA
                if hi2 > GANA:
                    hi2 = GANA
                hacer(tab, pts, lado, o, d, False, h, zp, zt, zs)
                v2 = -quiescencia(tab, pts, 1 - lado, -hi2, -lo2, ply + 1, qd - 1, h, ctx_i, w, f, B,
                                  mo, md, ms, zp, zt, zs)
                deshacer(tab, pts, lado, o, d, antes, pa)
                v = p * v1 + q * v2
        if v > mejor:
            mejor = v
        if v > alpha:
            alpha = v
        if alpha >= beta:
            break
    return mejor


@njit(cache=False)
def negamax(tab, pts, lado, prof, alpha, beta, ply, h, ctx_i, w, f, B, mo, md, ms,
            zp, zt, zs, tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos):
    """Expectiminimax en forma negamax con alfa-beta.

    Un ataque es un nodo de azar: valor = p·v(gana) + (1-p)·v(pierde). Las ventanas de los
    dos hijos salen de Star1 (Ballard, 1983) usando que todo valor está en [-1, 1].
    Las jugadas después de la primera se buscan con ventana nula (PVS) y las tranquilas
    tardías con una o dos capas menos (LMR); si mejoran alfa, se repiten completas.
    """
    ctx_i[0] += 1
    if ctx_i[0] > lim_nodos:            # se acabó el presupuesto de nodos: abortar
        ctx_i[1] = 1
        return 0.0
    if prof <= 0:
        return quiescencia(tab, pts, lado, alpha, beta, ply, prof + 3, h, ctx_i, w, f, B,
                           mo, md, ms, zp, zt, zs)
    alpha0 = alpha
    idx = h & TT_MASK
    tt_mov = -1
    if tt_k[idx] == h:
        tt_mov = tt_m[idx]
        if tt_d[idx] >= prof:
            v = tt_v[idx]
            fl = tt_f[idx]
            if fl == EXACTO:
                return v
            if fl == COTA_INF and v >= beta:
                return v
            if fl == COTA_SUP and v <= alpha:
                return v
    n = generar(tab, lado, mo, md, ply)
    if n == 0:                          # sin jugadas: pierde quien mueve
        return -GANA + ply * 1e-4
    ordenar(tab, lado, mo, md, ms, n, ply, tt_mov, killers, hist)
    mejor = -2.0
    mejor_cod = -1
    for i in range(n):
        o = mo[ply, i]
        d = md[ply, i]
        antes = tab[d]
        pa = pts[lado]
        cod = o * 81 + d
        red = 0
        if prof >= 3 and i >= 4 and antes == 0 and cod != killers[ply, 0] and cod != killers[ply, 1]:
            rd = d // 9
            prog_d = rd if lado == 0 else 8 - rd
            if prog_d <= 4:             # solo peones lejos de la meta
                red = 1
                if i >= 12 and prof >= 5:
                    red = 2
        a_lo = alpha
        a_hi = beta
        if i > 0:
            a_hi = alpha + 1e-7         # ventana nula
        while True:
            pr = prof - 1 - red
            if antes == 0:
                h2 = hacer(tab, pts, lado, o, d, True, h, zp, zt, zs)
                if pts[lado] >= 3:
                    v = GANA - ply * 1e-4
                else:
                    v = -negamax(tab, pts, 1 - lado, pr, -a_hi, -a_lo, ply + 1, h2, ctx_i, w, f, B,
                                 mo, md, ms, zp, zt, zs, tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos)
                deshacer(tab, pts, lado, o, d, antes, pa)
            else:
                p = prob_ataque(tab, lado, o, d)
                q = 1.0 - p
                lo1 = (a_lo - q * GANA) / p
                hi1 = (a_hi + q * GANA) / p
                wlo = lo1 if lo1 > -GANA else -GANA
                whi = hi1 if hi1 < GANA else GANA
                h2 = hacer(tab, pts, lado, o, d, True, h, zp, zt, zs)
                if pts[lado] >= 3:
                    v1 = GANA - ply * 1e-4
                else:
                    v1 = -negamax(tab, pts, 1 - lado, pr, -whi, -wlo, ply + 1, h2, ctx_i, w, f, B,
                                  mo, md, ms, zp, zt, zs, tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos)
                deshacer(tab, pts, lado, o, d, antes, pa)
                if ctx_i[1]:
                    return 0.0
                if v1 <= lo1:               # ni ganando el dado alcanza alfa
                    v = p * v1 + q * GANA
                elif v1 >= hi1:             # aun perdiendo el dado supera beta
                    v = p * v1 - q * GANA
                else:
                    lo2 = (a_lo - p * v1) / q
                    hi2 = (a_hi - p * v1) / q
                    if lo2 < -GANA:
                        lo2 = -GANA
                    if hi2 > GANA:
                        hi2 = GANA
                    h2 = hacer(tab, pts, lado, o, d, False, h, zp, zt, zs)
                    v2 = -negamax(tab, pts, 1 - lado, pr, -hi2, -lo2, ply + 1, h2, ctx_i, w, f, B,
                                  mo, md, ms, zp, zt, zs, tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos)
                    deshacer(tab, pts, lado, o, d, antes, pa)
                    v = p * v1 + q * v2
            if ctx_i[1]:
                return 0.0
            if v > alpha:
                if red > 0:                 # repetir sin reducción
                    red = 0
                    continue
                if a_hi < beta:             # repetir con ventana completa
                    a_hi = beta
                    continue
            break
        if v > mejor:
            mejor = v
            mejor_cod = cod
        if v > alpha:
            alpha = v
        if alpha >= beta:
            if antes == 0:
                if killers[ply, 0] != cod:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = cod
                hist[lado, cod] += prof * prof
            break
    fl = EXACTO
    if mejor <= alpha0:
        fl = COTA_SUP
    elif mejor >= beta:
        fl = COTA_INF
    if tt_k[idx] != h or tt_d[idx] <= prof:
        tt_k[idx] = h
        tt_v[idx] = mejor
        tt_d[idx] = prof
        tt_f[idx] = fl
        tt_m[idx] = mejor_cod
    return mejor


@njit(cache=False)
def raiz(tab, pts, lado, prof, h, ctx_i, w, f, B, mo, md, ms, zp, zt, zs,
         tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos, orden_o, orden_d, valores, nraiz):
    """Busca las jugadas de la raíz (PVS) en el orden dado. Regresa el índice de la mejor
    (-1 si se abortó antes de terminar la primera); `valores` queda con valores o cotas."""
    mejor = -3.0
    mejor_i = -1
    alpha = -GANA - 1.0
    beta = GANA + 1.0
    ply = 0
    for i in range(nraiz):
        o = orden_o[i]
        d = orden_d[i]
        antes = tab[d]
        pa = pts[lado]
        a_lo = alpha
        a_hi = beta
        if i > 0:
            a_hi = alpha + 1e-7
        while True:
            if antes == 0:
                h2 = hacer(tab, pts, lado, o, d, True, h, zp, zt, zs)
                if pts[lado] >= 3:
                    v = GANA
                else:
                    v = -negamax(tab, pts, 1 - lado, prof - 1, -a_hi, -a_lo, ply + 1, h2, ctx_i, w, f, B,
                                 mo, md, ms, zp, zt, zs, tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos)
                deshacer(tab, pts, lado, o, d, antes, pa)
            else:
                p = prob_ataque(tab, lado, o, d)
                q = 1.0 - p
                h2 = hacer(tab, pts, lado, o, d, True, h, zp, zt, zs)
                if pts[lado] >= 3:
                    v1 = GANA
                else:
                    v1 = -negamax(tab, pts, 1 - lado, prof - 1, -GANA, GANA, ply + 1, h2, ctx_i, w, f, B,
                                  mo, md, ms, zp, zt, zs, tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos)
                deshacer(tab, pts, lado, o, d, antes, pa)
                h2 = hacer(tab, pts, lado, o, d, False, h, zp, zt, zs)
                v2 = -negamax(tab, pts, 1 - lado, prof - 1, -GANA, GANA, ply + 1, h2, ctx_i, w, f, B,
                              mo, md, ms, zp, zt, zs, tt_k, tt_v, tt_d, tt_f, tt_m, killers, hist, lim_nodos)
                deshacer(tab, pts, lado, o, d, antes, pa)
                v = p * v1 + q * v2
            if ctx_i[1]:
                return mejor_i
            if i > 0 and v > alpha and a_hi < beta:
                a_hi = beta
                continue
            break
        valores[i] = v
        if v > mejor:
            mejor = v
            mejor_i = i
        if v > alpha:
            alpha = v
    return mejor_i


# ----------------------------------------------------------------------------------------
# Estado de la partida (se reinicia en cada preparar)
# ----------------------------------------------------------------------------------------
class _Ctx:
    pass


C = _Ctx()


def _nuevo_contexto():
    C.zp, C.zt, C.zs = _zobrist()
    C.tt_k = np.zeros(TT_SIZE, np.uint64)
    C.tt_v = np.zeros(TT_SIZE, np.float64)
    C.tt_d = np.full(TT_SIZE, -1, np.int16)
    C.tt_f = np.zeros(TT_SIZE, np.int8)
    C.tt_m = np.full(TT_SIZE, -1, np.int64)
    C.killers = np.full((MAXPLY, 2), -1, np.int64)
    C.hist = np.zeros((2, 81 * 81), np.float64)
    C.mo = np.zeros((MAXPLY, MAXMOV), np.int64)
    C.md = np.zeros((MAXPLY, MAXMOV), np.int64)
    C.ms = np.zeros((MAXPLY, MAXMOV), np.float64)
    C.f = np.zeros(NFEAT, np.float64)
    C.B = np.zeros(B_TAM, np.int64)
    C.ctx_i = np.zeros(2, np.int64)    # [nodos, abortado]
    C.w = PESOS.copy()
    C.nps = 300000.0                    # nodos/s; se recalibra con cada búsqueda


def _hash(tab, lado, pts):
    h = np.uint64(0)
    for sq in range(81):
        if tab[sq]:
            h ^= C.zp[tab[sq] - 1, sq]
    if lado == 1:
        h ^= C.zt
    h ^= C.zs[0, pts[0]]
    h ^= C.zs[1, pts[1]]
    return h


def _buscar(tab, pts, lado, presupuesto, prof_max=40, duro=None):
    """Profundización iterativa. `presupuesto`: tiempo objetivo; `duro`: tope absoluto.

    Regresa (origen, destino, valor, profundidad completada, nodos)."""
    inicio = time.perf_counter()
    if duro is None:
        duro = presupuesto * 2.5
    h = _hash(tab, lado, pts)
    n = generar(tab, lado, C.mo, C.md, 0)
    orden_o = C.mo[0, :n].copy()
    orden_d = C.md[0, :n].copy()
    valores = np.zeros(n, np.float64)
    if n == 1:
        return int(orden_o[0]), int(orden_d[0]), 0.0, 0, 0
    C.killers[:] = -1
    C.hist *= 0.25
    # orden inicial: anotaciones y ataques primero
    pri = np.zeros(n)
    meta = 8 if lado == 0 else 0
    for i in range(n):
        d = int(orden_d[i])
        if d // 9 == meta:
            pri[i] = 3.0
        elif tab[d] != 0:
            pri[i] = 1.0 + prob_ataque(tab, lado, int(orden_o[i]), d)
    idx = np.argsort(-pri, kind="stable")
    orden_o = orden_o[idx].copy()
    orden_d = orden_d[idx].copy()
    mejor_o, mejor_d, mejor_v, prof_hecha = int(orden_o[0]), int(orden_d[0]), 0.0, 0
    nodos_total = 0
    for prof in range(1, prof_max + 1):
        transcurrido = time.perf_counter() - inicio
        if prof > 1 and transcurrido > presupuesto * 0.45:
            break                       # la siguiente capa no cabría en el objetivo
        restante = duro - transcurrido
        lim = int(max(3000.0, restante * C.nps)) if prof > 1 else 10**9
        C.ctx_i[0] = 0
        C.ctx_i[1] = 0
        t0 = time.perf_counter()
        bi = raiz(tab, pts, lado, prof, h, C.ctx_i, C.w, C.f, C.B, C.mo, C.md, C.ms, C.zp, C.zt, C.zs,
                  C.tt_k, C.tt_v, C.tt_d, C.tt_f, C.tt_m, C.killers, C.hist, lim,
                  orden_o, orden_d, valores, n)
        dt = time.perf_counter() - t0
        nodos_total += int(C.ctx_i[0])
        if dt > 0.02:
            C.nps = 0.6 * C.nps + 0.4 * (C.ctx_i[0] / dt)
        if C.ctx_i[1]:
            # capa incompleta: solo la usamos si otra jugada ya superó a la mejor anterior
            if bi > 0:
                mejor_o, mejor_d, mejor_v = int(orden_o[bi]), int(orden_d[bi]), float(valores[bi])
            break
        mejor_o, mejor_d, mejor_v = int(orden_o[bi]), int(orden_d[bi]), float(valores[bi])
        prof_hecha = prof
        valores[bi] += 10.0             # la mejor primero; el resto por valor
        idx = np.argsort(-valores, kind="stable")
        valores[bi] -= 10.0
        orden_o = orden_o[idx].copy()
        orden_d = orden_d[idx].copy()
        valores = valores[idx].copy()
        if abs(mejor_v) >= GANA - 0.01:
            break                       # victoria o derrota segura: no hace falta más
    return mejor_o, mejor_d, mejor_v, prof_hecha, nodos_total


def _presupuesto(tiempo_restante, turno):
    """Tiempo objetivo y tope de la jugada número `turno` (contando las del libro): reparto
    parejo del reloj que queda entre las jugadas que faltan (estimadas), con menos tiempo en la
    apertura y 2.5 s de reserva."""
    reserva = 2.5
    disponible = tiempo_restante - reserva
    if disponible <= 0.05:
        return 0.02, 0.04
    faltan = max(10, 42 - turno)
    objetivo = disponible / faltan
    if turno <= 8:
        objetivo *= 0.5
    objetivo = min(1.5, objetivo)
    duro = min(objetivo * 2.0, disponible * 0.12)
    return objetivo, duro


def _calentar():
    """Compila todo numba fuera del reloj (búsquedas cortas, con y sin ataques)."""
    tab = np.zeros(81, np.int8)
    tab[:18] = 1
    tab[63:] = 2
    pts = np.zeros(2, np.int64)
    _buscar(tab, pts, 0, 0.05, prof_max=3)
    tab2 = np.zeros(81, np.int8)
    tab2[31] = 1    # e4
    tab2[21] = 1    # d3
    tab2[23] = 1    # f3
    tab2[41] = 2    # f5
    tab2[51] = 2    # g6
    prob_ataque(tab2, 0, 31, 41)
    _buscar(tab2, pts, 0, 0.05, prof_max=3)


ESPERA_PREPARAR = 14.0   # s que esperamos la compilación dentro de preparar (límite: 20 s)
LIBRO = {}               # clave de posición -> jugada en texto (data/aperturas.json)
PARTIDA = {"turno": 0}   # jugadas propias hechas en esta partida (libro incluido)


def _cargar_datos(carpeta):
    """Lee data/pesos.json y data/aperturas.json. Cualquier problema (falta el archivo, otros
    nombres de rasgos, números raros) deja los pesos de este archivo y el libro vacío."""
    global VENTAJA
    PESOS[:] = 0.0
    PESOS[:NFEAT_USADOS] = PESOS_AJUSTADOS
    VENTAJA = VENTAJA_ARCHIVO
    LIBRO.clear()
    if carpeta is None:
        return
    carpeta = Path(carpeta)
    try:
        d = json.loads((carpeta / "pesos.json").read_text(encoding="utf-8"))
        w = np.array(d["pesos"], dtype=np.float64)
        v = float(d.get("ventaja", VENTAJA_ARCHIVO))
        if (d.get("rasgos") == NOMBRES_RASGOS and w.shape == (NFEAT_USADOS,)
                and np.all(np.isfinite(w)) and 0.0 <= v <= 3.0):
            PESOS[:NFEAT_USADOS] = w
            VENTAJA = v
    except Exception:
        pass
    try:
        d = json.loads((carpeta / "aperturas.json").read_text(encoding="utf-8"))
        for k, v in d["posiciones"].items():
            LIBRO[str(k)] = str(v)
    except Exception:
        LIBRO.clear()


def clave_posicion(tab, pts, lado):
    """Clave del libro: las 81 casillas (0/1/2), quién mueve y los puntos."""
    return "".join(str(int(v)) for v in tab) + ("b" if lado == 0 else "n") + f"{int(pts[0])}{int(pts[1])}"


def _compilar():
    """Compila y deja listo el contexto de la partida (corre en un hilo aparte)."""
    try:
        _calentar()
        _nuevo_contexto()
        C.w[NFEAT - 1] = VENTAJA if mi_lado == 0 else -VENTAJA
        C.listo = True
    except Exception:
        C.listo = False


def preparar(config):
    """Lee data/ y compila numba fuera del reloj. Si la máquina fuera tan lenta que compilar no
    cupiera en ESPERA_PREPARAR, regresamos de todos modos (no perder por `preparar`) y la
    compilación termina durante nuestras primeras jugadas (ver `jugar`)."""
    global mi_lado
    mi_lado = 0 if config.color == "blancas" else 1
    PARTIDA["turno"] = 0
    _cargar_datos(getattr(config, "carpeta_datos", None))
    _nuevo_contexto()
    C.listo = False
    C.hilo = threading.Thread(target=_compilar, daemon=True)
    C.hilo.start()
    C.hilo.join(ESPERA_PREPARAR)


# ----------------------------------------------------------------------------------------
# Interfaz con el motor oficial
# ----------------------------------------------------------------------------------------
_COLUMNAS = "abcdefghi"
_CASILLA = {f"{_COLUMNAS[c]}{r + 1}": r * 9 + c for r in range(9) for c in range(9)}


def _desde_estado(estado):
    """Tablero, puntos y turno a partir de la API pública del motor."""
    tab = np.zeros(81, np.int8)
    for nombre in estado.piezas("blancas"):
        tab[_CASILLA[nombre]] = 1
    for nombre in estado.piezas("negras"):
        tab[_CASILLA[nombre]] = 2
    p = estado.puntos
    pts = np.array([p["blancas"], p["negras"]], np.int64)
    lado = 0 if estado.turno == "blancas" else 1
    return tab, pts, lado


def _jugada_segura(estado, legales):
    """Respaldo instantáneo (solo si algo falla): anotar, el mejor ataque o el mayor avance."""
    lado = 0 if estado.turno == "blancas" else 1
    meta = 8 if lado == 0 else 0
    mejor, mejor_v = legales[0], -1e9
    for j in legales:
        fila = j.destino // 9
        v = (fila if lado == 0 else 8 - fila) * 0.1
        if fila == meta:
            v += 100.0
        elif estado.es_ataque(j):
            v += 2.0 * estado.prob_ataque(j) - 0.5
        if v > mejor_v:
            mejor, mejor_v = j, v
    return mejor


def _del_libro(estado, legales):
    """Jugada del libro de aperturas para esta posición (o None)."""
    if not LIBRO:
        return None
    try:
        tab, pts, lado = _desde_estado(estado)
        texto = LIBRO.get(clave_posicion(tab, pts, lado))
        if texto is not None:
            for j in legales:
                if str(j) == texto:
                    return j
    except Exception:
        pass
    return None


def jugar(estado, tiempo_restante):
    legales = estado.jugadas()
    PARTIDA["turno"] += 1
    if len(legales) == 1:
        return legales[0]
    libro = _del_libro(estado, legales)
    if not getattr(C, "listo", False):
        # la compilación no terminó en preparar: esperarla un poco con nuestro reloj
        hilo = getattr(C, "hilo", None)
        if hilo is not None and hilo.is_alive():
            hilo.join(max(0.0, min(4.0, tiempo_restante - 22.0)))
        if not getattr(C, "listo", False):
            return libro if libro is not None else _jugada_segura(estado, legales)
    if libro is not None:
        reportar(libro=1)
        return libro
    try:
        tab, pts, lado = _desde_estado(estado)
        objetivo, duro = _presupuesto(tiempo_restante, PARTIDA["turno"])
        if tiempo_restante < 0.4:
            objetivo, duro = 0.01, 0.03
        t0 = time.perf_counter()
        o, d, v, prof, nodos = _buscar(tab, pts, lado, objetivo, duro=duro)
        for j in legales:
            if j.origen == o and j.destino == d:
                reportar(prof=prof, nodos=nodos, eval=round(v, 3), t=round(time.perf_counter() - t0, 3))
                return j
    except Exception:
        pass
    return _jugada_segura(estado, legales)
