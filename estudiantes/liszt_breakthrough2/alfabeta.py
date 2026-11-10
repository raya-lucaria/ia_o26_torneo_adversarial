"""Expectiminimax con poda alfa-beta (Star1) en numba para «Breakthrough de combate».

Idea: el bot piensa varias jugadas adelante, suponiendo que el rival también elige su mejor
jugada (negamax). Cada ataque es un nodo de azar: su valor es p·V(gana) + (1−p)·V(pierde),
calculado exacto, y Star1 poda también en esos nodos usando las cotas de la evaluación.

- Profundización iterativa (desde strategy.py): siempre hay una jugada lista.
- Tabla de transposición con hash Zobrist del motor (`nucleo.clave`).
- Orden de jugadas: jugada de la tabla, anotar, ataques por probabilidad, jugadas «killer».
- Quiescencia: al llegar a la profundidad 0 seguimos solo con ataques y jugadas que anotan,
  para no cortar a la mitad de un intercambio.
- Evaluación en puntos enteros (los pesos son enteros en el arreglo W).

Valores siempre desde el bando que mueve en ese nodo (negamax).
"""
import math

import numpy as np
from numba import njit

from motor import nucleo

GANA = 5200.0            # valor de una victoria (menos la profundidad, para preferir ganar antes)
MAXPLY = 64
MAXJ = 54
QPROF = 6                # medias jugadas máximas de quiescencia
TT_BITS = 20             # 2^20 entradas en la tabla de transposición (~40 MB)
TT_MASCARA = np.uint64((1 << TT_BITS) - 1)
NF = 20                  # número de características de la evaluación

# --- evaluación: modelo logístico ---
# La evaluación estima la probabilidad de que gane el bando que mueve:
#     P = 1 / (1 + exp(-Σ W[i]·F[i] / 1000))        (W en milésimas de logit, enteros)
# y la búsqueda usa 10000·P − 5000 (de −5000 a 5000). Como es una probabilidad, el promedio en
# los nodos de azar (p·V_gana + (1−p)·V_pierde) es exacto. Las victorias valen más que cualquier
# evaluación (GANA − profundidad > 5000).
#
# Características F (desde el bando que mueve, «mío − suyo» salvo que se diga otra cosa):
#   0      peones
#   1..7   peones con avance 1..7 (0 = fila inicial, 7 = a un paso de anotar)
#   8      apoyos (peones propios en diagonal de atrás)
#   9, 10  peones libres (cono sin rivales): cuántos y su avance total
#   11     puntos anotados
#   12     carrera a 3 puntos con peones libres: +1 la gano, −1 la pierdo, 0 nadie
#   13     tengo un peón a un paso de anotar (anoto ya): 0/1
#   14     peones del rival a un paso de anotar (debo capturarlos)
#   15, 16 mejor probabilidad de ataque disponible (×100): mía y del rival
#   17, 18 ataques con p ≥ 0.65: míos y del rival
#   19     constante (tener el turno)
# Pesos por defecto (a mano); los ajustados con partidas viven en data/pesos.json.
PESOS = np.array([350,
                  14, 35, 70, 122, 210, 350, 560,
                  42,
                  210, 140,
                  3500,
                  6000,
                  3000, -1750,
                  7, -7,
                  50, -90,
                  35], dtype=np.int64)


@njit(cache=True)
def _ins3(AC, base, d):
    """Mantiene en AC[base..base+2] las tres distancias más cortas, ordenadas."""
    if d < AC[base]:
        AC[base + 2] = AC[base + 1]
        AC[base + 1] = AC[base]
        AC[base] = d
    elif d < AC[base + 1]:
        AC[base + 2] = AC[base + 1]
        AC[base + 1] = d
    elif d < AC[base + 2]:
        AC[base + 2] = d


@njit(cache=True)
def _carrera(falta, AC, base):
    """Jugadas propias para sumar `falta` puntos con peones libres (>= 99: no alcanza)."""
    if falta <= 1:
        return AC[base]
    if falta == 2:
        return AC[base] + AC[base + 1]
    return AC[base] + AC[base + 1] + AC[base + 2]


@njit(cache=True)
def caracteristicas(tab, pts, c, AC, F):
    """Llena F (NF enteros) desde el bando `c`, que es el que mueve. AC: búfer de 40 enteros."""
    for i in range(40):
        AC[i] = 0
    for s in range(2):
        AC[s * 20 + 14] = 99
        AC[s * 20 + 15] = 99
        AC[s * 20 + 16] = 99
    for idx in range(81):
        pieza = tab[idx]
        if pieza == 0:
            continue
        s = pieza - 1
        base = s * 20
        r = idx // 9
        col = idx % 9
        if s == 0:
            adv = r
            fw = 1
        else:
            adv = 8 - r
            fw = -1
        enemigo = 2 - s
        AC[base] += 1
        if adv >= 1:
            AC[base + adv] += 1
        sup = nucleo.apoyo(tab, idx, s)
        AC[base + 8] += sup
        # cono de avance: ¿hay rivales que puedan cruzarse en el camino?
        libre = True
        k = 1
        rr = r + fw
        while 0 <= rr <= 8:
            lo = col - k if col - k > 0 else 0
            hi = col + k if col + k < 8 else 8
            for cc in range(lo, hi + 1):
                if tab[rr * 9 + cc] == enemigo:
                    libre = False
                    break
            if not libre:
                break
            k += 1
            rr += fw
        if adv == 7:
            AC[base + 11] += 1
        if libre:
            AC[base + 9] += 1
            AC[base + 10] += adv
            _ins3(AC, base + 14, 8 - adv)
        # ataques que este peón puede hacer ya
        rr = r + fw
        for t in range(2):
            cc = col - 1 if t == 0 else col + 1
            if cc < 0 or cc > 8:
                continue
            d = rr * 9 + cc
            if tab[d] == enemigo:
                pr = nucleo.prob_combate(sup, nucleo.apoyo(tab, d, 1 - s))
                q = int(pr * 100.0 + 0.5)
                if q > AC[base + 12]:
                    AC[base + 12] = q
                if q >= 65:
                    AC[base + 13] += 1
    o = 1 - c
    bc = c * 20
    bo = o * 20
    for k in range(11):
        F[k] = AC[bc + k] - AC[bo + k]
    F[11] = pts[c] - pts[o]
    tm = _carrera(3 - pts[c], AC, bc + 14)
    ts = _carrera(3 - pts[o], AC, bo + 14)
    if tm < 99 or ts < 99:
        F[12] = 1 if tm <= ts else -1  # muevo yo primero: empate de carrera a mi favor
    else:
        F[12] = 0
    F[13] = 1 if AC[bc + 11] > 0 else 0
    F[14] = AC[bo + 11]
    F[15] = AC[bc + 12]
    F[16] = AC[bo + 12]
    F[17] = AC[bc + 13]
    F[18] = AC[bo + 13]
    F[19] = 1


@njit(cache=True)
def evaluar(tab, pts, c, W, AC, F):
    """10000·P(gana el que mueve) − 5000, con P del modelo logístico."""
    caracteristicas(tab, pts, c, AC, F)
    s = 0
    for i in range(NF):
        s += W[i] * F[i]
    x = s / 1000.0
    if x > 30.0:
        x = 30.0
    elif x < -30.0:
        x = -30.0
    return 10000.0 / (1.0 + math.exp(-x)) - 5000.0


@njit(cache=True)
def _ordenar(tab, c, n, bo, bd, sc, mtt, k1, k2):
    """Calcula el puntaje de orden de cada jugada (mayor = se prueba antes)."""
    meta = 8 if c == 0 else 0
    for k in range(n):
        o = bo[k]
        d = bd[k]
        cod = o * 81 + d
        if cod == mtt:
            sc[k] = 1e9
        elif d // 9 == meta:
            sc[k] = 1e8 + nucleo.prob_jugada(tab, c, o, d)
        elif tab[d] != 0:
            p = nucleo.prob_jugada(tab, c, o, d)
            fila_v = d // 9 if c == 1 else 8 - d // 9  # qué tan cerca de anotar estaba la víctima
            sc[k] = 1e6 + p * 1000.0 + (8 - fila_v) * 20.0
        elif cod == k1 or cod == k2:
            sc[k] = 1e5
        else:
            sc[k] = (d // 9 if c == 0 else 8 - d // 9) * 1.0


@njit(cache=True)
def _siguiente(n, sc, i):
    """Selección parcial: deja en la posición i la jugada con mejor puntaje restante."""
    best = i
    for k in range(i + 1, n):
        if sc[k] > sc[best]:
            best = k
    return best


@njit(cache=True)
def negamax(tab, pts, c, prof, alpha, beta, ply,
            BO, BD, SC, KILL, TTK, TTV, TTD, TTF, TTM, ZP, ZT, ZPU, W, AC, F, ctr, out):
    """Valor (desde `c`) con ventana [alpha, beta]. prof <= 0 es quiescencia.

    ctr: [nodos, limite_nodos, abortado]. out: [origen, destino, valor] de la raíz.
    """
    ctr[0] += 1
    if ctr[0] > ctr[1]:
        ctr[2] = 1
        return 0.0
    if pts[0] >= 3:
        return (GANA - ply) if c == 0 else -(GANA - ply)
    if pts[1] >= 3:
        return (GANA - ply) if c == 1 else -(GANA - ply)
    bo = BO[ply]
    bd = BD[ply]
    sc = SC[ply]
    n = nucleo.generar(tab, c, bo, bd)
    if n == 0:
        return -(GANA - ply)  # sin jugadas: pierde quien mueve

    quies = prof <= 0
    alpha0 = alpha
    mejor = -1e18
    mtt = -1
    h = np.uint64(0)
    slot = 0
    if quies:
        estatico = evaluar(tab, pts, c, W, AC, F)
        if estatico >= beta or prof <= -QPROF:
            return estatico
        if estatico > alpha:
            alpha = estatico
        mejor = estatico
    else:
        h = nucleo.clave(tab, c, pts, ZP, ZT, ZPU)
        slot = np.int64(h & TT_MASCARA)
        if TTK[slot] == h:
            mtt = TTM[slot]
            if TTD[slot] >= prof and ply > 0:
                v = TTV[slot]
                f = TTF[slot]
                if f == 0:
                    return v
                if f == 1 and v >= beta:
                    return v
                if f == 2 and v <= alpha:
                    return v

    _ordenar(tab, c, n, bo, bd, sc, mtt, KILL[ply, 0], KILL[ply, 1])
    meta = 8 if c == 0 else 0
    mejor_cod = -1
    for i in range(n):
        b = _siguiente(n, sc, i)
        if b != i:  # intercambio para que la i-ésima sea la mejor restante
            to = bo[i]
            bo[i] = bo[b]
            bo[b] = to
            td = bd[i]
            bd[i] = bd[b]
            bd[b] = td
            ts = sc[i]
            sc[i] = sc[b]
            sc[b] = ts
        o = bo[i]
        d = bd[i]
        ataque = tab[d] != 0
        if quies and not ataque and d // 9 != meta:
            continue  # en quiescencia solo ataques y jugadas que anotan
        g_o = tab[o]
        g_d = tab[d]
        g_p = pts[c]
        if not ataque:
            nucleo.aplicar(tab, pts, c, o, d, True)
            v = -negamax(tab, pts, 1 - c, prof - 1, -beta, -alpha, ply + 1,
                         BO, BD, SC, KILL, TTK, TTV, TTD, TTF, TTM, ZP, ZT, ZPU, W, AC, F, ctr, out)
            tab[o] = g_o
            tab[d] = g_d
            pts[c] = g_p
        else:
            # Nodo de azar (Star1): cotas de los hijos en [-GANA, GANA]
            p = nucleo.prob_jugada(tab, c, o, d)
            q = 1.0 - p
            L = -GANA
            U = GANA
            A1 = (alpha - q * U) / p
            B1 = (beta - q * L) / p
            a1 = A1 if A1 > L else L
            b1 = B1 if B1 < U else U
            nucleo.aplicar(tab, pts, c, o, d, True)
            v1 = -negamax(tab, pts, 1 - c, prof - 1, -b1, -a1, ply + 1,
                          BO, BD, SC, KILL, TTK, TTV, TTD, TTF, TTM, ZP, ZT, ZPU, W, AC, F, ctr, out)
            tab[o] = g_o
            tab[d] = g_d
            pts[c] = g_p
            if ctr[2] != 0:
                return 0.0
            if v1 >= B1:
                v = p * v1 + q * L
            elif v1 <= A1:
                v = p * v1 + q * U
            else:
                A2 = (alpha - p * v1) / q
                B2 = (beta - p * v1) / q
                a2 = A2 if A2 > L else L
                b2 = B2 if B2 < U else U
                nucleo.aplicar(tab, pts, c, o, d, False)
                v2 = -negamax(tab, pts, 1 - c, prof - 1, -b2, -a2, ply + 1,
                              BO, BD, SC, KILL, TTK, TTV, TTD, TTF, TTM, ZP, ZT, ZPU, W, AC, F, ctr, out)
                tab[o] = g_o
                tab[d] = g_d
                pts[c] = g_p
                v = p * v1 + q * v2
        if ctr[2] != 0:
            return 0.0
        if v > mejor:
            mejor = v
            mejor_cod = o * 81 + d
            if ply == 0:
                out[0] = o
                out[1] = d
                out[2] = v
        if v > alpha:
            alpha = v
        if alpha >= beta:
            if not ataque and not quies:
                cod = o * 81 + d
                if KILL[ply, 0] != cod:
                    KILL[ply, 1] = KILL[ply, 0]
                    KILL[ply, 0] = cod
            break

    if not quies:
        if mejor <= alpha0:
            f = 2
        elif mejor >= beta:
            f = 1
        else:
            f = 0
        if TTK[slot] != h or TTD[slot] <= prof:
            TTK[slot] = h
            TTV[slot] = mejor
            TTD[slot] = prof
            TTF[slot] = f
            TTM[slot] = mejor_cod
    return mejor


class Buscador:
    """Guarda los búferes (se reservan una vez por partida) y hace la profundización iterativa."""

    def __init__(self, pesos=None):
        self.W = np.array(PESOS if pesos is None or len(pesos) != NF else pesos, dtype=np.int64)
        size = 1 << TT_BITS  # debe cuadrar con TT_MASCARA
        self.TTK = np.zeros(size, np.uint64)
        self.TTV = np.zeros(size, np.float64)
        self.TTD = np.full(size, -100, np.int64)
        self.TTF = np.zeros(size, np.int64)
        self.TTM = np.full(size, -1, np.int64)
        self.BO = np.zeros((MAXPLY + QPROF + 2, MAXJ), np.int64)
        self.BD = np.zeros((MAXPLY + QPROF + 2, MAXJ), np.int64)
        self.SC = np.zeros((MAXPLY + QPROF + 2, MAXJ), np.float64)
        self.KILL = np.full((MAXPLY + QPROF + 2, 2), -1, np.int64)
        self.ctr = np.zeros(3, np.int64)
        self.AC = np.zeros(40, np.int64)
        self.F = np.zeros(NF, np.int64)
        self.out = np.zeros(3, np.float64)
        self.tab = np.zeros(81, np.int8)
        self.pts = np.zeros(2, np.int64)
        self.nps = 300000.0  # nodos por segundo; se corrige midiendo

    def _una(self, tablero, color, puntos, prof, limite):
        self.tab[:] = tablero
        self.pts[:] = puntos
        self.ctr[0] = 0
        self.ctr[1] = limite
        self.ctr[2] = 0
        self.out[0] = -1
        v = negamax(self.tab, self.pts, color, prof, -GANA - 1.0, GANA + 1.0, 0,
                    self.BO, self.BD, self.SC, self.KILL,
                    self.TTK, self.TTV, self.TTD, self.TTF, self.TTM,
                    nucleo.Z_PIEZAS, nucleo.Z_TURNO, nucleo.Z_PUNTOS, self.W, self.AC, self.F,
                    self.ctr, self.out)
        return v, int(self.ctr[0]), self.ctr[2] != 0

    def buscar(self, tablero, color, puntos, segundos, prof_max=40):
        """Profundización iterativa. Regresa (origen, destino, info)."""
        import time
        t0 = time.perf_counter()
        fin = t0 + segundos
        mejor = None
        info = {"prof": 0, "nodos": 0, "eval": 0.0}
        self.KILL[:] = -1
        total = 0
        t_ant = None
        for prof in range(1, prof_max + 1):
            ahora = time.perf_counter()
            resto = fin - ahora
            if resto <= 0:
                break
            if mejor is not None and t_ant is not None and t_ant * 3.0 > resto:
                break  # la siguiente profundidad casi seguro no termina
            limite = max(2000, int(self.nps * resto))
            v, nodos, abortado = self._una(tablero, color, puntos, prof, limite)
            dt = time.perf_counter() - ahora
            total += nodos
            if dt > 0.01:
                self.nps = 0.7 * self.nps + 0.3 * (nodos / dt)
            if abortado:
                break
            mejor = (int(self.out[0]), int(self.out[1]))
            info = {"prof": prof, "nodos": total, "eval": round(float(v), 1)}
            t_ant = dt
            if abs(v) >= GANA - 100:
                break  # victoria o derrota ya vista
        if mejor is None:
            # ni la profundidad 1 terminó: la jugada que haya dejado la raíz, o None
            if self.out[0] >= 0:
                mejor = (int(self.out[0]), int(self.out[1]))
        info["nodos"] = total
        return mejor, info


def calentar():
    """Compila todo con los mismos tipos que en la partida (cuenta en `preparar`)."""
    b = Buscador()
    t = np.zeros(81, np.int8)
    t[0:18] = 1
    t[63:81] = 2
    t[40] = 1   # un peón en medio para que haya ataques y quiescencia
    t[48] = 2
    b.buscar(t, 0, np.zeros(2, np.int64), 0.05, prof_max=3)
