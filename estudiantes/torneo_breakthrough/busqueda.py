"""Búsqueda expectimax alfa-beta (*-minimax con poda Star1, Ballard 1983) en numba.

- Negamax: valores en [-1, 1] desde el jugador en turno (1 = gana seguro). La evaluación es
  tanh(...), así que queda dentro de esas cotas, como necesita Star1.
- Cada ataque es un nodo de azar con dos resultados de probabilidad conocida (gana p, pierde 1-p).
  Star1 transforma la ventana (alpha, beta) para cada resultado y poda sin perder exactitud.
- PVS (Reinefeld 1983): la primera jugada con la ventana completa y las demás con ventana nula;
  solo se repite la búsqueda si una supera a alpha.
- Quiescencia: anotar y ataques con p >= p mínima, también con ventanas Star1.
- Tabla de transposición (clave Zobrist), killers y history para ordenar.
- Caché de evaluación por clave Zobrist (ev_k, ev_v): la quiescencia no repite una evaluación.
- Victorias más cortas valen más: GANA - DIST·ply.
- Evaluación: la lineal aprendida (rasgos.py) con los pesos `pesos` (int16, ver rasgos.cuantizar).
Los parámetros de búsqueda viajan en ctx, así que el laboratorio compara variantes con el mismo
código. Este módulo no guarda estado: todos los arreglos los crea y pasa strategy.py.
"""

import numpy as np
from numba import njit

from rasgos import evaluar_lineal
from tablero_bits import CASILLAS, LADO, deshacer, generar, generar_tacticas, hacer, ocupada, prob

MAX_PLY = 96
MAX_JUG = 64
L = -1.0
U = 1.0
GANA = 1.0
DIST = 1e-4  # castigo por ply para preferir victorias rápidas (sigue dentro de [L, U])
EPS = 1e-7  # ancho de la ventana nula de PVS
EXACTO = 0
COTA_INF = 1  # el valor real es >= el guardado
COTA_SUP = 2  # el valor real es <= el guardado

# Índices de ctx (int64): contadores, límites, parámetros y la mejor jugada de la raíz.
C_NODOS = 0
C_QNODOS = 1
C_MAX = 2
C_ABORTA = 3
C_MEJOR = 4
C_QPMIN = 5  # p mínima (en %) de los ataques en la quiescencia
C_PVS = 6  # 1 = PVS
N_CTX = 16


@njit(cache=False)
def puntuar(bb, c, lista, n, puntajes, tt_jug, ply, killers, historia):
    """Orden: jugada de la TT, anotar, ataques (más p y objetivo más avanzado), killers, history."""
    meta = LADO - 1 if c == 0 else 0
    for i in range(n):
        m = lista[i]
        o = m // CASILLAS
        d = m % CASILLAS
        if m == tt_jug:
            puntajes[i] = 1 << 30
        elif d // LADO == meta:
            puntajes[i] = 1 << 29
        elif ocupada(bb, 1 - c, d):
            avance = d // LADO if c == 0 else LADO - 1 - d // LADO
            puntajes[i] = (1 << 28) + int(1000 * prob(bb, c, o, d)) * 16 + avance
        elif m == killers[ply, 0]:
            puntajes[i] = 1 << 27
        elif m == killers[ply, 1]:
            puntajes[i] = (1 << 27) - 1
        else:
            puntajes[i] = historia[c, m]


@njit(cache=False)
def siguiente(lista, puntajes, i, n):
    """Selección parcial: pone en `i` la jugada de mayor puntaje que falta."""
    b = i
    for j in range(i + 1, n):
        if puntajes[j] > puntajes[b]:
            b = j
    if b != i:
        lista[i], lista[b] = lista[b], lista[i]
        puntajes[i], puntajes[b] = puntajes[b], puntajes[i]
    return lista[i]


@njit(cache=False)
def quiescencia(bb, pts, c, alpha, beta, ply, qprof, h, movs, punt, ctx, pesos, ev_k, ev_v):
    ctx[C_QNODOS] += 1
    i_ev = h & np.uint64(ev_k.shape[0] - 1)  # caché de evaluación por clave Zobrist
    if ev_k[i_ev] == h:
        quieto = ev_v[i_ev]
    else:
        quieto = evaluar_lineal(bb, pts, c, pesos)
        ev_k[i_ev] = h
        ev_v[i_ev] = quieto
    if quieto >= beta or qprof <= 0 or ply >= MAX_PLY - 1:
        return quieto
    if quieto > alpha:
        alpha = quieto
    # Solo jugadas tácticas: anotar y ataques con p suficiente. (Si no hay ninguna se queda el
    # valor estático: quedarse sin jugadas lo detecta alfabeta, no la quiescencia.)
    lista = movs[ply]
    n = generar_tacticas(bb, c, lista)
    meta = LADO - 1 if c == 0 else 0
    k = 0
    for i in range(n):
        m = lista[i]
        d = m % CASILLAS
        if d // LADO == meta:
            punt[ply, k] = 1 << 20
            lista[k] = m
            k += 1
        else:  # un ataque
            p = prob(bb, c, m // CASILLAS, d)
            if p * 100.0 >= ctx[C_QPMIN]:
                punt[ply, k] = int(p * 1000)
                lista[k] = m
                k += 1
    mejor = quieto
    for i in range(k):
        m = siguiente(lista, punt[ply], i, k)
        o = m // CASILLAS
        d = m % CASILLAS
        if not ocupada(bb, 1 - c, d):  # anota caminando
            rival, anoto, h2 = hacer(bb, pts, c, o, d, True, h)
            if pts[c] >= 3:
                v = GANA - DIST * (ply + 1)
            else:
                v = -quiescencia(bb, pts, 1 - c, -beta, -alpha, ply + 1, qprof - 1, h2, movs, punt,
                                 ctx, pesos, ev_k, ev_v)
            deshacer(bb, pts, c, o, d, rival, anoto)
        else:
            p = prob(bb, c, o, d)
            q = 1.0 - p
            lo_a = (alpha - q * U) / p
            hi_a = (beta - q * L) / p
            rival, anoto, h2 = hacer(bb, pts, c, o, d, True, h)
            if pts[c] >= 3:
                a = GANA - DIST * (ply + 1)
            else:
                a = -quiescencia(bb, pts, 1 - c, -min(hi_a, U), -max(lo_a, L), ply + 1, qprof - 1,
                                 h2, movs, punt, ctx, pesos, ev_k, ev_v)
            deshacer(bb, pts, c, o, d, rival, anoto)
            if a <= lo_a:
                v = p * a + q * U
            elif a >= hi_a:
                v = p * a + q * L
            else:
                lo_b = (alpha - p * a) / q
                hi_b = (beta - p * a) / q
                rival, anoto, h2 = hacer(bb, pts, c, o, d, False, h)
                b = -quiescencia(bb, pts, 1 - c, -min(hi_b, U), -max(lo_b, L), ply + 1, qprof - 1,
                                 h2, movs, punt, ctx, pesos, ev_k, ev_v)
                deshacer(bb, pts, c, o, d, rival, anoto)
                v = p * a + q * b
        if v > mejor:
            mejor = v
            if v > alpha:
                alpha = v
                if v >= beta:
                    break
    return mejor


@njit(cache=False)
def tt_guardar(tt_k, tt_v, tt_p, tt_f, tt_m, h, prof, valor, bandera, jugada):
    i = h & np.uint64(tt_k.shape[0] - 1)
    if tt_k[i] != h or prof >= tt_p[i]:
        tt_k[i] = h
        tt_v[i] = valor
        tt_p[i] = prof
        tt_f[i] = bandera
        tt_m[i] = jugada


@njit(cache=False)
def alfabeta(bb, pts, c, prof, alpha, beta, ply, h, movs, punt, killers, historia,
             tt_k, tt_v, tt_p, tt_f, tt_m, ctx, qprof, pesos, ev_k, ev_v):
    """Valor desde `c`. En la raíz (ply 0) deja la mejor jugada en ctx[C_MEJOR]."""
    ctx[C_NODOS] += 1
    if ctx[C_NODOS] + ctx[C_QNODOS] >= ctx[C_MAX]:
        ctx[C_ABORTA] = 1
        return 0.0
    if prof <= 0 or ply >= MAX_PLY - 1:
        return quiescencia(bb, pts, c, alpha, beta, ply, qprof, h, movs, punt, ctx, pesos, ev_k, ev_v)
    alpha0 = alpha
    i_tt = h & np.uint64(tt_k.shape[0] - 1)
    tt_jug = -1
    if tt_k[i_tt] == h:
        tt_jug = tt_m[i_tt]
        if tt_p[i_tt] >= prof and ply > 0:
            v = tt_v[i_tt]
            f = tt_f[i_tt]
            if f == EXACTO or (f == COTA_INF and v >= beta) or (f == COTA_SUP and v <= alpha):
                return v
    lista = movs[ply]
    n = generar(bb, c, lista)
    if n == 0:
        return -(GANA - DIST * ply)
    puntuar(bb, c, lista, n, punt[ply], tt_jug, ply, killers, historia)
    mejor = -2.0
    mejor_jug = lista[0]
    for i in range(n):
        m = siguiente(lista, punt[ply], i, n)
        o = m // CASILLAS
        d = m % CASILLAS
        ataque = ocupada(bb, 1 - c, d)
        # Ventana (lo, hi): nula en PVS para toda jugada que no es la primera.
        lo = alpha
        hi = alpha + EPS if ctx[C_PVS] == 1 and i > 0 else beta
        while True:
            if not ataque:
                rival, anoto, h2 = hacer(bb, pts, c, o, d, True, h)
                if pts[c] >= 3:
                    v = GANA - DIST * (ply + 1)
                else:
                    v = -alfabeta(bb, pts, 1 - c, prof - 1, -hi, -lo, ply + 1, h2, movs, punt,
                                  killers, historia, tt_k, tt_v, tt_p, tt_f, tt_m, ctx, qprof,
                                  pesos, ev_k, ev_v)
                deshacer(bb, pts, c, o, d, rival, anoto)
            else:
                # Nodo de azar Star1: primero el resultado «gana» con su ventana transformada.
                p = prob(bb, c, o, d)
                q = 1.0 - p
                lo_a = (lo - q * U) / p
                hi_a = (hi - q * L) / p
                rival, anoto, h2 = hacer(bb, pts, c, o, d, True, h)
                if pts[c] >= 3:
                    a = GANA - DIST * (ply + 1)
                else:
                    a = -alfabeta(bb, pts, 1 - c, prof - 1, -min(hi_a, U), -max(lo_a, L), ply + 1,
                                  h2, movs, punt, killers, historia, tt_k, tt_v, tt_p, tt_f, tt_m,
                                  ctx, qprof, pesos, ev_k, ev_v)
                deshacer(bb, pts, c, o, d, rival, anoto)
                if a <= lo_a:
                    v = p * a + q * U  # cota superior: no llega a lo
                elif a >= hi_a:
                    v = p * a + q * L  # cota inferior: ya pasa hi
                else:
                    lo_b = (lo - p * a) / q
                    hi_b = (hi - p * a) / q
                    rival, anoto, h2 = hacer(bb, pts, c, o, d, False, h)
                    b = -alfabeta(bb, pts, 1 - c, prof - 1, -min(hi_b, U), -max(lo_b, L), ply + 1,
                                  h2, movs, punt, killers, historia, tt_k, tt_v, tt_p, tt_f, tt_m,
                                  ctx, qprof, pesos, ev_k, ev_v)
                    deshacer(bb, pts, c, o, d, rival, anoto)
                    v = p * a + q * b
            if ctx[C_ABORTA]:
                return 0.0
            if v <= alpha or hi >= beta or v >= beta:
                break
            hi = beta  # la ventana nula superó a alpha: se repite con la ventana completa
        if v > mejor:
            mejor = v
            mejor_jug = m
            if ply == 0:
                ctx[C_MEJOR] = m
            if v > alpha:
                alpha = v
                if v >= beta:
                    if not ataque:
                        if killers[ply, 0] != m:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = m
                        historia[c, m] += prof * prof
                    break
    if mejor <= alpha0:
        bandera = COTA_SUP
    elif mejor >= beta:
        bandera = COTA_INF
    else:
        bandera = EXACTO
    tt_guardar(tt_k, tt_v, tt_p, tt_f, tt_m, h, prof, mejor, bandera, mejor_jug)
    return mejor
