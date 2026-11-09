"""Partidas de simulación con política informada (numba).

Adaptado de `sistema/motor/nucleo.py` (permitido: docs/avanzado.md, «Modificar el motor»).
Solo se usa para VALORAR posiciones dentro de la búsqueda; la jugada que se regresa
siempre sale de `estado.jugadas()` del motor oficial.

Representación igual que el motor: `tablero` int8[81] (casilla = fila*9 + col,
0 vacía, 1 blanca, 2 negra), `puntos` int64[2], color 0 = Blancas, 1 = Negras.

Política de cada jugada simulada (en orden):
1. Decisiva: si un peón propio puede llegar a la meta, lo hace.
2. Antidecisiva: si un peón rival está a una casilla de su meta y podemos
   atacarlo, lo atacamos (con el mejor apoyo).
3. Si no, jugada al azar uniforme entre las legales.
Los ataques se resuelven con el generador de la simulación (no con los dados del
árbitro, que no se conocen).
"""

import numpy as np
from numba import njit

LADO = 9
CASILLAS = 81
MAX_JUGADAS = 54
PUNTOS_PARA_GANAR = 3
INTENTOS = 16
LIMITE_JUGADAS = 400  # seguridad: ninguna partida real dura tanto


@njit(cache=False)
def _apoyo(tab, casilla, c):
    fila = casilla // LADO - (1 if c == 0 else -1)
    if fila < 0 or fila >= LADO:
        return 0
    col = casilla % LADO
    pieza = c + 1
    total = 0
    if col > 0 and tab[fila * LADO + col - 1] == pieza:
        total += 1
    if col < LADO - 1 and tab[fila * LADO + col + 1] == pieza:
        total += 1
    return total


@njit(cache=False)
def _prob(tab, c, o, d):
    if tab[d] == 0:
        return 1.0
    p = 0.5 + 0.15 * (_apoyo(tab, o, c) - _apoyo(tab, d, 1 - c))
    return min(max(p, 0.2), 0.9)


@njit(cache=False)
def simular(tablero, puntos, color, n, semilla, para, tab, pts, origenes, destinos):
    """Juega `n` partidas desde la posición y regresa la fracción que gana `para`.

    `tab`, `pts`, `origenes` y `destinos` son búferes preasignados (se sobrescriben).
    """
    np.random.seed(semilla)
    victorias = 0.0
    for _ in range(n):
        for i in range(CASILLAS):
            tab[i] = tablero[i]
        pts[0] = puntos[0]
        pts[1] = puntos[1]
        c = color
        ganador = -1
        for _paso in range(LIMITE_JUGADAS):
            pieza = c + 1
            rival = 2 - c
            dir_ = 1
            meta = LADO - 1
            meta_rival = 0
            if c == 1:
                dir_ = -1
                meta = 0
                meta_rival = LADO - 1
            avance = dir_ * LADO
            o = -1
            d = -1
            # 1. Decisiva: peones propios en la fila anterior a la meta.
            base = (meta - dir_) * LADO
            for col in range(LADO):
                s = base + col
                if tab[s] != pieza:
                    continue
                f = s + avance
                if tab[f] == 0:
                    o = s
                    d = f
                    break
                if col > 0 and tab[f - 1] != pieza:
                    o = s
                    d = f - 1
                    if tab[f - 1] == 0:
                        break
                if col < LADO - 1 and tab[f + 1] != pieza:
                    o = s
                    d = f + 1
                    if tab[f + 1] == 0:
                        break
            # 2. Antidecisiva: rival a una casilla de su meta; lo atacamos.
            if o < 0:
                base = (meta_rival + dir_) * LADO
                mejor_p = -1.0
                for col in range(LADO):
                    s = base + col
                    if tab[s] != rival:
                        continue
                    atras = s - avance
                    if col > 0 and tab[atras - 1] == pieza:
                        p = _prob(tab, c, atras - 1, s)
                        if p > mejor_p:
                            mejor_p = p
                            o = atras - 1
                            d = s
                    if col < LADO - 1 and tab[atras + 1] == pieza:
                        p = _prob(tab, c, atras + 1, s)
                        if p > mejor_p:
                            mejor_p = p
                            o = atras + 1
                            d = s
            # 3. Al azar uniforme (muestreo por rechazo; si falla, todas las jugadas).
            if o < 0:
                for _intento in range(INTENTOS):
                    s = np.random.randint(0, CASILLAS)
                    if tab[s] != pieza:
                        continue
                    giro = np.random.randint(-1, 2)
                    col = s % LADO + giro
                    if col < 0 or col >= LADO:
                        continue
                    f = s + avance + giro
                    if tab[f] == pieza or (giro == 0 and tab[f] != 0):
                        continue
                    o = s
                    d = f
                    break
            if o < 0:
                m = 0
                for s in range(CASILLAS):
                    if tab[s] != pieza:
                        continue
                    col = s % LADO
                    f = s + avance
                    if col > 0 and tab[f - 1] != pieza:
                        origenes[m] = s
                        destinos[m] = f - 1
                        m += 1
                    if tab[f] == 0:
                        origenes[m] = s
                        destinos[m] = f
                        m += 1
                    if col < LADO - 1 and tab[f + 1] != pieza:
                        origenes[m] = s
                        destinos[m] = f + 1
                        m += 1
                if m == 0:
                    ganador = 1 - c
                    break
                k = np.random.randint(0, m)
                o = origenes[k]
                d = destinos[k]
            # Aplicar la jugada.
            gana = True
            if tab[d] != 0:
                gana = np.random.random() < _prob(tab, c, o, d)
            tab[o] = 0
            if gana:
                if d // LADO == meta:
                    pts[c] += 1
                    tab[d] = 0
                else:
                    tab[d] = pieza
            if pts[c] >= PUNTOS_PARA_GANAR:
                ganador = c
                break
            c = 1 - c
        if ganador == para:
            victorias += 1.0
        elif ganador < 0:
            victorias += 0.5  # nunca debería pasar; neutral
    return victorias / n


TAB = np.empty(CASILLAS, np.int8)
PTS = np.empty(2, np.int64)
ORIGENES = np.empty(MAX_JUGADAS, np.int64)
DESTINOS = np.empty(MAX_JUGADAS, np.int64)


def valorar(tablero, puntos, color, n, semilla, para):
    """Fracción de victorias de `para` (0/1) en `n` simulaciones desde la posición."""
    return simular(tablero, puntos, color, n, semilla, para, TAB, PTS, ORIGENES, DESTINOS)


def calentar():
    """Compila la simulación con los mismos tipos que en la partida (llamar en `preparar`)."""
    tab = np.zeros(CASILLAS, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    tab.flags.writeable = False  # como los arreglos de `Estado`
    pts.flags.writeable = False
    valorar(tab, pts, 0, 1, 0, 0)
