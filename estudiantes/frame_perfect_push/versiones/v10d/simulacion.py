"""Partidas de simulación con jugadas decisivas (numba).

Copia adaptada de `simular` en `sistema/motor/nucleo.py` (permitido: docs/avanzado.md,
«Modificar el motor»): mismas listas de peones y muestreo por rechazo, más dos reglas al
elegir cada jugada simulada:

- Decisiva: si un peón propio está en la fila anterior a su meta y puede avanzar sin
  combate (recto o en diagonal a casilla vacía), anota.
- Si no, jugada al azar uniforme, igual que el motor.

Solo sirve para VALORAR posiciones; la jugada que regresa el bot sale del motor oficial.
"""

import numpy as np
from numba import njit

LADO = 9
CASILLAS = 81
MAX_JUGADAS = 54
PUNTOS_PARA_GANAR = 3
INTENTOS = 16


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
def _quitar(lista, cuantos, indice, c, casilla):
    i = indice[casilla]
    ultimo = lista[c, cuantos[c] - 1]
    lista[c, i] = ultimo
    indice[ultimo] = i
    cuantos[c] -= 1


@njit(cache=False)
def simular(tablero, puntos, color, n, semilla, para, buf):
    """Fracción de `n` partidas que gana `para`. `buf` = búferes preasignados."""
    tab, pts, origenes, destinos, lista, cuantos, indice = buf
    np.random.seed(semilla)
    victorias = 0
    for _ in range(n):
        for s in range(CASILLAS):
            tab[s] = tablero[s]
        pts[0] = puntos[0]
        pts[1] = puntos[1]
        cuantos[0] = 0
        cuantos[1] = 0
        for s in range(CASILLAS):
            if tab[s] != 0:
                k = tab[s] - 1
                lista[k, cuantos[k]] = s
                indice[s] = cuantos[k]
                cuantos[k] += 1
        c = color
        ganador = -1
        while True:
            pieza = c + 1
            rival = 2 - c
            dir_ = 1 if c == 0 else -1
            avance = dir_ * LADO
            meta = LADO - 1 if c == 0 else 0
            o = -1
            d = -1
            # 1. Decisiva: anotar sin combate.
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
                if col > 0 and tab[f - 1] == 0:
                    o = s
                    d = f - 1
                    break
                if col < LADO - 1 and tab[f + 1] == 0:
                    o = s
                    d = f + 1
                    break
            # 3. Al azar uniforme (igual que el motor).
            if o < 0:
                for _intento in range(INTENTOS):
                    if cuantos[c] == 0:
                        break
                    prueba = lista[c, np.random.randint(0, cuantos[c])]
                    giro = np.random.randint(-1, 2)
                    col = prueba % LADO + giro
                    if col < 0 or col >= LADO:
                        continue
                    destino = prueba + avance + giro
                    v = tab[destino]
                    if v == pieza or (giro == 0 and v != 0):
                        continue
                    o = prueba
                    d = destino
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
            # Aplicar (mismas reglas que nucleo.aplicar), manteniendo las listas.
            gana = True
            if tab[d] != 0:
                gana = np.random.random() < _prob(tab, c, o, d)
                if gana:
                    _quitar(lista, cuantos, indice, 1 - c, d)
                    tab[d] = 0
                else:
                    _quitar(lista, cuantos, indice, c, o)
            tab[o] = 0
            if gana:
                if d // LADO == meta:
                    _quitar(lista, cuantos, indice, c, o)
                    pts[c] += 1
                else:
                    i = indice[o]
                    lista[c, i] = d
                    indice[d] = i
                    tab[d] = pieza
            if pts[c] >= PUNTOS_PARA_GANAR:
                ganador = c
                break
            c = 1 - c
        if ganador == para:
            victorias += 1
    return victorias / n


BUF = (
    np.empty(CASILLAS, np.int8),
    np.empty(2, np.int64),
    np.empty(MAX_JUGADAS, np.int64),
    np.empty(MAX_JUGADAS, np.int64),
    np.empty((2, CASILLAS), np.int64),
    np.empty(2, np.int64),
    np.empty(CASILLAS, np.int64),
)


def valorar(tablero, puntos, color, n, semilla, para):
    return simular(tablero, puntos, color, n, semilla, para, BUF)


def calentar():
    """Compila con los mismos tipos que en la partida (arreglos de solo lectura como `Estado`)."""
    tab = np.zeros(CASILLAS, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    tab.flags.writeable = False
    pts.flags.writeable = False
    valorar(tab, pts, 0, 1, 0, 0)
