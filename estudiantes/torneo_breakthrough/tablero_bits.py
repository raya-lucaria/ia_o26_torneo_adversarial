"""Reglas de Breakthrough de combate sobre bitboards por fila (numba).

bb[color, fila] es una máscara de 9 bits con las columnas que ocupa ese color (bit = columna).
Blancas = 0 avanzan hacia filas mayores; Negras = 1 hacia filas menores. Las casillas se numeran
como en el motor oficial (fila*9 + columna) y una jugada se codifica como origen*81 + destino.
La clave Zobrist usa las tablas del motor oficial, así que coincide con `estado.clave`.
"""

import numpy as np
from numba import njit

from motor import nucleo

LADO = 9
CASILLAS = 81
MASCARA = (1 << LADO) - 1
COLUMNAS = "abcdefghi"
Z_PIEZAS = nucleo.Z_PIEZAS
Z_TURNO = nucleo.Z_TURNO
Z_PUNTOS = nucleo.Z_PUNTOS


def desde_estado(estado):
    """(bb, puntos, color en turno) a partir de un `Estado`, con su API pública."""
    bb = np.zeros((2, LADO), np.int64)
    for color, nombre in enumerate(("blancas", "negras")):
        for casilla in estado.piezas(nombre):
            bb[color, int(casilla[1]) - 1] |= 1 << (ord(casilla[0]) - ord("a"))
    puntos = estado.puntos
    pts = np.array([puntos["blancas"], puntos["negras"]], np.int64)
    return bb, pts, 0 if estado.turno == "blancas" else 1


def texto(jugada):
    """Jugada codificada → texto como "e2-e3"."""
    o, d = divmod(int(jugada), CASILLAS)
    return f"{COLUMNAS[o % LADO]}{o // LADO + 1}-{COLUMNAS[d % LADO]}{d // LADO + 1}"


@njit(cache=False)
def ocupada(bb, color, casilla):
    """1 si `color` tiene un peón en `casilla`."""
    return (bb[color, casilla // LADO] >> (casilla % LADO)) & 1


@njit(cache=False)
def generar(bb, c, lista):
    """Jugadas legales de `c` en `lista` (codificadas); regresa cuántas son."""
    n = 0
    if c == 0:
        for f in range(LADO - 1):
            propio = bb[0, f]
            if propio == 0:
                continue
            sig_propio = bb[0, f + 1]
            ocup = sig_propio | bb[1, f + 1]
            for col in range(LADO):
                if (propio >> col) & 1 == 0:
                    continue
                o = f * LADO + col
                d0 = o + LADO
                if col > 0 and (sig_propio >> (col - 1)) & 1 == 0:
                    lista[n] = o * CASILLAS + d0 - 1
                    n += 1
                if (ocup >> col) & 1 == 0:
                    lista[n] = o * CASILLAS + d0
                    n += 1
                if col < LADO - 1 and (sig_propio >> (col + 1)) & 1 == 0:
                    lista[n] = o * CASILLAS + d0 + 1
                    n += 1
    else:
        for f in range(LADO - 1, 0, -1):
            propio = bb[1, f]
            if propio == 0:
                continue
            sig_propio = bb[1, f - 1]
            ocup = sig_propio | bb[0, f - 1]
            for col in range(LADO):
                if (propio >> col) & 1 == 0:
                    continue
                o = f * LADO + col
                d0 = o - LADO
                if col > 0 and (sig_propio >> (col - 1)) & 1 == 0:
                    lista[n] = o * CASILLAS + d0 - 1
                    n += 1
                if (ocup >> col) & 1 == 0:
                    lista[n] = o * CASILLAS + d0
                    n += 1
                if col < LADO - 1 and (sig_propio >> (col + 1)) & 1 == 0:
                    lista[n] = o * CASILLAS + d0 + 1
                    n += 1
    return n


@njit(cache=False)
def generar_tacticas(bb, c, lista):
    """Solo las jugadas de la quiescencia: ataques y jugadas que llegan a la meta."""
    n = 0
    paso = 1 if c == 0 else -1
    meta = LADO - 1 if c == 0 else 0
    for f in range(LADO):
        propio = bb[c, f]
        if propio == 0 or f == meta:
            continue
        f2 = f + paso
        rival = bb[1 - c, f2]
        if f2 == meta:  # todas las jugadas de la penúltima fila anotan (o atacan en la meta)
            libres_diag = ~bb[c, f2] & MASCARA
            izq = propio & (libres_diag << 1)
            der = propio & (libres_diag >> 1)
            recto = propio & ~(bb[0, f2] | bb[1, f2])
        else:
            izq = propio & (rival << 1)  # destino col-1 ocupado por el rival
            der = propio & (rival >> 1)  # destino col+1 ocupado por el rival
            recto = 0
        if (izq | der | recto) == 0:
            continue
        for col in range(LADO):
            bit = 1 << col
            if (izq | der | recto) & bit == 0:
                continue
            o = f * LADO + col
            d0 = f2 * LADO + col
            if izq & bit:
                lista[n] = o * CASILLAS + d0 - 1
                n += 1
            if recto & bit:
                lista[n] = o * CASILLAS + d0
                n += 1
            if der & bit:
                lista[n] = o * CASILLAS + d0 + 1
                n += 1
    return n


@njit(cache=False)
def apoyo(bb, casilla, c):
    """Peones de `c` en las dos diagonales de atrás de `casilla` (0, 1 o 2)."""
    f = casilla // LADO - (1 if c == 0 else -1)
    if f < 0 or f >= LADO:
        return 0
    col = casilla % LADO
    fila = bb[c, f]
    total = 0
    if col > 0:
        total += (fila >> (col - 1)) & 1
    if col < LADO - 1:
        total += (fila >> (col + 1)) & 1
    return total


@njit(cache=False)
def prob(bb, c, o, d):
    """p de que gane el ataque de `c` de `o` a `d` (la fórmula del juego, acotada)."""
    p = 0.5 + 0.15 * (apoyo(bb, o, c) - apoyo(bb, d, 1 - c))
    return min(max(p, 0.2), 0.9)


@njit(cache=False)
def hacer(bb, pts, c, o, d, gana, h):
    """Aplica la jugada en sitio. Regresa (había rival en d, anotó, clave nueva)."""
    fo, co = o // LADO, o % LADO
    fd, cd = d // LADO, d % LADO
    bb[c, fo] &= ~(1 << co)
    h ^= Z_PIEZAS[c, o]
    rival = (bb[1 - c, fd] >> cd) & 1
    if rival and not gana:
        return rival, 0, h ^ Z_TURNO  # sale el atacante; el defensor se queda
    if rival:
        bb[1 - c, fd] &= ~(1 << cd)
        h ^= Z_PIEZAS[1 - c, d]
    if fd == (LADO - 1 if c == 0 else 0):
        h ^= Z_PUNTOS[c, pts[c]]
        pts[c] += 1
        h ^= Z_PUNTOS[c, pts[c]]
        return rival, 1, h ^ Z_TURNO  # anota y sale del tablero
    bb[c, fd] |= 1 << cd
    h ^= Z_PIEZAS[c, d]
    return rival, 0, h ^ Z_TURNO


@njit(cache=False)
def deshacer(bb, pts, c, o, d, rival, anoto):
    fo, co = o // LADO, o % LADO
    fd, cd = d // LADO, d % LADO
    bb[c, fo] |= 1 << co
    if anoto:
        pts[c] -= 1
    else:
        bb[c, fd] &= ~(1 << cd)
    if rival:
        bb[1 - c, fd] |= 1 << cd


@njit(cache=False)
def clave(bb, c, pts):
    """Clave Zobrist de 64 bits (igual a `estado.clave`)."""
    h = np.uint64(0)
    for color in range(2):
        for f in range(LADO):
            fila = bb[color, f]
            for col in range(LADO):
                if (fila >> col) & 1:
                    h ^= Z_PIEZAS[color, f * LADO + col]
    if c == 1:
        h ^= Z_TURNO
    h ^= Z_PUNTOS[0, pts[0]]
    h ^= Z_PUNTOS[1, pts[1]]
    return h
