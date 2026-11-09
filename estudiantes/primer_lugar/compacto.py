"""Identidad exacta de 22 bytes y valores exportables de dos bytes."""
import math
import numpy as np


def identidad(estado):
    # Dos bits por casilla, incluidas las vacías. El último byte incluye turno
    # y ambos marcadores: nunca agrupamos estados por parecido en su valor.
    bits = ((estado._tablero[:, None] >> np.array([0, 1], dtype=np.int8)) & 1)
    meta = estado._color | (int(estado._puntos[0]) << 1) | (int(estado._puntos[1]) << 3)
    return np.packbits(bits).tobytes() + bytes([meta])


def reconstruir(clave):
    from motor import Estado
    if len(clave) != 22:
        raise ValueError('Identidad incompatible')
    bits = np.unpackbits(np.frombuffer(clave[:21], dtype=np.uint8))[:162].reshape(81, 2)
    tablero = (bits[:, 0] + 2 * bits[:, 1]).astype(np.int8)
    meta = clave[-1]
    return Estado(tablero, meta & 1, np.array([(meta >> 1) & 3, (meta >> 3) & 3], dtype=np.int64))


def cuantizar(valor):
    if not math.isfinite(valor) or not 0 <= valor <= 1:
        raise ValueError('Valor fuera de [0,1]')
    return int(round(valor * 65535))


def decuantizar(codigo):
    return int(codigo) / 65535.
