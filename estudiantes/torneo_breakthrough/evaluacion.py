"""Evaluación v2 (pesos a mano, la del prototipo A): valor en (-1, 1) desde el jugador en turno.

Por peón, en coordenadas relativas a su color (fila 0 = su fila de casa):
- valor de fila: W_PIEZA + FILA[fila]. La fila de casa vale casi tanto como estar muy avanzado
  (idea de Baier & Winands 2018): es la última defensa contra los corredores rivales;
- + W_APOYO por cada peón propio en sus diagonales de atrás (también bajan la p de los ataques);
- − W_AMENAZA·(1 − 0.3·apoyo)·valor de fila si un rival lo puede atacar;
- + W_PUNTO·PASADO[distancia] si ningún rival puede alcanzarlo: no hay peones rivales en su cono
  de avance (los peones solo avanzan, así que los rivales que ya lo rebasaron no cuentan).
Más W_PUNTO por punto de ventaja y tanh(ESCALA·total). Ver DISENO.md §2.7: estos pesos son el
punto de partida; la etapa 3 los reemplaza por pesos ajustados con datos.

El bot ya no la usa (juega con rasgos.py). Queda como referencia para herramientas/ajustar_pesos.py,
que reporta la v2 calibrada junto a los pesos aprendidos.
"""

import numpy as np
from numba import njit

from tablero_bits import LADO, MASCARA

W_PIEZA = 0.3
FILA = np.array([1.0, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6], np.float64)
W_APOYO = 0.12
W_AMENAZA = 0.25
W_PUNTO = 6.0
PASADO = np.array([0.0, 0.85, 0.7, 0.55, 0.42, 0.32, 0.24, 0.17, 0.12], np.float64)
ESCALA = 0.1
POP = np.array([bin(i).count("1") for i in range(1 << LADO)], np.int64)


@njit(cache=False)
def dilatar(m):
    """Columnas a distancia ≤ 1 de alguna columna de `m`."""
    return (m | (m << 1) | (m >> 1)) & MASCARA


@njit(cache=False)
def _fila(propios, atras, rivales_adelante, libres, base, pasado):
    """Suma de los peones de una fila (`propios`) con los términos de arriba."""
    if atras >= 0:
        a1 = (atras << 1) & MASCARA
        a2 = atras >> 1
        uno = propios & (a1 | a2)
        dos = propios & a1 & a2
    else:
        uno = 0
        dos = 0
    x = POP[propios] * base + W_APOYO * (POP[uno] + POP[dos])
    am = propios & (((rivales_adelante << 1) & MASCARA) | (rivales_adelante >> 1))
    if am:
        x -= W_AMENAZA * base * (POP[am & ~uno] + 0.7 * POP[am & uno & ~dos] + 0.4 * POP[am & dos])
    return x + W_PUNTO * pasado * POP[propios & libres]


@njit(cache=False)
def evaluar(bb, pts, c):
    total = 0.0
    # Blancas, de la fila 7 a la 0. `cono` = columnas que algún negro de más arriba alcanza:
    # cono(f) = dilatar(negros(f+1) | cono(f+1)).
    cono = 0
    for f in range(LADO - 2, -1, -1):
        cono = dilatar(bb[1, f + 1] | cono)
        if bb[0, f]:
            atras = bb[0, f - 1] if f > 0 else -1
            total += _fila(bb[0, f], atras, bb[1, f + 1], MASCARA & ~cono,
                           W_PIEZA + FILA[f], PASADO[LADO - 1 - f])
    # Negras, de la fila 1 a la 8 (su fila relativa es 8 - f).
    cono = 0
    for f in range(1, LADO):
        cono = dilatar(bb[0, f - 1] | cono)
        if bb[1, f]:
            atras = bb[1, f + 1] if f < LADO - 1 else -1
            total -= _fila(bb[1, f], atras, bb[0, f - 1], MASCARA & ~cono,
                           W_PIEZA + FILA[LADO - 1 - f], PASADO[f])
    total += W_PUNTO * (pts[0] - pts[1])
    if c == 1:
        total = -total
    return np.tanh(ESCALA * total)
