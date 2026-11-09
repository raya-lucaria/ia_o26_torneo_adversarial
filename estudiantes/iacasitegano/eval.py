"""Función de evaluación lineal: qué tan buena es una posición para `color`.

`features(estado, color)` calcula el vector de atributos (siempre "propio
menos rival"); `evaluar` solo hace el producto punto con `PESOS`. Separarlos
permite que `herramientas/entrenar.py` ajuste `PESOS` con datos de self-play
(`data/pesos.npy`) sin tocar la lógica de las features. `preparar()` en
`strategy.py` carga esos pesos si el archivo existe; si no, se usan los de
abajo (puestos a mano en la Fase 1-2, antes de entrenar con self-play).
"""

from __future__ import annotations

import numpy as np

from motor import Estado
from motor.jugada import casilla_de_texto
from motor.nucleo import LADO, PEONES, PUNTOS_PARA_GANAR

NOMBRES_FEATURES = ("puntos", "avance", "apoyo", "pasado", "columnas_vacias", "carrera")
# fmt: off
PESOS = np.array([
    1000.0,  # puntos: diferencia de puntos anotados
       3.0,  # avance: filas ya avanzadas por cada peón (suma)
       1.0,  # apoyo: peones propios que cubren a otro peón propio
       8.0,  # pasado: igual que avance, pero solo de peones sin nadie rival en su camino
      15.0,  # columnas_vacias: columnas RIVALES sin nadie menos las propias sin nadie
       0.0,  # carrera: distancia propia a la meta menos distancia rival (se entrena)
])
# fmt: on


def cargar_pesos(pesos: np.ndarray) -> None:
    """Reemplaza `PESOS` (lo llama `strategy.preparar` si existe `data/pesos.npy`)."""
    global PESOS
    pesos = np.asarray(pesos, dtype=float)
    # Compatibilidad durante la transición: un checkpoint de cinco features
    # conserva su comportamiento y empieza con peso cero para `carrera`.
    if pesos.shape == (len(NOMBRES_FEATURES) - 1,):
        pesos = np.concatenate((pesos, np.array([0.0])))
    if pesos.shape != (len(NOMBRES_FEATURES),):
        raise ValueError(f"se esperaban {len(NOMBRES_FEATURES)} pesos, llegaron {pesos.shape}")
    PESOS = pesos


def cota() -> float:
    """Cota real de |evaluar()| en una posición sin ganador, con los `PESOS` actuales.

    La usa `busqueda.py` para podar nodos de azar: acotar con un número inventado
    (p. ej. un millón) sería seguro pero inútil para podar porque nunca se acerca
    al valor real de una jugada. Se recalcula cada vez (no es una constante) porque
    unos pesos entrenados pueden tener otra magnitud que los de fábrica.
    """
    cota_por_feature = np.array(
        [
            PUNTOS_PARA_GANAR,
            PEONES * (LADO - 1),  # avance máximo: todos los peones en la última fila
            PEONES * 2,  # apoyo máximo: 2 por peón
            PEONES * (LADO - 1),  # pasado máximo: igual que avance
            LADO,  # columnas_vacias: como mucho las 9 columnas
            LADO,  # carrera: diferencia segura entre distancias 1..LADO
        ]
    )
    return float(np.dot(np.abs(PESOS), cota_por_feature))


def _rival(color: str) -> str:
    return "negras" if color == "blancas" else "blancas"


def _avance(fila: int, color: str) -> int:
    """Cuántas filas ya avanzó un peón en `fila` hacia su meta (0 a 7)."""
    return fila if color == "blancas" else (LADO - 1 - fila)


def _apoyo_propio(casillas: set[int], casilla: int, color: str) -> int:
    """Peones propios en las dos diagonales de atrás de `casilla` (0, 1 o 2).

    Misma regla que `nucleo.apoyo`, reimplementada aquí porque `Estado` no
    expone el tablero crudo (solo `piezas()`, que ya nos da las casillas).
    """
    fila = casilla // LADO
    fila_atras = fila - 1 if color == "blancas" else fila + 1
    if fila_atras < 0 or fila_atras >= LADO:
        return 0
    col = casilla % LADO
    total = 0
    if col > 0 and (fila_atras * LADO + col - 1) in casillas:
        total += 1
    if col < LADO - 1 and (fila_atras * LADO + col + 1) in casillas:
        total += 1
    return total


def _pasado(filas_rivales_por_columna: tuple[list[int], list[int]], casilla: int, color: str) -> bool:
    """True si ningún peón rival puede bloquear ni capturar a este peón antes de su meta.

    Revisa las tres columnas (la suya y las dos vecinas) en todas las filas que le
    faltan por recorrer: si no hay rival ahí, nada lo detiene.

    Ojo: esto es optimista (un solo defensor sin apoyo tampoco detiene de verdad,
    porque se le puede atacar en diagonal); por eso existe también la feature
    `columnas_vacias`, que no depende de dónde esté el rival.
    """
    fila = casilla // LADO
    col = casilla % LADO
    fila_minima, fila_maxima = filas_rivales_por_columna
    if color == "blancas":
        return all(fila_maxima[c] <= fila for c in range(max(0, col - 1), min(LADO, col + 2)))
    return all(fila_minima[c] >= fila for c in range(max(0, col - 1), min(LADO, col + 2)))


def _features_bando(estado: Estado, color: str, casillas_rivales: set[int]) -> np.ndarray:
    casillas = {casilla_de_texto(t) for t in estado.piezas(color)}
    fila_minima = [LADO] * LADO
    fila_maxima = [-1] * LADO
    for casilla_rival in casillas_rivales:
        fila_rival, columna_rival = divmod(casilla_rival, LADO)
        fila_minima[columna_rival] = min(fila_minima[columna_rival], fila_rival)
        fila_maxima[columna_rival] = max(fila_maxima[columna_rival], fila_rival)
    filas_rivales_por_columna = (fila_minima, fila_maxima)
    avance = apoyo = pasado = 0
    for c in casillas:
        a = _avance(c // LADO, color)
        avance += a
        apoyo += _apoyo_propio(casillas, c, color)
        if _pasado(filas_rivales_por_columna, c, color):
            pasado += a
    columnas_ocupadas = {c % LADO for c in casillas}
    columnas_vacias = LADO - len(columnas_ocupadas)
    return np.array([avance, apoyo, pasado, columnas_vacias], dtype=float)


def _distancia_meta(casillas: set[int], color: str) -> int:
    """Distancia del peón más avanzado a la fila meta; sin peones vale `LADO`."""
    if not casillas:
        return LADO
    avance_maximo = max(_avance(c // LADO, color) for c in casillas)
    return LADO - 1 - avance_maximo


def features(estado: Estado, color: str) -> np.ndarray:
    """Vector de atributos de `estado` para `color`, en el mismo orden que `PESOS`.

    Siempre "propio menos rival" (salvo `columnas_vacias`, invertida a propósito:
    más huecos rivales es bueno para nosotros). `evaluar = PESOS @ features`.
    """
    rival = _rival(color)
    puntos = estado.puntos
    casillas_propias = {casilla_de_texto(t) for t in estado.piezas(color)}
    casillas_rivales = {casilla_de_texto(t) for t in estado.piezas(rival)}
    propio = _features_bando(estado, color, casillas_rivales)
    ajeno = _features_bando(estado, rival, casillas_propias)
    avance, apoyo, pasado, columnas_vacias = propio - ajeno
    carrera = _distancia_meta(casillas_propias, color) - _distancia_meta(casillas_rivales, rival)
    return np.array(
        [puntos[color] - puntos[rival], avance, apoyo, pasado, -columnas_vacias, carrera]
    )


def evaluar(estado: Estado, color: str) -> float:
    """Valor de `estado` desde el punto de vista de `color` (más alto = mejor)."""
    return float(PESOS @ features(estado, color))
