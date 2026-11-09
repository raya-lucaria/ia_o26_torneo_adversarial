"""Rivales de práctica con otros estilos de juego, para buscar debilidades de nuestro bot.

Medimos casi siempre contra montecarlo y contra nosotros mismos, pero en la liga jugamos contra
otros equipos. Estos rivales juegan distinto: cada uno es un expectimax corto (su jugada, el dado,
la respuesta del otro, el dado…) con una evaluación sencilla cuyos pesos le dan su estilo:

- `agresivo`: un peón rival le vale el doble que uno propio, así que ataca desde p = 0.35;
- `corredor`: premia sobre todo a su peón más avanzado, así que empuja uno hasta la meta;
- `defensivo`: cuida su fila de casa y el apoyo de sus peones, y castiga que el rival avance;
- `equilibrado`: pesos parejos, como un expectimax típico.

Piensan 2 plies y, si les alcanza el tiempo de la jugada, 3. Usan el motor oficial y no comparten
código con nuestra búsqueda ni con nuestra evaluación. Solo los usa la arena; `herramientas/` no
corre en la liga.
"""

import random
import time
from collections import namedtuple

import numpy as np

from motor_rapido import LADO, desde_estado

Pesos = namedtuple(
    "Pesos",
    ["material", "material_rival", "avance", "avance_rival", "lider", "lider_rival", "apoyo",
     "casa"],
)
ESTILOS = {
    "agresivo": Pesos(3, 6, 0.5, 0.5, 0, 0, 0.5, 0),
    "corredor": Pesos(4, 4, 0.3, 0.3, 3, 1, 0.5, 0),
    "defensivo": Pesos(5, 5, 0.1, 1.0, 0, 2, 1.5, 1.5),
    "equilibrado": Pesos(5, 5, 0.5, 0.5, 1, 1, 1, 0.5),
}
PUNTO = 50.0  # cada punto anotado (de más) vale como 10 peones
GANAR = 10_000.0
PROF_MIN = 2  # siempre
PROF_MAX = 3  # solo si termina dentro del tiempo de la jugada
JUGADAS_POR_RELOJ = 30  # tiempo de la jugada = (reloj − reserva) / esto
JUGADA_MAX_S = 1.0  # aunque sobre reloj (en la arena por nodos el reloj no tiene límite)
RESERVA_S = 2.0
CRECIMIENTO = 30  # un ply más cuesta ~30 veces más (medido: 0.03 s → 0.9 s a media partida)
Rasgos = namedtuple("Rasgos", ["peones", "avance", "lider", "apoyados", "casa"])


def _rasgos(tablero: np.ndarray, color: int) -> Rasgos:
    """Rasgos de los peones de `color` en el tablero de 9×9 (fila 0 = fila «1»)."""
    propios = tablero == color + 1
    filas = np.nonzero(propios)[0]
    progreso = filas if color == 0 else LADO - 1 - filas  # filas avanzadas desde su casa
    detras = np.zeros_like(propios)  # casillas con un peón propio en una diagonal de atrás
    if color == 0:  # «atrás» de Blancas es la fila de abajo
        detras[1:, 1:] |= propios[:-1, :-1]
        detras[1:, :-1] |= propios[:-1, 1:]
    else:
        detras[:-1, 1:] |= propios[1:, :-1]
        detras[:-1, :-1] |= propios[1:, 1:]
    return Rasgos(
        peones=len(filas),
        avance=int(progreso.sum()),
        lider=int(progreso.max()) if len(filas) else 0,
        apoyados=int(np.count_nonzero(propios & detras)),
        casa=int(np.count_nonzero(propios[0 if color == 0 else LADO - 1])),
    )


def evaluar(tab: np.ndarray, pts: np.ndarray, yo: int, pesos: Pesos) -> float:
    """Valor de la posición para `yo` (0 = Blancas, 1 = Negras) con los pesos de un estilo."""
    tablero = tab.reshape(LADO, LADO)
    mio, suyo = _rasgos(tablero, yo), _rasgos(tablero, 1 - yo)
    return (
        PUNTO * (int(pts[yo]) - int(pts[1 - yo]))
        + pesos.material * mio.peones - pesos.material_rival * suyo.peones
        + pesos.avance * mio.avance - pesos.avance_rival * suyo.avance
        + pesos.lider * mio.lider - pesos.lider_rival * suyo.lider
        + pesos.apoyo * (mio.apoyados - suyo.apoyados)
        + pesos.casa * (mio.casa - suyo.casa)
    )


class _SinTiempo(Exception):
    """La búsqueda más profunda no terminó a tiempo: se queda la anterior."""


class Estilo:
    """Un rival de la arena con el estilo `nombre` (una clave de `ESTILOS`)."""

    def __init__(self, nombre: str):
        self.nombre = nombre
        self.pesos = ESTILOS[nombre]

    def preparar(self, config):
        self.color = config.color
        self.yo = 0 if config.color == "blancas" else 1
        # Desempata al azar entre jugadas iguales, sin repetir siempre la misma partida.
        self.rng = random.Random(f"{config.semilla}-{config.color}-{self.nombre}")

    def jugar(self, estado, tiempo_restante):
        inicio = time.perf_counter()
        para_jugada = max(0.0, tiempo_restante - RESERVA_S) / JUGADAS_POR_RELOJ
        limite = inicio + min(para_jugada, JUGADA_MAX_S)
        mejores = self._mejores(estado, PROF_MIN, None)
        for prof in range(PROF_MIN + 1, PROF_MAX + 1):
            ahora = time.perf_counter()
            if ahora + CRECIMIENTO * (ahora - inicio) > limite:  # no le alcanzaría: ni lo intenta
                break
            try:
                mejores = self._mejores(estado, prof, limite)
            except _SinTiempo:
                break
        return self.rng.choice(mejores)

    def _mejores(self, estado, prof, limite):
        """Las jugadas con el mejor valor esperado a `prof` plies."""
        valores = {j: self._esperado(estado, j, prof - 1, limite) for j in estado.jugadas()}
        tope = max(valores.values())
        return [j for j, v in valores.items() if v >= tope - 1e-9]

    def _esperado(self, estado, jugada, prof, limite):
        return sum(p * self._valor(hijo, prof, limite) for p, hijo in estado.siguiente(jugada))

    def _valor(self, estado, prof, limite):
        if limite is not None and time.perf_counter() > limite:
            raise _SinTiempo
        if estado.terminado():  # con más plies por delante, ganar (o perder) llegó antes
            return GANAR + prof if estado.ganador() == self.color else -GANAR - prof
        if prof == 0:
            tab, pts, _ = desde_estado(estado)
            return evaluar(tab, pts, self.yo, self.pesos)
        valores = [self._esperado(estado, j, prof - 1, limite) for j in estado.jugadas()]
        return max(valores) if estado.turno == self.color else min(valores)
