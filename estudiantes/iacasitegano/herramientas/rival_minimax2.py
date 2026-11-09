"""Rival de entrenamiento: expectiminimax puro de profundidad fija 2."""

from __future__ import annotations

import random


PROFUNDIDAD = 2


def preparar(config):
    global color, rng
    color = config.color
    rng = random.Random(config.semilla)


def _avance(casilla: str, bando: str) -> int:
    fila = int(casilla[1:])
    return fila if bando == "blancas" else 10 - fila


def _evaluar(estado, objetivo: str) -> float:
    rival = "negras" if objetivo == "blancas" else "blancas"
    puntos = 100.0 * (estado.puntos[objetivo] - estado.puntos[rival])
    avance = sum(_avance(c, objetivo) for c in estado.piezas(objetivo))
    avance -= sum(_avance(c, rival) for c in estado.piezas(rival))
    return puntos + avance


def _valor(estado, profundidad: int, objetivo: str) -> float:
    ganador = estado.ganador()
    if ganador is not None:
        return 10_000.0 if ganador == objetivo else -10_000.0
    jugadas = estado.jugadas()
    if not jugadas:
        ganador = "negras" if estado.turno == "blancas" else "blancas"
        return 10_000.0 if ganador == objetivo else -10_000.0
    if profundidad == 0:
        return _evaluar(estado, objetivo)
    valores = []
    for jugada in jugadas:
        esperado = sum(
            p * _valor(hijo, profundidad - 1, objetivo)
            for p, hijo in estado.siguiente(jugada)
        )
        valores.append(esperado)
    return (max if estado.turno == objetivo else min)(valores)


def jugar(estado, tiempo_restante):
    jugadas = estado.jugadas()
    for jugada in jugadas:
        if all(hijo.ganador() == color for _, hijo in estado.siguiente(jugada)):
            return jugada
    valores = [
        sum(p * _valor(hijo, PROFUNDIDAD - 1, color) for p, hijo in estado.siguiente(j))
        for j in jugadas
    ]
    mejor = max(valores)
    return rng.choice([j for j, v in zip(jugadas, valores, strict=True) if v == mejor])
