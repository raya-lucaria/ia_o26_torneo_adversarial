"""Rival de entrenamiento: anotación/ataque muy favorable, luego azar."""

from __future__ import annotations

import random


UMBRAL_ATAQUE_GRATIS = 0.80


def preparar(config):
    global color, rng
    color = config.color
    rng = random.Random(config.semilla)


def jugar(estado, tiempo_restante):
    jugadas = estado.jugadas()
    for jugada in jugadas:
        if all(hijo.ganador() == color for _, hijo in estado.siguiente(jugada)):
            return jugada
    favorables = [
        j for j in jugadas
        if estado.es_ataque(j) and estado.prob_ataque(j) >= UMBRAL_ATAQUE_GRATIS
    ]
    if favorables:
        mejor = max(estado.prob_ataque(j) for j in favorables)
        return rng.choice([j for j in favorables if estado.prob_ataque(j) == mejor])
    return rng.choice(jugadas)
