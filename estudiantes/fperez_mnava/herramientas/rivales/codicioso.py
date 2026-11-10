"""Rival de prueba «codicioso»: gana si puede; si no, ataca con p >= 0.5; si no, avanza sin atacar."""

import random


def preparar(config):
    global rng
    rng = random.Random(config.semilla)


def jugar(estado, tiempo_restante):
    jugadas = estado.jugadas()
    for j in jugadas:  # 1. una jugada que gana la partida sin importar el dado
        if all(hijo.ganador() == estado.turno for _, hijo in estado.siguiente(j)):
            return j
    buenos = [(estado.prob_ataque(j), j) for j in jugadas if estado.es_ataque(j)]
    buenos = [(p, j) for p, j in buenos if p >= 0.5]  # 2. el mejor ataque disponible
    if buenos:
        mejor = max(p for p, _ in buenos)
        return rng.choice([j for p, j in buenos if p == mejor])
    sin_ataque = [j for j in jugadas if not estado.es_ataque(j)]  # 3. avanzar sin atacar
    return rng.choice(sin_ataque or jugadas)
