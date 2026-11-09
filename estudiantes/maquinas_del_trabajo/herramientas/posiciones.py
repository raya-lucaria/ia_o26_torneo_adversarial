"""Posiciones de prueba hechas con el motor oficial (solo para pruebas y herramientas).

- `muestrear`: posiciones no terminales de partidas al azar, más posiciones sintéticas (peones
  regados al azar) para cubrir casos raros: pocos peones, bandos sin peones, 2–2 puntos.
- `expectimax`: la búsqueda de referencia, lenta y sin podas, hecha con el motor oficial.

Los combates de estas partidas se resuelven con un `random.Random` propio: no tienen nada que
ver con los dados del árbitro.
"""

import random

from motor import Estado
from motor.jugada import nombre_casilla

from evaluacion import PESOS_F2
from herramientas.desde_python import evaluar
from motor_rapido import desde_estado

LADO = 9
PERDIDA_POR_PLY = 0.0005  # el valor de una victoria baja esto por ply (decisión 4 del plan)


def posicion(blancas: list[str], negras: list[str], turno: str, puntos=(0, 0)) -> Estado:
    """Posición armada a mano, p. ej. `posicion(["e8"], ["b9"], "blancas", (2, 0))`."""
    return Estado.desde_dict(
        {
            "blancas": blancas,
            "negras": negras,
            "turno": turno,
            "puntos": {"blancas": puntos[0], "negras": puntos[1]},
        }
    )


def expectimax(estado: Estado, prof: int, pesos=PESOS_F2, ply: int = 0) -> float:
    """Valor para el bando que mueve, sin podas: el máximo, entre sus jugadas, del valor esperado."""
    ganador = estado.ganador()
    if ganador is not None:
        valor = 1.0 - PERDIDA_POR_PLY * ply
        return valor if ganador == estado.turno else -valor
    if prof == 0:
        tab, pts, color = desde_estado(estado)
        return evaluar(tab, pts, color, pesos)
    return max(
        sum(p * -expectimax(hijo, prof - 1, pesos, ply + 1) for p, hijo in estado.siguiente(j))
        for j in estado.jugadas()
    )


def posicion_sintetica(rng: random.Random) -> Estado | None:
    """Peones al azar (0–18 por bando, nunca en su fila meta), 0–2 puntos y turno al azar.

    Regresa None si la posición ya terminó (el bando en turno no tiene peones).
    """
    casillas = rng.sample(range(LADO * LADO), LADO * LADO)
    usadas = set()
    listas = {}
    for nombre, fila_meta in (("blancas", LADO - 1), ("negras", 0)):
        cuantos = rng.randint(0, 18)
        elegidas = [c for c in casillas if c // LADO != fila_meta and c not in usadas][:cuantos]
        usadas.update(elegidas)
        listas[nombre] = [nombre_casilla(c) for c in elegidas]
    estado = Estado.desde_dict(
        {
            **listas,
            "turno": rng.choice(["blancas", "negras"]),
            "puntos": {"blancas": rng.randint(0, 2), "negras": rng.randint(0, 2)},
        }
    )
    return None if estado.terminado() else estado


def muestrear(
    n: int, semilla: int, sinteticas: float = 0.3, ply_min: int = 0, ply_max: int = 10**9
) -> list[Estado]:
    """`n` posiciones no terminales: de partidas al azar (entre `ply_min` y `ply_max`) y sintéticas.

    Mismos argumentos → mismas posiciones.
    """
    rng = random.Random(semilla)
    posiciones: list[Estado] = []
    estado, ply = None, 0
    while len(posiciones) < n:
        if rng.random() < sinteticas:
            sintetica = posicion_sintetica(rng)
            if sintetica is not None:
                posiciones.append(sintetica)
            continue
        if estado is None or estado.terminado() or ply > ply_max:
            estado, ply = Estado.inicial(), 0
        jugada = rng.choice(estado.jugadas())
        gana = rng.random() < estado.prob_ataque(jugada) if estado.es_ataque(jugada) else None
        estado = estado.aplicar(jugada, gana)
        ply += 1
        if not estado.terminado() and ply_min <= ply <= ply_max:
            posiciones.append(estado)
    return posiciones
