
"""Bot hibrido de Monte Carlo.

Blancas: simulaciones Monte Carlo probabilisticas.
Negras: simulaciones guiadas por progreso y anotacion.

Las reglas del tablero y los dados son las del motor oficial.
"""

import itertools
import random
import time

import numpy as np
from numba import njit
from motor import Estado, nucleo, reportar, simular_al_azar


SIMULACIONES = 10
RESERVA = 0.30


@njit(cache=True)
def simular_guiado(tablero, puntos, turno, n, semilla, objetivo):
    """Fraccion de victorias usando jugadas ponderadas."""

    np.random.seed(semilla)

    origenes = np.empty(54, np.int64)
    destinos = np.empty(54, np.int64)
    pesos = np.empty(54, np.float64)

    victorias = 0

    for _ in range(n):
        tab = tablero.copy()
        pts = puntos.copy()
        jugador = turno

        while True:
            cantidad = nucleo.generar(
                tab, jugador, origenes, destinos
            )

            if cantidad == 0:
                ganador = 1 - jugador
                break

            peso_total = 0.0
            meta = 8 if jugador == 0 else 0

            for k in range(cantidad):
                destino = destinos[k]
                fila = destino // 9

                avance = (
                    fila if jugador == 0 else 8 - fila
                )

                # Dar preferencia a peones mas avanzados.
                x = avance / 8.0

                peso = (
                    1.0
                    + 0.25 * avance
                    + 8.0 * x * x * x * x
                )

                # Priorizar anotaciones, especialmente el tercer punto.
                if fila == meta:
                    peso += (
                        45.0 if pts[jugador] == 2
                        else 22.0
                    )

                # Evaluar ataques segun el dado oficial.
                if tab[destino] != 0:
                    prob = nucleo.prob_jugada(
                        tab,
                        jugador,
                        origenes[k],
                        destino,
                    )

                    peso *= 0.4 + 1.1 * prob

                pesos[k] = peso
                peso_total += peso

            # Seleccion aleatoria ponderada.
            ruleta = np.random.random() * peso_total
            elegido = cantidad - 1

            for k in range(cantidad):
                ruleta -= pesos[k]

                if ruleta <= 0.0:
                    elegido = k
                    break

            origen = origenes[elegido]
            destino = destinos[elegido]

            prob = nucleo.prob_jugada(
                tab, jugador, origen, destino
            )

            gana_ataque = np.random.random() < prob

            nucleo.aplicar(
                tab,
                pts,
                jugador,
                origen,
                destino,
                gana_ataque,
            )

            if pts[jugador] >= 3:
                ganador = jugador
                break

            jugador = 1 - jugador

        if ganador == objetivo:
            victorias += 1

    return victorias / n


def codificar(estado):
    """Convertir el estado visible a arreglos para Numba."""

    tab = np.zeros(81, dtype=np.int8)

    for indice, bando in enumerate(
        ("blancas", "negras"), 1
    ):
        for casilla in estado.piezas(bando):
            columna = ord(casilla[0]) - ord("a")
            fila = int(casilla[1]) - 1
            tab[fila * 9 + columna] = indice

    p = estado.puntos

    pts = np.array(
        [p["blancas"], p["negras"]],
        dtype=np.int64,
    )

    turno = 0 if estado.turno == "blancas" else 1

    return tab, pts, turno


def preparar(config):
    global color, objetivo, rng, reserva

    color = config.color
    objetivo = 0 if color == "blancas" else 1

    rng = random.Random(config.semilla)
    reserva = RESERVA * config.reloj_s

    # Compilar los simuladores antes de comenzar la partida.
    tablero, puntos, turno = codificar(
        Estado.inicial()
    )

    simular_guiado(
        tablero,
        puntos,
        turno,
        1,
        0,
        objetivo,
    )

    simular_al_azar(
        Estado.inicial(),
        1,
        0,
        para=color,
    )


def jugar(estado, tiempo_restante):
    jugadas = estado.jugadas()

    if len(jugadas) == 1:
        return jugadas[0]

    disponible = tiempo_restante - reserva

    limite = (
        time.perf_counter()
        + max(0.0, disponible) / 25.0
    )

    sucesores = {
        j: estado.siguiente(j)
        for j in jugadas
    }

    # Buscar victorias inmediatas garantizadas.
    for j in jugadas:
        if all(
            hijo.ganador() == color
            for _, hijo in sucesores[j]
        ):
            return j

    if disponible <= 0:
        return rng.choice(jugadas)

    # Preparar una sola vez los estados sucesores.
    preparados = {
    j: [
        (
            prob,
            codificar(hijo) if color == "negras" else None,
            hijo.ganador(),
        )
        for prob, hijo in sucesores[j]
    ]
    for j in jugadas
    }

    total = {j: 0.0 for j in jugadas}
    muestras = {j: 0 for j in jugadas}

    # Monte Carlo con politica distinta segun el color.
    for hechas, j in enumerate(
        itertools.cycle(jugadas), 1
    ):
        valor = 0.0

        # CORRECCION:
        # El indice identifica el resultado exacto del dado.
        for indice, (prob, datos, ganador) in enumerate(
            preparados[j]
        ):
            if ganador is not None:
                estimacion = (
                    1.0 if ganador == color else 0.0
                )

            elif color == "blancas":
                # Monte Carlo probabilistico oficial.
                estimacion = simular_al_azar(
                    sucesores[j][indice][1],
                    20,
                    rng.randrange(2**31),
                    para=color,
                )

            else:
                # Monte Carlo guiado.
                estimacion = simular_guiado(
                    *datos,
                    SIMULACIONES,
                    rng.randrange(2**31),
                    objetivo,
                )

            valor += prob * estimacion

        total[j] += valor
        muestras[j] += 1

        if (
            hechas >= len(jugadas)
            and time.perf_counter() >= limite
        ):
            break

    mejor = max(
        jugadas,
        key=lambda j: total[j] / muestras[j],
    )

    reportar(
        muestras=sum(muestras.values()),
        eval=total[mejor] / muestras[mejor],
    )

    return mejor
