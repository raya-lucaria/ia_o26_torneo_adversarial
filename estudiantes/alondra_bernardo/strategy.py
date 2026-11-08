"""Expectiminimax con corte (alfa-beta + Star1 + quietud) en numba, con profundización iterativa."""

import json
import time
from pathlib import Path

import numpy as np

from motor import Estado, reportar
import rapido

RESERVA = 0.05         # fracción del reloj que nunca usamos (con 0.1 sobraban ~6 s)
RESERVA_MINIMA = 1.0   # segundos; protege relojes cortos
FRENO = 2.0            # no empezar otra profundidad si lo usado × FRENO rebasa el objetivo
TOPE = 1.5             # una búsqueda se aborta al pasar TOPE × objetivo
USO = 1.1              # el freno deja parte del objetivo sin usar; esto lo compensa
LMR = True             # reducción de jugadas tardías (197-163 en la arena)
SOBRECOSTO = 0.05      # s por jugada que el árbitro cuenta aunque no pensemos (~40 ms en Docker)
MIN_RESTANTES = 10.0   # con 6, las partidas largas jugaban el final con ~10 ms por jugada


def jugadas_restantes(estado):
    """Estimación conservadora de cuántas jugadas propias faltan, según las piezas.

    Ajustada al percentil 75 de 96 partidas de m7 en Docker (con 36 piezas faltan
    38 jugadas propias; con 30, 15; con 24, 7). La fórmula anterior seguía la
    mediana y dejaba las partidas largas, que son las que perdíamos, con
    0.04-0.17 s por jugada al final.
    """
    piezas = len(estado.piezas("blancas")) + len(estado.piezas("negras"))
    return max(MIN_RESTANTES, 3.0 + 0.085 * (piezas - 16) ** 2)


def preparar(config):
    global color, reserva, ctx, nps
    color = config.color
    reserva = max(RESERVA * config.reloj_s, RESERVA_MINIMA)
    ruta = Path(config.carpeta_datos) / "pesos.json"
    pesos = (json.loads(ruta.read_text())["pesos"] if ruta.exists()
             else rapido.pesos_iniciales())
    if len(pesos) > rapido.RASGOS:
        raise ValueError("pesos.json no coincide con los rasgos de rapido.py")
    ctx = rapido.Contexto(pesos)
    ctx.control[5] = int(LMR)
    # Compilar numba aquí (fuera del reloj) y medir nodos por segundo.
    inicial = Estado.inicial()
    rapido.buscar(ctx, inicial._tablero, inicial._puntos, inicial._color, 2, 10**9)
    t = time.perf_counter()
    rapido.buscar(ctx, inicial._tablero, inicial._puntos, inicial._color, 6, 10**9)
    nps = max(2e5, ctx.control[0] / max(1e-3, time.perf_counter() - t))
    ctx.limpiar()


def jugar(estado, tiempo_restante):
    global nps
    inicio = time.perf_counter()
    jugadas = estado.jugadas()
    if len(jugadas) == 1:
        return jugadas[0]
    restantes = jugadas_restantes(estado)
    # El costo fijo de cada jugada futura se aparta antes de repartir el reloj.
    disponible = tiempo_restante - reserva - SOBRECOSTO * restantes
    if disponible <= 0.05:
        disponible = 0.05
    objetivo = USO * disponible / restantes
    tope = min(TOPE * objetivo, 0.5 * disponible)
    ctx.historial *= 0.5  # lo de turnos pasados pesa menos

    mejor = None
    valor = 0.0
    profundidad = 0
    nodos = 0
    for prof in range(1, rapido.MAX_PLY):
        usado = time.perf_counter() - inicio
        if prof > 1 and usado * FRENO > objetivo:
            break
        limite = max(1000, int((tope - usado) * nps))
        t = time.perf_counter()
        v, cod = rapido.buscar(ctx, estado._tablero, estado._puntos, estado._color, prof, limite)
        dt = time.perf_counter() - t
        nodos += ctx.control[0]
        if dt > 0.02:
            nps = 0.7 * nps + 0.3 * ctx.control[0] / dt
        if cod is None:
            # Abortada: control[3] es la mejor de las jugadas de la raíz que sí se terminaron
            # a esta profundidad (la mejor anterior va primero por la tabla). Con la ventana
            # completa de la raíz su valor es exacto, así que vale más que la profundidad anterior.
            if ctx.control[3] >= 0 and mejor is not None:
                mejor, profundidad = int(ctx.control[3]), prof - 0.5
            break
        mejor, valor, profundidad = cod, v, prof
        if abs(v) > 0.999:  # final forzado: ya no cambia
            break
    if mejor is None:
        # Sin tiempo ni para la profundidad 1: un ply sin quietud cuesta ~una evaluación
        # por jugada, mejor que la primera jugada legal.
        ctx.control[4] = 0
        _, mejor = rapido.buscar(ctx, estado._tablero, estado._puntos, estado._color, 1, 10**6)
        ctx.control[4] = rapido.QUIETUD
        if mejor is None:
            return jugadas[0]
    reportar(prof=profundidad, valor=round(float(valor), 3), nodos=int(nodos),
             objetivo=round(objetivo, 3))
    return rapido.texto(mejor)
