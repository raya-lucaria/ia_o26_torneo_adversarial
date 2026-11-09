"""Bot «T-Regex»: expectiminimax con poda alfa-beta/Star1 y profundización iterativa.

La búsqueda vive en `busqueda.py` (compilada con numba). Aquí solo:
1. convertimos el `Estado` del motor a nuestros arreglos,
2. repartimos el reloj y profundizamos mientras alcance el tiempo,
3. regresamos la mejor jugada de la última profundidad terminada.
"""

import time

import numpy as np

import busqueda as bq
from motor import Estado, Jugada, reportar
from motor.jugada import casilla_de_texto

RESERVA_S = 2.0  # segundos del reloj que nunca gastamos (comunicación, máquina más lenta)
JUGADAS_ESPERADAS = 42  # jugadas propias que suele durar una partida (medido: 39 a 49)
MIN_RESTANTES = 10  # aunque la partida se alargue, suponemos que faltan al menos estas
MAX_PROF = 40


def preparar(config):
    global color, ctx, nodos_por_s, hechas
    hechas = 0  # jugadas que llevamos en esta partida
    color = 0 if config.color == "blancas" else 1
    ctx = bq.nuevo_contexto()
    # Compilamos numba aquí (fuera del reloj) con una búsqueda corta desde la posición inicial.
    tab, pts = a_arreglos(Estado.inicial())
    inicio = time.perf_counter()
    bq.buscar(ctx, tab, pts, 0, 3, -1, 1 << 62)
    nodos_por_s = 200_000.0  # se corrige con lo que midamos en cada jugada
    print(f"[t_regex] compilado y búsqueda de prueba en {time.perf_counter() - inicio:.2f} s")


def a_arreglos(estado):
    tab = np.zeros(bq.CASILLAS, np.int8)
    for valor, nombre in ((1, "blancas"), (2, "negras")):
        for texto in estado.piezas(nombre):
            tab[casilla_de_texto(texto)] = valor
    pts = np.array([estado.puntos["blancas"], estado.puntos["negras"]], np.int64)
    return tab, pts


def jugar(estado, tiempo_restante):
    global nodos_por_s, hechas
    inicio = time.perf_counter()
    hechas += 1
    jugadas = estado.jugadas()
    if len(jugadas) == 1:
        return jugadas[0]

    # Presupuesto: lo disponible repartido entre las jugadas que creemos que faltan.
    disponible = max(tiempo_restante - RESERVA_S, 0.0)
    presupuesto = disponible / max(MIN_RESTANTES, JUGADAS_ESPERADAS - hechas)
    tope = min(2 * presupuesto, disponible * 0.3)  # nunca pasamos de aquí: se aborta

    tab, pts = a_arreglos(estado)
    # Apertura: si la posición está en el libro (búsqueda profunda precalculada), no pensamos.
    del_libro = bq.jugada_de_libro(tab, pts, color)
    if del_libro >= 0:
        j = Jugada(del_libro // bq.CASILLAS, del_libro % bq.CASILLAS)
        if j in jugadas:
            reportar(libro=1)
            return j
    bq.nueva_jugada(ctx)
    mejor, valor, prof_hecha, nodos_total = -1, 0.0, 0, 0
    dt_previo = 0.0
    for prof in range(1, MAX_PROF + 1):
        restante = tope - (time.perf_counter() - inicio)
        limite = max(int(restante * nodos_por_s), 1000) if prof > 1 else 1 << 62
        t0 = time.perf_counter()
        jugada, v, completas, nodos, abortada = bq.buscar(ctx, tab, pts, color, prof, mejor, limite)
        dt = time.perf_counter() - t0
        nodos_total += nodos
        if dt > 0.01:
            nodos_por_s = 0.5 * nodos_por_s + 0.5 * nodos / dt
        if abortada:
            # Iteración incompleta: sirve si alcanzó a terminar la jugada que iba primero
            # (la mejor de la profundidad anterior); lo que la superó se evaluó completo.
            if completas > 0:
                mejor = jugada
            break
        mejor, valor, prof_hecha = jugada, v, prof
        if abs(v) > 0.95:  # victoria o derrota segura: no hace falta ver más
            break
        # ¿Alcanza para la siguiente? Estimamos su costo con el crecimiento de esta.
        crecimiento = dt / dt_previo if dt_previo > 0.002 else 6.0
        dt_previo = dt
        if time.perf_counter() - inicio + dt * min(max(crecimiento, 2.0), 30.0) > presupuesto:
            break

    reportar(prof=prof_hecha, nodos=int(nodos_total), eval=round((valor + 1) / 2, 3))
    j = Jugada(int(mejor // bq.CASILLAS), int(mejor % bq.CASILLAS))
    return j if j in jugadas else jugadas[0]
