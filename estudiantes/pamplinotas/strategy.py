"""Bot «pamplinotas»: expectiminimax con corte, alfa-beta y *-minimax, con profundización
iterativa y tabla de transposición, sobre un motor propio compilado con numba (`buscador.py`).

Entrada y salida son las del torneo: `preparar(config)` una vez por partida y
`jugar(estado, tiempo_restante)` cada turno, que regresa el texto de una jugada legal.
"""

import json
import time

from motor import Estado, reportar

import buscador

RESERVA_FRACCION = 0.07  # parte del reloj que nunca gastamos pensando (comunicación, máquina lenta)
RESERVA_MAX_S = 3.5
JUGADAS_PROPIAS_ESPERADAS = 42  # las partidas contra montecarlo duran 41 jugadas propias en promedio
MIN_JUGADAS_RESTANTES = 10  # al final de la partida no supongas que ya casi acaba
FACTOR_EXTRA = 1.35  # se permite rebasar el presupuesto de una jugada hasta este factor
RAMIFICACION_EFECTIVA = 3.0  # cuánto crece cada nivel de profundidad, para no empezar uno que no cabe
PROF_MAX = 30

buscador_global = None
reserva_s = RESERVA_MAX_S
carga_s = 0.0  # cuánto tardó preparar (compilar numba); se informa en la primera jugada
inicio_carga = time.perf_counter()
jugadas_hechas = 0


def cargar_pesos(carpeta_datos):
    """Pesos de la evaluación ajustados offline (`data/pesos.json`)."""
    with open(carpeta_datos / "pesos.json", encoding="utf-8") as f:
        return buscador.pesos_desde_json(json.load(f))


def preparar(config):
    """Compila numba (unos 3 s en Docker, fuera del reloj) y deja listas las tablas de la partida."""
    global buscador_global, reserva_s, jugadas_hechas, carga_s
    reserva_s = min(RESERVA_MAX_S, RESERVA_FRACCION * config.reloj_s)
    jugadas_hechas = 0
    buscador_global = buscador.Buscador(cargar_pesos(config.carpeta_datos))
    tab, color, pts = buscador.tablero_desde_estado(Estado.inicial())
    buscador_global.buscar(tab, color, pts, 3, time.perf_counter() + 5.0)
    buscador_global.limpiar_tablas()
    carga_s = round(time.perf_counter() - inicio_carga, 2)


def presupuesto(tiempo_restante, reserva, hechas, esperadas=JUGADAS_PROPIAS_ESPERADAS, minimo=MIN_JUGADAS_RESTANTES):
    """Segundos para esta jugada: lo disponible repartido entre las jugadas que faltan."""
    disponible = tiempo_restante - reserva
    if disponible <= 0:
        return 0.0
    faltan = max(minimo, esperadas - hechas)
    return disponible / faltan


def elegir(motor, tab, color, pts, inicio, objetivo):
    """Profundización iterativa dentro del presupuesto. Regresa (jugada, valor, profundidad, nodos);
    la jugada es la de la última búsqueda completa, o −1 si ni la profundidad 1 terminó."""
    limite_duro = inicio + objetivo * FACTOR_EXTRA
    mejor, valor, prof_hecha, nodos = -1, 0.0, 0, 0
    for prof in range(1, PROF_MAX + 1):
        transcurrido = time.perf_counter() - inicio
        if prof > 1 and transcurrido * RAMIFICACION_EFECTIVA > objetivo:
            break
        j, v, abortada = motor.buscar(tab, color, pts, prof, limite_duro)
        nodos += motor.nodos()
        if abortada:
            break
        mejor, valor, prof_hecha = j, v, prof
        if abs(v) >= buscador.M - buscador.MAX_PLY:
            break  # final forzado a la vista: no hace falta mirar más lejos
    return mejor, valor, prof_hecha, nodos


def jugar(estado, tiempo_restante):
    global jugadas_hechas
    inicio = time.perf_counter()
    jugadas_hechas += 1
    tab, color, pts = buscador.tablero_desde_estado(estado)
    legales = estado.jugadas()
    if len(legales) == 1:
        return legales[0]

    objetivo = presupuesto(tiempo_restante, reserva_s, jugadas_hechas)
    if objetivo <= 0.0:
        # Solo queda la reserva: una búsqueda mínima, que es casi instantánea.
        j, v, abortada = buscador_global.buscar(tab, color, pts, 2, inicio + 0.05)
        return buscador.jugada_a_texto(j) if not abortada else str(legales[0])

    mejor, valor, prof_hecha, nodos = elegir(buscador_global, tab, color, pts, inicio, objetivo)
    if mejor < 0:
        return str(legales[0])  # ni la profundidad 1 terminó (máquina muy lenta)
    informe = dict(prof=prof_hecha, nodos=nodos, eval=round(valor / buscador.ESCALA_V, 3))
    if jugadas_hechas == 1:
        informe["carga_s"] = carga_s
    reportar(**informe)
    return buscador.jugada_a_texto(mejor)
