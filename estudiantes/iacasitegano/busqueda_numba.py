"""La misma búsqueda que `busqueda.py` (expectiminimax exacto + poda alfa-beta
+ Star1 en los nodos de azar), pero compilada con numba sobre el tablero
crudo (`int8[81]`), no sobre objetos `Estado`. Medido: profundidad 3-4 en
Python puro ya cuesta demasiado para competir con `montecarlo` (que simula
partidas completas, no solo unos ply) — esto busca 1-2 órdenes de magnitud
más de nodos/segundo para llegar más lejos.

`elegir` recibe y regresa objetos `Estado`/`Jugada` normales: convierte a
arreglos crudos, llama al núcleo numba, y convierte la jugada de vuelta. Así
`strategy.py` no cambia su forma de llamar a la búsqueda.

Tabla de transposición: un arreglo fijo de numpy (tabla hash de mapeo directo,
"siempre reemplaza" en colisión), no un `numba.typed.Dict` — mismo patrón de
tipos (arreglos) que ya usa el resto del archivo, sin arriesgar compatibilidad
de `cache=True` con un tipo de dato más exótico. La llave combina el hash
Zobrist del tablero (`nucleo.clave`, ya numba) con la profundidad restante:
un valor calculado a profundidad 2 no sirve para una pregunta a profundidad 4.

# ponytail: solo se guardan valores EXACTOS (el ciclo de jugadas terminó sin
# cortar por poda): uno podado a medias es una cota, no el valor real, y
# cachearlo podría hacer que una búsqueda futura con otra ventana alfa/beta
# use un valor que no le corresponde. Mismo criterio que `busqueda.py`.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit

from motor import Estado, Jugada
from motor.jugada import casilla_de_texto
from motor.nucleo import (
    CASILLAS,
    LADO,
    MAX_JUGADAS,
    PUNTOS_PARA_GANAR,
    VACIA,
    Z_PIEZAS,
    Z_PUNTOS,
    Z_TURNO,
    aplicar,
    apoyo,
    clave,
    generar,
    prob_jugada,
)

import eval as ev

PROFUNDIDAD_MAXIMA = 12
MARGEN_SEGURIDAD = 2.0
FACTOR_INICIAL = 20.0
# Piso pesimista del factor de crecimiento entre profundidades: medido en partidas reales,
# un salto de profundidad puede costar ~123x (no los ~15-30x "típicos") cuando una posición
# se vuelve táctica de golpe (muchos ataques simultáneos = la poda Star1 corta mucho menos).
# Confiar solo en la ÚLTIMA transición medida nos hizo perder una partida por reloj
# (profundidad 5→6 pasó de 0.08s a 9.35s). El piso evita repetirlo: nunca asumimos que el
# próximo salto será más barato que esto, sin importar qué tan barata haya sido la historia.
FACTOR_PESIMISTA = 150.0
TT_BITS = 20
TT_TAM = 1 << TT_BITS  # potencia de 2: el índice es un AND, no un módulo
TT_MASCARA = np.uint64(TT_TAM - 1)
TT_VACIA = np.uint64(0)  # una posición real con llave exactamente 0 es, en la práctica, imposible
QUIESCENCIA_MAX = 0  # medida con A/B real: no compensa (ver nota abajo). Apagada por default.
# Fase 2 medida con A/B real y rechazada: dejar la implementación disponible para
# futuras pruebas, pero el campeón corre sin este cambio.
ORDENAMIENTO_JUGADAS = False


@njit(cache=True)
def _features_numba(tablero, puntos, color):
    """Igual que `eval.features`, pero sobre el tablero crudo (ver ahí el porqué de cada una)."""
    rival = 1 - color
    avance_p = apoyo_p = pasado_p = 0
    avance_r = apoyo_r = pasado_r = 0
    avance_max_p = -1
    avance_max_r = -1
    columnas_p = np.zeros(LADO, np.int8)
    columnas_r = np.zeros(LADO, np.int8)
    for c in range(CASILLAS):
        pieza = tablero[c]
        if pieza == VACIA:
            continue
        fila = c // LADO
        col = c % LADO
        es_propio = (pieza - 1) == color
        bando = color if es_propio else rival
        a = fila if bando == 0 else (LADO - 1 - fila)
        ap = apoyo(tablero, c, bando)
        # Pasado: nadie del otro color en las 3 columnas (col-1, col, col+1) en las
        # filas que le faltan por recorrer.
        pasado = True
        fila_desde = fila + 1 if bando == 0 else 0
        fila_hasta = LADO if bando == 0 else fila
        for f in range(fila_desde, fila_hasta):
            base = f * LADO
            for dc in (-1, 0, 1):
                cc = col + dc
                if cc < 0 or cc >= LADO:
                    continue
                if tablero[base + cc] == (2 - bando):  # pieza del color contrario a `bando`
                    pasado = False
        if es_propio:
            avance_p += a
            if a > avance_max_p:
                avance_max_p = a
            apoyo_p += ap
            if pasado:
                pasado_p += a
            columnas_p[col] = 1
        else:
            avance_r += a
            if a > avance_max_r:
                avance_max_r = a
            apoyo_r += ap
            if pasado:
                pasado_r += a
            columnas_r[col] = 1
    vacias_p = LADO - columnas_p.sum()
    vacias_r = LADO - columnas_r.sum()
    distancia_p = LADO if avance_max_p < 0 else LADO - 1 - avance_max_p
    distancia_r = LADO if avance_max_r < 0 else LADO - 1 - avance_max_r
    salida = np.empty(6, np.float64)
    salida[0] = puntos[color] - puntos[rival]
    salida[1] = avance_p - avance_r
    salida[2] = apoyo_p - apoyo_r
    salida[3] = pasado_p - pasado_r
    salida[4] = vacias_r - vacias_p
    salida[5] = distancia_p - distancia_r
    return salida


@njit(cache=True)
def _evaluar_numba(tablero, puntos, color, pesos):
    return np.dot(pesos, _features_numba(tablero, puntos, color))


@njit(cache=True)
def _amenaza_anotacion(tablero, color):
    """Si el bando tiene una jugada que llega a su fila de anotación."""
    origenes = np.empty(MAX_JUGADAS, np.int64)
    destinos = np.empty(MAX_JUGADAS, np.int64)
    n = generar(tablero, color, origenes, destinos)
    fila_meta = LADO - 1 if color == 0 else 0
    for i in range(n):
        if destinos[i] // LADO == fila_meta:
            return True
    return False


@njit(cache=True)
def _es_pasado(tablero, color, destino):
    """Aproximación barata de peón pasado para desempatar jugadas tranquilas."""
    fila = destino // LADO
    col = destino % LADO
    rival = 1 - color
    if color == 0:
        filas = range(fila + 1, LADO)
    else:
        filas = range(0, fila)
    for f in filas:
        base = f * LADO
        for dc in (-1, 0, 1):
            cc = col + dc
            if 0 <= cc < LADO and tablero[base + cc] == (2 - rival):
                return False
    return True


@njit(cache=True)
def _prioridad_jugada(tablero, color, origen, destino, amenaza_rival, origen_sugerido,
                      destino_sugerido):
    """Prioridad lexicográfica aproximada; nunca altera el valor minimax."""
    fila_meta = LADO - 1 if color == 0 else 0
    ataque = tablero[destino] != VACIA
    puntuacion = 0.0

    # 1-2. Victoria inmediata y ataque que anota si gana.
    if destino // LADO == fila_meta:
        puntuacion += 1_000_000.0 if not ataque else 900_000.0

    # 3. Ataques con mayor probabilidad.
    if ataque:
        puntuacion += 500_000.0 + 100_000.0 * prob_jugada(tablero, color, origen, destino)

    # 4. Si había una anotación rival disponible, premiar la jugada que la bloquea.
    # Se asume el resultado favorable del propio ataque solo para ordenar; el árbol
    # sigue evaluando todas las ramas exactamente.
    tablero_despues = tablero.copy()
    tablero_despues[destino] = tablero_despues[origen]
    tablero_despues[origen] = VACIA
    if ataque and destino // LADO != fila_meta:
        # El defensor desaparece en la rama favorable, que es la relevante para
        # reconocer una defensa táctica; la rama desfavorable se evalúa aparte.
        pass
    if amenaza_rival and not _amenaza_anotacion(tablero_despues, 1 - color):
        puntuacion += 100_000.0

    # 5. Carrera pasada y apoyo; el avance también desempata tranquilas.
    avance = destino // LADO if color == 0 else LADO - 1 - (destino // LADO)
    puntuacion += 100.0 * avance
    if _es_pasado(tablero_despues, color, destino):
        puntuacion += 10_000.0
    puntuacion += 10.0 * apoyo(tablero_despues, destino, color)

    # 6. Ganadora de la iteración anterior; por debajo de las tácticas urgentes.
    if origen == origen_sugerido and destino == destino_sugerido:
        puntuacion += 1.0
    return puntuacion


@njit(cache=True)
def _ordenar_jugadas_numba(tablero, color, origenes, destinos, n, origen_sugerido,
                           destino_sugerido):
    amenaza_rival = _amenaza_anotacion(tablero, 1 - color)
    prioridades = np.empty(MAX_JUGADAS, np.float64)
    for i in range(n):
        prioridades[i] = _prioridad_jugada(
            tablero, color, origenes[i], destinos[i], amenaza_rival,
            origen_sugerido, destino_sugerido,
        )
    # Selection sort: n es pequeño y numba lo compila sin crear objetos Python.
    for i in range(n):
        mejor = i
        for j in range(i + 1, n):
            if prioridades[j] > prioridades[mejor]:
                mejor = j
        if mejor != i:
            prioridades[i], prioridades[mejor] = prioridades[mejor], prioridades[i]
            origenes[i], origenes[mejor] = origenes[mejor], origenes[i]
            destinos[i], destinos[mejor] = destinos[mejor], destinos[i]


_FIRMA_VALOR = (
    "float64(int8[::1], int64[::1], int64, int64, int64, float64, float64, float64[::1], float64,"
    " int64[::1], int64[::1], uint64[::1], float64[::1])"
)
_FIRMA_QUIESCENCIA = (
    "float64(int8[::1], int64[::1], int64, int64, float64, float64, float64[::1], float64, int64,"
    " int64[::1], int64[::1])"
)


# Quiescencia: en la hoja (profundidad 0), si todavía hay ataques disponibles, el eval
# estático puede estar "a medio combate" (ignora que la jugada siguiente puede ganar o
# perder una pieza). Sigue SOLO los ataques (nunca jugadas tranquilas) unos turnos más,
# con "stand pat": el que mueve siempre puede quedarse con el valor estático si atacar
# no le conviene (igual que en una quiescencia de ajedrez).
#
# Medida con A/B real contra montecarlo (dos semillas, 40 partidas, tope de 2 ataques):
# mismo número de victorias que sin ella, pero MENOS puntos totales (p.ej. 13 vs 19 en
# una semilla) — sale más cara de lo que compensa en precisión, porque cada hoja de la
# búsqueda principal ahora dispara un sub-árbol completo, y eso resta profundidad neta
# dentro del mismo presupuesto de tiempo. Con `QUIESCENCIA_MAX = 0` (de fábrica) esta
# función se reduce exactamente al eval estático de antes: el mecanismo queda listo y
# validado (ver pruebas diferenciales) por si en el futuro, con más profundidad
# disponible o features distintas, sí compensa — pero no se activa sin medirlo de nuevo.
@njit(_FIRMA_QUIESCENCIA, cache=True)
def _quiescencia_numba(tablero, puntos, color, objetivo, alfa, beta, pesos, ganancia, restante,
                        origenes, destinos):
    if puntos[0] >= PUNTOS_PARA_GANAR:
        return ganancia if objetivo == 0 else -ganancia
    if puntos[1] >= PUNTOS_PARA_GANAR:
        return ganancia if objetivo == 1 else -ganancia
    n = generar(tablero, color, origenes, destinos)
    if ORDENAMIENTO_JUGADAS:
        _ordenar_jugadas_numba(tablero, color, origenes, destinos, n, -1, -1)
    if n == 0:
        ganador = 1 - color
        return ganancia if ganador == objetivo else -ganancia

    estatico = _evaluar_numba(tablero, puntos, objetivo, pesos)
    if restante == 0:
        return estatico

    es_max = color == objetivo
    if es_max:
        if estatico >= beta:
            return estatico
        if estatico > alfa:
            alfa = estatico
    else:
        if estatico <= alfa:
            return estatico
        if estatico < beta:
            beta = estatico

    mejor = estatico
    for i in range(n):
        origen, destino = origenes[i], destinos[i]
        if tablero[destino] == VACIA:
            continue  # solo ataques: una jugada tranquila no cambia la urgencia del combate
        o2 = np.empty(MAX_JUGADAS, np.int64)
        d2 = np.empty(MAX_JUGADAS, np.int64)
        p1 = prob_jugada(tablero, color, origen, destino)
        p2w = 1.0 - p1
        t_gana = tablero.copy()
        pt_gana = puntos.copy()
        aplicar(t_gana, pt_gana, color, origen, destino, True)
        v1 = _quiescencia_numba(
            t_gana, pt_gana, 1 - color, objetivo, -ganancia, ganancia, pesos, ganancia,
            restante - 1, o2, d2,
        )
        mejor_caso = p1 * v1 + p2w * ganancia
        peor_caso = p1 * v1 - p2w * ganancia
        if mejor_caso <= alfa:
            v = mejor_caso
        elif peor_caso >= beta:
            v = peor_caso
        else:
            o3 = np.empty(MAX_JUGADAS, np.int64)
            d3 = np.empty(MAX_JUGADAS, np.int64)
            t_pierde = tablero.copy()
            pt_pierde = puntos.copy()
            aplicar(t_pierde, pt_pierde, color, origen, destino, False)
            v2 = _quiescencia_numba(
                t_pierde, pt_pierde, 1 - color, objetivo, -ganancia, ganancia, pesos, ganancia,
                restante - 1, o3, d3,
            )
            v = p1 * v1 + p2w * v2
        if es_max:
            if v > mejor:
                mejor = v
            if mejor > alfa:
                alfa = mejor
        else:
            if v < mejor:
                mejor = v
            if mejor < beta:
                beta = mejor
        if alfa >= beta:
            break
    return mejor


# Una sola función auto-recursiva (no dos que se llaman entre sí): numba sabe inferir
# bien la recursión de una función consigo misma con firma explícita, pero no la
# recursión mutua entre dos funciones separadas ("cannot type infer runaway
# recursion"). La lógica de cada jugada (una o dos ramas de azar, con la poda Star1
# que acota la rama que no hace falta ver) va en línea, dentro del ciclo por jugada.
@njit(_FIRMA_VALOR, cache=True)
def _valor_numba(tablero, puntos, color, objetivo, profundidad, alfa, beta, pesos, ganancia,
                  origenes, destinos, tt_claves, tt_valores):
    # Igual que `estado.ganador()`: puntos primero, si no, generamos las jugadas para
    # ver si al que le toca ya no tiene ninguna (entonces gana el otro). Generamos UNA
    # sola vez y reusamos `n` para enumerar abajo si no hay poda ni final de partida.
    if puntos[0] >= PUNTOS_PARA_GANAR:
        return ganancia if objetivo == 0 else -ganancia
    if puntos[1] >= PUNTOS_PARA_GANAR:
        return ganancia if objetivo == 1 else -ganancia
    n = generar(tablero, color, origenes, destinos)
    if ORDENAMIENTO_JUGADAS:
        _ordenar_jugadas_numba(tablero, color, origenes, destinos, n, -1, -1)
    if n == 0:
        ganador = 1 - color
        return ganancia if ganador == objetivo else -ganancia
    if profundidad == 0:
        return _quiescencia_numba(
            tablero, puntos, color, objetivo, alfa, beta, pesos, ganancia, QUIESCENCIA_MAX,
            origenes, destinos,
        )

    llave = clave(tablero, color, puntos, Z_PIEZAS, Z_TURNO, Z_PUNTOS) ^ np.uint64(profundidad)
    idx = llave & TT_MASCARA
    if tt_claves[idx] == llave:
        return tt_valores[idx]

    es_max = color == objetivo
    mejor = -ganancia - 1.0 if es_max else ganancia + 1.0
    exacto = True
    for i in range(n):
        origen, destino = origenes[i], destinos[i]
        o2 = np.empty(MAX_JUGADAS, np.int64)
        d2 = np.empty(MAX_JUGADAS, np.int64)
        if tablero[destino] == VACIA:
            t2 = tablero.copy()
            p2 = puntos.copy()
            aplicar(t2, p2, color, origen, destino, True)
            v = _valor_numba(
                t2, p2, 1 - color, objetivo, profundidad - 1, alfa, beta, pesos, ganancia, o2, d2,
                tt_claves, tt_valores,
            )
        else:
            p1 = prob_jugada(tablero, color, origen, destino)
            p2w = 1.0 - p1
            t_gana = tablero.copy()
            pt_gana = puntos.copy()
            aplicar(t_gana, pt_gana, color, origen, destino, True)
            v1 = _valor_numba(
                t_gana, pt_gana, 1 - color, objetivo, profundidad - 1, -ganancia, ganancia,
                pesos, ganancia, o2, d2, tt_claves, tt_valores,
            )
            mejor_caso = p1 * v1 + p2w * ganancia
            peor_caso = p1 * v1 - p2w * ganancia
            if mejor_caso <= alfa:
                v = mejor_caso  # no le gana a alfa ni en el mejor caso: no hace falta ver "pierde"
            elif peor_caso >= beta:
                v = peor_caso  # ya es demasiado bueno/malo para el rival de arriba: tampoco
            else:
                o3 = np.empty(MAX_JUGADAS, np.int64)
                d3 = np.empty(MAX_JUGADAS, np.int64)
                t_pierde = tablero.copy()
                pt_pierde = puntos.copy()
                aplicar(t_pierde, pt_pierde, color, origen, destino, False)
                v2 = _valor_numba(
                    t_pierde, pt_pierde, 1 - color, objetivo, profundidad - 1, -ganancia,
                    ganancia, pesos, ganancia, o3, d3, tt_claves, tt_valores,
                )
                v = p1 * v1 + p2w * v2
        if es_max:
            if v > mejor:
                mejor = v
            if mejor > alfa:
                alfa = mejor
        else:
            if v < mejor:
                mejor = v
            if mejor < beta:
                beta = mejor
        if alfa >= beta:
            exacto = False  # quedaron jugadas sin revisar: esto es una cota, no el valor real
            break

    if exacto and llave != TT_VACIA:
        tt_claves[idx] = llave
        tt_valores[idx] = mejor
    return mejor


@njit(cache=True)
def _mejor_jugada_numba(tablero, puntos, color, profundidad, pesos, ganancia, origen_sugerido,
                         destino_sugerido):
    """Igual que un nodo MAX de `_valor_numba`, pero devolviendo también QUÉ jugada ganó
    (`_valor_numba` por sí solo regresa el valor, no el índice: no sirve para elegir).
    Tabla de transposición nueva por llamada: se reinicia en cada decisión de jugada.

    `origen_sugerido`/`destino_sugerido` (-1 si no hay): la jugada ganadora de la
    profundidad anterior en esta misma decisión. Probarla primero no cambia el
    resultado (el orden de las jugadas nunca cambia el valor de alfa-beta), pero
    aprieta `alfa` antes, así que las jugadas siguientes se podan más."""
    tt_claves = np.zeros(TT_TAM, np.uint64)
    tt_valores = np.zeros(TT_TAM, np.float64)
    origenes = np.empty(MAX_JUGADAS, np.int64)
    destinos = np.empty(MAX_JUGADAS, np.int64)
    n = generar(tablero, color, origenes, destinos)
    if ORDENAMIENTO_JUGADAS:
        _ordenar_jugadas_numba(
            tablero, color, origenes, destinos, n, origen_sugerido, destino_sugerido,
        )
    if origen_sugerido >= 0:
        for i in range(n):
            if origenes[i] == origen_sugerido and destinos[i] == destino_sugerido:
                if i != 0:
                    origenes[0], origenes[i] = origenes[i], origenes[0]
                    destinos[0], destinos[i] = destinos[i], destinos[0]
                break
    mejor_valor = -ganancia - 1.0
    mejor_i = 0
    alfa = -ganancia
    for i in range(n):
        origen, destino = origenes[i], destinos[i]
        o2 = np.empty(MAX_JUGADAS, np.int64)
        d2 = np.empty(MAX_JUGADAS, np.int64)
        if tablero[destino] == VACIA:
            t2 = tablero.copy()
            p2 = puntos.copy()
            aplicar(t2, p2, color, origen, destino, True)
            v = _valor_numba(
                t2, p2, 1 - color, color, profundidad - 1, alfa, ganancia, pesos, ganancia, o2,
                d2, tt_claves, tt_valores,
            )
        else:
            p1 = prob_jugada(tablero, color, origen, destino)
            p2w = 1.0 - p1
            t_gana = tablero.copy()
            pt_gana = puntos.copy()
            aplicar(t_gana, pt_gana, color, origen, destino, True)
            v1 = _valor_numba(
                t_gana, pt_gana, 1 - color, color, profundidad - 1, -ganancia, ganancia, pesos,
                ganancia, o2, d2, tt_claves, tt_valores,
            )
            mejor_caso = p1 * v1 + p2w * ganancia
            if mejor_caso <= alfa:
                v = mejor_caso
            else:
                o3 = np.empty(MAX_JUGADAS, np.int64)
                d3 = np.empty(MAX_JUGADAS, np.int64)
                t_pierde = tablero.copy()
                pt_pierde = puntos.copy()
                aplicar(t_pierde, pt_pierde, color, origen, destino, False)
                v2 = _valor_numba(
                    t_pierde, pt_pierde, 1 - color, color, profundidad - 1, -ganancia, ganancia,
                    pesos, ganancia, o3, d3, tt_claves, tt_valores,
                )
                v = p1 * v1 + p2w * v2
        if v > mejor_valor:
            mejor_valor = v
            mejor_i = i
            alfa = mejor_valor
    return origenes[mejor_i], destinos[mejor_i], mejor_valor


def _tablero_desde_estado(estado: Estado) -> tuple[np.ndarray, np.ndarray, int]:
    tablero = np.zeros(CASILLAS, np.int8)
    for t in estado.piezas("blancas"):
        tablero[casilla_de_texto(t)] = 1
    for t in estado.piezas("negras"):
        tablero[casilla_de_texto(t)] = 2
    puntos = np.array([estado.puntos["blancas"], estado.puntos["negras"]], np.int64)
    color = 0 if estado.turno == "blancas" else 1
    return tablero, puntos, color


def elegir(estado: Estado, color: str, limite_tiempo: float) -> tuple[Jugada, float, int]:
    """Misma interfaz que `busqueda.elegir`: sube la profundidad mientras alcance el tiempo."""
    tablero, puntos, color_idx = _tablero_desde_estado(estado)
    pesos = np.asarray(ev.PESOS, dtype=np.float64)
    ganancia = ev.cota() + 100.0

    jugadas = estado.jugadas()
    mejor_jugada, mejor_valor, mejor_profundidad = jugadas[0], 0.0, 0
    tiempo_anterior = None
    profundidad = 1
    origen_sugerido, destino_sugerido = -1, -1  # nada todavía: es la primera profundidad

    while profundidad <= PROFUNDIDAD_MAXIMA:
        inicio = time.perf_counter()
        o, d, valor = _mejor_jugada_numba(
            tablero, puntos, color_idx, profundidad, pesos, ganancia, origen_sugerido,
            destino_sugerido,
        )
        duracion = time.perf_counter() - inicio
        mejor_jugada = Jugada(int(o), int(d))
        mejor_valor, mejor_profundidad = valor, profundidad
        origen_sugerido, destino_sugerido = int(o), int(d)
        if time.perf_counter() >= limite_tiempo:
            break
        # Sin contador de nodos expuesto desde numba (no vale la pena el overhead):
        # usamos la razón de TIEMPO entre profundidades consecutivas como factor de
        # crecimiento medido, igual de válido que contar nodos para esta decisión. Pero
        # nunca confiamos en que el próximo salto sea más barato que el piso pesimista:
        # una posición tranquila puede volverse táctica de una profundidad a la siguiente.
        factor_medido = duracion / tiempo_anterior if tiempo_anterior else FACTOR_INICIAL
        factor = max(factor_medido, FACTOR_PESIMISTA)
        tiempo_anterior = duracion
        if limite_tiempo - time.perf_counter() < duracion * factor * MARGEN_SEGURIDAD:
            break
        profundidad += 1

    return mejor_jugada, mejor_valor, mejor_profundidad


def calentar() -> None:
    """Fuerza la compilación de numba (tarda segundos): llamar en `preparar`, fuera del reloj."""
    tablero, puntos, color = _tablero_desde_estado(Estado.inicial())
    pesos = np.asarray(ev.PESOS, dtype=np.float64)
    _mejor_jugada_numba(tablero, puntos, color, 1, pesos, ev.cota() + 100.0, -1, -1)
