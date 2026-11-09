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

# Solo se guardan valores EXACTOS: el ciclo termina sin corte propio y el
# resultado queda dentro de la ventana alfa/beta de entrada. Fuera de ella,
# aunque termine el ciclo local, los hijos pueden haber devuelto cotas. No
# se reutilizan esas cotas como valores exactos en una ventana diferente.
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

import eval3 as e3

PROFUNDIDAD_MAXIMA = 12
MARGEN_SEGURIDAD = 2.0
FACTOR_INICIAL = 20.0
# Piso pesimista del factor de crecimiento entre profundidades: medido en partidas reales,
# un salto de profundidad puede costar ~123x (no los ~15-30x "típicos") cuando una posición
# se vuelve táctica de golpe (muchos ataques simultáneos = la poda Star1 corta mucho menos).
# Confiar solo en la ÚLTIMA transición medida nos hizo perder una partida por reloj
# (profundidad 5→6 pasó de 0.08s a 9.35s). El piso evita repetirlo: nunca asumimos que el
# próximo salto será más barato que esto, sin importar qué tan barata haya sido la historia.
FACTOR_PESIMISTA = 150.0  # ya no se usa: lo reemplazó el tope de nodos (ver `elegir`)
# Búsqueda anytime (INFORME.md 8i). `FRACCION_TOPE`: qué parte del tiempo que queda se
# apuesta en la siguiente profundidad. No es 1.0 porque los nodos por segundo de la
# profundidad anterior son una estimación: si la posición se vuelve táctica, los nodos se
# vuelven más caros y el tiempo real se pasa. Con 0.8 queda margen y, si aun así aborta, no
# se pierde la jugada (se juega la de la última profundidad completa).
FRACCION_TOPE = 0.8
TOPE_MINIMO = 20_000  # nunca abortamos antes de esto: por debajo, el overhead domina
TOPE_PRIMERA_PROFUNDIDAD = 2_000_000  # profundidad 1 es barata; tope alto = nunca aborta
ULTIMA_BUSQUEDA = {"nodos": 0, "abortos": 0, "parciales": 0}  # diagnóstico de `elegir`
# Pesos de eval3 (259). Viven aquí y no en `eval.py` porque eval3 no comparte vector con la
# evaluación anterior; `strategy.preparar` los carga con `cargar_pesos`. El default es cero
# salvo la diferencia de puntos: con todo en cero la evaluación sería constante y la búsqueda
# elegiría la primera jugada legal, lo que parece "funcionar" y no juega a nada.
PESOS = e3.empaquetar(np.zeros(e3.N_FEATURES, np.float64), activas=())
PESOS[e3.I_PUNTOS] = 1.0   # `puntos` y `filas` no tienen bandera: se calculan siempre
GANANCIA = 1.0  # utilidad de victoria; las hojas son 2*sigmoid(logit)-1
TT_BITS = 20
TT_TAM = 1 << TT_BITS  # potencia de 2: el índice es un AND, no un módulo
TT_MASCARA = np.uint64(TT_TAM - 1)
TT_VACIA = np.uint64(0)  # una posición real con llave exactamente 0 es, en la práctica, imposible
QUIESCENCIA_MAX = 0  # medida con A/B real: no compensa (ver nota abajo). Apagada por default.
# Fase 2 medida con A/B real y rechazada: dejar la implementación disponible para
# futuras pruebas, pero el campeón corre sin este cambio.
ORDENAMIENTO_JUGADAS = True  # F_GOOGLE


CAL_B0 = 0.0      # F_CAL: calibracion ajustada con partidas oficiales vs montecarlo
CAL_STM = 0.26912704441311125


@njit(cache=True)
def _evaluar_numba(tablero, puntos, color, pesos, mueve_objetivo):
    """Utilidad esperada en las unidades de los terminales (F_CAL: pesos reajustados con
    partidas oficiales + termino de turno; ver INFORME). tanh(logit/2) = 2*P(victoria)-1."""
    logit = e3.evaluar_nb(tablero, puntos, color, pesos) + CAL_B0
    if mueve_objetivo:
        logit += CAL_STM
    else:
        logit -= CAL_STM
    return np.tanh(0.5 * logit)


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
    """F_GOOGLE: orden BARATO literal de la receta (capturas primero, luego el avance de los
    peones mas cercanos a la meta). Sin copiar el tablero ni generar jugadas rivales: una
    clave por jugada y una insercion. La version anterior (rechazada) costaba mas de lo que
    podaba. Nunca cambia el valor minimax (solo el orden de exploracion)."""
    fila_meta = LADO - 1 if color == 0 else 0
    claves = np.empty(MAX_JUGADAS, np.float64)
    for i in range(n):
        d = destinos[i]
        fila = d // LADO
        avance = fila if color == 0 else LADO - 1 - fila
        k = 1.0 * avance
        if fila == fila_meta:
            k += 1_000_000.0
        if tablero[d] != VACIA:
            k += 1000.0 + 100.0 * prob_jugada(tablero, color, origenes[i], d)
        if origenes[i] == origen_sugerido and d == destino_sugerido:
            k += 10_000_000.0
        claves[i] = k
    for i in range(1, n):
        k = claves[i]
        o = origenes[i]
        d = destinos[i]
        j = i - 1
        while j >= 0 and claves[j] < k:
            claves[j + 1] = claves[j]
            origenes[j + 1] = origenes[j]
            destinos[j + 1] = destinos[j]
            j -= 1
        claves[j + 1] = k
        origenes[j + 1] = o
        destinos[j + 1] = d


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
@njit(cache=True)
def _quiescencia_numba(tablero, puntos, color, objetivo, alfa, beta, pesos, ganancia, restante,
                        origenes, destinos, contador, tope):
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

    estatico = _evaluar_numba(tablero, puntos, objetivo, pesos, color == objetivo)
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
            restante - 1, o2, d2, contador, tope,
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
                restante - 1, o3, d3, contador, tope,
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


# Una sola función auto-recursiva (no dos que se llaman entre sí): numba infiere la
# especialización al primer uso desde `preparar()` (después de copiar el caché), evitando
# compilarla al importar el módulo. La lógica de cada jugada (una o dos ramas de azar, con la poda Star1
# que acota la rama que no hace falta ver) va en línea, dentro del ciclo por jugada.
@njit(cache=True)
def _valor_numba(tablero, puntos, color, objetivo, profundidad, alfa, beta, pesos, ganancia,
                  origenes, destinos, tt_claves, tt_valores, contador, tope):
    """`contador` es un arreglo de 2: `contador[0]` nodos visitados, `contador[1]` bandera
    de aborto. Al pasar `tope` nodos se marca la bandera y TODOS los niveles regresan de
    inmediato; el valor que devuelven en ese caso es basura y `elegir` descarta la
    profundidad entera (ver ahí). `tope <= 0` significa sin tope."""
    contador[0] += 1
    if contador[1] != 0:
        return 0.0  # ya se abortó más abajo: salimos sin tocar nada
    if tope > 0 and contador[0] > tope:
        contador[1] = 1
        return 0.0
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
            origenes, destinos, contador, tope,
        )

    alfa_entrada, beta_entrada = alfa, beta
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
                tt_claves, tt_valores, contador, tope,
            )
        else:
            p1 = prob_jugada(tablero, color, origen, destino)
            p2w = 1.0 - p1
            t_gana = tablero.copy()
            pt_gana = puntos.copy()
            aplicar(t_gana, pt_gana, color, origen, destino, True)
            v1 = _valor_numba(
                t_gana, pt_gana, 1 - color, objetivo, profundidad - 1, -ganancia, ganancia,
                pesos, ganancia, o2, d2, tt_claves, tt_valores, contador, tope,
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
                    ganancia, pesos, ganancia, o3, d3, tt_claves, tt_valores, contador, tope,
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
        if contador[1] != 0:
            # Abortado en un subárbol: `mejor` está a medio calcular. No es una cota, es
            # basura — salimos sin escribir la TT (la profundidad entera se descarta).
            return 0.0
        if alfa >= beta:
            exacto = False  # quedaron jugadas sin revisar: esto es una cota, no el valor real
            break

    # Acabar el ciclo sin un corte propio no basta: si la búsqueda falla por
    # debajo/encima de la ventana de entrada, los hijos pueden haber devuelto
    # cotas. Solo el resultado interior a esa ventana se guarda como exacto.
    if (exacto and alfa_entrada < mejor < beta_entrada
            and llave != TT_VACIA and contador[1] == 0):
        tt_claves[idx] = llave
        tt_valores[idx] = mejor
    return mejor


@njit(cache=True)
def _mejor_jugada_numba(tablero, puntos, color, profundidad, pesos, ganancia, origen_sugerido,
                         destino_sugerido, contador, tope):
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
    completadas = 0  # jugadas de la raíz cuyo subárbol terminó (no se abortó a media)
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
                d2, tt_claves, tt_valores, contador, tope,
            )
        else:
            p1 = prob_jugada(tablero, color, origen, destino)
            p2w = 1.0 - p1
            t_gana = tablero.copy()
            pt_gana = puntos.copy()
            aplicar(t_gana, pt_gana, color, origen, destino, True)
            v1 = _valor_numba(
                t_gana, pt_gana, 1 - color, color, profundidad - 1, -ganancia, ganancia, pesos,
                ganancia, o2, d2, tt_claves, tt_valores, contador, tope,
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
                    pesos, ganancia, o3, d3, tt_claves, tt_valores, contador, tope,
                )
                v = p1 * v1 + p2w * v2
        if contador[1] != 0:
            # Abortado DENTRO de esta jugada de la raíz: su `v` está a medias y se
            # descarta. Pero las `completadas` anteriores sí tienen valor válido, y
            # `mejor_i`/`mejor_valor` ya apuntan a la mejor de ellas: se devuelven junto
            # con la cuenta, y `elegir` decide si las aprovecha (ver INFORME.md 8j).
            return origenes[mejor_i], destinos[mejor_i], mejor_valor, 1, completadas
        completadas += 1
        if v > mejor_valor:
            mejor_valor = v
            mejor_i = i
            alfa = mejor_valor
    return origenes[mejor_i], destinos[mejor_i], mejor_valor, 0, completadas


def _tablero_desde_estado(estado: Estado) -> tuple[np.ndarray, np.ndarray, int]:
    tablero = np.zeros(CASILLAS, np.int8)
    for t in estado.piezas("blancas"):
        tablero[casilla_de_texto(t)] = 1
    for t in estado.piezas("negras"):
        tablero[casilla_de_texto(t)] = 2
    puntos = np.array([estado.puntos["blancas"], estado.puntos["negras"]], np.int64)
    color = 0 if estado.turno == "blancas" else 1
    return tablero, puntos, color



def cota_eval3(W) -> float:
    """Cota del LOGIT crudo de eval3; no es la ganancia terminal de este prototipo.

    Se conserva para diagnósticos de la escala anterior. En este prototipo Star1
    usa el intervalo de utilidad [-1, 1], no esta cota de logits. Para las tablas (patrones y
    filas) se usa `max|W|` por el total de peones, no la suma de los |W| de las 243
    entradas: el histograma reparte como mucho PEONES peones propios y PEONES rivales
    entre las entradas, así que `max|W| * 2 * PEONES` es una cota válida y mucho más
    ajustada que sumar todo.
    """
    from motor.nucleo import PEONES, PUNTOS_PARA_GANAR
    W = np.asarray(W, dtype=np.float64)
    pat = np.abs(W[e3.I_PATRONES:e3.I_FILAS]).max(initial=0.0)
    fil = np.abs(W[e3.I_FILAS:e3.I_AMENAZA]).max(initial=0.0)
    total = abs(W[e3.I_PUNTOS]) * PUNTOS_PARA_GANAR
    total += (pat + fil) * 2.0 * PEONES
    total += (abs(W[e3.I_AMENAZA]) + abs(W[e3.I_AMENAZA + 1])) * PEONES  # prob <= 1 por peón
    total += abs(W[e3.I_AMENAZA + 2]) * 2.0 * PEONES         # apoyo <= 2 por peón
    total += abs(W[e3.I_LIBRES]) * 2.0 * PEONES
    total += abs(W[e3.I_LIBRES + 1]) * 2.0 * PEONES * LADO   # distancia ajustada acotada
    total += abs(W[e3.I_MOVILIDAD]) * float(MAX_JUGADAS)
    # Familia `extra`: `pasado` suma a lo más PEONES peones por (LADO-1) de avance en cada
    # bando; `vacias` y `carrera` viven en [-LADO, LADO] por construcción.
    total += abs(W[e3.I_EXTRA]) * 2.0 * PEONES * (LADO - 1)
    total += abs(W[e3.I_EXTRA + 1]) * float(LADO)
    total += abs(W[e3.I_EXTRA + 2]) * float(LADO)
    # `material`: diferencia de peones vivos, en [-PEONES, PEONES].
    total += abs(W[e3.I_MATERIAL]) * float(PEONES)
    # `marcador`: construcción antisimétrica, cada celda contribuye como mucho ±1 por peso.
    total += np.abs(W[e3.I_MARCADOR:e3.I_MARCADOR + 3]).sum()
    # `exposicion`: p_ataque·(1−p_recaptura) por peón, en [0,1]; propio menos rival, PEONES
    # de cada lado como mucho.
    total += abs(W[e3.I_EXPOSICION]) * float(PEONES)
    return float(total)


def cargar_pesos(W) -> None:
    """Fija los pesos logísticos de eval3 y la utilidad terminal de victoria.

    Las hojas pasan por tanh y están en [-1, 1]; los terminales son +/-1.
    Rechaza un vector del largo equivocado: pasar el `pesos.npy` de
    7 de la evaluación anterior a un kernel que indexa 259 es lectura fuera de rango en
    numba, sin aviso."""
    global PESOS, GANANCIA
    W = np.ascontiguousarray(W, dtype=np.float64)
    if W.shape == (e3.N_PESOS_V1,):
        # Un archivo de la VERSIÓN 1 (el que ya está en eval3_listo.flag). Se traduce al
        # formato actual con las familias nuevas apagadas: el comportamiento es idéntico al
        # de antes de que este archivo creciera (ver eval3.desde_v1 y test_compat_v1.py).
        W = e3.desde_v1(W)
    elif W.shape != (e3.N_PESOS,):
        raise ValueError(f"eval3 espera {e3.N_PESOS} pesos empaquetados (o "
                         f"{e3.N_PESOS_V1} de la versión 1), llegaron {W.shape}")
    PESOS = W
    GANANCIA = 1.0


def _pesos_y_ganancia():
    """Pesos logísticos y utilidad terminal en unidades de victoria."""
    return PESOS, GANANCIA


def elegir(estado: Estado, color: str, limite_tiempo: float) -> tuple[Jugada, float, int]:
    """Misma interfaz que `busqueda.elegir`, pero ANYTIME: abre la siguiente profundidad sin
    predecir lo que va a costar y, si se pasa del tope de nodos, la abandona y se queda con
    la jugada de la última profundidad COMPLETA (ver INFORME.md 8i).

    El bot anterior predecía: solo abría la profundidad siguiente si le alcanzaba para
    `duracion × 300` (`FACTOR_PESIMISTA × MARGEN_SEGURIDAD`). Medido, eso lo dejaba usando
    ~1% de su presupuesto y jugando a profundidad 3 (INFORME.md 8h). Aquí el tope se
    calibra con los nodos por segundo REALES de la última profundidad completa y el tiempo
    que de verdad queda, así que no hay constante que adivine nada."""
    tablero, puntos, color_idx = _tablero_desde_estado(estado)
    pesos, ganancia = _pesos_y_ganancia()

    jugadas = estado.jugadas()
    mejor_jugada, mejor_valor, mejor_profundidad = jugadas[0], 0.0, 0
    nodos_por_segundo = 0.0
    profundidad = 1
    origen_sugerido, destino_sugerido = -1, -1  # nada todavía: es la primera profundidad
    nodos_totales = 0
    abortos = 0
    parciales = 0  # profundidades abortadas que aun así aportaron una jugada

    while profundidad <= PROFUNDIDAD_MAXIMA:
        restante = limite_tiempo - time.perf_counter()
        if restante <= 0.0:
            break
        # Tope de nodos para ESTA profundidad: lo que caben en el tiempo que queda, según
        # la velocidad medida recién. La primera profundidad no tiene medición todavía, y
        # es la barata: va con un tope fijo generoso (y es la que garantiza que siempre
        # haya una jugada completa de respaldo).
        if nodos_por_segundo > 0.0:
            tope = int(nodos_por_segundo * restante * FRACCION_TOPE)
            if tope < TOPE_MINIMO:
                tope = TOPE_MINIMO
        else:
            tope = TOPE_PRIMERA_PROFUNDIDAD
        contador = np.zeros(2, np.int64)

        inicio = time.perf_counter()
        o, d, valor, abortado, completadas = _mejor_jugada_numba(
            tablero, puntos, color_idx, profundidad, pesos, ganancia, origen_sugerido,
            destino_sugerido, contador, tope,
        )
        duracion = time.perf_counter() - inicio
        nodos_totales += int(contador[0])

        if abortado:
            abortos += 1
            # Profundidad incompleta. La jugada que se quedó a media se descarta siempre,
            # pero las que SÍ terminaron su subárbol tienen valor válido: la raíz prueba
            # primero la mejor de la profundidad anterior, así que con 2 o más terminadas
            # estamos eligiendo entre esa y al menos una alternativa, ya a esta
            # profundidad. Con una sola terminada no hay con qué comparar (sería la misma
            # jugada de siempre), así que se conserva la última profundidad completa.
            if completadas >= 2:
                mejor_jugada = Jugada(int(o), int(d))
                mejor_valor = valor
                parciales += 1
                # `mejor_profundidad` NO se actualiza: esta profundidad quedó incompleta y
                # reportarla como alcanzada sería mentirle al diagnóstico.
            break

        mejor_jugada = Jugada(int(o), int(d))
        mejor_valor, mejor_profundidad = valor, profundidad
        origen_sugerido, destino_sugerido = int(o), int(d)
        if duracion > 0.0:
            nodos_por_segundo = contador[0] / duracion
        profundidad += 1

    # Diagnóstico de la última decisión, para que `strategy.jugar` lo reporte junto con
    # `prof`/`eval` (así se ve en `-v` y en el visor). `abortos` tiene que ser 0 casi
    # siempre: si sube, el tope de nodos está mal calibrado.
    ULTIMA_BUSQUEDA["nodos"] = nodos_totales
    ULTIMA_BUSQUEDA["abortos"] = abortos
    ULTIMA_BUSQUEDA["parciales"] = parciales
    return mejor_jugada, mejor_valor, mejor_profundidad


def calentar() -> None:
    """Fuerza la compilación de numba (tarda segundos): llamar en `preparar`, fuera del reloj."""
    tablero, puntos, color = _tablero_desde_estado(Estado.inicial())
    pesos, ganancia = _pesos_y_ganancia()
    _mejor_jugada_numba(tablero, puntos, color, 1, pesos, ganancia, -1, -1,
                        np.zeros(2, np.int64), TOPE_PRIMERA_PROFUNDIDAD)  # 5 valores
