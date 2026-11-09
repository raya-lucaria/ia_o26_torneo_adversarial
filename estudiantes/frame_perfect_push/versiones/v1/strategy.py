"""Expectiminimax conservador para Breakthrough de combate."""

import math
import time

from motor import reportar

PESOS = {
    "puntos": 650.0,
    "material": 32.0,
    "avance": 58.0,
    "apoyo": 8.0,
    "ataque": 16.0,
    "riesgo": 34.0,
    "movilidad": 1.2,
    "amenaza": 210.0,
}

VALOR_TERMINAL = 1_000_000.0
PROFUNDIDADES = (1, 2)
NODOS_MAXIMOS = 2400
REVISION_NODOS = 16
PRESUPUESTO_MAXIMO = 0.25
RESERVA_RELOJ = 1.5


class _BusquedaDetenida(Exception):
    pass


def preparar(config):
    return None


def _posicion(casilla):
    return int(casilla[1]) - 1, ord(casilla[0]) - ord("a")


def _apoyo(posicion, propias, color):
    fila, columna = posicion
    atras = fila - (1 if color == "blancas" else -1)
    if not 0 <= atras < 9:
        return 0
    return sum(
        (atras, diagonal) in propias
        for diagonal in (columna - 1, columna + 1)
        if 0 <= diagonal < 9
    )


def _probabilidad_ataque(origen, destino, propias, rivales, color):
    apoyo_atacante = _apoyo(origen, propias, color)
    rival = "negras" if color == "blancas" else "blancas"
    apoyo_defensor = _apoyo(destino, rivales, rival)
    return min(0.9, max(0.2, 0.5 + 0.15 * (apoyo_atacante - apoyo_defensor)))


def _rasgos_color(piezas, contrarias, color):
    propias = {_posicion(casilla) for casilla in piezas}
    rivales = {_posicion(casilla) for casilla in contrarias}
    direccion = 1 if color == "blancas" else -1
    meta = 8 if color == "blancas" else 0
    avance = 0.0
    apoyos = 0.0
    ataques = 0.0
    amenaza = 0.0

    for fila, columna in propias:
        progreso = max(0, fila - 1) if color == "blancas" else max(0, 7 - fila)
        avance += (progreso / 7) ** 3
        apoyos += _apoyo((fila, columna), propias, color)
        siguiente_fila = fila + direccion
        if not 0 <= siguiente_fila < 9:
            continue

        mejor_anotacion = 0.0
        recto = (siguiente_fila, columna)
        if recto not in propias and recto not in rivales and siguiente_fila == meta:
            mejor_anotacion = 1.0
        for siguiente_columna in (columna - 1, columna + 1):
            if not 0 <= siguiente_columna < 9:
                continue
            destino = (siguiente_fila, siguiente_columna)
            if destino in propias:
                continue
            if destino in rivales:
                probabilidad = _probabilidad_ataque(
                    (fila, columna), destino, propias, rivales, color
                )
                ataques += probabilidad
                if siguiente_fila == meta:
                    mejor_anotacion = max(mejor_anotacion, probabilidad)
            elif siguiente_fila == meta:
                mejor_anotacion = 1.0
        amenaza += mejor_anotacion

    color_rival = "negras" if color == "blancas" else "blancas"
    direccion_rival = 1 if color_rival == "blancas" else -1
    riesgo_por_peon = {}
    for fila, columna in propias:
        fila_atacante = fila - direccion_rival
        for columna_atacante in (columna - 1, columna + 1):
            origen = (fila_atacante, columna_atacante)
            if not (0 <= fila_atacante < 9 and 0 <= columna_atacante < 9):
                continue
            if origen in rivales:
                probabilidad = _probabilidad_ataque(
                    origen, (fila, columna), rivales, propias, color_rival
                )
                riesgo_por_peon[(fila, columna)] = max(
                    riesgo_por_peon.get((fila, columna), 0.0), probabilidad
                )

    return {
        "material": float(len(propias)),
        "avance": avance,
        "apoyo": apoyos,
        "ataque": ataques,
        "riesgo": sum(riesgo_por_peon.values()),
        "amenaza": amenaza,
    }


def _evaluar(estado, perspectiva):
    blancas = _rasgos_color(estado.piezas("blancas"), estado.piezas("negras"), "blancas")
    negras = _rasgos_color(estado.piezas("negras"), estado.piezas("blancas"), "negras")
    propio, rival = (blancas, negras) if perspectiva == "blancas" else (negras, blancas)
    otro = "negras" if perspectiva == "blancas" else "blancas"
    diferencia_puntos = estado.puntos[perspectiva] - estado.puntos[otro]
    movilidad = len(estado.jugadas()) * (1 if estado.turno == perspectiva else -1)
    return (
        PESOS["puntos"] * diferencia_puntos
        + PESOS["material"] * (propio["material"] - rival["material"])
        + PESOS["avance"] * (propio["avance"] - rival["avance"])
        + PESOS["apoyo"] * (propio["apoyo"] - rival["apoyo"])
        + PESOS["ataque"] * (propio["ataque"] - rival["ataque"])
        + PESOS["riesgo"] * (rival["riesgo"] - propio["riesgo"])
        + PESOS["movilidad"] * movilidad
        + PESOS["amenaza"] * (propio["amenaza"] - rival["amenaza"])
    )


def _revisar_recursos(contexto):
    contexto["nodos"] += 1
    if contexto["nodos"] > NODOS_MAXIMOS:
        raise _BusquedaDetenida
    if contexto["nodos"] % REVISION_NODOS == 0 and time.perf_counter() >= contexto["limite"]:
        raise _BusquedaDetenida


def _valor(estado, profundidad, contexto):
    _revisar_recursos(contexto)
    ganador = estado.ganador()
    if ganador is not None:
        return VALOR_TERMINAL if ganador == contexto["perspectiva"] else -VALOR_TERMINAL
    if profundidad == 0:
        return _evaluar(estado, contexto["perspectiva"])

    jugadas = estado.jugadas()
    if not jugadas:
        return -VALOR_TERMINAL if estado.turno == contexto["perspectiva"] else VALOR_TERMINAL

    maximiza = estado.turno == contexto["perspectiva"]
    mejor = -math.inf if maximiza else math.inf
    for jugada in jugadas:
        valor = _valor_esperado(estado, jugada, profundidad, contexto)
        mejor = max(mejor, valor) if maximiza else min(mejor, valor)
    return mejor


def _valor_esperado(estado, jugada, profundidad, contexto):
    return sum(
        probabilidad * _valor(hijo, profundidad - 1, contexto)
        for probabilidad, hijo in estado.siguiente(jugada)
    )


def _presupuesto(tiempo_restante, cantidad_piezas):
    try:
        restante = float(tiempo_restante)
    except (TypeError, ValueError):
        restante = 3.0
    if not math.isfinite(restante):
        restante = 3.0
    utilizable = max(0.0, restante - RESERVA_RELOJ)
    turnos_estimados = max(10, min(28, cantidad_piezas))
    return min(PRESUPUESTO_MAXIMO, utilizable * 0.5 / turnos_estimados)


def _buscar(estado, legales, tiempo_restante):
    perspectiva = estado.turno
    mejor_completa = legales[0]
    mejor_valor = -math.inf
    limite = time.perf_counter() + _presupuesto(
        tiempo_restante, len(estado.piezas("blancas")) + len(estado.piezas("negras"))
    )
    nodos = 0
    profundidad_completa = 0

    for jugada in legales:
        if all(hijo.ganador() == perspectiva for _, hijo in estado.siguiente(jugada)):
            return jugada, VALOR_TERMINAL, nodos, 0

    for profundidad in PROFUNDIDADES:
        contexto = {"limite": limite, "nodos": nodos, "perspectiva": perspectiva}
        mejor_iteracion = legales[0]
        valor_iteracion = -math.inf
        try:
            for jugada in legales:
                valor = _valor_esperado(estado, jugada, profundidad, contexto)
                if valor > valor_iteracion:
                    mejor_iteracion, valor_iteracion = jugada, valor
        except _BusquedaDetenida:
            nodos = contexto["nodos"]
            break
        mejor_completa, mejor_valor = mejor_iteracion, valor_iteracion
        nodos = contexto["nodos"]
        profundidad_completa = profundidad
        if time.perf_counter() >= limite or nodos >= NODOS_MAXIMOS:
            break

    return mejor_completa, mejor_valor, nodos, profundidad_completa


def jugar(estado, tiempo_restante):
    legales = estado.jugadas()
    if not legales:
        return None
    respaldo = legales[0]
    try:
        mejor, valor, nodos, profundidad = _buscar(estado, legales, tiempo_restante)
        reportar(nodos=nodos, prof=profundidad, eval=valor)
        return mejor if mejor in legales else respaldo
    except Exception:
        return respaldo