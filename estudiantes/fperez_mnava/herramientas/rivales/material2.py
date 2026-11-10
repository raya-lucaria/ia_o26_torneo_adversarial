"""Rival de prueba «material2»: expectimax a 2 capas con evaluación de material y avance.

Representa al equipo típico: búsqueda corta y evaluación «peones + puntos». Si queda poco
reloj baja a 1 capa para no perder por tiempo.
"""


def preparar(config):
    pass


def _avance(estado, color):
    filas = (int(c[1]) for c in estado.piezas(color))
    return sum(f - 1 for f in filas) if color == "blancas" else sum(9 - f for f in filas)


def _evaluar(estado, yo):
    otro = "negras" if yo == "blancas" else "blancas"
    pts = estado.puntos
    material = len(estado.piezas(yo)) - len(estado.piezas(otro))
    return 10 * (pts[yo] - pts[otro]) + material + 0.05 * (_avance(estado, yo) - _avance(estado, otro))


def _valor(estado, prof, yo):
    g = estado.ganador()
    if g is not None:
        return 1000 if g == yo else -1000
    if prof == 0:
        return _evaluar(estado, yo)
    valores = [
        sum(p * _valor(hijo, prof - 1, yo) for p, hijo in estado.siguiente(j))
        for j in estado.jugadas()
    ]
    return max(valores) if estado.turno == yo else min(valores)


def jugar(estado, tiempo_restante):
    yo = estado.turno
    prof = 2 if tiempo_restante > 3.0 else 1
    jugadas = estado.jugadas()
    valores = [
        sum(p * _valor(hijo, prof - 1, yo) for p, hijo in estado.siguiente(j)) for j in jugadas
    ]
    return jugadas[valores.index(max(valores))]
