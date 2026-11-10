"""Reparto del reloj: cuánto pensar en cada jugada (Python puro).

Tenemos 30 s para toda la partida, y el reloj también corre mientras el árbitro nos pasa la
jugada (unos milisegundos por jugada con Docker). Por eso:
- nunca tocamos una reserva fija de 3.5 s (así una partida larga no pasa de 27 s, el 90 % del
  reloj, donde el torneo ya avisa con ⚠);
- descontamos un costo fijo de 15 ms por cada jugada que falta (el p90 medido con Docker);
- repartimos el resto entre las jugadas que estimamos que faltan. No suponemos «40 jugadas»:
  la estimación depende de los peones que nos quedan y de los puntos ya anotados.

`blando` es lo que queremos pensar; `duro` es el corte que nunca se pasa (3 veces `blando`).
`ESCALA_TIEMPO` multiplica `blando`: la estimación de jugadas es conservadora y las partidas
terminaban con 4–8 s sin usar además de la reserva.
"""

RESERVA_S = 3.5
COSTO_FIJO_S = 0.015
ESCALA_TIEMPO = 1.0


def jugadas_restantes(peones_propios: int, puntos_totales: int) -> int:
    """Cuántas jugadas nuestras creemos que faltan (entre 8 y 70)."""
    return min(max(int(1.5 * peones_propios) + 6 - 2 * puntos_totales, 8), 70)


def presupuesto(
    tiempo_restante: float, peones_propios: int, puntos_totales: int, escala: float = ESCALA_TIEMPO
) -> tuple[float, float]:
    """(blando, duro) en segundos para esta jugada. (0, 0) si ya solo queda la reserva."""
    n = jugadas_restantes(peones_propios, puntos_totales)
    util = tiempo_restante - RESERVA_S - COSTO_FIJO_S * n
    if util <= 0:
        return 0.0, 0.0
    blando = min(escala * util / n, util)
    duro = min(3 * blando, tiempo_restante - RESERVA_S)
    return blando, duro
