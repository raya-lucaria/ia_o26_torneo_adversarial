"""SPRT para comparar dos versiones del bot sin engañarnos con el ruido de los dados.

Las partidas de la arena van en pares (misma semilla, colores invertidos), así que la unidad es
el par: su puntaje es 0, 0.5 o 1 (ninguna, una o las dos ganadas). Los dos resultados de un par
están correlacionados; usar el par como observación lo toma en cuenta.

Prueba secuencial de razón de verosimilitud (Wald) con la aproximación normal que usa Fishtest
(«GSPRT»): con puntaje medio esperado s0 bajo H0 y s1 bajo H1, y la varianza σ² medida de los
puntajes de los pares,

    LLR = N · (s1 − s0) · (2·x̄ − s0 − s1) / (2·σ²).

Se juega hasta que LLR cruza una cota: ≥ ln((1 − β)/α) acepta H1 (la nueva es mejor por al
menos `elo1`), ≤ ln(β/(1 − α)) acepta H0 (no es mejor que `elo0`). Este juego no tiene empates.
"""

import math

VARIANZA_MINIMA = 1e-4  # evita dividir entre cero si todos los pares salen iguales
# Con pocos pares la varianza medida no es confiable (dos pares iguales dan varianza 0 y un LLR
# enorme): antes de MIN_PARES el LLR vale 0 y no se decide nada.
MIN_PARES = 30


def puntaje_esperado(elo: float) -> float:
    """Puntaje medio que corresponde a una diferencia de Elo."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo_de(puntaje: float) -> float:
    """Diferencia de Elo que corresponde a un puntaje medio (±inf en 0 y 1)."""
    if puntaje <= 0.0:
        return -math.inf
    if puntaje >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / puntaje - 1.0)


def media_y_varianza(pares: list[float]) -> tuple[float, float]:
    n = len(pares)
    media = sum(pares) / n
    return media, sum((x - media) ** 2 for x in pares) / n


def llr(pares: list[float], elo0: float = 0.0, elo1: float = 35.0) -> float:
    """Logaritmo de la razón de verosimilitud de H1 contra H0, con los puntajes de los pares."""
    if len(pares) < MIN_PARES:
        return 0.0
    s0, s1 = puntaje_esperado(elo0), puntaje_esperado(elo1)
    media, varianza = media_y_varianza(pares)
    return len(pares) * (s1 - s0) * (2 * media - s0 - s1) / (2 * max(varianza, VARIANZA_MINIMA))


def cotas(alfa: float = 0.1, beta: float = 0.1) -> tuple[float, float]:
    """(inferior, superior) del LLR: abajo se acepta H0, arriba H1."""
    return math.log(beta / (1 - alfa)), math.log((1 - beta) / alfa)


def decision(valor: float, alfa: float = 0.1, beta: float = 0.1) -> str:
    """'H1' (es mejor), 'H0' (no es mejor) o '' (todavía no se sabe)."""
    inferior, superior = cotas(alfa, beta)
    if valor >= superior:
        return "H1"
    if valor <= inferior:
        return "H0"
    return ""


def intervalo_elo(pares: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """(Elo, mínimo, máximo) con intervalo de confianza del 95 % sobre los pares."""
    media, varianza = media_y_varianza(pares)
    error = z * math.sqrt(varianza / len(pares))
    return elo_de(media), elo_de(media - error), elo_de(media + error)
