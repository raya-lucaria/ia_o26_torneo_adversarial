"""Bot «frame_perfect_push» v9: Monte Carlo con bandido UCB1, azar exacto y bloqueo de derrota inmediata.

Idea:
- Cada jugada legal es un «brazo». Una tirada valora un desenlace de la jugada con
  partidas al azar del motor (`simular_al_azar`, compilado con numba).
- Azar exacto: un ataque tiene dos desenlaces (gana con p, pierde con 1 - p). Se
  visitan en proporción EXACTA a su probabilidad (muestreo estratificado, sin dado
  propio), así el promedio converge a p·V(gana) + (1 - p)·V(pierde).
- UCB1 reparte las tiradas: más simulaciones para las jugadas prometedoras.
- Táctica antes de simular: si una jugada gana segura, se toma; un desenlace que deja
  al rival ganar seguro en su siguiente jugada vale 0.
- Reloj adaptativo: se deja de pensar antes si la partida ya está decidida y se piensa
  más si las dos mejores jugadas están empatadas.
- Seguridad: jugada legal de respaldo; cualquier excepción la devuelve.
"""

import math
import random
import time

from motor import Estado, reportar, simular_al_azar

SIMS_POR_TIRADA = 40  # partidas al azar por tirada
SIMS_RELOJ_BAJO = 4  # por tirada cuando el presupuesto no alcanza la ronda inicial
SIMS_POR_SEGUNDO = 20_000  # estimación conservadora (máquina lenta) para esa decisión
TIRADAS_POR_PASO = 4  # tiradas seguidas al brazo elegido por UCB1
C_UCB = 0.5  # exploración de UCB1 (valores en [0, 1])
RESERVA_S = 2.0  # reloj que nunca gastamos (escala con reloj_s)
DIVISOR = 19  # cada jugada usa (reloj - reserva) / DIVISOR
TOPE_S = 1.8  # máximo por jugada (escala con reloj_s)
MIN_S = 0.02  # mínimo por jugada cuando queda poco reloj
DECIDIDA = 0.08  # media < 0.08 o > 0.92: la jugada ya está clara
EMPATE = 0.02  # diferencia de medias entre las dos más simuladas para pensar más
EXTENSION = 1.6  # multiplica el presupuesto cuando hay empate
RELOJ_PARA_EXTENDER_S = 6.0  # solo se extiende con reloj de sobra (escala con reloj_s)

color = "blancas"
rival = "negras"
rng = random.Random(0)
escala = 1.0  # reloj_s / 30: permite probar con reloj reducido


def preparar(config):
    global color, rival, rng, escala
    color = config.color
    rival = "negras" if color == "blancas" else "blancas"
    rng = random.Random(config.semilla)
    escala = config.reloj_s / 30.0
    simular_al_azar(Estado.inicial(), 1, 0, para=color)  # calentar numba fuera del reloj


def jugar(estado, tiempo_restante):
    inicio = time.perf_counter()
    legales = estado.jugadas()
    respaldo = legales[0]
    try:
        return _decidir(estado, legales, tiempo_restante, inicio)
    except Exception:
        return respaldo


class _Brazo:
    """Una jugada. `resultados`: [probabilidad, estado, valor fijo o None, visitas]."""

    __slots__ = ("jugada", "resultados", "n", "w")

    def __init__(self, estado, jugada):
        self.jugada = jugada
        self.resultados = [[p, s, _valor_fijo(s), 0] for p, s in estado.siguiente(jugada)]
        self.n = 0
        self.w = 0.0

    def tirar(self, sims=SIMS_POR_TIRADA):
        resultados = self.resultados
        if len(resultados) == 1:
            r = resultados[0]
        else:
            n = self.n + 1
            r = max(resultados, key=lambda x: x[0] * n - x[3])  # el más atrasado
        r[3] += 1
        v = r[2]
        if v is None:
            v = simular_al_azar(r[1], sims, rng.getrandbits(31), para=color)
        self.n += 1
        self.w += v


def _gana_seguro(estado, jugador):
    """True si `jugador` (en turno, con 2 puntos) anota el tercero sin combate.

    Barato: una sola generación de jugadas. No cubre ganar dejando al rival sin
    jugadas (raro); en ese caso se simula como cualquier otra posición.
    """
    meta = 8 if jugador == "blancas" else 0
    for j in estado.jugadas():
        if j.destino // 9 == meta and not estado.es_ataque(j):
            return True
    return False


def _valor_fijo(s):
    """Valor exacto de un desenlace si se conoce sin simular; si no, None."""
    ganador = s.ganador()
    if ganador is not None:
        return 1.0 if ganador == color else 0.0
    turno = s.turno  # quien juega en `s` y ya tiene 2 puntos puede ganar seguro
    if s.puntos[turno] == 2 and _gana_seguro(s, turno):
        return 1.0 if turno == color else 0.0
    return None


def _decidir(estado, legales, tiempo_restante, inicio):
    # 1. Victoria segura: tomarla.
    for j in legales:
        if all(s.ganador() == color for _, s in estado.siguiente(j)):
            return j
    if len(legales) == 1:
        return legales[0]

    # 2. Presupuesto de tiempo.
    disponible = tiempo_restante - RESERVA_S * escala
    presupuesto = min(TOPE_S * escala, max(MIN_S, disponible / DIVISOR))
    limite = inicio + presupuesto

    # 3. Bandido UCB1: una tirada por brazo y luego al más prometedor.
    brazos = [_Brazo(estado, j) for j in legales]
    # Con poco reloj, la ronda inicial (una tirada por jugada) debe ser barata.
    costo_ronda = len(legales) * SIMS_POR_TIRADA / SIMS_POR_SEGUNDO
    sims_inicial = SIMS_POR_TIRADA if presupuesto >= costo_ronda else SIMS_RELOJ_BAJO
    for b in brazos:
        b.tirar(sims_inicial)
    total = len(brazos)
    mitad = inicio + presupuesto / 2
    revisado_mitad = False
    extendido = False
    while True:
        ahora = time.perf_counter()
        if ahora >= mitad and not revisado_mitad:
            revisado_mitad = True
            m = _mas_simulada(brazos)
            media = m.w / m.n
            if media < DECIDIDA or media > 1.0 - DECIDIDA:
                break  # partida decidida: ahorrar reloj
        if ahora >= limite:
            if extendido or not _empate(brazos) or disponible < RELOJ_PARA_EXTENDER_S * escala:
                break
            extendido = True
            limite = inicio + presupuesto * EXTENSION
        log_n = math.log(total)
        mejor = max(brazos, key=lambda b: b.w / b.n + C_UCB * math.sqrt(log_n / b.n))
        for _ in range(TIRADAS_POR_PASO):
            mejor.tirar()
        total += TIRADAS_POR_PASO

    # 4. La jugada más simulada (desempate: mejor promedio).
    mejor = _mas_simulada(brazos)
    reportar(tiradas=total, eval=round(mejor.w / mejor.n, 3), ext=int(extendido))
    return mejor.jugada


def _mas_simulada(brazos):
    return max(brazos, key=lambda b: (b.n, b.w / b.n))


def _empate(brazos):
    """True si las dos jugadas más simuladas tienen medias casi iguales y nada está decidido."""
    if len(brazos) < 2:
        return False
    a, b = sorted(brazos, key=lambda x: x.n, reverse=True)[:2]
    ma, mb = a.w / a.n, b.w / b.n
    return abs(ma - mb) < EMPATE and DECIDIDA < ma < 1.0 - DECIDIDA
