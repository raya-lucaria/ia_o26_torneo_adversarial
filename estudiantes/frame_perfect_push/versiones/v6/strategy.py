"""Bot «frame_perfect_push» v6: Monte Carlo con bandido UCB1, azar exacto y respuesta del rival.

Idea:
- Cada jugada legal es un «brazo». Una tirada valora un desenlace de la jugada con
  partidas al azar del motor (`simular_al_azar`, compilado con numba).
- Azar exacto: un ataque tiene dos desenlaces (gana con p, pierde con 1 - p). Se
  visitan en proporción EXACTA a su probabilidad (muestreo estratificado, sin dado
  propio), así el promedio converge a p·V(gana) + (1 - p)·V(pierde).
- UCB1 reparte las tiradas: más simulaciones para las jugadas prometedoras.
- Táctica antes de simular: si una jugada gana segura, se toma; un desenlace que deja
  al rival ganar seguro en su siguiente jugada vale 0.
- Fase 2 (respuesta del rival): para las mejores jugadas de la fase 1, cada desenlace
  se valora con un segundo bandido donde el RIVAL elige (minimiza nuestro valor); la
  jugada vale sum(p · valor de la mejor respuesta del rival). Así no suponemos que el
  rival juega al azar en su siguiente jugada.
- Seguridad: jugada legal de respaldo; cualquier excepción la devuelve.
"""

import math
import random
import time

from motor import Estado, reportar, simular_al_azar

SIMS_POR_TIRADA = 20  # partidas al azar por tirada
TIRADAS_POR_PASO = 4  # tiradas seguidas al brazo elegido por UCB1
C_UCB = 0.5  # exploración de UCB1 (valores en [0, 1])
RESERVA_S = 3.0  # reloj que nunca gastamos (escala con reloj_s)
DIVISOR = 22  # cada jugada usa (reloj - reserva) / DIVISOR
TOPE_S = 1.5  # máximo por jugada (escala con reloj_s)
MIN_S = 0.02  # mínimo por jugada cuando queda poco reloj
FRACCION_FASE1 = 0.4  # parte del presupuesto para el bandido de la raíz
CANDIDATAS = 4  # jugadas que pasan a la fase 2

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

    def tirar(self):
        resultados = self.resultados
        if len(resultados) == 1:
            r = resultados[0]
        else:
            n = self.n + 1
            r = max(resultados, key=lambda x: x[0] * n - x[3])  # el más atrasado
        r[3] += 1
        v = r[2]
        if v is None:
            v = simular_al_azar(r[1], SIMS_POR_TIRADA, rng.getrandbits(31), para=color)
        self.n += 1
        self.w += v


class _Respuestas:
    """Bandido del rival desde un desenlace `s`: cada brazo es una respuesta suya."""

    __slots__ = ("brazos", "total", "fijo")

    def __init__(self, s, fijo):
        self.fijo = fijo
        self.brazos = [] if fijo is not None else [_Brazo(s, j) for j in s.jugadas()]
        self.total = 0
        for b in self.brazos:
            b.tirar()
            self.total += 1

    def tirar(self):
        if self.fijo is not None or not self.brazos:
            return
        log_n = math.log(self.total)
        # El rival minimiza nuestro valor: UCB sobre 1 - promedio.
        b = max(self.brazos, key=lambda b: 1.0 - b.w / b.n + C_UCB * math.sqrt(log_n / b.n))
        b.tirar()
        self.total += 1

    def valor(self):
        if self.fijo is not None:
            return self.fijo
        if not self.brazos:
            return 1.0  # el rival no tiene jugadas: ganamos
        b = max(self.brazos, key=lambda b: (b.n, -b.w / b.n))  # su respuesta más explorada
        return b.w / b.n


def _gana_seguro(estado, jugador):
    """True si `jugador` (en turno) tiene una jugada que gana con todo desenlace."""
    for j in estado.jugadas():
        if all(s.ganador() == jugador for _, s in estado.siguiente(j)):
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
    limite = inicio + presupuesto * FRACCION_FASE1

    # 3. Bandido UCB1: una tirada por brazo y luego al más prometedor.
    brazos = [_Brazo(estado, j) for j in legales]
    for b in brazos:
        b.tirar()
    total = len(brazos)
    while time.perf_counter() < limite:
        log_n = math.log(total)
        mejor = max(brazos, key=lambda b: b.w / b.n + C_UCB * math.sqrt(log_n / b.n))
        for _ in range(TIRADAS_POR_PASO):
            mejor.tirar()
        total += TIRADAS_POR_PASO

    # 4. Fase 2: respuesta del rival para las mejores candidatas.
    candidatas = sorted(brazos, key=lambda b: (b.n, b.w / b.n), reverse=True)[:CANDIDATAS]
    arboles = [[(r[0], _Respuestas(r[1], r[2])) for r in b.resultados] for b in candidatas]
    limite2 = inicio + presupuesto
    k = 0
    while time.perf_counter() < limite2:
        for _, resp in arboles[k % len(arboles)]:
            resp.tirar()
        k += 1
    valores = [sum(p * resp.valor() for p, resp in arbol) for arbol in arboles]
    i = max(range(len(candidatas)), key=lambda i: valores[i])
    mejor = candidatas[i]
    reportar(tiradas=total, fase2=k, eval=round(valores[i], 3))
    return mejor.jugada
