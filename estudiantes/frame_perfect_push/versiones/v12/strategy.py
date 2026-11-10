"""Bot «frame_perfect_push» v12: Monte Carlo con bandido UCB1, azar exacto, bloqueo de derrota inmediata y libro de aperturas.

Idea:
- Cada jugada legal es un «brazo». Una tirada valora un desenlace de la jugada con
  partidas al azar del motor (`simular_al_azar`, compilado con numba).
- Azar exacto: un ataque tiene dos desenlaces (gana con p, pierde con 1 - p). Se
  visitan en proporción EXACTA a su probabilidad (muestreo estratificado, sin dado
  propio), así el promedio converge a p·V(gana) + (1 - p)·V(pierde).
- UCB1 reparte las tiradas: más simulaciones para las jugadas prometedoras.
- Táctica antes de simular: si una jugada gana segura, se toma; un desenlace que deja
  al rival ganar seguro en su siguiente jugada vale 0.
- Amenaza en dos jugadas: si el rival tiene 2 puntos y puede llevar un peón a la fila
  anterior a su meta, se calcula EXACTAMENTE con qué probabilidad sobrevivimos a su
  siguiente jugada (capturando ese peón o ganando antes). Esa probabilidad es una cota
  superior del valor del desenlace (las partidas al azar casi nunca ven esa carrera).
- Libro de aperturas (`data/libro.json`, generado offline con
  `herramientas/generar_libro.py`): si la posición está en el libro, se juega su jugada
  al instante (si es legal) y el reloj ahorrado queda para el resto de la partida.
- Seguridad: jugada legal de respaldo; cualquier excepción la devuelve.
"""

import json
import math
import random
import time
from pathlib import Path

from motor import Estado, reportar, simular_al_azar

SIMS_POR_TIRADA = 40  # partidas al azar por tirada
SIMS_RELOJ_BAJO = 4  # por tirada cuando el presupuesto no alcanza la ronda inicial
SIMS_POR_SEGUNDO = 20_000  # estimación conservadora (máquina lenta) para esa decisión
TIRADAS_POR_PASO = 4  # tiradas seguidas al brazo elegido por UCB1
C_UCB = 0.5  # exploración de UCB1 (valores en [0, 1])
RESERVA_S = 3.0  # reloj que nunca gastamos (escala con reloj_s)
DIVISOR = 22  # cada jugada usa (reloj - reserva) / DIVISOR
TOPE_S = 1.5  # máximo por jugada (escala con reloj_s)
MIN_S = 0.02  # mínimo por jugada cuando queda poco reloj

color = "blancas"
rival = "negras"
rng = random.Random(0)
escala = 1.0  # reloj_s / 30: permite probar con reloj reducido
libro = {}  # clave de posición (str) -> jugada en texto


def preparar(config):
    global libro
    _fijar_color(config.color, config.semilla, config.reloj_s)
    libro = _cargar_libro(config.carpeta_datos)
    simular_al_azar(Estado.inicial(), 1, 0, para=color)  # calentar numba fuera del reloj


def _fijar_color(nuevo_color, semilla, reloj_s=30.0):
    global color, rival, rng, escala
    color = nuevo_color
    rival = "negras" if color == "blancas" else "blancas"
    rng = random.Random(semilla)
    escala = reloj_s / 30.0


def _cargar_libro(carpeta):
    """Libro de aperturas; si falta o está dañado, se juega sin libro."""
    try:
        with open(Path(carpeta) / "libro.json", encoding="utf-8") as f:
            return {str(k): str(v) for k, v in json.load(f)["jugadas"].items()}
    except Exception:
        return {}


def jugar(estado, tiempo_restante):
    inicio = time.perf_counter()
    legales = estado.jugadas()
    respaldo = legales[0]
    try:
        texto = libro.get(str(estado.clave))
        if texto is not None:
            for j in legales:
                if str(j) == texto:
                    reportar(libro=1)
                    return j
        return _decidir(estado, legales, tiempo_restante, inicio)
    except Exception:
        return respaldo


class _Brazo:
    """Una jugada. `resultados`: [probabilidad, estado, valor fijo o None, visitas, tope]."""

    __slots__ = ("jugada", "resultados", "n", "w")

    def __init__(self, estado, jugada):
        self.jugada = jugada
        self.resultados = []
        for p, s in estado.siguiente(jugada):
            fijo = _valor_fijo(s)
            tope = 1.0
            if fijo is None and s.turno == rival and s.puntos[rival] == 2:
                tope = _supervivencia(s)
                if tope == 0.0:
                    fijo = 0.0
            self.resultados.append([p, s, fijo, 0, tope])
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
            v = min(r[4], simular_al_azar(r[1], sims, rng.getrandbits(31), para=color))
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


def _anota_seguro(tablero, jugador):
    """True si `jugador` tiene un peón en la fila anterior a su meta con casilla libre
    adelante (recto o diagonal): si le toca, anota sin combate."""
    if jugador == "blancas":
        pieza, fila, paso = 1, 7, 9
    else:
        pieza, fila, paso = 2, 1, -9
    for col in range(9):
        c = fila * 9 + col
        if tablero[c] != pieza:
            continue
        f = c + paso
        if tablero[f] == 0 or (col > 0 and tablero[f - 1] == 0) or (col < 8 and tablero[f + 1] == 0):
            return True
    return False


def _supervivencia(s):
    """Rival en turno con 2 puntos: cota exacta de nuestra probabilidad de sobrevivir
    si avanza sin combate un peón a la fila anterior a su meta y luego anota.

    Para cada avance así del rival, buscamos nuestra mejor respuesta (con sus
    desenlaces exactos) y medimos la probabilidad de que después el rival ya no pueda
    anotar seguro. El rival elige el avance que más nos daña.
    """
    fila_dos = 2 if rival == "negras" else 6  # a dos filas de su meta
    tope = 1.0
    for m in s.jugadas():
        if m.origen // 9 != fila_dos or s.es_ataque(m):
            continue
        s1 = s.aplicar(m)
        g1 = s1.ganador()
        if g1 == rival:
            return 0.0
        if g1 is not None:
            continue
        mejor = 0.0
        for r in s1.jugadas():
            viva = 0.0
            for p, t in s1.siguiente(r):
                g = t.ganador()
                if g == color or (g is None and not _anota_seguro(t._tablero, rival)):
                    viva += p
            if viva > mejor:
                mejor = viva
                if mejor >= 1.0:
                    break
        if mejor < tope:
            tope = mejor
            if tope == 0.0:
                return 0.0
    return tope


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

    # 3. Bandido UCB1 hasta el límite.
    brazos, total = _buscar(estado, legales, presupuesto, limite)

    # 4. La jugada más simulada (desempate: mejor promedio).
    mejor = max(brazos, key=lambda b: (b.n, b.w / b.n))
    reportar(tiradas=total, eval=round(mejor.w / mejor.n, 3))
    return mejor.jugada


def _buscar(estado, legales, presupuesto, limite):
    """Bandido UCB1: una tirada por brazo y luego al más prometedor hasta `limite`."""
    brazos = [_Brazo(estado, j) for j in legales]
    # Con poco reloj, la ronda inicial (una tirada por jugada) debe ser barata.
    costo_ronda = len(legales) * SIMS_POR_TIRADA / SIMS_POR_SEGUNDO
    sims_inicial = SIMS_POR_TIRADA if presupuesto >= costo_ronda else SIMS_RELOJ_BAJO
    for b in brazos:
        b.tirar(sims_inicial)
    total = len(brazos)
    if all(r[2] is not None for b in brazos for r in b.resultados):
        return brazos, total  # todo valor es exacto: simular no cambia nada
    while time.perf_counter() < limite:
        log_n = math.log(total)
        mejor = max(brazos, key=lambda b: b.w / b.n + C_UCB * math.sqrt(log_n / b.n))
        for _ in range(TIRADAS_POR_PASO):
            mejor.tirar()
        total += TIRADAS_POR_PASO

    return brazos, total


def analizar(estado, segundos, nuevo_color, semilla):
    """Para el generador del libro (offline): mejor jugada con `segundos` de búsqueda."""
    _fijar_color(nuevo_color, semilla)
    legales = estado.jugadas()
    for j in legales:
        if all(s.ganador() == color for _, s in estado.siguiente(j)):
            return j, 1.0, 0
    brazos, total = _buscar(estado, legales, segundos, time.perf_counter() + segundos)
    mejor = max(brazos, key=lambda b: (b.n, b.w / b.n))
    return mejor.jugada, mejor.w / mejor.n, total
