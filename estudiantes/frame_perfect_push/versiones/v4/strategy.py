"""Bot «frame_perfect_push» v3: MCTS (UCT) con nodos de azar explícitos.

Idea:
- Árbol de búsqueda Monte Carlo. Cada nodo es una posición; cada arista, una jugada.
- Un ataque tiene dos desenlaces (gana con probabilidad p, pierde con 1 - p). Los
  dos estados salen de `estado.siguiente(j)` y se visitan en proporción EXACTA a su
  probabilidad (muestreo estratificado, sin dado propio): el valor de la arista
  converge al valor esperado p·V(gana) + (1 - p)·V(pierde).
- Las hojas nuevas se valoran con partidas al azar del motor (`simular_al_azar`,
  compilado con numba).
- El árbol se reutiliza entre turnos de la misma partida.
- Seguridad: siempre hay una jugada legal de respaldo; cualquier excepción la devuelve.
"""

import gc
import math
import random
import time

from motor import Estado, reportar, simular_al_azar

import simulacion

C_UCT = 0.7  # exploración de UCB1 (valores en [0, 1])
C_RAIZ = 0.5  # exploración del bandido de la raíz
SIMS_POR_HOJA = 8  # partidas al azar por hoja nueva
RESERVA_S = 3.0  # reloj que nunca gastamos
DIVISOR = 22  # cada jugada usa (reloj - reserva) / DIVISOR
TOPE_S = 1.5  # máximo por jugada
MIN_S = 0.02  # mínimo por jugada cuando queda poco reloj
MAX_NODOS = 250_000  # cota de memoria del árbol
POLITICA_INFORMADA = True  # simulacion.py en vez de simular_al_azar

color = "blancas"
rng = random.Random(0)
_raiz = None  # raíz del árbol del turno anterior
_elegida = None  # arista que jugamos en el turno anterior
_nodos = 0
escala = 1.0  # reloj_s / 30: permite probar con reloj reducido


class _Nodo:
    """Posición del árbol. `n` = visitas; `hijos` = aristas ya expandidas."""

    __slots__ = ("estado", "turno", "ganador", "pendientes", "hijos", "n")

    def __init__(self, estado):
        global _nodos
        _nodos += 1
        self.estado = estado
        self.turno = estado.turno
        puntos = estado._puntos
        if puntos[0] >= 3:
            self.ganador = "blancas"
        elif puntos[1] >= 3:
            self.ganador = "negras"
        else:
            self.ganador = None  # «sin jugadas» se detecta al expandir
        self.pendientes = None  # se generan al visitar el nodo por segunda vez
        self.hijos = []
        self.n = 0


class _Arista:
    """Jugada desde un nodo. `w` acumula el valor para quien la juega."""

    __slots__ = ("jugada", "resultados", "n", "w")

    def __init__(self, estado, jugada):
        # Cada resultado: [probabilidad, estado, nodo (perezoso), visitas]
        self.jugada = jugada
        self.resultados = [[p, s, None, 0] for p, s in estado.siguiente(jugada)]
        self.n = 0
        self.w = 0.0


def _ordenar(estado):
    """Jugadas sin expandir; se expanden desde el final (prioridad al final)."""
    meta = 8 if estado.turno == "blancas" else 0
    tablero = estado._tablero.tobytes()
    jugadas = estado.jugadas()
    rng.shuffle(jugadas)
    jugadas.sort(key=lambda j: 2 if j.destino // 9 == meta else (1 if tablero[j.destino] else 0))
    return jugadas


def _resultado(arista):
    """Desenlace a visitar: el más atrasado respecto a su probabilidad exacta."""
    resultados = arista.resultados
    if len(resultados) == 1:
        r = resultados[0]
    else:
        total = arista.n + 1
        r = max(resultados, key=lambda x: x[0] * total - x[3])
    r[3] += 1
    if r[2] is None:
        r[2] = _Nodo(r[1])
    return r[2]


def _seleccionar(nodo):
    log_n = math.log(nodo.n)
    mejor = None
    mejor_valor = -1.0
    for a in nodo.hijos:
        valor = a.w / a.n + C_UCT * math.sqrt(log_n / a.n)
        if valor > mejor_valor:
            mejor, mejor_valor = a, valor
    return mejor


def _valor_blancas(nodo):
    """Fracción de victorias de Blancas desde `nodo` (exacta si terminó)."""
    if nodo.ganador is not None:
        return 1.0 if nodo.ganador == "blancas" else 0.0
    estado = nodo.estado
    if POLITICA_INFORMADA:
        turno = 0 if nodo.turno == "blancas" else 1
        return simulacion.valorar(
            estado._tablero, estado._puntos, turno, SIMS_POR_HOJA, rng.getrandbits(31), 0
        )
    return simular_al_azar(estado, SIMS_POR_HOJA, rng.getrandbits(31), para="blancas")


def _iterar(raiz):
    nodos = [raiz]
    aristas = []
    nodo = raiz
    while nodo.ganador is None:
        if nodo.pendientes is None:
            nodo.pendientes = _ordenar(nodo.estado)
            if not nodo.pendientes and not nodo.hijos:
                nodo.ganador = "negras" if nodo.turno == "blancas" else "blancas"
                break
        if nodo.pendientes and _nodos < MAX_NODOS:
            arista = _Arista(nodo.estado, nodo.pendientes.pop())
            nodo.hijos.append(arista)
            aristas.append(arista)
            nodo = _resultado(arista)
            nodos.append(nodo)
            break
        if not nodo.hijos:
            break  # sin memoria para expandir: valorar aquí
        arista = _seleccionar(nodo)
        aristas.append(arista)
        nodo = _resultado(arista)
        nodos.append(nodo)
    v = _valor_blancas(nodo)
    for n in nodos:
        n.n += 1
    for arista, padre in zip(aristas, nodos):
        arista.n += 1
        arista.w += v if padre.turno == "blancas" else 1.0 - v


def _reusar(estado):
    """Busca la posición actual dos niveles abajo de la jugada anterior."""
    if _elegida is None:
        return None
    clave = estado.clave
    for r in _elegida.resultados:
        nodo = r[2]
        if nodo is None:
            continue
        for a in nodo.hijos:
            for r2 in a.resultados:
                if r2[2] is not None and r2[2].estado.clave == clave:
                    return r2[2]
    return None


def _contar(nodo):
    total = 1
    pila = [nodo]
    while pila:
        actual = pila.pop()
        for a in actual.hijos:
            for r in a.resultados:
                if r[2] is not None:
                    total += 1
                    pila.append(r[2])
    return total


def preparar(config):
    global color, rng, _raiz, _elegida, _nodos, escala
    color = config.color
    escala = config.reloj_s / 30.0
    rng = random.Random(config.semilla)
    _raiz = _elegida = None
    _nodos = 0
    gc.disable()  # el árbol no tiene ciclos: la memoria se libera por conteo de referencias
    simular_al_azar(Estado.inicial(), 1, 0, para=color)  # calentar numba fuera del reloj
    simulacion.calentar()


def jugar(estado, tiempo_restante):
    inicio = time.perf_counter()
    legales = estado.jugadas()
    respaldo = legales[0]
    try:
        return _decidir(estado, legales, tiempo_restante, inicio)
    except Exception:
        return respaldo


def _decidir(estado, legales, tiempo_restante, inicio):
    # 1. Victoria segura (todo desenlace gana): tomarla.
    for j in legales:
        if all(s.ganador() == color for _, s in estado.siguiente(j)):
            return j
    if len(legales) == 1:
        return legales[0]

    # 2. Presupuesto de tiempo.
    disponible = tiempo_restante - RESERVA_S * escala
    presupuesto = min(TOPE_S * escala, max(MIN_S, disponible / DIVISOR))
    limite = inicio + presupuesto

    # 3. Bandido UCB1 en la raíz: cada jugada es un brazo; cada tirada valora un
    #    desenlace (estratificado por su probabilidad exacta) con simulaciones.
    yo = 0 if color == "blancas" else 1
    brazos = [_Arista(estado, j) for j in legales]
    total = 0
    for a in brazos:  # una tirada por brazo
        _tirar(a, yo)
        total += 1
    while time.perf_counter() < limite:
        log_n = math.log(total)
        mejor = None
        mejor_valor = -1.0
        for a in brazos:
            v = a.w / a.n + C_RAIZ * math.sqrt(log_n / a.n)
            if v > mejor_valor:
                mejor, mejor_valor = a, v
        for _ in range(4):
            _tirar(mejor, yo)
            total += 1

    mejor = max(brazos, key=lambda a: (a.n, a.w / a.n))
    reportar(tiradas=total, eval=round(mejor.w / mejor.n, 3))
    return mejor.jugada


def _tirar(arista, yo):
    resultados = arista.resultados
    if len(resultados) == 1:
        r = resultados[0]
    else:
        n = arista.n + 1
        r = max(resultados, key=lambda x: x[0] * n - x[3])
    r[3] += 1
    s = r[1]
    puntos = s._puntos
    if puntos[yo] >= 3:
        v = 1.0
    elif puntos[1 - yo] >= 3:
        v = 0.0
    else:
        turno = 0 if s.turno == "blancas" else 1
        v = simulacion.valorar(s._tablero, puntos, turno, SIMS_POR_HOJA, rng.getrandbits(31), yo)
    arista.n += 1
    arista.w += v
