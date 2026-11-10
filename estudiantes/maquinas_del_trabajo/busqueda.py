"""Búsqueda del bot: expectiminimax en forma negamax, con alfa-beta y Star1, en numba.

- Nodos de decisión: negamax *fail-soft* con ventana (α, β). Los valores van de −1 a 1 y son
  `2·P(ganar) − 1` del bando que mueve.
- Nodos de azar: cada ataque tiene dos resultados (gana con probabilidad p, pierde con 1 − p).
  Star1 (Ballard, 1983) busca primero el más probable con una ventana reducida, y el segundo
  solo si todavía puede cambiar la decisión. Así se poda también bajo azar.
- Victoria o derrota segura: ±(1 − 0.0005·ply), por encima de cualquier evaluación (tope 0.95);
  entre dos victorias se prefiere la más corta y entre dos derrotas la más larga.
- Último punto en las hojas: si el siguiente punto decide la partida y un peón lo anota sin
  combate en la jugada que sigue, la hoja ya sabe quién gana (`_ultimo_punto`).
- Caché de evaluaciones: a una hoja se llega muchas veces por otro orden de jugadas (~1 de cada
  3). Su evaluación se guarda por clave Zobrist y se reusa: el valor es el mismo, solo más rápido.

Todo el estado de la búsqueda vive en una `Memoria` de arreglos preasignados (nada se crea
por nodo): el tablero de cada ply, las jugadas generadas, la tabla de transposición, etc.
"""

import time
from collections import namedtuple

import numba
import numpy as np
from numba import njit

from evaluacion import I_AVANCE, I_MATERIAL, PESOS_F2, evaluar
from motor_rapido import (
    CASILLAS,
    LADO,
    MAX_JUGADAS,
    PUNTOS_PARA_GANAR,
    VACIA,
    apoyo,
    clave,
    generar,
    hacer,
    njit_interna,
    paso,
    prob_jugada,
    tiene_jugada,
)

MAX_PLY = 64  # ningún camino de la búsqueda pasa de aquí (protege los arreglos por ply)
PROF_MAX = 60
BITS_TT = 22  # 4 M lugares en la tabla de transposición (≈ 88 MB)
# 256 K lugares en la caché de evaluaciones (4 MB). Medido en la Mac: 2^18 reusa el 33 % de las
# hojas y da +23 % de nodos/s; 2^16 +19 %, 2^20 +14 % (una tabla más grande ya no cabe en la
# caché del procesador).
BITS_EV = 18
PERDIDA_POR_PLY = 0.0005
EXACTO, INFERIOR, SUPERIOR = 1, 2, 3  # tipos de valor guardado en la TT
MAX_NODOS = 2**62
L, U = -1.0, 1.0  # cotas de cualquier valor: Star1 las necesita
VENTANA_ASPIRACION = 0.1
# Se empieza otra profundidad si, según cuánto creció la anterior, cabría en FACTOR_ITERACION
# veces lo que queda del tiempo blando. Una iteración cortada igual aporta su mejor jugada.
FACTOR_ITERACION = 4.0  # medido en la arena: 75 % contra 58 % con 1.5

Resultado = namedtuple("Resultado", ["jugada", "valor", "prof", "nodos", "segundos"])

Memoria = namedtuple(
    "Memoria",
    [
        "tabs",  # int8[MAX_PLY+2, 81]: tablero de cada ply
        "ptss",  # int64[MAX_PLY+2, 2]: puntos de cada ply
        "claves",  # uint64[MAX_PLY+2]: clave Zobrist de cada ply
        "origenes",  # int64[MAX_PLY+2, 54]: jugadas generadas en cada ply
        "destinos",
        "puntajes",  # int64[MAX_PLY+2, 54]: para ordenar las jugadas
        "killers",  # int64[MAX_PLY+2, 2]: jugadas tranquilas que cortaron en ese ply
        "historia",  # int64[81·81]: qué tanto ha cortado cada jugada tranquila
        "tt_clave",  # tabla de transposición: un lugar por índice
        "tt_valor",
        "tt_prof",
        "tt_tipo",
        "tt_jugada",
        "tt_gen",
        "pesos",  # float64: pesos de la evaluación (N_BASE, o N_PESOS con tabla de patrones)
        "contadores",  # int64[3]: nodos, abortado, generación
        "limites",  # float64[2]: hora límite (perf_counter) y máximo de nodos
        "opciones",  # int64[5]: usar TT, ordenar, ataques que se espera perder al final,
        # último punto en las hojas, caché de evaluaciones
        "raiz",  # int64[2]: mejor jugada de la raíz que subió α (−1 si ninguna), jugadas terminadas
        "ev_clave",  # uint64[2^BITS_EV]: caché de evaluaciones, clave de la hoja en cada lugar
        "ev_valor",  # float64[2^BITS_EV]: y su evaluación
    ],
)


def crear_memoria(bits_tt: int = BITS_TT, pesos=None) -> Memoria:
    """Memoria nueva. Se crea una vez (en `preparar`) y se reusa toda la partida."""
    lugares = 2**bits_tt
    plies = MAX_PLY + 2
    return Memoria(
        tabs=np.zeros((plies, CASILLAS), np.int8),
        ptss=np.zeros((plies, 2), np.int64),
        claves=np.zeros(plies, np.uint64),
        origenes=np.zeros((plies, MAX_JUGADAS), np.int64),
        destinos=np.zeros((plies, MAX_JUGADAS), np.int64),
        puntajes=np.zeros((plies, MAX_JUGADAS), np.int64),
        killers=np.full((plies, 2), -1, np.int64),
        historia=np.zeros(CASILLAS * CASILLAS, np.int64),
        tt_clave=np.zeros(lugares, np.uint64),
        tt_valor=np.zeros(lugares, np.float64),
        tt_prof=np.zeros(lugares, np.int8),
        tt_tipo=np.zeros(lugares, np.int8),
        tt_jugada=np.zeros(lugares, np.int16),
        tt_gen=np.zeros(lugares, np.uint8),
        pesos=np.array(PESOS_F2 if pesos is None else pesos, dtype=np.float64),
        contadores=np.zeros(3, np.int64),
        limites=np.array([np.inf, float(MAX_NODOS)]),
        # PVS y LMR se probaron y se quitaron: PVS subía los nodos y LMR no ganó en la arena
        # (ver INFORME); su código, además, alargaba `preparar`.
        opciones=np.array([1, 1, 1, 1, 1], np.int64),
        raiz=np.array([-1, 0], np.int64),
        # Clave 0 en los lugares vacíos: ninguna posición real tiene esa clave (1 en 2^64).
        ev_clave=np.zeros(2**BITS_EV, np.uint64),
        ev_valor=np.zeros(2**BITS_EV, np.float64),
    )


@njit_interna
def ahora():
    """Reloj de pared dentro de numba (para cortar la búsqueda a tiempo)."""
    with numba.objmode(t="float64"):
        t = time.perf_counter()
    return t


# --- Tabla de transposición ------------------------------------------------------------------
# Cubetas de 2 lugares: el lugar 0 guarda lo más profundo (de esta búsqueda) y el lugar 1 lo
# más reciente. Cada lugar guarda la clave completa, así que una colisión de índice no se
# confunde con la misma posición.


@njit_interna
def _cubeta(m, h):
    return np.int64(h & np.uint64(m.tt_clave.shape[0] - 2))


@njit_interna
def tt_buscar(m, h):
    """Índice del lugar de la TT que tiene la posición `h`, o −1 si no está."""
    i = _cubeta(m, h)
    if m.tt_tipo[i] != 0 and m.tt_clave[i] == h:
        return i
    if m.tt_tipo[i + 1] != 0 and m.tt_clave[i + 1] == h:
        return i + 1
    return -1


@njit_interna
def tt_guardar(m, h, prof, valor, tipo, jugada):
    """Guarda un resultado en la TT (en el lugar 0 si lo merece; si no, en el lugar 1)."""
    i = _cubeta(m, h)
    generacion = m.contadores[2]
    if not (m.tt_clave[i] == h or m.tt_gen[i] != generacion or prof >= m.tt_prof[i]):
        i += 1
    m.tt_clave[i] = h
    m.tt_valor[i] = valor
    m.tt_prof[i] = prof
    m.tt_tipo[i] = tipo
    m.tt_jugada[i] = jugada
    m.tt_gen[i] = generacion


# --- Orden de jugadas ------------------------------------------------------------------------
# Mientras antes se busque la mejor jugada, más poda alfa-beta. El orden: jugadas que anotan,
# la jugada de la TT, ataques (los más rentables primero), killers y, al final, las jugadas
# tranquilas según cuánto han cortado antes (historia).

ANOTA = 3_000_000_000
DE_LA_TT = 2_000_000_000
ATAQUE = 10_000_000
KILLER = 1_000_000


@njit_interna
def _valor_peon(m, casilla, duenio):
    """Valor aproximado de un peón para ordenar ataques: material + avance."""
    fila = casilla // LADO
    avance = fila if duenio == 0 else LADO - 1 - fila
    return m.pesos[I_MATERIAL] + m.pesos[I_AVANCE + avance]


@njit_interna
def ordenar(m, ply, n, color, jugada_tt):
    """Llena `m.puntajes[ply, :n]` (más alto = se busca antes)."""
    tab = m.tabs[ply]
    meta = LADO - 1 if color == 0 else 0
    for i in range(n):
        o = m.origenes[ply, i]
        d = m.destinos[ply, i]
        cod = o * CASILLAS + d
        if d // LADO == meta:
            puntaje = ANOTA
        elif cod == jugada_tt:
            puntaje = DE_LA_TT
        elif tab[d] != VACIA:
            p = prob_jugada(tab, color, o, d)
            ganancia = p * _valor_peon(m, d, 1 - color) - (1.0 - p) * _valor_peon(m, o, color)
            if ganancia < 0.0 and m.opciones[2] != 0:
                puntaje = int(1000.0 * ganancia)  # ataque que se espera perder: al final
            else:
                puntaje = ATAQUE + int(1000.0 * ganancia)
        elif cod == m.killers[ply, 0]:
            puntaje = KILLER
        elif cod == m.killers[ply, 1]:
            puntaje = KILLER - 1
        else:
            puntaje = min(m.historia[cod], KILLER - 1)
        m.puntajes[ply, i] = puntaje


# --- El último punto en las hojas ------------------------------------------------------------
# Con 2 puntos, el que anote el siguiente gana. En una hoja eso se sabe sin buscar más cuando un
# peón está a una jugada de la meta y puede anotar sin combate (recto o en diagonal a una casilla
# vacía). La evaluación no lo ve si la meta tiene peones cerca: su carrera decidida pide
# corredores sin nadie alrededor. Nadie puede tapar las casillas de la meta del rival (los peones
# no retroceden), así que la única defensa es capturar al peón.


@njit_interna
def _anota_sin_combate(tab, c, duenio):
    """True si el peón de `duenio` en `c`, a una jugada de su meta, puede anotar sin atacar."""
    frente = c + paso(duenio) * LADO
    col = c % LADO
    if tab[frente] == VACIA:
        return True
    if col > 0 and tab[frente - 1] == VACIA:
        return True
    return col < LADO - 1 and tab[frente + 1] == VACIA


@njit_interna
def _ultimo_punto(tab, pts, color):
    """1 si `color` (el que mueve) anota el último punto en su jugada; −1 si el rival lo anota
    en la suya haga lo que haga `color`; 0 si no se sabe sin buscar."""
    rival = 1 - color
    if pts[color] == PUNTOS_PARA_GANAR - 1:
        fila = LADO - 2 if color == 0 else 1  # a una jugada de la meta
        cerca = False
        for c in range(fila * LADO, (fila + 1) * LADO):
            if tab[c] == color + 1:
                if _anota_sin_combate(tab, c, color):
                    return 1
                cerca = True
        if cerca:  # quizá anote atacando antes que el rival: no se sabe
            return 0
    if pts[rival] == PUNTOS_PARA_GANAR - 1:
        fila = LADO - 2 if rival == 0 else 1
        amenazas = 0
        for c in range(fila * LADO, (fila + 1) * LADO):
            if tab[c] == rival + 1 and _anota_sin_combate(tab, c, rival):
                # `apoyo` desde el que mueve cuenta sus peones que pueden atacar la casilla `c`.
                if apoyo(tab, c, color) == 0:
                    return -1
                amenazas += 1
        if amenazas >= 2:  # solo puede capturar a uno
            return -1
    return 0


@njit
def negamax(m, ply, prof, alfa, beta, color):
    """Valor de la posición `m.tabs[ply]` para `color` (el que mueve), *fail-soft*.

    Si el valor real es ≤ α regresa una cota superior (≤ α); si es ≥ β, una cota inferior
    (≥ β); si no, el valor exacto. Con el aborto encendido regresa 0.0 y nadie lo usa.
    """
    nodos = m.contadores[0] + 1
    m.contadores[0] = nodos
    if nodos >= m.limites[1] or ((nodos & 4095) == 0 and ahora() > m.limites[0]):
        m.contadores[1] = 1
    if m.contadores[1] != 0:
        return 0.0

    tab = m.tabs[ply]
    pts = m.ptss[ply]
    if pts[1 - color] >= PUNTOS_PARA_GANAR:  # el que acaba de mover ganó
        return -(1.0 - PERDIDA_POR_PLY * ply)
    if prof <= 0 or ply >= MAX_PLY:
        if not tiene_jugada(tab, color):
            return -(1.0 - PERDIDA_POR_PLY * ply)
        if m.opciones[3] != 0:
            ultimo = _ultimo_punto(tab, pts, color)
            if ultimo > 0:  # anota en el ply siguiente
                return 1.0 - PERDIDA_POR_PLY * (ply + 1)
            if ultimo < 0:  # el rival anota dos plies después
                return -(1.0 - PERDIDA_POR_PLY * (ply + 2))
        if m.opciones[4] == 0:
            return evaluar(tab, pts, color, m.pesos)
        h = m.claves[ply]  # incluye tablero, puntos y quién mueve: todo lo que lee evaluar
        lugar = np.int64(h & np.uint64(m.ev_clave.shape[0] - 1))
        if m.ev_clave[lugar] == h:
            return m.ev_valor[lugar]
        valor = evaluar(tab, pts, color, m.pesos)
        m.ev_clave[lugar] = h
        m.ev_valor[lugar] = valor
        return valor

    h = m.claves[ply]
    usar_tt = m.opciones[0] != 0
    jugada_tt = -1
    if usar_tt:
        lugar = tt_buscar(m, h)
        if lugar >= 0:
            jugada_tt = m.tt_jugada[lugar]
            # En la raíz nunca se corta: ahí necesitamos la jugada, no solo el valor.
            if ply > 0 and m.tt_prof[lugar] >= prof:
                guardado = m.tt_valor[lugar]
                tipo = m.tt_tipo[lugar]
                if (
                    tipo == EXACTO
                    or (tipo == INFERIOR and guardado >= beta)
                    or (tipo == SUPERIOR and guardado <= alfa)
                ):
                    return guardado

    origenes = m.origenes[ply]
    destinos = m.destinos[ply]
    puntajes = m.puntajes[ply]
    n = generar(tab, color, origenes, destinos)
    if n == 0:  # sin jugadas: pierde el que mueve
        return -(1.0 - PERDIDA_POR_PLY * ply)
    ordenar_jugadas = m.opciones[1] != 0
    if ordenar_jugadas:
        ordenar(m, ply, n, color, jugada_tt)

    alfa_original = alfa
    tab_hijo = m.tabs[ply + 1]
    pts_hijo = m.ptss[ply + 1]
    mejor = -2.0
    mejor_cod = -1
    for i in range(n):
        if ordenar_jugadas:  # selección: trae al lugar i la de mayor puntaje que queda
            k = i
            for j in range(i + 1, n):
                if puntajes[j] > puntajes[k]:
                    k = j
            if k != i:
                origenes[i], origenes[k] = origenes[k], origenes[i]
                destinos[i], destinos[k] = destinos[k], destinos[i]
                puntajes[i], puntajes[k] = puntajes[k], puntajes[i]
        o = origenes[i]
        d = destinos[i]
        sin_combate = tab[d] == VACIA
        if sin_combate:
            # Sin combate `gana` no importa. Se pasa una variable y no la constante True:
            # con la constante numba compilaría `hacer` dos veces (más tiempo de preparar).
            m.claves[ply + 1] = hacer(tab, pts, color, o, d, sin_combate, h, tab_hijo, pts_hijo)
            valor = -negamax(m, ply + 1, prof - 1, -beta, -alfa, 1 - color)
        else:
            # Star1: primero el resultado más probable (A), con la ventana que garantiza
            # que si A ya decide (corte o falla), B no se busca.
            p = prob_jugada(tab, color, o, d)
            gana_a = p >= 0.5
            pa = p if gana_a else 1.0 - p
            pb = 1.0 - pa
            m.claves[ply + 1] = hacer(tab, pts, color, o, d, gana_a, h, tab_hijo, pts_hijo)
            a_a = max((alfa - pb * U) / pa, L)
            b_a = min((beta - pb * L) / pa, U)
            v_a = -negamax(m, ply + 1, prof - 1, -b_a, -a_a, 1 - color)
            if v_a >= b_a:
                valor = pa * v_a + pb * L  # cota inferior, ya ≥ β
            elif v_a <= a_a:
                valor = pa * v_a + pb * U  # cota superior, ya ≤ α
            else:
                m.claves[ply + 1] = hacer(tab, pts, color, o, d, not gana_a, h, tab_hijo, pts_hijo)
                a_b = max((alfa - pa * v_a) / pb, L)
                b_b = min((beta - pa * v_a) / pb, U)
                v_b = -negamax(m, ply + 1, prof - 1, -b_b, -a_b, 1 - color)
                valor = pa * v_a + pb * v_b
        if m.contadores[1] != 0:
            return 0.0
        cod = o * CASILLAS + d
        if ply == 0:
            m.raiz[1] += 1
        if valor > mejor:
            mejor = valor
            mejor_cod = cod
        if valor > alfa:
            alfa = valor
            if ply == 0:
                m.raiz[0] = cod
        if alfa >= beta:
            if sin_combate and ordenar_jugadas:  # una jugada tranquila cortó: recordarla
                if m.killers[ply, 0] != cod:
                    m.killers[ply, 1] = m.killers[ply, 0]
                    m.killers[ply, 0] = cod
                m.historia[cod] += prof * prof
            break

    if usar_tt:
        if mejor <= alfa_original:
            tipo = SUPERIOR
        elif mejor >= beta:
            tipo = INFERIOR
        else:
            tipo = EXACTO
        tt_guardar(m, h, prof, mejor, tipo, mejor_cod)
    return mejor


def _preparar_raiz(m: Memoria, tab, pts, color: int) -> None:
    """Pone la posición en el ply 0 y reinicia contadores y resultado de la raíz."""
    m.tabs[0] = tab
    m.ptss[0] = pts
    m.claves[0] = clave(m.tabs[0], color, m.ptss[0])
    m.contadores[0] = 0
    m.contadores[1] = 0
    m.raiz[0] = -1
    m.raiz[1] = 0


def buscar(m: Memoria, tab, pts, color: int, prof: int, alfa=-1.0, beta=1.0) -> tuple[float, int]:
    """Valor y código de la mejor jugada a profundidad fija, sin límite de tiempo."""
    _preparar_raiz(m, tab, pts, color)
    m.limites[0] = np.inf
    m.limites[1] = float(MAX_NODOS)
    valor = negamax(m, 0, int(prof), float(alfa), float(beta), int(color))
    return valor, int(m.raiz[0])


def _raiz(m: Memoria, prof: int, alfa: float, beta: float, color: int) -> float:
    m.raiz[0] = -1
    m.raiz[1] = 0
    return negamax(m, 0, prof, float(alfa), float(beta), color)


def profundizar(
    m: Memoria, tab, pts, color: int, blando: float, duro: float,
    prof_max: int = PROF_MAX, max_nodos: int = MAX_NODOS,
) -> Resultado:
    """Profundización iterativa: la mejor jugada que alcance a ver en el tiempo dado.

    - `blando`: segundos que queremos usar. No se empieza una profundidad que, según cuánto
      creció la anterior, no alcanzaría a terminar.
    - `duro`: segundos que nunca se pasan; la búsqueda se corta adentro de numba.
    - `max_nodos`: corte por nodos (el resultado no depende del reloj; sirve para pruebas).

    Siempre hay respuesta: la profundidad 1 se busca sin límite (tarda microsegundos).
    """
    inicio = time.perf_counter()
    color = int(color)
    _preparar_raiz(m, tab, pts, color)
    m.contadores[2] = (m.contadores[2] + 1) % 256  # nueva generación en la TT
    m.killers[:] = -1
    m.historia[:] //= 4  # lo que cortó en turnos anteriores vale menos ahora
    n = generar(m.tabs[0], color, m.origenes[0], m.destinos[0])
    if n <= 1:  # una sola jugada (o ninguna): no hay nada que pensar
        jugada = int(m.origenes[0, 0] * CASILLAS + m.destinos[0, 0]) if n == 1 else -1
        return Resultado(jugada, 0.0, 0, 0, time.perf_counter() - inicio)

    m.limites[:] = (np.inf, float(MAX_NODOS))
    valor = _raiz(m, 1, L, U, color)
    jugada = int(m.raiz[0])
    prof = 1
    nodos_anteriores = int(m.contadores[0])
    if duro > 0.01:
        m.limites[:] = (inicio + duro, float(max_nodos))
        for d in range(2, prof_max + 1):
            inicio_iteracion = time.perf_counter()
            nodos_al_empezar = int(m.contadores[0])
            alfa = max(valor - VENTANA_ASPIRACION, L)
            beta = min(valor + VENTANA_ASPIRACION, U)
            v = _raiz(m, d, alfa, beta, color)
            ya_mejor = -1  # jugada que superó la ventana de aspiración (si fue el caso)
            if m.contadores[1] == 0 and (v <= alfa or v >= beta):
                if v >= beta:
                    ya_mejor = int(m.raiz[0])
                v = _raiz(m, d, L, U, color)  # falló la aspiración: ventana completa
            if m.contadores[1] != 0:  # se acabó el tiempo (o los nodos) a media iteración
                # Una jugada de esta iteración terminada y mejor que la anterior sí sirve.
                if m.raiz[0] >= 0:
                    jugada = int(m.raiz[0])
                elif ya_mejor >= 0:
                    jugada = ya_mejor
                break
            valor, jugada, prof = v, int(m.raiz[0]), d
            gastados = int(m.contadores[0]) - nodos_al_empezar
            crecimiento = min(max(gastados / max(nodos_anteriores, 1), 2.0), 30.0)
            nodos_anteriores = gastados
            tardo = time.perf_counter() - inicio_iteracion
            if tardo * crecimiento > FACTOR_ITERACION * (blando - (time.perf_counter() - inicio)):
                break
    return Resultado(jugada, valor, prof, int(m.contadores[0]), time.perf_counter() - inicio)
