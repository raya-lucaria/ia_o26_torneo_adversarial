"""Bot potential_man: UCT en la raíz, expectimax corto y carrera a 3 puntos.

SIMULACIONES (quietas) y SIMULACIONES_ATAQUE se pueden bajar si el reloj
se acerca a 27 s. Rango pensado: 24–40.
"""

import json
import math
import pickle
import random
import time
from pathlib import Path

from motor import Estado, reportar, simular_al_azar

SIMULACIONES = 28
SIMULACIONES_ATAQUE = 36
RESERVA = 0.22
PESO_WINRATE = 0.55
PESO_HEUR = 0.45
C_UCT = 0.85
W_PTS = 3.4
W_MAT = 0.30
W_AVANCE = 0.028
W_APOYO = 0.07
W_CARRERA = 0.25
# Avance 4–7 un poco más caro: premia acercarse a la meta sin aplastar el medio juego.
PESOS_AVANCE = (0.0, 1.0, 2.0, 3.5, 6.0, 11.0, 20.0, 34.0)
BONUS_ANOTA = 0.16
BONUS_PARA = 0.22
PENAL_SUELTA = 0.28
PENAL_COINFLIP = 0.18
P_PARA_MIN = 0.50
TOPE_EVAL = 20000
TOPE_SIM = 8000
LIBRO_PLIES = 8


def preparar(config):
    global color, rival, rng, reserva, _eval_cache, _sim_cache, n_turno, libro, tablas, lado_apertura
    color = config.color
    rival = "negras" if color == "blancas" else "blancas"
    rng = random.Random(config.semilla)
    # Un ala por partida: en los primeros turnos se empuja ese lado.
    lado_apertura = "izquierda" if rng.random() < 0.5 else "derecha"
    reserva = RESERVA * config.reloj_s
    _eval_cache = {}
    _sim_cache = {}
    n_turno = 0
    libro, tablas = {}, None
    _cargar_datos(getattr(config, "carpeta_datos", None))
    simular_al_azar(Estado.inicial(), 1, 0, para=color)


def _cargar_datos(carpeta):
    """Libro y tablas son opcionales: si no están, el bot juega igual."""
    global libro, tablas
    if not carpeta:
        return
    base = Path(carpeta)
    apertura = base / "opening.json"
    if apertura.is_file():
        try:
            datos = json.loads(apertura.read_text(encoding="utf-8"))
            if isinstance(datos, dict):
                libro = datos
        except (OSError, json.JSONDecodeError):
            libro = {}
    tablas_path = base / "tablas.pkl"
    if tablas_path.is_file():
        try:
            with tablas_path.open("rb") as archivo:
                cargado = pickle.load(archivo)
            if isinstance(cargado, dict):
                tablas = cargado
        except (OSError, pickle.UnpicklingError, EOFError):
            tablas = None


def _casilla(texto):
    return (int(texto[1]) - 1) * 9 + (ord(texto[0]) - 97)


def _avance(casilla, para):
    fila = casilla // 9
    return fila if para == "blancas" else 8 - fila


def _apoyos(casilla, propias, para):
    fila, col = divmod(casilla, 9)
    atras = fila - 1 if para == "blancas" else fila + 1
    if atras < 0 or atras > 8:
        return 0
    n = 0
    for dc in (-1, 1):
        c = col + dc
        if 0 <= c <= 8 and (atras * 9 + c) in propias:
            n += 1
    return n


def _vias_anotar(piezas, propias, ajenas, para):
    """Peón → todas las casillas de la última fila donde puede anotar."""
    fila_from = 7 if para == "blancas" else 1
    fila_to = 8 if para == "blancas" else 0
    vias = {}
    for c in piezas:
        if c // 9 != fila_from:
            continue
        col = c % 9
        destinos = []
        for dc in (-1, 0, 1):
            nc = col + dc
            if nc < 0 or nc > 8:
                continue
            dest = fila_to * 9 + nc
            if dest in propias:
                continue
            if dc == 0 and dest in ajenas:
                continue
            destinos.append(dest)
        if destinos:
            vias[c] = destinos
    return vias


def _tempos_peon(casilla, para, ajenas):
    av = _avance(casilla, para)
    t = 8 - av
    fila, col = divmod(casilla, 9)
    df = 1 if para == "blancas" else -1
    nf = fila + df
    if 0 <= nf <= 8 and (nf * 9 + col) in ajenas:
        t += 0.6
    return t


def _tempos_a_meta(piezas, ajenas, para, puntos):
    faltan = 3 - puntos
    if faltan <= 0:
        return 0.0
    if not piezas:
        return 30.0
    tempos = sorted(_tempos_peon(c, para, ajenas) for c in piezas)
    if len(tempos) < faltan:
        tempos.extend([12.0] * (faltan - len(tempos)))
    return sum(tempos[:faltan])


def _piezas_int(estado, quien):
    return [_casilla(c) for c in estado.piezas(quien)]


def _guardar_eval(clave, valor):
    if len(_eval_cache) > TOPE_EVAL:
        _eval_cache.clear()
    _eval_cache[clave] = valor
    return valor


def _hay_peon_nuestro(casilla, set_propias, solo_diagonales):
    """Un peón nuestro llega a esa casilla en un paso (recto y diagonal, o solo diagonal)."""
    fila, col = divmod(casilla, 9)
    df = 1 if color == "blancas" else -1
    origen_f = fila - df
    if origen_f < 0 or origen_f > 8:
        return False
    deltas = (-1, 1) if solo_diagonales else (-1, 0, 1)
    for dc in deltas:
        oc = col + dc
        if 0 <= oc <= 8 and (origen_f * 9 + oc) in set_propias:
            return True
    return False


def _penal_camino_libre(ajenas, set_propias, set_ajenas):
    """Peón rival avanzado, recto vacío, y no lo podemos capturar ni bloquear."""
    df = 1 if rival == "blancas" else -1
    penal = 0.0
    for c in ajenas:
        av = _avance(c, rival)
        if av < 6 or _hay_peon_nuestro(c, set_propias, solo_diagonales=True):
            continue
        fila, col = divmod(c, 9)
        nf = fila + df
        if nf < 0 or nf > 8:
            continue
        delante = nf * 9 + col
        if delante in set_propias or delante in set_ajenas:
            continue
        if _hay_peon_nuestro(delante, set_propias, solo_diagonales=False):
            continue
        penal += 0.08 if av == 6 else 0.12
    return min(0.20, penal)


def evaluar(estado):
    """Carrera a 3 puntos + estructura. La caché vive toda la partida."""
    clave = estado.clave
    cached = _eval_cache.get(clave)
    if cached is not None:
        return cached

    ganador = estado.ganador()
    if ganador == color:
        return _guardar_eval(clave, 1.0)
    if ganador is not None:
        return _guardar_eval(clave, -1.0)

    pts = estado.puntos
    propias = _piezas_int(estado, color)
    ajenas = _piezas_int(estado, rival)
    set_propias = set(propias)
    set_ajenas = set(ajenas)

    avance_yo = 0.0
    for c in propias:
        av = _avance(c, color)
        avance_yo += PESOS_AVANCE[av]
        if av >= 4:
            avance_yo += 6.0 * _apoyos(c, set_propias, color) * W_APOYO

    avance_rival = 0.0
    for c in ajenas:
        av = _avance(c, rival)
        avance_rival += PESOS_AVANCE[av]
        if av >= 4:
            avance_rival += 6.0 * _apoyos(c, set_ajenas, rival) * W_APOYO

    estructura = (
        W_PTS * (pts[color] - pts[rival])
        + (0.6 if pts[color] >= 2 else 0.0)
        - (0.6 if pts[rival] >= 2 else 0.0)
        + W_AVANCE * (avance_yo - avance_rival)
        + W_MAT * (len(propias) - len(ajenas))
    )

    ty = _tempos_a_meta(propias, set_ajenas, color, pts[color])
    tr = _tempos_a_meta(ajenas, set_propias, rival, pts[rival])
    carrera = max(-4.0, min(4.0, (tr - ty) * W_CARRERA))

    vias_r = _vias_anotar(ajenas, set_ajenas, set_propias, rival)
    vias_y = _vias_anotar(propias, set_propias, set_ajenas, color)
    tactica = 0.0
    if estado.turno == rival and vias_r:
        tactica -= 0.62 * len(vias_r)
        if pts[rival] >= 2:
            tactica -= 0.45
    elif estado.turno == color and vias_r:
        tactica -= 0.38 * len(vias_r)
        if pts[rival] >= 2:
            tactica -= 0.30
    if estado.turno == color and vias_y:
        tactica += 0.28 * len(vias_y)
        if pts[color] >= 2:
            tactica += 0.25
    # Suave: no saca jugadas. Solo baja la eval si el peón ya va suelto.
    tactica -= _penal_camino_libre(ajenas, set_propias, set_ajenas)

    valor = math.tanh(estructura / 3.6) + 0.18 * math.tanh(carrera) + tactica
    return _guardar_eval(clave, max(-1.0, min(1.0, valor)))


def _mezcla(winrate, heuristica):
    return PESO_WINRATE * winrate + PESO_HEUR * (heuristica + 1.0) / 2.0


def _simular(estado, n):
    """Misma (posición, n, semilla) no se vuelve a tirar."""
    ganador = estado.ganador()
    if ganador == color:
        return 1.0
    if ganador is not None:
        return 0.0
    semilla = rng.randrange(2**31)
    clave = (estado.clave, n, semilla)
    previa = _sim_cache.get(clave)
    if previa is not None:
        return previa
    wr = simular_al_azar(estado, n, semilla, para=color)
    if len(_sim_cache) > TOPE_SIM:
        _sim_cache.clear()
    _sim_cache[clave] = wr
    return wr


def _valor_hijo(hijo, n_sims):
    return _mezcla(_simular(hijo, n_sims), evaluar(hijo))


def _ev_heuristico(estado, j):
    return sum(p * evaluar(hijo) for p, hijo in estado.siguiente(j))


def _anota(estado, hijo):
    return hijo.puntos[color] > estado.puntos[color] or hijo.ganador() == color


def _amenazas_rival(estado):
    propias = set(_piezas_int(estado, color))
    ajenas = _piezas_int(estado, rival)
    return set(_vias_anotar(ajenas, set(ajenas), propias, rival))


def _es_parada(estado, j, amenazas):
    return bool(amenazas) and estado.es_ataque(j) and j.destino in amenazas


def _bloqueos(estado, jugadas, ajenas):
    """Ocupar el recto de un peón rival que ya está a 1–2 pasos de anotar."""
    df = 1 if rival == "blancas" else -1
    rectos = set()
    for c in ajenas:
        if _avance(c, rival) < 6:
            continue
        fila, col = divmod(c, 9)
        nf = fila + df
        if 0 <= nf <= 8:
            rectos.add(nf * 9 + col)
    return [j for j in jugadas if j.destino in rectos]


def _intercambios(estado, jugadas):
    """Ataques con p alto contra un peón ya adelantado."""
    buenos = []
    for j in jugadas:
        if not estado.es_ataque(j):
            continue
        if _avance(j.destino, rival) < 5:
            continue
        if estado.prob_ataque(j) >= 0.65:
            buenos.append(j)
    return buenos


def _bonus_tactico(estado, j, amenazas):
    ramas = estado.siguiente(j)
    bonus = 0.0
    pts_r = estado.puntos[rival]
    anota_algun = False
    for p, hijo in ramas:
        if _anota(estado, hijo):
            anota_algun = True
            bonus += p * BONUS_ANOTA
            if hijo.ganador() == color:
                bonus += p * 0.5
        if hijo.ganador() is None:
            inm = _amenazas_rival(hijo)
            if inm:
                extra = PENAL_SUELTA * (1.4 if pts_r >= 2 else 1.0)
                bonus -= p * extra * min(2, len(inm))
            elif amenazas:
                bonus += p * 0.06
    p_atk = estado.prob_ataque(j) if estado.es_ataque(j) else 1.0
    if _es_parada(estado, j, amenazas):
        # p=0.50 casi neutro; por debajo se penaliza, no se premia.
        bonus += BONUS_PARA * (p_atk - P_PARA_MIN) * 2.4
        if p_atk < 0.40:
            bonus -= PENAL_COINFLIP * (0.45 - p_atk)
        elif pts_r >= 2 and p_atk >= P_PARA_MIN:
            bonus += 0.08
    elif estado.es_ataque(j):
        if _avance(j.destino, rival) >= 6 and p_atk >= 0.65:
            bonus += 0.06 * p_atk
        if p_atk <= P_PARA_MIN and not anota_algun:
            bonus -= PENAL_COINFLIP * (P_PARA_MIN + 0.05 - p_atk)
    return bonus


def _n_sims(es_ataque, n_candidatos):
    base = SIMULACIONES if n_candidatos <= 18 else max(24, SIMULACIONES - (n_candidatos - 18) // 2)
    if es_ataque:
        return min(40, max(base, SIMULACIONES_ATAQUE if n_candidatos <= 22 else base + 6))
    return min(40, base)


def medir_jugada(estado, j, amenazas, n_candidatos):
    ramas = estado.siguiente(j)
    if len(ramas) == 1:
        ev = _valor_hijo(ramas[0][1], _n_sims(False, n_candidatos))
    else:
        n = _n_sims(True, n_candidatos)
        ev = 0.0
        for p, hijo in ramas:
            ev += p * _valor_hijo(hijo, n)
    return ev + _bonus_tactico(estado, j, amenazas)


def _ucb(j, total, muestras, n_raiz):
    n = muestras[j]
    if n <= 0:
        return float("inf")
    return total[j] / n + C_UCT * math.sqrt(math.log(n_raiz + 1.0) / n)


def _unicos(seq):
    vistos = []
    for j in seq:
        if j not in vistos:
            vistos.append(j)
    return vistos


def _candidatos(estado, jugadas, amenazas):
    pts = estado.puntos
    quietas_anotan = []
    mates = []
    paradas = []
    for j in jugadas:
        ramas = estado.siguiente(j)
        if all(hijo.ganador() == color for _, hijo in ramas):
            mates.append(j)
            continue
        if len(ramas) == 1 and _anota(estado, ramas[0][1]):
            quietas_anotan.append(j)
        if _es_parada(estado, j, amenazas):
            paradas.append(j)

    if mates:
        return mates, "mate"

    if amenazas:
        ajenas = set(_piezas_int(estado, rival))
        bloqueos = _bloqueos(estado, jugadas, ajenas)
        cambios = _intercambios(estado, jugadas)
        urgentes = _unicos(paradas + quietas_anotan + bloqueos + cambios)
        if pts[rival] >= 2 and urgentes:
            return urgentes, "para"
        if paradas or bloqueos:
            return urgentes, "para"

    if quietas_anotan:
        return quietas_anotan, "anota"
    return jugadas, "uct"


def _expectimax(estado, prof):
    """2 plies como mucho, solo las 3 mejores por heurística. Sin rollouts."""
    ganador = estado.ganador()
    if ganador == color:
        return 1.0
    if ganador is not None:
        return -1.0
    jugadas = estado.jugadas()
    if prof <= 0 or not jugadas:
        return evaluar(estado)
    maximiza = estado.turno == color
    ordenadas = sorted(jugadas, key=lambda j: _ev_heuristico(estado, j), reverse=maximiza)[:3]
    valores = []
    for j in ordenadas:
        valores.append(sum(p * _expectimax(hijo, prof - 1) for p, hijo in estado.siguiente(j)))
    return max(valores) if maximiza else min(valores)


def _expectimax_jugada(estado, j):
    # Nuestra jugada + respuesta del rival (profundidad 1 en el hijo = 2 plies).
    return sum(p * _expectimax(hijo, 1) for p, hijo in estado.siguiente(j))


def _del_libro(estado, jugadas):
    if n_turno > LIBRO_PLIES or (not libro and not tablas):
        return None
    clave = str(estado.clave)
    texto = None
    if isinstance(libro, dict):
        texto = libro.get(clave) or libro.get(str(n_turno))
    if texto is None and isinstance(tablas, dict):
        texto = tablas.get(estado.clave) or tablas.get(clave) or tablas.get(n_turno)
    if not isinstance(texto, str):
        return None
    for j in jugadas:
        if str(j) == texto:
            return j
    return None


def _orden_exacto(est, jugadas):
    """Primero anotar o capturar; dentro de cada grupo, la heurística del bando que mueve."""
    maximiza = est.turno == color
    fila_meta = 8 if est.turno == "blancas" else 0

    def clave(j):
        urgente = est.es_ataque(j) or j.destino // 9 == fila_meta
        heur = _ev_heuristico(est, j)
        return (1 if urgente else 0, heur if maximiza else -heur)

    return sorted(jugadas, key=clave, reverse=True)


def _bonus_apertura(estado, j):
    """+0.10 en los primeros 10 turnos si la jugada abre el ala elegida."""
    if n_turno > 10:
        return 0.0
    lado = set(range(4)) if lado_apertura == "izquierda" else set(range(5, 9))
    col = j.destino % 9
    if col in lado:
        return 0.10
    if not estado.es_ataque(j):
        return 0.0
    # Captura en una columna pegada al ala (para la izquierda, la 4; para la derecha, la 4).
    borde = {c + d for c in lado for d in (-1, 1) if 0 <= c + d <= 8 and c + d not in lado}
    return 0.10 if col in borde else 0.0


class _TopeBusqueda(Exception):
    """El alfa-beta se pasó de nodos o de tiempo; el resultado no es confiable."""


def es_final_manejable(estado):
    """Pocos peones (≤ 8), o todos a lo más a 6 pasos de la meta."""
    propias = _piezas_int(estado, color)
    ajenas = _piezas_int(estado, rival)
    if len(propias) + len(ajenas) <= 8:
        return True
    set_propias, set_ajenas = set(propias), set(ajenas)
    tempos = [_tempos_peon(c, color, set_ajenas) for c in propias]
    tempos += [_tempos_peon(c, rival, set_propias) for c in ajenas]
    return bool(tempos) and max(tempos) <= 6


def buscar_exacto(estado, limite_nodos=80000, limite_s=0.8):
    """Expectiminimax con poda. None si no cabe en nodos o en tiempo.

    En ataques el valor es la esperanza exacta (las dos ramas se exploran).
    En jugadas quietas, alfa-beta poda desde el punto de vista de `color`.
    """
    nodos = [0]
    deadline = time.perf_counter() + limite_s

    def corte():
        if nodos[0] > limite_nodos or time.perf_counter() >= deadline:
            raise _TopeBusqueda

    def valor(est, alpha, beta):
        nodos[0] += 1
        corte()
        ganador = est.ganador()
        if ganador == color:
            return 1.0
        if ganador is not None:
            return -1.0
        jugadas = est.jugadas()
        if not jugadas:
            return -1.0 if est.turno == color else 1.0
        maximiza = est.turno == color
        orden = _orden_exacto(est, jugadas)
        if maximiza:
            mejor = -2.0
            for j in orden:
                mejor = max(mejor, _resultado(est, j, alpha, beta))
                alpha = max(alpha, mejor)
                if alpha >= beta:
                    break
            return mejor
        mejor = 2.0
        for j in orden:
            mejor = min(mejor, _resultado(est, j, alpha, beta))
            beta = min(beta, mejor)
            if alpha >= beta:
                break
        return mejor

    def _resultado(est, j, alpha, beta):
        ramas = est.siguiente(j)
        if len(ramas) == 1:
            return valor(ramas[0][1], alpha, beta)
        # Esperanza exacta: cada cara del dado se valora en la ventana completa.
        return sum(p * valor(hijo, -1.0, 1.0) for p, hijo in ramas)

    try:
        jugadas = estado.jugadas()
        if not jugadas:
            return None
        mejor_j = None
        mejor_v = -2.0
        alpha = -1.0
        for j in _orden_exacto(estado, jugadas):
            v = _resultado(estado, j, alpha, 1.0)
            if v > mejor_v:
                mejor_v = v
                mejor_j = j
            alpha = max(alpha, v)
        if mejor_j is None:
            return None
        return mejor_j, mejor_v
    except _TopeBusqueda:
        return None


def _presupuesto(disponible, n_candidatos):
    # Más candidatas piden un poco más de reloj, sin comerse el medio juego.
    divisor = max(12.0, 26.0 - min(n_candidatos, 24) * 0.35)
    return min(1.35, max(0.05, disponible / divisor))


def jugar(estado, tiempo_restante):
    global n_turno
    n_turno += 1

    jugadas = estado.jugadas()
    del_libro = _del_libro(estado, jugadas)
    if del_libro is not None:
        reportar(muestras=0, eval=0.5, modo="libro", riesgo=0.0)
        return del_libro

    amenazas = _amenazas_rival(estado)
    candidatos, modo0 = _candidatos(estado, jugadas, amenazas)

    if modo0 == "mate":
        reportar(muestras=0, eval=1.0, modo="mate", riesgo=0.0)
        return candidatos[0]
    if modo0 == "anota" and len(candidatos) == 1:
        reportar(muestras=0, eval=0.85, modo="anota", riesgo=0.0)
        return candidatos[0]

    disponible = tiempo_restante - reserva
    n_peones = len(estado.piezas(color)) + len(estado.piezas(rival))
    # Atajo intacto. Contra montecarlo la carrera termina con ~20 peones, así que
    # no se ensancha el umbral: un árbol más grande solo abortaría y robaría tiempo al UCT.
    if es_final_manejable(estado) and disponible > 0.3 and n_peones <= 10:
        if n_peones <= 8:
            limite_s = min(1.2, disponible * 0.55)
        else:
            limite_s = min(0.12, disponible * 0.2)
        hallado = buscar_exacto(estado, limite_s=limite_s)
        if hallado is not None:
            jugada, valor = hallado
            reportar(muestras=0, eval=valor, modo="exacto", riesgo=float(len(amenazas)))
            return jugada

    limite = time.perf_counter() + _presupuesto(max(0.0, disponible), len(candidatos))

    if disponible <= 0:
        mejor = max(
            candidatos,
            key=lambda j: _ev_heuristico(estado, j) + _bonus_tactico(estado, j, amenazas),
        )
        reportar(muestras=0, eval=evaluar(estado), modo=modo0, riesgo=0.0)
        return mejor

    total = {}
    muestras = {}
    n_cand = len(candidatos)
    for j in candidatos:
        # Una muestra virtual con la heurística: el UCT no arranca a ciegas.
        previo = _ev_heuristico(estado, j) + _bonus_tactico(estado, j, amenazas) + _bonus_apertura(estado, j)
        total[j] = (previo + 1.0) / 2.0
        muestras[j] = 1.0

    for j in candidatos:
        if time.perf_counter() >= limite:
            break
        total[j] += medir_jugada(estado, j, amenazas, n_cand)
        muestras[j] += 1.0

    n_raiz = sum(muestras.values())
    while time.perf_counter() < limite:
        j = max(candidatos, key=lambda x: _ucb(x, total, muestras, n_raiz))
        total[j] += medir_jugada(estado, j, amenazas, n_cand)
        muestras[j] += 1.0
        n_raiz += 1.0

    extra = {}
    if time.perf_counter() + 0.04 < limite and len(candidatos) > 1:
        mejores = sorted(candidatos, key=lambda j: total[j] / muestras[j], reverse=True)[:2]
        for j in mejores:
            if time.perf_counter() >= limite:
                break
            extra[j] = 0.12 * _expectimax_jugada(estado, j)

    def _nota(j):
        return total[j] / muestras[j] + extra.get(j, 0.0)

    mejor = max(candidatos, key=_nota)
    reportar(
        muestras=int(round(sum(muestras.values()))),
        eval=_nota(mejor),
        modo=modo0,
        riesgo=float(len(amenazas)),
    )
    return mejor


# Más simulaciones en ataques (28/36), UCT con prior heurístico, expectimax de 2 plies
# solo en las 2 mejores si queda tiempo, caché de eval entre turnos, parada escalada
# por p, bloqueo del paso, libro opcional, y alfa-beta exacto si el final cabe en nodos.
# La eval resta un poco si un peón rival va libre y no se puede capturar ni bloquear.
