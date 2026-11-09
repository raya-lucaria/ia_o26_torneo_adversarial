"""Bot «mc_aprendiz»: Modelo principal de competición para el torneo de IA Breakthrough.

Arquitectura híbrida que integra:
- Búsqueda adaptativa mediante UCB1 (Multi-Armed Bandit) para asignar rollouts.
- Evaluación estratégica multidimensional con pesos parametrizados y aprendibles (TD(0)).
- Resolución analítica exacta de los nodos de azar (capturas y probabilidades reales).
- Detección táctica inmediata de victorias terminales y amenazas críticas.
- Tabla de transposición rápida basada en llaves Zobrist de 64 bits (`estado.clave`).
- Gestión estricta del reloj de partida y respaldo legal infalible.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from motor import Estado, reportar, simular_al_azar

# Configuración por defecto si los archivos JSON no están disponibles
CONFIG_DEFECTO = {
    "c_exploracion": 0.85,
    "peso_heuristica": 0.40,
    "sims_por_hoja": 10,
    "max_cache": 5000,
    "fraccion_turno": 25.0,
    "reserva_reloj": 0.30,
}

PESOS_DEFECTO = {
    "dif_puntos": 2.107807,
    "dif_peones": 0.058300,
    "avance_promedio": 0.496565,
    "avance_maximo": 0.350000,
    "amenaza_anotacion": 0.400000,
    "movilidad_relativa": 0.150000,
    "apoyo_mutuo": 0.120000,
    "vulnerabilidad": 0.180000,
    "bloqueo_rival": 0.250000,
    "control_centro": 0.080000,
}

NOMBRES_CARACTERISTICAS = (
    "dif_puntos",
    "dif_peones",
    "avance_promedio",
    "avance_maximo",
    "amenaza_anotacion",
    "movilidad_relativa",
    "apoyo_mutuo",
    "vulnerabilidad",
    "bloqueo_rival",
    "control_centro",
)


def _fila_col(casilla: str) -> tuple[int, int]:
    return int(casilla[1]), ord(casilla[0]) - ord("a")


def _avance_peon(fila: int, col_j: str) -> int:
    return (fila - 1) if col_j == "blancas" else (9 - fila)


def extraer_caracteristicas(estado: Estado, perspectiva: str) -> list[float]:
    """Extrae vector de características normalizadas en [-1, 1] desde `perspectiva`."""
    ganador = estado.ganador()
    if ganador is not None:
        val = 1.0 if ganador == perspectiva else -1.0
        return [val] * len(NOMBRES_CARACTERISTICAS)

    rival = "negras" if perspectiva == "blancas" else "blancas"
    pts = estado.puntos
    mias = estado.piezas(perspectiva)
    suyas = estado.piezas(rival)
    n_mias = len(mias)
    n_suyas = len(suyas)

    dif_pts = (pts[perspectiva] - pts[rival]) / 3.0
    dif_peones = (n_mias - n_suyas) / 18.0

    av_mios = [_avance_peon(_fila_col(c)[0], perspectiva) for c in mias] if n_mias else [0]
    av_suyos = [_avance_peon(_fila_col(c)[0], rival) for c in suyas] if n_suyas else [0]

    prom_yo = sum(av_mios) / max(1, n_mias)
    prom_el = sum(av_suyos) / max(1, n_suyas)
    dif_prom = (prom_yo - prom_el) / 8.0

    max_yo = max(av_mios) if av_mios else 0
    max_el = max(av_suyos) if av_suyos else 0
    dif_max = (max_yo - max_el) / 8.0

    amenazas_yo = sum(1 for a in av_mios if a >= 7)
    amenazas_el = sum(1 for a in av_suyos if a >= 7)
    dif_amenaza = max(-1.0, min(1.0, (amenazas_yo - amenazas_el) / 3.0))

    if estado.turno == perspectiva:
        mov_yo = len(estado.jugadas())
        mov_el = max(1, n_suyas)
    else:
        mov_el = len(estado.jugadas())
        mov_yo = max(1, n_mias)
    movilidad_rel = max(-1.0, min(1.0, (mov_yo - mov_el) / 25.0))

    pos_mias = {_fila_col(c) for c in mias}
    pos_suyas = {_fila_col(c) for c in suyas}

    def _peones_apoyados(pos, col_j):
        df = 1 if col_j == "blancas" else -1
        ap = 0
        for f, c in pos:
            f_det = f - df
            if (f_det, c - 1) in pos or (f_det, c + 1) in pos:
                ap += 1
        return ap

    apoyo_yo = _peones_apoyados(pos_mias, perspectiva) / max(1, n_mias)
    apoyo_el = _peones_apoyados(pos_suyas, rival) / max(1, n_suyas)
    apoyo_mutuo = apoyo_yo - apoyo_el

    def _peones_amenazados(prop, riv, col_j):
        df_riv = -1 if col_j == "blancas" else 1
        am = 0
        for f, c in prop:
            f_r = f - df_riv
            if (f_r, c - 1) in riv or (f_r, c + 1) in riv:
                am += 1
        return am

    am_yo = _peones_amenazados(pos_mias, pos_suyas, perspectiva) / max(1, n_mias)
    am_el = _peones_amenazados(pos_suyas, pos_mias, rival) / max(1, n_suyas)
    vulnerabilidad = am_el - am_yo

    bloqueo = 0.0
    for f, c in pos_suyas:
        if _avance_peon(f, rival) >= 6:
            df = 1 if perspectiva == "blancas" else -1
            f_b = f + df
            if (f_b, c) in pos_mias or (f_b, c - 1) in pos_mias or (f_b, c + 1) in pos_mias:
                bloqueo += 0.5
            else:
                bloqueo -= 0.5
    bloqueo_rival = max(-1.0, min(1.0, bloqueo))

    centro_yo = sum(1 for _, c in pos_mias if 2 <= c <= 6) / max(1, n_mias)
    centro_el = sum(1 for _, c in pos_suyas if 2 <= c <= 6) / max(1, n_suyas)
    control_centro = centro_yo - centro_el

    return [
        float(dif_pts),
        float(dif_peones),
        float(dif_prom),
        float(dif_max),
        float(dif_amenaza),
        float(movilidad_rel),
        float(apoyo_mutuo),
        float(vulnerabilidad),
        float(bloqueo_rival),
        float(control_centro),
    ]


def preparar(config):
    global color, rival, rng, reserva, parametros, vector_pesos, cache_heuristica
    color = config.color
    rival = "negras" if color == "blancas" else "blancas"
    rng = random.Random(config.semilla)

    # 1. Carga de hiperparámetros
    parametros = dict(CONFIG_DEFECTO)
    carpeta_propia = Path(__file__).parent
    ruta_cfg = carpeta_propia / "config.json"
    if ruta_cfg.is_file():
        try:
            parametros.update(json.loads(ruta_cfg.read_text()))
        except Exception:
            pass

    # 2. Carga de pesos aprendidos
    pesos_dict = dict(PESOS_DEFECTO)
    ruta_pesos = carpeta_propia / "pesos.json"
    if ruta_pesos.is_file():
        try:
            cargados = json.loads(ruta_pesos.read_text())
            for k in NOMBRES_CARACTERISTICAS:
                if k in cargados:
                    pesos_dict[k] = float(cargados[k])
        except Exception:
            pass

    vector_pesos = [pesos_dict[k] for k in NOMBRES_CARACTERISTICAS]
    reserva = float(parametros.get("reserva_reloj", 0.30)) * config.reloj_s
    cache_heuristica = {}

    # Calentamiento JIT de numba
    simular_al_azar(Estado.inicial(), 1, 0, para=color)


def evaluar_aprendido(estado: Estado) -> float:
    """Evalúa la posición con los pesos aprendidos mapeando a probabilidad [0, 1]."""
    ganador = estado.ganador()
    if ganador == color:
        return 1.0
    if ganador is not None:
        return 0.0

    clave = estado.clave
    if clave in cache_heuristica:
        return cache_heuristica[clave]

    feats = extraer_caracteristicas(estado, color)
    score = sum(w * x for w, x in zip(vector_pesos, feats))
    # Sigmoide logística simétrica centrada
    val = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, score))))

    if len(cache_heuristica) < int(parametros.get("max_cache", 10000)):
        cache_heuristica[clave] = val

    return val


def evaluar_hoja(estado: Estado, limite: float) -> float:
    """Combina valor aprendido (con caché rápida) con simulación Monte Carlo fresca."""
    if estado.terminado():
        return 1.0 if estado.ganador() == color else 0.0

    heur = evaluar_aprendido(estado)
    if time.perf_counter() >= limite:
        return heur

    sims = int(parametros.get("sims_por_hoja", 15))
    peso_h = float(parametros.get("peso_heuristica", 0.15))
    mc = simular_al_azar(estado, sims, rng.randrange(2**31), para=color)
    return peso_h * heur + (1.0 - peso_h) * mc


def valor_esperado_jugada(estado: Estado, jugada, limite: float) -> float:
    """Calcula la ganancia esperada considerando todas las transiciones del motor."""
    siguientes = estado.siguiente(jugada)
    return sum(p * evaluar_hoja(hijo, limite) for p, hijo in siguientes)


def jugar(estado: Estado, tiempo_restante: float):
    disponible = tiempo_restante - reserva
    fraccion = float(parametros.get("fraccion_turno", 25.0))
    limite = time.perf_counter() + max(0.0, disponible) / fraccion
    jugadas = estado.jugadas()

    if not jugadas:
        return None

    # 1. Poda táctica terminal: victoria inmediata garantizada
    for j in jugadas:
        if all(hijo.ganador() == color for _, hijo in estado.siguiente(j)):
            return j

    # 2. Anotación segura inmediata: si hay una pieza a una casilla de dar punto y hay espacio libre, tomarlo
    meta = 9 if color == "blancas" else 1
    anotaciones_seguras = [
        j for j in jugadas
        if (j.destino // 9 + 1) == meta and not estado.es_ataque(j)
    ]
    if anotaciones_seguras:
        # Preferir avance recto si está libre; de lo contrario cualquier diagonal sin combate
        rectas = [j for j in anotaciones_seguras if j.destino % 9 == j.origen % 9]
        return rectas[0] if rectas else anotaciones_seguras[0]

    if disponible <= 0.0:
        return rng.choice(jugadas)

    # 2. Inicialización de estadísticas del bandit UCB1
    recompensas = {j: 0.0 for j in jugadas}
    visitas = {j: 0 for j in jugadas}
    total_visitas = 0
    c_ucb = float(parametros.get("c_exploracion", 0.85))

    # Fase de inicialización: 1 visita para cada jugada
    for j in jugadas:
        ev = valor_esperado_jugada(estado, j, limite)
        recompensas[j] += ev
        visitas[j] += 1
        total_visitas += 1
        if time.perf_counter() >= limite:
            break

    # Fase de exploración y explotación adaptativa con UCB1
    while time.perf_counter() < limite:
        log_n = math.log(max(1, total_visitas))
        mejor_j = None
        mejor_ucb = -float("inf")

        for j in jugadas:
            n_j = visitas[j]
            if n_j == 0:
                mejor_j = j
                break
            q_j = recompensas[j] / n_j
            ucb = q_j + c_ucb * math.sqrt(log_n / n_j)
            if ucb > mejor_ucb:
                mejor_ucb = ucb
                mejor_j = j

        if mejor_j is None:
            mejor_j = rng.choice(jugadas)

        ev = valor_esperado_jugada(estado, mejor_j, limite)
        recompensas[mejor_j] += ev
        visitas[mejor_j] += 1
        total_visitas += 1

    # Criterio final: jugada con mayor soporte empírico y recompensa
    def criterio_final(j):
        n = visitas[j]
        if n == 0:
            return 0.0
        return (recompensas[j] / n) + 0.005 * math.log(1 + n)

    mejor_jugada = max(jugadas, key=criterio_final)
    q_mejor = recompensas[mejor_jugada] / max(1, visitas[mejor_jugada])
    reportar(muestras=total_visitas, eval=q_mejor, modelo="mc_aprendiz")
    return mejor_jugada
