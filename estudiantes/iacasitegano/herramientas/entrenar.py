"""Genera partidas de self-play (y partidas contra `random`/`montecarlo`, para
variar las posiciones que vemos) con nuestra propia búsqueda, y ajusta los
`PESOS` de `eval.py` por regresión lineal (con un poco de regularización)
contra el resultado final de cada partida. Guarda el resultado como un
candidato `data/candidato_<timestamp>.npy`; nunca reemplaza directamente el
campeón `data/pesos.npy`.

Nada en `herramientas/` corre en la liga (ver docs/avanzado.md). Para
regenerar los pesos:

    cd estudiantes/iacasitegano
    python herramientas/entrenar.py

Primera corrida (60 partidas, solo contra uno mismo): no alcanzó para ganarle
a montecarlo. Segunda corrida (300 partidas, rivales variados, ridge):
tampoco — confirmó que el techo no eran los pesos, era la profundidad de
búsqueda en Python. Tercer cambio, el que de verdad movió la aguja (ver
`INFORME.md` sección 7): portar la búsqueda a numba (`busqueda_numba.py`),
~200x más rápido en profundidad 4. Cuarta corrida (500 partidas, profundidad
4): primera victoria real contra montecarlo. Quinta corrida: profundidad 5
(ya barata con la tabla de transposición dentro de numba) — entrenar con
partidas de una versión más débil que la que va a usar esos pesos no tenía
sentido, y profundidad 5 ya es la que de verdad juega el bot en partidas
reales con tiempo típico.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np

CARPETA_EQUIPO = Path(__file__).resolve().parent.parent
CARPETA_ESTUDIANTES = CARPETA_EQUIPO.parent
sys.path.insert(0, str(CARPETA_EQUIPO))
sys.path.insert(0, str(CARPETA_EQUIPO.parent.parent / "sistema"))

from motor import Estado, Jugada  # noqa: E402

import busqueda_numba as bn  # noqa: E402
import eval as ev  # noqa: E402

PARTIDAS_POR_SEMILLA = 100
PROFUNDIDAD = 5  # barata gracias a numba+TT (~0.1-0.15s/jugada); antes del port, 2 era el límite
SEMILLAS = (1234, 5678, 9012)
LAMBDA_RIDGE = 1.0  # regularización L2: frena pesos inestables por colinealidad entre features
RELOJ_RIVAL_S = 30.0  # reloj "de a de veras" para random/montecarlo, igual que una partida oficial
# Fracción de partidas contra cada rival (adentro, siempre jugamos nosotros con nuestra búsqueda).
OPONENTES = {
    "uno_mismo": 0.5,
    "random": 0.2,
    "montecarlo": 0.2,
    "voraz": 0.05,
    "minimax2": 0.05,
}
RIVALES_LOCALES = {
    "voraz": CARPETA_EQUIPO / "herramientas" / "rival_voraz.py",
    "minimax2": CARPETA_EQUIPO / "herramientas" / "rival_minimax2.py",
}


def _cargar_strategy(equipo: str) -> ModuleType:
    ruta = RIVALES_LOCALES.get(equipo, CARPETA_ESTUDIANTES / equipo / "strategy.py")
    spec = importlib.util.spec_from_file_location(f"strategy_{equipo}", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class _ConfigRival:
    def __init__(self, color: str, semilla: int):
        self.color = color
        self.semilla = semilla
        self.reloj_s = RELOJ_RIVAL_S


def _jugador_propio(ganancia: float):
    def mover(estado: Estado, _reloj: float) -> Jugada:
        tablero, puntos, color_idx = bn._tablero_desde_estado(estado)
        pesos = np.ascontiguousarray(ev.PESOS, dtype=np.float64)
        o, d, _ = bn._mejor_jugada_numba(
            tablero, puntos, color_idx, PROFUNDIDAD, pesos, ganancia, -1, -1
        )
        return Jugada(int(o), int(d))

    return mover


def _jugador_rival(modulo: ModuleType, color: str, semilla: int):
    modulo.preparar(_ConfigRival(color, semilla))

    def mover(estado: Estado, reloj: float):
        return modulo.jugar(estado, reloj)

    return mover


def _jugar_partida(
    rng: random.Random, ganancia: float, modulo_rival: ModuleType | None
) -> list[tuple[np.ndarray, float]]:
    """Una partida (nosotros vs. `modulo_rival`, o contra uno mismo si es `None`).

    Regresa [(features, resultado), ...] solo de NUESTRAS jugadas (+1 si ganamos
    esa partida, -1 si perdimos): es la única perspectiva que de verdad jugamos.
    """
    estado = Estado.inicial()
    nuestro_color = rng.choice(["blancas", "negras"])
    mover_propio = _jugador_propio(ganancia)
    reloj = {"blancas": RELOJ_RIVAL_S, "negras": RELOJ_RIVAL_S}
    mover_rival = (
        None
        if modulo_rival is None
        else _jugador_rival(modulo_rival, "negras" if nuestro_color == "blancas" else "blancas",
                             rng.randrange(2**31))
    )

    registro: list[tuple[np.ndarray, str]] = []
    while not estado.terminado():
        color = estado.turno
        if color == nuestro_color:
            registro.append((ev.features(estado, color), color))
            jugada = mover_propio(estado, reloj[color])
        else:
            inicio = time.perf_counter()
            jugada = (mover_rival or mover_propio)(estado, reloj[color])
            reloj[color] = max(0.0, reloj[color] - (time.perf_counter() - inicio))
        probs, hijos = zip(*estado.siguiente(jugada), strict=True)
        estado = rng.choices(hijos, weights=probs)[0]

    ganador = estado.ganador()
    return [(x, 1.0 if color == ganador else -1.0) for x, color in registro]


def _elegir_rival(rng: random.Random, modulos: dict[str, ModuleType | None]) -> ModuleType | None:
    nombres = list(OPONENTES.keys())
    pesos = list(OPONENTES.values())
    return modulos[rng.choices(nombres, weights=pesos)[0]]


def generar_datos(partidas: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(semilla)
    ruta_pesos_actuales = CARPETA_EQUIPO / "data" / "pesos.npy"
    if ruta_pesos_actuales.exists():  # jugamos con los últimos pesos aceptados, no los de fábrica
        ev.cargar_pesos(np.load(ruta_pesos_actuales))
    ganancia = ev.cota() + 100.0
    modulos: dict[str, ModuleType | None] = {"uno_mismo": None}
    if OPONENTES.get("random"):
        modulos["random"] = _cargar_strategy("random")
    if OPONENTES.get("montecarlo"):
        modulos["montecarlo"] = _cargar_strategy("montecarlo")
    for nombre in ("voraz", "minimax2"):
        if OPONENTES.get(nombre):
            modulos[nombre] = _cargar_strategy(nombre)

    xs: list[np.ndarray] = []
    ys: list[float] = []
    inicio = time.perf_counter()
    for i in range(partidas):
        rival = _elegir_rival(rng, modulos)
        for x, y in _jugar_partida(rng, ganancia, rival):
            xs.append(x)
            ys.append(y)
        if (i + 1) % 10 == 0 or i + 1 == partidas:
            print(
                f"  {i + 1}/{partidas} partidas, {len(xs)} posiciones, "
                f"{time.perf_counter() - inicio:.0f}s"
            )
    return np.array(xs), np.array(ys)


def ajustar_pesos(xs: np.ndarray, ys: np.ndarray, lam: float) -> np.ndarray:
    """Ridge: minimiza ||Xw - y||² + lam·||w||². Cierra en forma con la ecuación normal."""
    n_features = xs.shape[1]
    return np.linalg.solve(xs.T @ xs + lam * np.eye(n_features), xs.T @ ys)


def main() -> None:
    total = PARTIDAS_POR_SEMILLA * len(SEMILLAS)
    print(f"Generando {total} partidas (semillas: {SEMILLAS}; "
          f"oponentes: {OPONENTES}, profundidad propia {PROFUNDIDAD})...")
    lotes = [generar_datos(PARTIDAS_POR_SEMILLA, semilla) for semilla in SEMILLAS]
    xs = np.concatenate([lote[0] for lote in lotes])
    ys = np.concatenate([lote[1] for lote in lotes])
    print(f"{len(xs)} posiciones. Ajustando pesos (ridge, lambda={LAMBDA_RIDGE})...")
    pesos = ajustar_pesos(xs, ys, LAMBDA_RIDGE)
    print("Pesos (antes -> entrenados):")
    for nombre, antes, nuevo in zip(ev.NOMBRES_FEATURES, ev.PESOS, pesos, strict=True):
        print(f"  {nombre}: {antes:.2f} -> {nuevo:.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = CARPETA_EQUIPO / "data" / f"candidato_{timestamp}.npy"
    np.save(destino, pesos)
    print(f"Candidato guardado en {destino}; pesos.npy no se modifica")


if __name__ == "__main__":
    main()
