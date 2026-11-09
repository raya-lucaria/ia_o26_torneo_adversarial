"""Liga interna: el gate antes de aceptar cualquier cambio de eval o búsqueda.

"Se ve mejor" no alcanza (ya nos pasó dos veces esta sesión: quiescencia y orden de
jugadas corrigieron algo real pero, medidos, no ganaron partidas de más — ver
INFORME.md sección 8b). Esto automatiza lo que hasta ahora hacíamos a mano con
`./torneo jugar --semilla ...`: juega el bot actual contra `random`, `montecarlo`, y
—si se le da `--checkpoint`— contra una versión anterior de los pesos, y reporta
victorias y puntos totales de cada lado. No corre en la liga oficial.

Uso:
    cd estudiantes/iacasitegano
    python herramientas/liga_interna.py
    python herramientas/liga_interna.py --partidas 40 --profundidad 4
    python herramientas/liga_interna.py --checkpoint data/pesos_anterior.npy

Antes de guardar un `data/pesos.npy` nuevo como el "bueno", vale la pena copiar el
viejo a algo como `data/pesos_anterior.npy` y correr esto con `--checkpoint` para
confirmar que el nuevo de verdad le gana al viejo, no solo a random/montecarlo.

# ponytail: sin la suite fija de ~50 posiciones de regresión que menciona el plan
# (necesita un criterio de "correcto" por posición, no solo ganar partidas — más
# trabajo del que rinde todavía). Esto compara por RESULTADO de partidas completas,
# que ya es el gate que de verdad nos ha hecho falta hasta ahora.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

CARPETA_EQUIPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARPETA_EQUIPO))
sys.path.insert(0, str(CARPETA_EQUIPO.parent.parent / "sistema"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from motor import Estado, Jugada  # noqa: E402

import busqueda_numba as bn  # noqa: E402
import eval as ev  # noqa: E402
from entrenar import RELOJ_RIVAL_S, _cargar_strategy, _ConfigRival  # noqa: E402

PARTIDAS_DEFAULT = 40
PROFUNDIDAD_DEFAULT = 4
SEMILLA = 2026


def _jugador_con_pesos(pesos: np.ndarray, profundidad: int, ganancia: float):
    pesos = np.asarray(pesos, dtype=float)
    if pesos.shape == (len(ev.NOMBRES_FEATURES) - 1,):
        pesos = np.concatenate((pesos, np.array([0.0])))
    if pesos.shape != (len(ev.NOMBRES_FEATURES),):
        raise ValueError(f"checkpoint con forma incompatible: {pesos.shape}")
    pesos = np.ascontiguousarray(pesos, dtype=np.float64)

    def mover(estado: Estado, _reloj: float) -> Jugada:
        tablero, puntos, color_idx = bn._tablero_desde_estado(estado)
        o, d, _ = bn._mejor_jugada_numba(
            tablero, puntos, color_idx, profundidad, pesos, ganancia, -1, -1
        )
        return Jugada(int(o), int(d))

    return mover


def _jugador_rival(modulo: ModuleType, color: str, semilla: int):
    modulo.preparar(_ConfigRival(color, semilla))

    def mover(estado: Estado, reloj: float) -> Jugada:
        return modulo.jugar(estado, reloj)

    return mover


def _jugar_partida(rng: random.Random, mover_a, mover_b, color_a: str) -> tuple[str, dict]:
    """`mover_a` juega `color_a`, `mover_b` el otro color. Regresa (ganador 'a'/'b', puntos)."""
    estado = Estado.inicial()
    reloj = {"blancas": RELOJ_RIVAL_S, "negras": RELOJ_RIVAL_S}
    while not estado.terminado():
        color = estado.turno
        mover = mover_a if color == color_a else mover_b
        inicio = time.perf_counter()
        jugada = mover(estado, reloj[color])
        reloj[color] = max(0.0, reloj[color] - (time.perf_counter() - inicio))
        probs, hijos = zip(*estado.siguiente(jugada), strict=True)
        estado = rng.choices(hijos, weights=probs)[0]
    ganador_color = estado.ganador()
    ganador = "a" if ganador_color == color_a else "b"
    return ganador, estado.puntos


def jugar_liga(nombre_a, fabrica_a, nombre_b, fabrica_b, partidas: int, semilla: int) -> None:
    """`fabrica_x(rng, color, semilla_juego)` debe regresar un `mover(estado, reloj)` fresco."""
    rng = random.Random(semilla)
    victorias_a = victorias_b = 0
    puntos_a = puntos_b = 0
    inicio = time.perf_counter()
    for i in range(partidas):
        color_a = "blancas" if i % 2 == 0 else "negras"
        color_b = "negras" if color_a == "blancas" else "blancas"
        semilla_juego = rng.randrange(2**31)
        mover_a = fabrica_a(color_a, semilla_juego)
        mover_b = fabrica_b(color_b, semilla_juego)
        ganador, puntos = _jugar_partida(rng, mover_a, mover_b, color_a)
        if ganador == "a":
            victorias_a += 1
        else:
            victorias_b += 1
        puntos_a += puntos[color_a]
        puntos_b += puntos[color_b]
    duracion = time.perf_counter() - inicio
    print(
        f"  {nombre_a} {victorias_a} - {victorias_b} {nombre_b}"
        f"  (puntos: {puntos_a} - {puntos_b}, {partidas} partidas, {duracion:.0f}s)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partidas", type=int, default=PARTIDAS_DEFAULT)
    ap.add_argument("--profundidad", type=int, default=PROFUNDIDAD_DEFAULT)
    ap.add_argument("--checkpoint", type=Path, default=None, help="pesos.npy anterior, para comparar")
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument(
        "--rival",
        choices=("ambos", "random", "montecarlo"),
        default="ambos",
        help="rival externo a ejecutar; permite continuar sin repetir tandas ya confirmadas",
    )
    args = ap.parse_args()

    ruta_pesos_actuales = CARPETA_EQUIPO / "data" / "pesos.npy"
    if ruta_pesos_actuales.exists():  # si no, se queda con los PESOS a mano de eval.py
        ev.cargar_pesos(np.load(ruta_pesos_actuales))
    ganancia = ev.cota() + 100.0
    pesos_actuales = np.ascontiguousarray(ev.PESOS, dtype=np.float64)

    def fabrica_actual(_color, _semilla_juego):
        return _jugador_con_pesos(pesos_actuales, args.profundidad, ganancia)

    rivales = ("random", "montecarlo") if args.rival == "ambos" else (args.rival,)
    for nombre_rival in rivales:
        modulo = _cargar_strategy(nombre_rival)

        def fabrica_rival(color, semilla_juego, modulo=modulo):
            return _jugador_rival(modulo, color, semilla_juego)

        print(f"iacasitegano (profundidad {args.profundidad}) vs {nombre_rival}:")
        jugar_liga("iacasitegano", fabrica_actual, nombre_rival, fabrica_rival,
                   args.partidas, args.semilla)

    if args.checkpoint is not None:
        pesos_anteriores = np.ascontiguousarray(np.load(args.checkpoint), dtype=np.float64)

        def fabrica_checkpoint(_color, _semilla_juego):
            return _jugador_con_pesos(pesos_anteriores, args.profundidad, ganancia)

        print(f"iacasitegano (pesos actuales) vs iacasitegano ({args.checkpoint.name}):")
        jugar_liga("actual", fabrica_actual, "checkpoint", fabrica_checkpoint,
                   args.partidas, args.semilla)


if __name__ == "__main__":
    main()
