"""Datos para entrenar la evaluación: posiciones de partidas, sus rasgos y quién ganó.

Uso (desde estudiantes/maquinas_del_trabajo/; los datos van a `datos_entrenamiento/`, que el
`.gitignore` de la carpeta deja fuera de git: no se suben). Así se generaron los datos de los
pesos actuales (`data/pesos.npz`, ver herramientas/entrenar.py): jugando con los pesos que había
antes, los del commit 354df4b (la primera tabla de patrones).
    mkdir -p datos_entrenamiento
    git show 354df4b:estudiantes/maquinas_del_trabajo/data/pesos.npz \
        > datos_entrenamiento/pesos_354df4b.npz
    ../../.venv/bin/python -m herramientas.datos --rival nuestro --partidas 12000 --nodos 20000 \
        --semilla 2001 --pesos datos_entrenamiento/pesos_354df4b.npz \
        --salida datos_entrenamiento/r2_autojuego.npz
    ../../.venv/bin/python -m herramientas.datos --rival montecarlo --partidas 1500 --nodos 20000 \
        --reloj-rival 4 --semilla 2002 --pesos datos_entrenamiento/pesos_354df4b.npz \
        --salida datos_entrenamiento/r2_montecarlo.npz
Las partidas de autojuego se repiten igual con la misma semilla (nuestro bot juega por nodos);
las de montecarlo no del todo, porque montecarlo piensa por reloj.

Cada partida empieza con unas jugadas al azar (para que no todas sean iguales) y luego juega
nuestro bot con un límite de nodos (rápido y repetible) contra sí mismo o contra la copia de
`montecarlo` con reloj corto. Por cada posición se guardan sus rasgos (`evaluacion.rasgos`), el
lugar en la tabla de patrones de cada peón (`evaluacion.patrones`),
desde el bando que mueve, si ese bando terminó ganando y el valor que le dio nuestra búsqueda
(NaN si movió montecarlo): es lo que necesita la regresión logística de
`herramientas/entrenar.py`. Los combates usan un `random.Random` propio.
"""

import argparse
import random
import sys
import time
from collections import namedtuple
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from motor import Estado

from busqueda import crear_memoria, profundizar
from evaluacion import MAX_PEONES, N_BASE, patrones, rasgos
from herramientas.rival_montecarlo import Montecarlo
from motor_rapido import desde_estado, texto_jugada

CARPETA = Path(__file__).resolve().parent.parent
# Por posición: rasgos (N_BASE), quién ganó, valor de la búsqueda, por peón el lugar de su
# patrón y su signo (+1 del bando que mueve, −1 del rival; relleno −1 y 0), y la posición misma
# (tablero, puntos y bando que mueve): con ella `entrenar` recalcula los rasgos si cambian.
Datos = namedtuple("Datos", ["X", "y", "v", "indices", "signos", "tabs", "ptss", "colores"])


def partida_para_datos(
    semilla: int, rival: str, nodos: int, reloj_rival: float = 4.0, pesos=None
) -> Datos:
    """Los datos (ver `Datos`) de cada posición de una partida."""
    rng = random.Random(semilla)
    al_azar = rng.randint(2, 8)  # jugadas al azar del principio
    nuestros = {"blancas", "negras"}
    rival_mc = None
    if rival == "montecarlo":
        color_rival = "negras" if semilla % 2 == 0 else "blancas"
        nuestros.discard(color_rival)
        rival_mc = Montecarlo()
        rival_mc.preparar(
            SimpleNamespace(
                color=color_rival, semilla=semilla, carpeta_datos=CARPETA / "data",
                reloj_s=reloj_rival,
            )
        )
    memorias = {color: crear_memoria(bits_tt=18, pesos=pesos) for color in nuestros}
    usado_rival = 0.0
    filas, turnos, valores, indices, signos, posiciones = [], [], [], [], [], []
    salida = np.empty(N_BASE)
    indices_peon = np.empty(MAX_PEONES, np.int64)
    signos_peon = np.empty(MAX_PEONES, np.int64)
    estado = Estado.inicial()
    ply = 0
    while not estado.terminado():
        tab, pts, color = desde_estado(estado)
        if ply < al_azar:
            texto = str(rng.choice(estado.jugadas()))
        else:
            rasgos(tab, pts, color, salida)
            filas.append(salida.copy())
            n = patrones(tab, color, indices_peon, signos_peon)
            indices.append(np.concatenate([indices_peon[:n], np.full(MAX_PEONES - n, -1)]))
            signos.append(np.concatenate([signos_peon[:n], np.zeros(MAX_PEONES - n, np.int64)]))
            turnos.append(estado.turno)
            posiciones.append((tab, pts, color))
            if estado.turno in nuestros:
                r = profundizar(memorias[estado.turno], tab, pts, color, 100.0, 100.0, max_nodos=nodos)
                texto = texto_jugada(r.jugada)
                valores.append(r.valor)
            else:
                valores.append(np.nan)
                inicio = time.perf_counter()
                texto = str(rival_mc.jugar(estado, max(0.0, reloj_rival - usado_rival)))
                usado_rival += time.perf_counter() - inicio
        jugada = {str(j): j for j in estado.jugadas()}[texto]
        gana = rng.random() < estado.prob_ataque(jugada) if estado.es_ataque(jugada) else None
        estado = estado.aplicar(jugada, gana)
        ply += 1
    ganador = estado.ganador()
    return Datos(
        X=np.array(filas, dtype=np.float32).reshape(-1, N_BASE),
        y=np.array([t == ganador for t in turnos], dtype=np.uint8),
        v=np.array(valores, dtype=np.float32),
        indices=np.array(indices, dtype=np.int32).reshape(-1, MAX_PEONES),
        signos=np.array(signos, dtype=np.int8).reshape(-1, MAX_PEONES),
        tabs=np.array([t for t, _, _ in posiciones], dtype=np.int8).reshape(-1, 81),
        ptss=np.array([p for _, p, _ in posiciones], dtype=np.int8).reshape(-1, 2),
        colores=np.array([c for _, _, c in posiciones], dtype=np.int8),
    )


@cache
def _pesos_de(archivo):
    """Se lee una vez por proceso (no viaja la tabla completa en cada trabajo)."""
    return None if archivo is None else np.load(archivo)["pesos"]


def _jugar(trabajo):
    semilla, rival, nodos, reloj_rival, archivo = trabajo
    return partida_para_datos(semilla, rival, nodos, reloj_rival, _pesos_de(archivo))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partidas", type=int, default=1000)
    parser.add_argument("--rival", choices=["nuestro", "montecarlo"], default="nuestro")
    parser.add_argument("--nodos", type=int, default=20_000, help="límite de nodos por jugada")
    parser.add_argument("--reloj-rival", type=float, default=4.0)
    parser.add_argument("--procesos", type=int, default=6)
    parser.add_argument("--semilla", type=int, default=1)
    parser.add_argument("--pesos", type=Path, help="pesos de nuestro bot (.npz); si no, los a mano")
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args(argv)

    trabajos = [
        (args.semilla * 1_000_003 + k, args.rival, args.nodos, args.reloj_rival, args.pesos)
        for k in range(args.partidas)
    ]
    inicio = time.perf_counter()
    with Pool(args.procesos) as pool:
        partidas = pool.map(_jugar, trabajos, chunksize=4)
    juntos = {campo: np.concatenate([getattr(d, campo) for d in partidas]) for campo in Datos._fields}
    y = juntos["y"]
    ids = np.concatenate([np.full(len(d.y), k, np.int32) for k, d in enumerate(partidas)])
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.salida, partida=ids, **juntos)
    print(
        f"{len(partidas)} partidas, {len(y)} posiciones → {args.salida}"
        f" ({time.perf_counter() - inicio:.0f} s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
