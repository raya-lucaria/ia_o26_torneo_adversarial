"""Genera posiciones de entrenamiento para ajustar los pesos de la evaluación (offline).

Juega partidas del bot contra sí mismo con poco tiempo por jugada. Para que las partidas no se
repitan, las primeras medias jugadas son al azar y de vez en cuando se juega una jugada al azar.
De cada posición guarda sus características (alfabeta.caracteristicas, desde el bando que mueve)
y si ese bando terminó ganando la partida.

Uso (desde la raíz del repo, con .venv activado):
    python estudiantes/liszt_breakthrough/herramientas/generar_datos.py \
        --partidas 400 --seg 0.05 --salida datos_ronda1.npz [--pesos estudiantes/liszt_breakthrough/data/pesos.json]
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[1]
RAIZ = CARPETA.parents[1]
sys.path[:0] = [str(RAIZ / "sistema"), str(CARPETA)]

import numpy as np  # noqa: E402

PESOS = None
buscador = None


def _init(pesos):
    global PESOS, buscador
    import alfabeta as ab
    PESOS = pesos
    buscador = ab.Buscador(pesos=pesos)


def una_partida(args):
    semilla, seg, al_azar_ini, eps = args
    import alfabeta as ab
    from motor import Estado
    rng = random.Random(semilla)
    AC = np.zeros(40, np.int64)
    F = np.zeros(ab.NF, np.int64)
    e = Estado.inicial()
    filas = []
    movers = []
    medias = 0
    n_azar = rng.randint(2, al_azar_ini)
    while not e.terminado():
        jugadas = e.jugadas()
        if medias < n_azar or rng.random() < eps:
            j = rng.choice(jugadas)
        else:
            ab.caracteristicas(e._tablero, e._puntos, e._color, AC, F)
            filas.append(F.copy())
            movers.append(e._color)
            mov, _info = buscador.buscar(e._tablero, e._color, e._puntos, seg)
            j = next((x for x in jugadas if mov and x.origen == mov[0] and x.destino == mov[1]),
                     rng.choice(jugadas))
        if e.es_ataque(j):
            e = e.aplicar(j, gana_ataque=rng.random() < e.prob_ataque(j))
        else:
            e = e.aplicar(j)
        medias += 1
    ganador = 0 if e.ganador() == "blancas" else 1
    y = [1 if m == ganador else 0 for m in movers]
    return filas, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidas", type=int, default=400)
    ap.add_argument("--seg", type=float, default=0.05, help="segundos por jugada")
    ap.add_argument("--azar-inicial", type=int, default=10, help="máx. medias jugadas al azar al inicio")
    ap.add_argument("--eps", type=float, default=0.05, help="prob. de jugada al azar después")
    ap.add_argument("--pesos", default=None, help="pesos.json con los que juega el bot")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    pesos = None
    if a.pesos:
        with open(a.pesos, encoding="utf-8") as f:
            pesos = json.load(f)["pesos"]
    tareas = [(a.semilla * 100003 + i, a.seg, a.azar_inicial, a.eps) for i in range(a.partidas)]
    # Compilar todo aquí, una vez: así los procesos solo leen la caché de numba (escribirla
    # desde varios procesos a la vez puede fallar).
    import alfabeta as ab
    from motor import Estado
    e0 = Estado.inicial()
    ab.caracteristicas(e0._tablero, e0._puntos, e0._color, np.zeros(40, np.int64),
                       np.zeros(ab.NF, np.int64))
    ab.calentar()
    t0 = time.time()
    X, Y, G = [], [], []
    with Pool(a.procs, initializer=_init, initargs=(pesos,)) as p:
        for g, (filas, y) in enumerate(p.imap_unordered(una_partida, tareas, chunksize=2)):
            X.extend(filas)
            Y.extend(y)
            G.extend([g] * len(y))
            if (g + 1) % 50 == 0:
                print(f"  {g + 1}/{a.partidas} partidas · {len(Y)} posiciones · {time.time() - t0:.0f} s",
                      flush=True)
    np.savez_compressed(a.salida, X=np.array(X, np.int64), y=np.array(Y, np.int8),
                        g=np.array(G, np.int32))
    print(f"Listo: {len(Y)} posiciones de {a.partidas} partidas en {a.salida} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
