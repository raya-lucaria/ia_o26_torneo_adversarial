"""Genera posiciones etiquetadas para ajustar los pesos de la evaluación (no corre en la liga).

El bot juega contra sí mismo con pocos nodos por jugada. De cada posición guardamos sus
rasgos (vistos desde quien está en turno) y si ese jugador terminó ganando la partida.
Para variar las partidas, las primeras jugadas son al azar y de vez en cuando se juega una
jugada al azar (esas posiciones sí se guardan; la jugada al azar solo cambia el rumbo).

Uso (desde la raíz del repo; escribe en resultados/, que no se sube):
    docker run --rm --cpus=8 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/generar_partidas.py \
        --partidas 3000 --nodos 40000 --salida /torneo/resultados/entrenamiento/partidas.npz
"""

import argparse
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import busqueda as bq  # noqa: E402

JUGADAS_AL_AZAR = 4  # plies iniciales al azar
PROB_AZAR = 0.03  # probabilidad de una jugada al azar en el resto de la partida


def elegir(ctx, tab, pts, color, nodos):
    bq.nueva_jugada(ctx)
    mejor, gastados = -1, 0
    for prof in range(1, 41):
        limite = max(nodos - gastados, 1000) if prof > 1 else 1 << 62
        jugada, v, completas, n, abortada = bq.buscar(ctx, tab, pts, color, prof, mejor, limite)
        gastados += n
        if abortada:
            if completas > 0:
                mejor = jugada
            break
        mejor = jugada
        if abs(v) > 0.95 or gastados >= nodos:
            break
    return mejor


def partida(args):
    semilla, nodos = args
    rng = random.Random(semilla)
    ctx = bq.nuevo_contexto()
    tab = np.zeros(81, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    origenes = np.zeros((1, bq.MAX_JUGADAS), np.int64)
    destinos = np.zeros((1, bq.MAX_JUGADAS), np.int64)
    filas, turnos = [], []
    color = 0
    for ply in range(400):
        if pts[1 - color] >= 3:
            ganador = 1 - color
            break
        n = bq.generar(tab, color, 0, origenes, destinos)
        if n == 0:
            ganador = 1 - color
            break
        x = np.empty(bq.NUM_RASGOS)
        bq.rasgos(tab, pts, color, x, ctx[0][6], ctx[0][7])
        filas.append(x)
        turnos.append(color)
        if ply < JUGADAS_AL_AZAR or rng.random() < PROB_AZAR:
            k = rng.randrange(n)
            jugada = origenes[0, k] * 81 + destinos[0, k]
        else:
            jugada = elegir(ctx, tab, pts, color, nodos)
        o, d = jugada // 81, jugada % 81
        gana = True
        if tab[d] != 0:
            gana = rng.random() < bq.prob_ataque(tab, color, o, d)
        bq.hacer(tab, pts, color, o, d, gana, np.uint64(0))
        color = 1 - color
    y = np.array([1.0 if t == ganador else 0.0 for t in turnos])
    return np.array(filas), y, np.full(len(y), semilla)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidas", type=int, default=3000)
    ap.add_argument("--nodos", type=int, default=40_000)
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()
    trabajos = [(args.semilla + k, args.nodos) for k in range(args.partidas)]
    xs, ys, ids = [], [], []
    with Pool(args.procesos) as pool:
        for i, (x, y, g) in enumerate(pool.imap_unordered(partida, trabajos)):
            xs.append(x)
            ys.append(y)
            ids.append(g)
            if (i + 1) % 200 == 0:
                print(f"{i + 1} partidas", flush=True)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida, x=np.concatenate(xs), y=np.concatenate(ys),
                        partida=np.concatenate(ids))
    print(f"{sum(len(y) for y in ys)} posiciones de {args.partidas} partidas → {salida}")


if __name__ == "__main__":
    main()
