"""Genera datos de entrenamiento para la evaluación por autojuego (offline, no corre en la liga).

Juega partidas del bot contra sí mismo con búsqueda a profundidad fija y, en cada posición,
guarda los rasgos de `strategy.rasgos` (desde la perspectiva de quien mueve) y la etiqueta
1 si quien mueve terminó ganando. Los dados se tiran con `random` (p del motor).

Uso (desde estudiantes/<equipo>/, con el .venv del torneo activo):
    python herramientas/autojuego.py datos_r2 --partidas 3000 --procs 4 --prof 3 --semilla 31

Escribe datos_r2_w<i>.npz (uno por proceso, se va guardando cada 100 partidas).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARPETA))


def trabajador(args):
    idx, n_partidas, prof, eps, semilla, salida = args
    import strategy as bot  # el mismo motor y la misma evaluación que juegan en la liga

    bot._nuevo_contexto()
    rng = random.Random(semilla * 1000 + idx)
    nf = bot.NFEAT_USADOS
    f = np.zeros(bot.NFEAT, np.float64)
    B = np.zeros(bot.B_TAM, np.int64)
    X, Y, G = [], [], []
    t0 = time.time()
    for g in range(n_partidas):
        tab = np.zeros(81, np.int8)
        tab[:18] = 1
        tab[63:] = 2
        pts = np.zeros(2, np.int64)
        lado, ply, ganador = 0, 0, -1
        aleatorias = rng.randint(0, 10)       # aperturas variadas
        filas, lados = [], []
        bot.C.tt_k[:] = 0
        bot.C.hist[:] = 0
        while True:
            n = bot.generar(tab, lado, bot.C.mo, bot.C.md, 0)
            if n == 0:
                ganador = 1 - lado
                break
            bot.rasgos(tab, pts, lado, f, B)
            filas.append(f[:nf].copy())
            lados.append(lado)
            if ply < aleatorias or rng.random() < eps:
                k = rng.randrange(n)
                o, d = int(bot.C.mo[0, k]), int(bot.C.md[0, k])
            else:
                o, d, _, _, _ = bot._buscar(tab, pts, lado, 10.0, prof_max=prof)
            gana = True
            if tab[d] != 0:
                gana = rng.random() < bot.prob_ataque(tab, lado, o, d)
            bot.hacer(tab, pts, lado, o, d, gana, np.uint64(0), bot.C.zp, bot.C.zt, bot.C.zs)
            if pts[lado] >= 3:
                ganador = lado
                break
            lado = 1 - lado
            ply += 1
        for fila, l in zip(filas, lados):
            X.append(fila)
            Y.append(1.0 if l == ganador else 0.0)
            G.append(semilla * 100000 + idx * 10000 + g)
        if (g + 1) % 100 == 0 or g + 1 == n_partidas:
            np.savez_compressed(f"{salida}_w{idx}.npz", X=np.array(X), Y=np.array(Y), G=np.array(G))
            if idx == 0:
                print(f"{g + 1}/{n_partidas} partidas ({time.time() - t0:.0f} s)", flush=True)
    return len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("salida")
    ap.add_argument("--partidas", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--prof", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--semilla", type=int, default=1)
    a = ap.parse_args()
    por = a.partidas // a.procs
    with mp.get_context("spawn").Pool(a.procs) as pool:
        n = pool.map(trabajador, [(i, por, a.prof, a.eps, a.semilla, a.salida) for i in range(a.procs)])
    print(f"{sum(n)} posiciones guardadas en {a.salida}_w*.npz")


if __name__ == "__main__":
    main()
