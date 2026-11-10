"""Ajusta los pesos de la evaluación (Texel): regresión logística sobre posiciones de autojuego.

    python herramientas/ajustar.py --partidas 3000 --nodos 3000 --rondas 3 --procesos 10

Cada ronda juega partidas con los pesos actuales, etiqueta cada posición con quién ganó y reajusta.
El resultado queda en data/pesos.npy. Los rasgos son blancas menos negras, así que no hay intercepto.
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

CARPETA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARPETA))
import mi_motor  # noqa: E402

JUGADAS_AL_AZAR = 10  # plies iniciales al azar, para variar las partidas
EPSILON = 0.05  # probabilidad de jugada al azar después, para ver posiciones menos pulidas


def _inicial(tab, pts):
    tab[:] = 0
    tab[:18] = 1
    tab[-18:] = 2
    pts[:] = 0


def jugar_partida(buf, nodos, rng):
    """Una partida de autojuego. Regresa (rasgos por posición, 1 si ganaron blancas)."""
    tab, pts, mo, md, f, mascaras, ctx, Z = buf[0], buf[1], buf[2], buf[3], buf[10], buf[11], buf[9], buf[13]
    _inicial(tab, pts)
    buf[5][:] = 0  # TT limpia: cada partida es independiente
    turno, ply, filas = 0, 0, []
    while True:
        n = mi_motor.generar(tab, turno, mo, md, 90)
        if n == 0:
            return filas, 1 - turno == 0
        if ply >= JUGADAS_AL_AZAR:
            mi_motor.rasgos(tab, pts, turno, f, mascaras)
            filas.append(f.copy())
        h = int(mi_motor.clave_de(tab, pts, turno, Z))
        if ply < JUGADAS_AL_AZAR or rng.random() < EPSILON:
            i = rng.integers(n)
            o, d = int(mo[90, i]), int(md[90, i])
        else:
            o, d = -1, -1
            for prof in range(1, 30):
                ctx[:] = 0
                ctx[1] = nodos
                mi_motor.buscar_raiz(buf, turno, h, prof)
                if ctx[4] > 0:
                    o, d = int(ctx[3]) // 81, int(ctx[3]) % 81
                if ctx[2]:
                    break
        gana = True
        if tab[d] != 0:
            gana = rng.random() < mi_motor.prob_ataque(tab, turno, o, d)
        mi_motor.hacer(tab, pts, turno, o, d, gana, h, Z)
        if pts[turno] >= mi_motor.PUNTOS_PARA_GANAR:
            return filas, turno == 0
        turno = 1 - turno
        ply += 1


def _trabajo(args):
    pesos, cuantas, nodos, semilla = args
    buf = mi_motor.crear_buffers(pesos, bits_tt=18)
    rng = np.random.default_rng(semilla)
    X, y = [], []
    for _ in range(cuantas):
        filas, blancas = jugar_partida(buf, nodos, rng)
        X.extend(filas)
        y.extend([float(blancas)] * len(filas))
    return np.array(X), np.array(y)


def ajustar(X, y, inicio, l2):
    def costo(w):
        s = X @ w
        perdida = np.logaddexp(0, -s) * y + np.logaddexp(0, s) * (1 - y)
        p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        return perdida.mean() + l2 * (w @ w), X.T @ (p - y) / len(y) + 2 * l2 * w

    r = minimize(costo, inicio, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x, r.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidas", type=int, default=2000)
    ap.add_argument("--nodos", type=int, default=3000)
    ap.add_argument("--rondas", type=int, default=3)
    ap.add_argument("--procesos", type=int, default=10)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--salida", default=str(CARPETA / "data" / "pesos.npy"))
    args = ap.parse_args()
    salida = Path(args.salida)
    pesos = np.load(salida) if salida.exists() else mi_motor.PESOS_INICIALES.copy()
    if len(pesos) != mi_motor.N_RASGOS:
        pesos = mi_motor.PESOS_INICIALES.copy()
    with mp.Pool(args.procesos) as pool:
        for ronda in range(args.rondas):
            trabajos = [
                (pesos, args.partidas // args.procesos, args.nodos, [ronda, i, np.random.SeedSequence().entropy % 2**32])
                for i in range(args.procesos)
            ]
            partes = pool.map(_trabajo, trabajos)
            X = np.vstack([p[0] for p in partes])
            y = np.concatenate([p[1] for p in partes])
            pesos, costo = ajustar(X, y, pesos, args.l2)
            np.save(salida, pesos)
            print(f"ronda {ronda}: {len(y)} posiciones · blancas ganan {y.mean():.3f} · costo {costo:.4f}", flush=True)
            print(np.round(pesos, 3).tolist(), flush=True)


if __name__ == "__main__":
    main()
