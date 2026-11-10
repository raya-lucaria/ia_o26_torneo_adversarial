"""Genera partidas de autojuego para ajustar la evaluación (método de Texel). Offline.

Cada partida: apertura al azar de `--apertura` plies (sin ataques), luego nuestro motor con
`--nodos` nodos por jugada contra sí mismo, con la evaluación lineal y los pesos de `--pesos`
(por omisión data/pesos.npy). Con probabilidad `--epsilon`, una jugada al azar en las primeras 20
jugadas después de la apertura, para que haya variedad. Dados propios (np.random.default_rng).
Guarda cada posición (bitboards, puntos, turno, ply) con el resultado final de su partida en un
.npz.

Uso (desde la raíz del repo):
  python estudiantes/torneo_breakthrough/herramientas/generar_partidas.py --partidas 10000 \\
      --salida laboratorio/resultados/texel/partidas_v2.npz
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))


def jugar_bloque(args):
    inicio, cuantas, nodos, plies_apertura, epsilon, semilla, pesos = args
    import numpy as np

    import tablero_bits as tb
    from strategy import Motor

    m = Motor(np.load(pesos))
    lista = np.zeros(64, np.int64)
    filas = {"bb": [], "pts": [], "turno": [], "ply": [], "partida": [], "resultado": []}
    for g in range(inicio, inicio + cuantas):
        rng = np.random.default_rng([semilla, g])
        bb = np.zeros((2, tb.LADO), np.int64)
        bb[0, 0] = bb[0, 1] = bb[1, 7] = bb[1, 8] = tb.MASCARA
        pts = np.zeros(2, np.int64)
        c = 0
        m.tt_k[:] = 0
        m.historia[:] = 0
        posiciones = []
        ganador = -1
        for ply in range(400):
            n = tb.generar(bb, c, lista)
            if n == 0:
                ganador = 1 - c
                break
            if ply < plies_apertura:
                quietas = [int(x) for x in lista[:n] if not tb.ocupada(bb, 1 - c, int(x) % 81)]
                jug = quietas[int(rng.integers(len(quietas)))] if quietas else int(lista[0])
            elif ply < plies_apertura + 40 and rng.random() < epsilon:
                jug = int(lista[int(rng.integers(n))])
            else:
                posiciones.append((bb.copy(), pts.copy(), c, ply))
                jug, _, _, _ = m.pensar(bb, pts, c, 0.0, max_nodos=nodos)
                if jug < 0:
                    jug = int(lista[0])
            o, d = divmod(jug, 81)
            gana = True
            if tb.ocupada(bb, 1 - c, d):
                gana = bool(rng.random() < tb.prob(bb, c, o, d))
            tb.hacer(bb, pts, c, o, d, gana, np.uint64(0))
            if pts[c] >= 3:
                ganador = c
                break
            c = 1 - c
        for b, p, t, ply in posiciones:
            filas["bb"].append(b.astype(np.int16))
            filas["pts"].append(p.astype(np.int8))
            filas["turno"].append(t)
            filas["ply"].append(ply)
            filas["partida"].append(g)
            filas["resultado"].append(1 if ganador == 0 else 0)  # 1 = ganan Blancas
    return {k: np.array(v) for k, v in filas.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partidas", type=int, default=2000)
    ap.add_argument("--nodos", type=int, default=60_000)
    ap.add_argument("--apertura", type=int, default=6)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--procesos", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--bloque", type=int, default=20)
    ap.add_argument("--pesos", default=str(BOT / "data" / "pesos.npy"), help="pesos de la evaluación")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    import numpy as np

    trabajo = [(i, min(a.bloque, a.partidas - i), a.nodos, a.apertura, a.epsilon, a.semilla,
                a.pesos) for i in range(0, a.partidas, a.bloque)]
    t = time.time()
    partes = []
    with mp.get_context("spawn").Pool(a.procesos) as pool:
        for k, r in enumerate(pool.imap_unordered(jugar_bloque, trabajo), 1):
            partes.append(r)
            if k % 10 == 0:
                print(f"  {k * a.bloque}/{a.partidas} partidas, {time.time() - t:.0f}s", flush=True)
    datos = {k: np.concatenate([p[k] for p in partes]) for k in partes[0]}
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.salida, **datos)
    n = len(datos["resultado"])
    blancas = 100 * datos["resultado"].mean()
    print(f"{a.partidas} partidas, {n} posiciones, gana Blancas {blancas:.1f}% de las posiciones, "
          f"{time.time() - t:.0f}s → {a.salida}")


if __name__ == "__main__":
    main()
