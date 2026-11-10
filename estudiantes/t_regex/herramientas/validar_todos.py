"""Compara un candidato (cambios de constantes) contra el bot actual frente a TODOS los rivales.

Para no sobreajustar a un solo rival, el candidato y el bot actual juegan contra cada versión
vieja (v1…v8) y contra montecarlo, y entre una y otra contra random (que no se nos olvide
ganarle). Los dos juegan exactamente las mismas partidas (mismos dados, mismos colores), así
que la diferencia se mide en pares y tiene menos ruido. Al final, también uno contra otro.

Uso (desde estudiantes/t_regex, en la imagen de bots):
    python herramientas/validar_todos.py --pa QS_PROF=4 --pares 100
"""

import argparse
import sys

import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arena  # noqa: E402

VERSIONES = [f"../t_regex_v{k}" for k in range(1, 9)] + ["montecarlo:1700"]


PLIES_LIBRO = 100  # posiciones para el libro: solo antes de este ply


def posiciones_nuestras(historia, color_nuestro):
    """(clave, tablero, puntos, color, ply) de las posiciones donde nos tocaba mover."""
    mod = arena.cargar(".")
    tab = np.zeros(81, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    color, vistas = 0, []
    for ply, (jugada, gana) in enumerate(historia[:PLIES_LIBRO]):
        if color == color_nuestro:
            vistas.append((int(mod.hash_completo(tab, color, pts)), tab.copy(), pts.copy(),
                           color, ply))
        mod.hacer(tab, pts, color, jugada // 81, jugada % 81, gana, np.uint64(0))
        color = 1 - color
    return vistas


def jugar(trabajo):
    etiqueta, quien, a, b, semilla, nodos, a_es_blancas, guardar = trabajo
    blancas, negras = (a, b) if a_es_blancas else (b, a)
    ganador, _, _, historia = arena.partida((blancas, negras, semilla, nodos))
    vistas = []
    if guardar:  # en el duelo directo, los dos lados somos nosotros
        colores = (0, 1) if etiqueta == "directo" else (0 if a_es_blancas else 1,)
        for c in colores:
            vistas += posiciones_nuestras(historia, c)
    return etiqueta, quien, semilla, a_es_blancas, (ganador == 0) == a_es_blancas, vistas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", required=True, help="cambios del candidato: QS_PROF=4,QS_PMIN=0.5")
    ap.add_argument("--base", default="", help="cambios del bot de referencia (default: actual)")
    ap.add_argument("--pares", type=int, default=100, help="juegos de dados por versión")
    ap.add_argument("--pares-random", type=int, default=25)
    ap.add_argument("--pares-directo", type=int, default=100, help="candidato contra actual")
    ap.add_argument("--nodos", type=int, default=100_000)
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=50_000, help="otra = dados nunca vistos")
    ap.add_argument("--posiciones", help="npz donde guardar nuestras posiciones (para el libro)")
    args = ap.parse_args()
    cand, base = (".", args.pa), (".", args.base)
    trabajos = []
    semilla = args.semilla
    orden = []
    for v in VERSIONES:  # cada versión, y luego random
        for rival, pares in ((v, args.pares), ("random", args.pares_random)):
            etiqueta = rival if rival != "random" else f"random (tras {Path(v).name})"
            orden.append(etiqueta)
            for k in range(pares):
                for a_es_blancas in (True, False):
                    for quien, yo in (("cand", cand), ("base", base)):
                        trabajos.append((etiqueta, quien, yo, (rival, ""), semilla + k,
                                         args.nodos, a_es_blancas, bool(args.posiciones)))
            semilla += pares
    for k in range(args.pares_directo):
        for a_es_blancas in (True, False):
            trabajos.append(("directo", "cand", cand, base, semilla + k, args.nodos,
                             a_es_blancas, bool(args.posiciones)))

    res = defaultdict(dict)  # etiqueta -> {(semilla, color): {quien: ganó}}
    todas = []
    with Pool(args.procesos) as pool:
        for etiqueta, quien, s, color, gano, vistas in pool.imap_unordered(jugar, trabajos,
                                                                          chunksize=2):
            res[etiqueta].setdefault((s, color), {})[quien] = gano
            todas += vistas
    if args.posiciones:
        np.savez_compressed(args.posiciones,
                            claves=np.array([v[0] for v in todas], np.uint64),
                            tableros=np.array([v[1] for v in todas], np.int8),
                            puntos=np.array([v[2] for v in todas], np.int64),
                            colores=np.array([v[3] for v in todas], np.int8),
                            plies=np.array([v[4] for v in todas], np.int16))
        print(f"{len(todas)} posiciones nuestras → {args.posiciones}")

    print(f"candidato: {args.pa}   referencia: {args.base or 'bot actual'}   "
          f"nodos/jugada: {args.nodos}")
    tot = defaultdict(lambda: [0, 0, 0])  # grupo -> [gana cand, gana base, partidas]
    for etiqueta in orden:
        pares = list(res[etiqueta].values())
        c = sum(p["cand"] for p in pares)
        b = sum(p["base"] for p in pares)
        n = len(pares)
        print(f"  {etiqueta:28s} candidato {c / n:6.1%}   actual {b / n:6.1%}   "
              f"({c - b:+d} de {n})")
        grupo = "random" if etiqueta.startswith("random") else "versiones+montecarlo"
        for i, x in enumerate((c, b, n)):
            tot[grupo][i] += x
    for grupo, (c, b, n) in tot.items():
        # Diferencia pareada: las dos juegan las mismas partidas.
        difs = [p["cand"] - p["base"] for e in orden
                if e.startswith("random") == (grupo == "random") for p in res[e].values()]
        media = sum(difs) / n
        var = sum((d - media) ** 2 for d in difs) / max(n - 1, 1)
        error = 1.96 * (var / n) ** 0.5
        print(f"TOTAL {grupo:22s} candidato {c / n:6.1%}   actual {b / n:6.1%}   "
              f"diferencia {media:+.1%} ± {error:.1%} (95 %)")
    d = list(res["directo"].values())
    if d:
        g = sum(p["cand"] for p in d)
        tasa = g / len(d)
        print(f"directo: candidato gana {g}/{len(d)} = {tasa:.1%} ± "
              f"{1.96 * (tasa * (1 - tasa) / len(d)) ** 0.5:.1%} contra el actual")


if __name__ == "__main__":
    main()
