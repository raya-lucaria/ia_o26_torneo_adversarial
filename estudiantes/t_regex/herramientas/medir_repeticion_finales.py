"""¿Se repiten las posiciones de final entre partidas? (no corre en la liga)

Juega dos grupos de partidas con semillas distintas (nuestro bot contra montecarlo, random y
versiones anteriores) y guarda el hash de cada posición con pocos peones. Cuenta qué fracción
de las posiciones del grupo B ya aparecían en el grupo A: es la tasa de aciertos que tendría
una tabla de finales construida con A al jugar partidas nuevas.

Uso (desde estudiantes/t_regex, dentro de la imagen de bots):
    python herramientas/medir_repeticion_finales.py --partidas 200 --peones 14
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arena  # noqa: E402

RIVALES = ["montecarlo:300", "random", "../t_regex_v3", "../t_regex_v5", "."]


def jugar(args):
    semilla, nodos, peones_max = args
    rival = RIVALES[semilla % len(RIVALES)]
    yo, otro = (".", ""), (rival, "")
    blancas, negras = (yo, otro) if semilla % 2 == 0 else (otro, yo)
    _, _, _, historia = arena.partida((blancas, negras, semilla, nodos))
    mod = arena.cargar(".")
    tab = np.zeros(81, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    color = 0
    claves = set()
    for jugada, gana in historia:
        if (tab != 0).sum() <= peones_max:
            claves.add(int(mod.hash_completo(tab, color, pts)))
        mod.hacer(tab, pts, color, jugada // 81, jugada % 81, gana, np.uint64(0))
        color = 1 - color
    return claves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidas", type=int, default=200)
    ap.add_argument("--peones", type=int, default=14, help="total de peones en el tablero")
    ap.add_argument("--nodos", type=int, default=30_000)
    args = ap.parse_args()
    grupos = []
    with Pool(8) as pool:
        for base in (0, 10**6):
            trabajos = [(base + k, args.nodos, args.peones) for k in range(args.partidas)]
            grupo = set()
            for c in pool.imap_unordered(jugar, trabajos):
                grupo |= c
            grupos.append(grupo)
    a, b = grupos
    comunes = len(a & b)
    print(f"posiciones con ≤ {args.peones} peones: grupo A {len(a)}, grupo B {len(b)}")
    print(f"de B ya estaban en A: {comunes} ({comunes / max(len(b), 1):.2%})")


if __name__ == "__main__":
    main()
