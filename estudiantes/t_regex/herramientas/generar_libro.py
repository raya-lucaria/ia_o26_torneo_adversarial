"""Genera el libro de aperturas data/libro.npz (no corre en la liga).

El libro guarda, para posiciones de apertura, la jugada que elige una búsqueda mucho más
profunda que la que cabe en el reloj de la partida. Se llave con el hash Zobrist del estado
(el mismo que `estado.clave`).

Árbol que cubrimos, para cada color que podamos llevar:
- En nuestro turno: buscamos a fondo y solo seguimos la jugada elegida.
- En el turno rival: en su primera jugada, todas; después, la que elegiría una búsqueda
  profunda (un rival fuerte, o nuestra propia versión) y las K que una búsqueda corta
  considera mejores para él.
Las aperturas no tienen ataques hasta que los peones se encuentran; si hay un ataque en el
camino, seguimos el resultado «gana» y el «pierde».

Uso (desde la raíz del repo; con todos los núcleos, tarda del orden de una hora):
    docker run --rm --cpus=8 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/generar_libro.py --salida data/libro.npz
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import busqueda as bq  # noqa: E402

NODOS_NUESTRA = 5_000_000  # búsqueda de la jugada del libro (~6 s, profundidad ~9)
NODOS_RIVAL = 400_000  # búsqueda para elegir las K jugadas probables del rival
K_RIVAL = 3  # más la jugada de la búsqueda profunda: 4 por turno rival
PLIES_MAX = 8  # hasta qué ply de la partida cubrimos


def buscar_profundo(tab, pts, color, nodos):
    """Profundización iterativa hasta `nodos`. Regresa (jugada, valor, profundidad)."""
    ctx = bq.nuevo_contexto()
    mejor, valor, prof_hecha, gastados = -1, 0.0, 0, 0
    for prof in range(1, 41):
        limite = max(nodos - gastados, 1000) if prof > 1 else 1 << 62
        jugada, v, completas, n, abortada = bq.buscar(ctx, tab, pts, color, prof, mejor, limite)
        gastados += n
        if abortada:
            if completas > 0:
                mejor = jugada
            break
        mejor, valor, prof_hecha = jugada, v, prof
        if abs(v) > 0.95 or gastados >= nodos:
            break
    return mejor, valor, prof_hecha


def tarea_nuestra(args):
    tab, pts, color = args
    return buscar_profundo(tab, pts, color, NODOS_NUESTRA)


def tarea_rival(args):
    """Jugadas probables del rival: la que elige una búsqueda profunda (lo que haría un rival
    fuerte, o nuestro propio bot con ese color) y las K mejores según una búsqueda corta."""
    tab, pts, color = args
    profunda, _, _ = buscar_profundo(tab, pts, color, NODOS_NUESTRA)
    origenes = np.zeros((1, bq.MAX_JUGADAS), np.int64)
    destinos = np.zeros((1, bq.MAX_JUGADAS), np.int64)
    n = bq.generar(tab, color, 0, origenes, destinos)
    todas = [int(origenes[0, i] * 81 + destinos[0, i]) for i in range(n)]
    # Valor de cada jugada desde el rival: búsqueda del hijo con presupuesto repartido.
    valores = []
    for j in todas:
        o, d = j // 81, j % 81
        t2, p2 = tab.copy(), pts.copy()
        bq.hacer(t2, p2, color, o, d, True, np.uint64(0))
        _, v, _ = buscar_profundo(t2, p2, 1 - color, NODOS_RIVAL // max(n, 1))
        valores.append((-v, j))
    valores.sort(reverse=True)
    elegidas = [profunda] + [j for _, j in valores if j != profunda]
    return elegidas[:K_RIVAL + 1]


def hijos(tab, pts, color, jugada):
    """Posiciones que resultan de la jugada (dos si es ataque)."""
    o, d = jugada // 81, jugada % 81
    resultados = [True, False] if tab[d] != 0 else [True]
    for gana in resultados:
        t2, p2 = tab.copy(), pts.copy()
        bq.hacer(t2, p2, color, o, d, gana, np.uint64(0))
        yield t2, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--plies", type=int, default=PLIES_MAX)
    ap.add_argument("--procesos", type=int, default=8)
    args = ap.parse_args()

    tab0 = np.zeros(81, np.int8)
    tab0[:18] = 1
    tab0[-18:] = 2
    pts0 = np.zeros(2, np.int64)
    libro = {}  # clave → (jugada, valor, profundidad)
    # Frontera: (tab, pts, color en turno, ply, ¿es nuestro turno?) para cada color nuestro.
    frontera = [(tab0, pts0, 0, 0, True), (tab0, pts0, 0, 0, False)]
    inicio = time.time()
    with Pool(args.procesos) as pool:
        while frontera:
            nuestras = [f for f in frontera if f[4]]
            rivales = [f for f in frontera if not f[4]]
            # Quitamos repetidas (transposiciones) y las que ya están en el libro.
            vistas = {}
            for f in nuestras:
                h = int(bq.hash_completo(f[0], f[2], f[1]))
                if h not in libro and h not in vistas:
                    vistas[h] = f
            nuestras = list(vistas.values())
            nueva = []
            resultados = pool.map(tarea_nuestra, [(f[0], f[1], f[2]) for f in nuestras])
            for f, (jugada, valor, prof) in zip(nuestras, resultados, strict=True):
                tab, pts, color, ply, _ = f
                libro[int(bq.hash_completo(tab, color, pts))] = (jugada, valor, prof)
                if ply + 1 < args.plies:
                    for t2, p2 in hijos(tab, pts, color, jugada):
                        nueva.append((t2, p2, 1 - color, ply + 1, False))
            elegidas = pool.map(tarea_rival, [(f[0], f[1], f[2]) for f in rivales
                                              if f[3] > 0])
            k = 0
            for f in rivales:
                tab, pts, color, ply, _ = f
                if ply == 0:  # primera jugada del rival: todas
                    o = np.zeros((1, bq.MAX_JUGADAS), np.int64)
                    d = np.zeros((1, bq.MAX_JUGADAS), np.int64)
                    n = bq.generar(tab, color, 0, o, d)
                    jugadas = [int(o[0, i] * 81 + d[0, i]) for i in range(n)]
                else:
                    jugadas = elegidas[k]
                    k += 1
                if ply + 1 < args.plies:
                    for j in jugadas:
                        for t2, p2 in hijos(tab, pts, color, j):
                            nueva.append((t2, p2, 1 - color, ply + 1, True))
            frontera = nueva
            print(f"{len(libro)} posiciones en el libro, frontera {len(frontera)}, "
                  f"{(time.time() - inicio) / 60:.1f} min", flush=True)

    claves = np.array(list(libro.keys()), np.uint64)
    jugadas = np.array([v[0] for v in libro.values()], np.int16)
    valores = np.array([v[1] for v in libro.values()])
    profs = np.array([v[2] for v in libro.values()], np.int8)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.salida, claves=claves, jugadas=jugadas, valores=valores,
                        profundidades=profs)
    print(f"{len(claves)} posiciones → {args.salida}  (profundidad media {profs.mean():.1f})")


if __name__ == "__main__":
    main()
