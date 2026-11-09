"""Puebla el libro (data/libro.npz) explorando el árbol del juego al azar, sin rivales.

Por rondas:
1. Recorre el árbol al azar hasta una hoja (una partida terminada): en cada turno, de
   nosotros o del rival, una jugada legal al azar y, si es ataque, los dados al azar. La mitad
   de los recorridos sale de la posición inicial y la otra mitad de una posición de un
   recorrido anterior, tomando otra jugada al azar: así el árbol se abre en ramas nuevas en vez
   de repetir caminos sueltos.
2. De cada recorrido se queda con las posiciones que importan para ganar:
   - las últimas `--finales` antes de la hoja: ahí se decide la partida (hay una jugada que
     gana o una que evita que el otro gane);
   - las tempranas (antes de `--temprano`), las que más se parecen a una partida real.
3. A cada una le busca la mejor jugada con una búsqueda profunda (`--nodos-libro`). La
   búsqueda llega hasta las hojas cuando la partida está cerca de acabar: si encuentra una
   victoria segura se detiene y esa es la jugada que se guarda; si el rival amenaza ganar,
   la mejor jugada es la que lo evita.
4. Guarda el libro cada 15 minutos y al final de cada ronda, con un respaldo por hora.

Uso (desde la raíz del repo; corre durante horas):
    docker run --rm --cpus=8 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/explorar_arbol.py --rondas 1000
"""

import argparse
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import busqueda as bq  # noqa: E402
from ampliar_libro import ARCHIVO, GUARDAR_CADA_S, cargar_libro, guardar_libro, respaldar  # noqa: E402
from generar_libro import buscar_profundo  # noqa: E402

GANA = 0.95  # |valor| por encima de esto: la búsqueda encontró una victoria (o derrota) segura


def recorrido(semilla, inicio=None):
    """Un recorrido al azar hasta una hoja, desde `inicio` = (tab, pts, color, ply) o desde la
    posición inicial. Regresa (posiciones, ganador): cada posición es
    (clave, tab, pts, color, ply, jugada_al_azar)."""
    rng = random.Random(semilla)
    if inicio is None:
        tab = np.zeros(81, np.int8)
        tab[:18] = 1
        tab[-18:] = 2
        pts = np.zeros(2, np.int64)
        color, primero = 0, 0
    else:
        tab, pts, color, primero = inicio[0].copy(), inicio[1].copy(), inicio[2], inicio[3]
    origenes = np.zeros((1, 54), np.int64)
    destinos = np.zeros((1, 54), np.int64)
    posiciones = []
    for ply in range(primero, primero + 400):
        if pts[1 - color] >= 3:
            return posiciones, 1 - color
        n = bq.generar(tab, color, 0, origenes, destinos)
        if n == 0:
            return posiciones, 1 - color
        i = rng.randrange(n)
        o, d = int(origenes[0, i]), int(destinos[0, i])
        clave = int(bq.hash_completo(tab, color, pts))
        posiciones.append((clave, tab.copy(), pts.copy(), color, ply, o * 81 + d))
        gana = True
        if tab[d] != 0:
            gana = rng.random() < bq.prob_ataque(tab, color, o, d)
        bq.hacer(tab, pts, color, o, d, gana, np.uint64(0))
        color = 1 - color
    raise RuntimeError("recorrido demasiado largo")


def explorar(args):
    """Un recorrido; regresa (elegidas, ramas). Elegidas: (clave, tab, pts, color, ply, defensa),
    con `defensa` = la jugada al azar dejó que el rival ganara en su siguiente turno. Ramas:
    unas posiciones del camino de donde pueden salir recorridos futuros."""
    semilla, inicio, finales, temprano = args
    posiciones, ganador = recorrido(semilla, inicio)
    total = len(posiciones)
    elegidas = []
    for k, (clave, tab, pts, color, ply, _) in enumerate(posiciones):
        if ply < temprano or k >= total - finales:
            defensa = color != ganador and k >= total - 2
            elegidas.append((clave, tab, pts, color, ply, defensa))
    rng = random.Random(semilla + 1)
    ramas = [p[1:5] for p in rng.sample(posiciones, min(4, total))]
    return elegidas, ramas


def calcular(args):
    tab, pts, color, nodos = args
    jugada, valor, _ = buscar_profundo(tab, pts, color, nodos)
    return jugada, valor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rondas", type=int, default=10)
    ap.add_argument("--recorridos", type=int, default=2000, help="recorridos al azar por ronda")
    ap.add_argument("--finales", type=int, default=8, help="posiciones antes de la hoja")
    ap.add_argument("--temprano", type=int, default=12, help="y todas las de antes de este ply")
    ap.add_argument("--nodos-libro", type=int, default=2_000_000)
    ap.add_argument("--procesos", type=int, default=8)
    args = ap.parse_args()
    semilla = int(time.time()) % 1_000_000 * 1000
    rng = random.Random(semilla)
    ramas = []  # posiciones de recorridos anteriores desde donde abrir ramas nuevas
    for ronda in range(args.rondas):
        libro = cargar_libro()
        inicio = time.time()
        muestra, defensas = {}, set()
        vistas = ya_estaban = 0
        trabajos = [(semilla + k, rng.choice(ramas) if ramas and rng.random() < 0.5 else None,
                     args.finales, args.temprano) for k in range(args.recorridos)]
        semilla += args.recorridos
        nuevas_ramas = []
        with Pool(args.procesos) as pool:
            for elegidas, de_aqui in pool.imap_unordered(explorar, trabajos, chunksize=16):
                nuevas_ramas += de_aqui
                for clave, tab, pts, color, ply, defensa in elegidas:
                    vistas += 1
                    if clave in libro:
                        ya_estaban += 1
                        continue
                    muestra.setdefault(clave, (tab, pts, color))
                    if defensa:
                        defensas.add(clave)
        ramas = (ramas + nuevas_ramas)[-50_000:]
        claves = list(muestra)
        trabajos = [(*muestra[c], args.nodos_libro) for c in claves]
        ganadoras = perdidas = 0
        ultimo = time.time()
        with Pool(args.procesos) as pool:
            for c, (jugada, valor) in zip(claves, pool.imap(calcular, trabajos, chunksize=4),
                                          strict=True):
                libro[c] = jugada
                ganadoras += valor > GANA
                perdidas += valor < -GANA
                if time.time() - ultimo > GUARDAR_CADA_S:
                    guardar_libro(libro)
                    respaldar()
                    ultimo = time.time()
                    print(f"  guardado parcial: libro {len(libro)}", flush=True)
        guardar_libro(libro)
        respaldar()
        print(f"ronda {ronda + 1}: {args.recorridos} recorridos, {vistas} posiciones elegidas "
              f"({ya_estaban} ya en el libro), {len(claves)} agregadas: {ganadoras} con victoria "
              f"segura, {len(defensas)} donde había que evitar que el otro ganara, {perdidas} "
              f"perdidas pase lo que pase → libro {len(libro)} "
              f"({ARCHIVO.stat().st_size / 1e6:.1f} MB), {(time.time() - inicio) / 60:.0f} min",
              flush=True)


if __name__ == "__main__":
    main()
