"""Agrega al libro (data/libro.npz) las posiciones de partidas que ya se jugaron.

Fuentes:
- `--posiciones`: archivos npz que guarda `validar_todos.py --posiciones` (arena).
- `--partidas`: carpetas de resultados del árbitro real (`./torneo jugar`); de cada partida
  toma las posiciones donde movía un bot nuestro (nombre que empieza con «t_regex»).

Como `ampliar_libro.py`, solo agrega las que no están, aparecen antes del ply 100 y se
repitieron (≥ 2 veces) o son tempranas: las que probablemente vuelvan a salir. Las busca a
fondo en paralelo y guarda el libro cada 15 minutos.

Uso (desde estudiantes/t_regex, en la imagen de bots):
    python herramientas/libro_de_partidas.py --posiciones /torneo/resultados/medio/pos.npz \
        --partidas /torneo/resultados/2026-11-01_00-30-00
"""

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import busqueda as bq  # noqa: E402
from ampliar_libro import ARCHIVO, GUARDAR_CADA_S, cargar_libro, guardar_libro, respaldar  # noqa: E402
from generar_libro import buscar_profundo  # noqa: E402
from motor import Jugada  # noqa: E402

PLIES = 100


def de_arena(archivo):
    with np.load(archivo) as d:
        yield from zip(d["claves"].tolist(), d["tableros"], d["puntos"], d["colores"].tolist(),
                       d["plies"].tolist(), strict=True)


def de_arbitro(carpeta):
    for archivo in sorted(Path(carpeta).glob("*.json")):
        partida = json.loads(archivo.read_text())
        nuestros = {c for c in (0, 1)
                    if partida["meta"][("blancas", "negras")[c]].startswith("t_regex")}
        tab = np.zeros(81, np.int8)
        tab[:18] = 1
        tab[-18:] = 2
        pts = np.zeros(2, np.int64)
        for ply, j in enumerate(partida["jugadas"][:PLIES]):
            color = 0 if j["color"] == "blancas" else 1
            if color in nuestros:
                yield int(bq.hash_completo(tab, color, pts)), tab.copy(), pts.copy(), color, ply
            jugada = Jugada.desde_texto(j["jugada"])
            gana = j["combate"]["gano"] if j["combate"] else True
            bq.hacer(tab, pts, color, jugada.origen, jugada.destino, gana, np.uint64(0))


def calcular(args):
    tab, pts, color, nodos = args
    return buscar_profundo(tab, pts, color, nodos)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posiciones", nargs="*", default=[])
    ap.add_argument("--partidas", nargs="*", default=[])
    ap.add_argument("--temprano", type=int, default=20)
    ap.add_argument("--nodos-libro", type=int, default=2_000_000)
    ap.add_argument("--procesos", type=int, default=8)
    args = ap.parse_args()
    inicio = time.time()
    libro = cargar_libro()
    conteo, muestra = Counter(), {}
    total = en_libro = 0
    fuentes = [de_arena(a) for a in args.posiciones] + [de_arbitro(c) for c in args.partidas]
    for fuente in fuentes:
        for clave, tab, pts, color, ply in fuente:
            total += 1
            if clave in libro:
                en_libro += 1
                continue
            conteo[clave] += 1
            if clave not in muestra or ply < muestra[clave][3]:
                muestra[clave] = (tab, pts, color, ply)
    elegidas = [c for c, n in conteo.items() if n >= 2 or muestra[c][3] < args.temprano]
    print(f"{total} posiciones nuestras ({en_libro} ya en el libro, "
          f"{en_libro / max(total, 1):.1%}); {len(conteo)} distintas nuevas, "
          f"{len(elegidas)} elegidas", flush=True)
    trabajos = [(*muestra[c][:3], args.nodos_libro) for c in elegidas]
    ultimo = time.time()
    with Pool(args.procesos) as pool:
        for c, jugada in zip(elegidas, pool.imap(calcular, trabajos, chunksize=4), strict=True):
            libro[c] = jugada
            if time.time() - ultimo > GUARDAR_CADA_S:
                guardar_libro(libro)
                respaldar()
                ultimo = time.time()
                print(f"  guardado parcial: libro {len(libro)}", flush=True)
    guardar_libro(libro)
    respaldar()
    print(f"{len(elegidas)} agregadas → libro {len(libro)} "
          f"({ARCHIVO.stat().st_size / 1e6:.1f} MB), {(time.time() - inicio) / 60:.0f} min",
          flush=True)


if __name__ == "__main__":
    main()
