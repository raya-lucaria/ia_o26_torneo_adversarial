"""Amplía el libro (data/libro.npz) con posiciones que de verdad aparecen en partidas.

El libro inicial (`generar_libro.py`) cubre un árbol fijo de aperturas. Este agrega, por
rondas, las posiciones a las que llegan las partidas reales:
1. Juega partidas de NUESTRO bot (con el libro actual) contra un grupo variado de rivales
   (versiones viejas, montecarlo, random) y anota cada posición en la que nos toca mover, que
   no está en el libro y que aparece antes del ply `--plies`.
2. Elige las que se repitieron (≥ 2 veces) o son tempranas: son las que probablemente vuelvan
   a aparecer en la liga. Las finales casi no se repiten (medido: 1.3 % con ≤ 24 peones).
3. Las busca a fondo (`--nodos-libro`, más profundo que en partida) en paralelo y las agrega.
4. Guarda el libro tras cada ronda (se puede interrumpir sin perder nada) y repite.

Uso (desde la raíz del repo; corre durante horas):
    docker run --rm --cpus=8 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/ampliar_libro.py --rondas 20
"""

import argparse
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arena  # noqa: E402
import busqueda as bq  # noqa: E402
from generar_libro import buscar_profundo  # noqa: E402

# Rivales variados: cada versión vieja tiene otra evaluación (y otras aperturas), montecarlo con
# distintas muestras va de razonable a errático, y random lleva a posiciones raras.
RIVALES = ["../t_regex_v1", "../t_regex_v2", "../t_regex_v3", "../t_regex_v4",
           "../t_regex_v5", "../t_regex_v6", "../t_regex_v7", "../t_regex_v8",
           "montecarlo:100", "montecarlo:500", "montecarlo:1700", "random", "."]
ARCHIVO = Path(__file__).resolve().parent.parent / "data" / "libro.npz"
RESPALDOS = Path("/torneo/resultados/respaldos_libro")
GUARDAR_CADA_S = 15 * 60  # durante el cálculo, guardamos el libro cada 15 minutos
RESPALDO_CADA_S = 60 * 60  # y una copia con fecha cada hora


def cargar_libro():
    if not ARCHIVO.exists():
        return {}
    with np.load(ARCHIVO) as d:
        return dict(zip(d["claves"].tolist(), d["jugadas"].tolist(), strict=True))


def guardar_libro(libro):
    claves = np.array(list(libro.keys()), np.uint64)
    jugadas = np.array(list(libro.values()), np.int16)
    tmp = ARCHIVO.with_name("libro_tmp.npz")
    np.savez_compressed(tmp, claves=claves, jugadas=jugadas)
    tmp.replace(ARCHIVO)  # reemplazo atómico: nunca queda un libro a medias


_ultimo_respaldo = [0.0]


def respaldar():
    """Copia con fecha del libro, una vez por hora (por si algo corrompe el principal)."""
    if time.time() - _ultimo_respaldo[0] < RESPALDO_CADA_S:
        return
    RESPALDOS.mkdir(parents=True, exist_ok=True)
    destino = RESPALDOS / f"libro_{time.strftime('%Y-%m-%d_%H-%M')}.npz"
    destino.write_bytes(ARCHIVO.read_bytes())
    _ultimo_respaldo[0] = time.time()
    print(f"  respaldo → {destino.name}", flush=True)


def recolectar(args):
    """Una partida contra un rival; regresa las posiciones (clave, tab, pts, color, ply)
    donde nos tocaba mover, antes de `plies`. Usa el libro que el bot cargó al importar."""
    semilla, nodos, plies = args
    rival = RIVALES[semilla % len(RIVALES)]
    yo, otro = (".", ""), (rival, "")
    nuestro_color = semilla // len(RIVALES) % 2
    blancas, negras = (yo, otro) if nuestro_color == 0 else (otro, yo)
    _, _, _, historia = arena.partida((blancas, negras, semilla, nodos))
    mod = arena.cargar(".")
    tab = np.zeros(81, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    color = 0
    vistas = []
    for ply, (jugada, gana) in enumerate(historia):
        if ply >= plies:
            break
        if color == nuestro_color:
            clave = int(mod.hash_completo(tab, color, pts))
            vistas.append((clave, tab.copy(), pts.copy(), color, ply))
        mod.hacer(tab, pts, color, jugada // 81, jugada % 81, gana, np.uint64(0))
        color = 1 - color
    return vistas


def calcular(args):
    tab, pts, color, nodos = args
    jugada, _, _ = buscar_profundo(tab, pts, color, nodos)
    return jugada


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rondas", type=int, default=10)
    ap.add_argument("--partidas", type=int, default=800, help="partidas por ronda")
    ap.add_argument("--nodos", type=int, default=60_000, help="nodos por jugada al recolectar")
    ap.add_argument("--nodos-libro", type=int, default=2_000_000)
    ap.add_argument("--plies", type=int, default=40, help="solo posiciones antes de este ply")
    ap.add_argument("--temprano", type=int, default=16, help="antes de este ply, aunque no se repita")
    ap.add_argument("--max-por-ronda", type=int, default=6000)
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--rivales", nargs="+", help="reemplaza la lista RIVALES (p. ej. ../t_regex_v2 random)")
    args = ap.parse_args()
    if args.rivales:
        RIVALES[:] = args.rivales  # los procesos del Pool (fork) heredan la lista ya cambiada
    print("rivales:", " ".join(RIVALES), flush=True)
    semilla = int(time.time()) % 1_000_000 * 100
    for ronda in range(args.rondas):
        libro = cargar_libro()
        inicio = time.time()
        conteo, muestra = Counter(), {}
        en_libro = total = 0
        trabajos = [(semilla + k, args.nodos, args.plies) for k in range(args.partidas)]
        semilla += args.partidas
        with Pool(args.procesos) as pool:  # procesos nuevos: cargan el libro de esta ronda
            for vistas in pool.imap_unordered(recolectar, trabajos):
                for clave, tab, pts, color, ply in vistas:
                    total += 1
                    if clave in libro:
                        en_libro += 1
                        continue
                    conteo[clave] += 1
                    if clave not in muestra or ply < muestra[clave][3]:
                        muestra[clave] = (tab, pts, color, ply)
        elegidas = [c for c, n in conteo.items() if n >= 2 or muestra[c][3] < args.temprano]
        elegidas.sort(key=lambda c: (-conteo[c], muestra[c][3]))  # repetidas y tempranas primero
        elegidas = elegidas[: args.max_por_ronda]
        trabajos = [(*muestra[c][:3], args.nodos_libro) for c in elegidas]
        ultimo = time.time()
        with Pool(args.procesos) as pool:
            for c, jugada in zip(elegidas, pool.imap(calcular, trabajos, chunksize=4),
                                 strict=True):
                libro[c] = jugada
                if time.time() - ultimo > GUARDAR_CADA_S:
                    guardar_libro(libro)
                    respaldar()
                    ultimo = time.time()
                    print(f"  guardado parcial: libro {len(libro)}", flush=True)
        guardar_libro(libro)
        respaldar()
        print(f"ronda {ronda + 1}: aciertos del libro {en_libro}/{total} = "
              f"{en_libro / max(total, 1):.1%} de nuestras jugadas antes del ply {args.plies}; "
              f"{len(conteo)} posiciones nuevas, {len(elegidas)} agregadas"
              f" → libro {len(libro)} ({ARCHIVO.stat().st_size / 1e6:.1f} MB), "
              f"{(time.time() - inicio) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
