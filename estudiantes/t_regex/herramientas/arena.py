"""Arena offline: enfrenta dos versiones de `busqueda.py` en muchas partidas (no corre en la liga).

Cada jugada usa profundización iterativa con un presupuesto fijo de NODOS (no de tiempo), así
la comparación no depende de la carga de la máquina. Los dados imitan al árbitro: un generador
por color, y cada juego de dados se juega dos veces cambiando colores.

Uso (dentro de la imagen de bots, con todos los núcleos):
    docker run --rm --cpus=8 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/arena.py \
        --a . --b ../t_regex_v1 --pares 100 --nodos 300000
"""

import argparse
import importlib.util
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import itertools

import numpy as np

from motor import Estado, simular_al_azar

MC_SIMULACIONES = 20  # como estudiantes/montecarlo: partidas al azar por muestra
_modulos = {}


class Montecarlo:
    """Réplica de estudiantes/montecarlo (Monte Carlo plano) con `muestras` fijas por jugada.

    En la liga montecarlo hace unas 1700 muestras por jugada (lo reporta en el visor).
    """

    def __init__(self, muestras, semilla):
        self.muestras = muestras
        self.rng = random.Random(semilla)

    def elegir(self, tab, pts, color):
        estado = Estado(tab.copy(), color, pts.copy())
        nombre = estado.turno
        jugadas = estado.jugadas()
        for j in jugadas:
            if all(hijo.ganador() == nombre for _, hijo in estado.siguiente(j)):
                return j.origen * 81 + j.destino
        total = {j: 0.0 for j in jugadas}
        cuenta = {j: 0 for j in jugadas}
        for hechas, j in enumerate(itertools.cycle(jugadas), 1):
            probs, hijos = zip(*estado.siguiente(j), strict=True)
            hijo = self.rng.choices(hijos, weights=probs)[0]
            total[j] += simular_al_azar(
                hijo, MC_SIMULACIONES, self.rng.randrange(2**31), para=nombre
            )
            cuenta[j] += 1
            if hechas >= max(self.muestras, len(jugadas)):
                break
        mejor = max(jugadas, key=lambda j: total[j] / cuenta[j])
        return mejor.origen * 81 + mejor.destino


class Azar:
    """Como estudiantes/random: una jugada legal al azar."""

    def __init__(self, semilla):
        self.rng = random.Random(semilla)

    def elegir(self, tab, pts, color):
        jugadas = Estado(tab.copy(), color, pts.copy()).jugadas()
        j = self.rng.choice(jugadas)
        return j.origen * 81 + j.destino


def cargar(carpeta, cambios=""):
    """Importa `busqueda.py` de `carpeta` con un nombre único (para tener dos versiones).

    `cambios` = "W_LIBRE=2.0,ESCALA=3" cambia constantes del módulo antes de compilar:
    numba lee las globales al compilar, así que cada combinación es otra versión.
    """
    llave = (str(Path(carpeta).resolve()), cambios)
    if llave not in _modulos:
        nombre = f"busqueda_{len(_modulos)}"
        spec = importlib.util.spec_from_file_location(nombre, Path(llave[0]) / "busqueda.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[nombre] = mod
        spec.loader.exec_module(mod)
        for par in filter(None, cambios.split(",")):
            k, v = par.split("=")
            if "[" in k:  # un elemento de un arreglo: PESOS[3]=0.6
                k, i = k.rstrip("]").split("[")
                arreglo = getattr(mod, k).copy()
                arreglo[int(i)] = float(v)
                setattr(mod, k, arreglo)
            else:
                assert hasattr(mod, k), f"{k} no existe en busqueda.py"
                setattr(mod, k, float(v))
        _modulos[llave] = mod
    return _modulos[llave]


def elegir(bot, tab, pts, color, nodos):
    """Profundización iterativa hasta gastar `nodos`. Regresa (o*81+d, profundidad)."""
    mod, ctx = bot
    if hasattr(mod, "jugada_de_libro"):
        del_libro = mod.jugada_de_libro(tab, pts, color)
        if del_libro >= 0:
            return del_libro, 0
    mod.nueva_jugada(ctx)
    mejor, gastados, prof_hecha = -1, 0, 0
    for prof in range(1, 41):
        limite = max(nodos - gastados, 1000) if prof > 1 else 1 << 62
        jugada, v, completas, n, abortada = mod.buscar(ctx, tab, pts, color, prof, mejor, limite)
        gastados += n
        if abortada:
            if completas > 0:
                mejor = jugada
            break
        mejor, prof_hecha = jugada, prof
        if abs(v) > 0.95 or gastados >= nodos:
            break
    return mejor, prof_hecha


def partida(args):
    """Una partida. Regresa 0 si ganan las blancas, 1 si las negras, y la duración en plies."""
    blancas, negras, semilla, nodos = args
    # Contexto nuevo (tabla limpia): en la liga cada partida es un proceso nuevo.
    # La carpeta «montecarlo:N» es la réplica de montecarlo con N muestras por jugada y
    # «random» la de random.
    bots = []
    for carpeta, cambios in (blancas, negras):
        if carpeta.startswith("montecarlo"):
            bots.append(Montecarlo(int(carpeta.split(":")[1]), semilla))
        elif carpeta == "random":
            bots.append(Azar(semilla))
        else:
            m = cargar(carpeta, cambios)
            bots.append((m, m.nuevo_contexto()))
    mod = next(b[0] for b in bots if isinstance(b, tuple))
    tab = np.zeros(81, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    dados = (random.Random(semilla * 2), random.Random(semilla * 2 + 1))
    color = 0
    profs = ([], [])
    historia = []  # jugadas (o*81+d, ¿ganó el ataque?) para --guardar
    origenes = np.zeros((1, 54), np.int64)
    destinos = np.zeros((1, 54), np.int64)
    for ply in range(400):
        if pts[1 - color] >= 3:
            return 1 - color, ply, profs, historia
        if mod.generar(tab, color, 0, origenes, destinos) == 0:
            return 1 - color, ply, profs, historia
        if isinstance(bots[color], (Montecarlo, Azar)):
            jugada, prof = bots[color].elegir(tab, pts, color), 0
        else:
            jugada, prof = elegir(bots[color], tab, pts, color, nodos)
        profs[color].append(prof)
        o, d = jugada // 81, jugada % 81
        gana = True
        if tab[d] != 0:
            gana = dados[color].random() < mod.prob_ataque(tab, color, o, d)
        historia.append((int(jugada), bool(gana)))
        mod.hacer(tab, pts, color, o, d, gana, np.uint64(0))
        color = 1 - color
    raise RuntimeError("partida demasiado larga")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pa", default="", help="cambios de constantes para A: W_LIBRE=2,ESCALA=3")
    ap.add_argument("--pb", default="", help="cambios de constantes para B")
    ap.add_argument("--pares", type=int, default=50, help="juegos de dados (2 partidas cada uno)")
    ap.add_argument("--nodos", type=int, default=300_000)
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=1000)
    ap.add_argument("--guardar", help="JSON donde guardar las partidas que perdió A")
    args = ap.parse_args()
    trabajos = []
    for k in range(args.pares):
        s = args.semilla + k
        a, b = (args.a, args.pa), (args.b, args.pb)
        trabajos.append((a, b, s, args.nodos))
        trabajos.append((b, a, s, args.nodos))
    victorias_a = 0
    victorias_blancas = 0
    profs_a, profs_b = [], []
    perdidas = []
    with Pool(args.procesos) as pool:
        for i, (ganador, plies, profs, jugadas) in enumerate(pool.imap(partida, trabajos)):
            a_es_blancas = i % 2 == 0
            profs_a.extend(profs[0 if a_es_blancas else 1])
            profs_b.extend(profs[1 if a_es_blancas else 0])
            gano_a = (ganador == 0) == a_es_blancas
            if not gano_a:
                perdidas.append({"a_es": "blancas" if a_es_blancas else "negras",
                                 "jugadas": jugadas})
            victorias_a += gano_a
            victorias_blancas += ganador == 0
            if (i + 1) % 20 == 0:
                print(f"{i + 1} partidas: A {victorias_a}  B {i + 1 - victorias_a}", flush=True)
    total = len(trabajos)
    tasa = victorias_a / total
    error = 1.96 * (tasa * (1 - tasa) / total) ** 0.5
    print(f"A = {args.a} {args.pa}  B = {args.b} {args.pb}  nodos/jugada = {args.nodos}")
    print(f"A gana {victorias_a}/{total} = {tasa:.1%} ± {error:.1%} (95 %)")
    print(f"Profundidad media: A {np.mean(profs_a):.2f}  B {np.mean(profs_b):.2f}"
          "  (0 = montecarlo)")
    print(f"Blancas ganan {victorias_blancas}/{total} = {victorias_blancas / total:.1%}")
    if args.guardar:
        import json

        Path(args.guardar).write_text(json.dumps(perdidas))
        print(f"{len(perdidas)} partidas perdidas por A → {args.guardar}")


if __name__ == "__main__":
    main()
