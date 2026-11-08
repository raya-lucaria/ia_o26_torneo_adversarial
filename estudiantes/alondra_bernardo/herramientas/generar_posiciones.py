"""Genera posiciones de autojuego etiquetadas con la búsqueda (offline, en paralelo).

Cada partida se juega con rapido.py a profundidad fija; con probabilidad
--azar una jugada es al azar, para ver posiciones variadas. Cada posición
se etiqueta con el valor que da una búsqueda a --prof desde ella (no con
quién ganó la partida: los dados hacen esa etiqueta muy ruidosa).
Con --contra-mc S, uno de los dos lados es Monte Carlo plano con S segundos
por jugada (como el rival del curso), para ver las posiciones que crea un
rival distinto de nosotros; igual se etiquetan con nuestra búsqueda.

Desde la raíz del repo, con el .venv activo:
python estudiantes/alondra_bernardo/herramientas/generar_posiciones.py \
    --partidas 2000 --prof 4 --salida /ruta/posiciones.npz
"""

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EQUIPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EQUIPO))
from motor import Estado
from arena import montecarlo
import rapido

AJUSTES = {}


def iniciar(pesos, prof, azar, omitir, lmr, contra_mc):
    ctx = rapido.Contexto(pesos)
    ctx.control[5] = int(lmr)
    AJUSTES.update(ctx=ctx, prof=prof, azar=azar, omitir=omitir, contra_mc=contra_mc)


def partida(semilla):
    ctx, prof = AJUSTES["ctx"], AJUSTES["prof"]
    rng = random.Random(semilla)
    estado = Estado.inicial()
    lado_mc = semilla % 2 if AJUSTES["contra_mc"] > 0 else -1
    filas = []
    n = 0
    while estado.ganador() is None:
        jugadas = estado.jugadas()
        ctx.limpiar()  # no toca control[5]
        v, cod = rapido.buscar(ctx, estado._tablero, estado._puntos, estado._color, prof, 10**12)
        if n >= AJUSTES["omitir"]:
            filas.append((estado._tablero.copy(), estado._puntos.copy(), estado._color, (v + 1) / 2))
        if estado._color == lado_mc:
            jugada = montecarlo(estado, AJUSTES["contra_mc"], rng)
        elif rng.random() < AJUSTES["azar"]:
            jugada = rng.choice(jugadas)
        else:
            jugada = next(j for j in jugadas if str(j) == rapido.texto(cod))
        probs, hijos = zip(*estado.siguiente(jugada))
        estado = rng.choices(hijos, weights=probs)[0]
        n += 1
    ganador = 0 if estado.ganador() == "blancas" else 1
    return [f + (float(f[2] == ganador),) for f in filas]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partidas", type=int, default=1000)
    p.add_argument("--prof", type=int, default=4)
    p.add_argument("--azar", type=float, default=0.1)
    p.add_argument("--omitir", type=int, default=2, help="primeras jugadas sin guardar")
    p.add_argument("--pesos", type=Path, help="pesos.json; si falta, los iniciales")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--procesos", type=int, default=8)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--lmr", action="store_true", help="reducción de jugadas tardías")
    p.add_argument("--contra-mc", type=float, default=0.0,
                   help="segundos por jugada del lado Monte Carlo (0: autojuego)")
    a = p.parse_args()
    pesos = (json.loads(a.pesos.read_text())["pesos"] if a.pesos else rapido.pesos_iniciales())
    semillas = [a.semilla * 1_000_003 + i for i in range(a.partidas)]
    with Pool(a.procesos, initializer=iniciar, initargs=(pesos, a.prof, a.azar, a.omitir, a.lmr, a.contra_mc)) as pool:
        filas = [f for partida_filas in pool.imap_unordered(partida, semillas, chunksize=4)
                 for f in partida_filas]
    np.savez_compressed(
        a.salida,
        tableros=np.array([f[0] for f in filas], np.int8),
        puntos=np.array([f[1] for f in filas], np.int64),
        colores=np.array([f[2] for f in filas], np.int64),
        etiquetas=np.array([f[3] for f in filas], np.float64),
        resultados=np.array([f[4] for f in filas], np.float64),
    )
    print(f"{len(filas)} posiciones de {a.partidas} partidas en {a.salida}")


if __name__ == "__main__":
    main()
