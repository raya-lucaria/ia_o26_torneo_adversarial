"""Arena local rápida: dos juegos de pesos (o Monte Carlo plano) a presupuesto fijo de nodos.

Cada jugada usa profundización iterativa hasta --nodos nodos. Colores
alternados, en paralelo. Sirve para filtrar ideas antes de gastar Docker.

python estudiantes/alondra_bernardo/herramientas/arena.py A.json B.json --partidas 400
(usa «inicial» para los pesos a mano y «mc:SEGUNDOS» para Monte Carlo plano;
«A.json+lmr» activa la reducción de jugadas tardías «+q6» fija la quietud en 6 y «+x2» da el doble de nodos)
"""

import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

EQUIPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EQUIPO))
from motor import Estado, simular_al_azar
import rapido

JUGADORES = {}


def cargar(nombre):
    if nombre == "inicial":
        return rapido.pesos_iniciales()
    return json.loads(Path(nombre).read_text())["pesos"]


def iniciar(a, b, nodos):
    for k, nombre in (("a", a), ("b", b)):
        if nombre.startswith("mc:"):
            JUGADORES[k] = ("mc", float(nombre[3:]))
            continue
        ruta, *extras = nombre.split("+")
        ctx = rapido.Contexto(cargar(ruta))
        ctx.multiplo = 1.0
        for extra in extras:
            if extra == "lmr":
                ctx.control[5] = 1
            elif extra.startswith("x"):
                ctx.multiplo = float(extra[1:])
            elif extra.startswith("q"):
                ctx.control[4] = int(extra[1:])
        JUGADORES[k] = ("busqueda", ctx)
    JUGADORES["nodos"] = nodos


def con_busqueda(ctx, estado, nodos):
    ctx.limpiar()
    mejor = None
    for prof in range(1, rapido.MAX_PLY):
        v, cod = rapido.buscar(ctx, estado._tablero, estado._puntos, estado._color, prof, nodos)
        if cod is None:
            break
        mejor = cod
        if abs(v) > 0.999:
            break
    jugadas = estado.jugadas()
    if mejor is None:
        return jugadas[0]
    return next(j for j in jugadas if str(j) == rapido.texto(mejor))


def montecarlo(estado, segundos, rng):
    color = estado.turno
    jugadas = estado.jugadas()
    for j in jugadas:
        if all(h.ganador() == color for _, h in estado.siguiente(j)):
            return j
    total = {j: 0.0 for j in jugadas}
    n = {j: 0 for j in jugadas}
    fin = time.perf_counter() + segundos
    i = 0
    while True:
        j = jugadas[i % len(jugadas)]
        i += 1
        probs, hijos = zip(*estado.siguiente(j))
        h = rng.choices(hijos, weights=probs)[0]
        total[j] += simular_al_azar(h, 20, rng.randrange(2**31), para=color)
        n[j] += 1
        if i >= len(jugadas) and time.perf_counter() >= fin:
            break
    return max(jugadas, key=lambda j: total[j] / n[j])


def partida(k):
    rng = random.Random(k)
    a_blancas = k % 2 == 0
    e = Estado.inicial()
    while e.ganador() is None:
        tipo, x = JUGADORES["a" if (e.turno == "blancas") == a_blancas else "b"]
        j = con_busqueda(x, e, int(JUGADORES["nodos"] * x.multiplo)) if tipo == "busqueda" else montecarlo(e, x, rng)
        probs, hijos = zip(*e.siguiente(j))
        e = rng.choices(hijos, weights=probs)[0]
    return (e.ganador() == "blancas") == a_blancas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--partidas", type=int, default=200)
    p.add_argument("--nodos", type=int, default=200_000)
    p.add_argument("--procesos", type=int, default=9)
    p.add_argument("--semilla", type=int, default=0)
    a = p.parse_args()
    with Pool(a.procesos, initializer=iniciar, initargs=(a.a, a.b, a.nodos)) as pool:
        r = pool.map(partida, range(a.semilla, a.semilla + a.partidas))
    print(f"{a.a} vs {a.b}: {sum(r)}-{len(r) - sum(r)} ({a.nodos} nodos por jugada)")


if __name__ == "__main__":
    main()
