"""Genera posiciones para ajustar los pesos de la evaluación (no corre en la liga).

Juega partidas offline de nuestro bot contra sí mismo a profundidad fija con un poco de azar y
guarda de cada posición el tablero y dos etiquetas en [0, 1]:

- `YB`: la probabilidad de ganar que estima nuestra búsqueda a profundidad `--prof` (como
  TD-Gammon: poco ruido, pero hereda los puntos ciegos de esa búsqueda);
- `YR`: 1 si quien movía terminó ganando la partida (insesgada, pero los dados la hacen ruidosa).

También guarda `G`, la partida de cada fila, para poder calcular la etiqueta TD(λ).

`ajustar_pesos.py` las mezcla. Se guardan tableros y no rasgos para poder cambiar los rasgos sin
volver a jugar.

    PYTHONPATH=sistema:estudiantes/pamplinotas:estudiantes/pamplinotas/herramientas python \\
        estudiantes/pamplinotas/herramientas/generar_posiciones.py --partidas 2400 \\
        --pesos estudiantes/pamplinotas/data/pesos.json --salida partidas.npz
"""

import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np
from motor import Estado, Jugada

import buscador


def probabilidad_desde_valor(v):
    """EVAL = ESCALA_V·(2P − 1); los finales (±M) se llevan a 0 o 1."""
    return min(1.0, max(0.0, (v / buscador.ESCALA_V + 1.0) / 2.0))


def jugar_partida(args):
    semilla, prof, epsilon, pesos = args
    rng = random.Random(semilla)
    w = buscador.pesos_desde_json(pesos)
    # La etiqueta sale de una búsqueda completa a profundidad fija: sin LMR.
    bots = {"blancas": buscador.Buscador(w, lmr=0), "negras": buscador.Buscador(w, lmr=0)}
    dados = {"blancas": random.Random(semilla * 2 + 1), "negras": random.Random(semilla * 2 + 2)}
    e = Estado.inicial()
    filas = []  # (tablero, color que mueve, puntos, etiqueta de búsqueda, nombre del color)
    n = 0
    while not e.terminado() and n < 300:
        color = e.turno
        tab, c, pts = buscador.tablero_desde_estado(e)
        j, v, _ = bots[color].buscar(tab, c, pts, prof, time.perf_counter() + 60)
        if n >= 2:
            filas.append((tab.copy(), c, pts.copy(), probabilidad_desde_valor(v), color))
        if rng.random() < epsilon:
            texto = str(rng.choice(e.jugadas()))
        else:
            texto = buscador.jugada_a_texto(j)
        jugada = Jugada.desde_texto(texto)
        if e.es_ataque(jugada):
            e = e.aplicar(jugada, gana_ataque=dados[color].random() < e.prob_ataque(jugada))
        else:
            e = e.aplicar(jugada)
        n += 1
    ganador = e.ganador()
    if ganador is None:
        return [], [], [], [], []
    return (
        [t for t, _, _, _, _ in filas],
        [c for _, c, _, _, _ in filas],
        [p for _, _, p, _, _ in filas],
        [yb for _, _, _, yb, _ in filas],
        [1.0 if color == ganador else 0.0 for _, _, _, _, color in filas],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidas", type=int, default=2400)
    ap.add_argument("--prof", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.08)
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", required=True)
    o = ap.parse_args()
    with open(o.pesos, encoding="utf-8") as f:
        pesos = json.load(f)
    tareas = [(o.semilla * 1_000_000 + i, o.prof, o.epsilon, pesos) for i in range(o.partidas)]
    T, C, P, YB, YR, G = [], [], [], [], [], []
    partidas_ok = 0
    t = time.perf_counter()
    with mp.Pool() as pool:
        for k, r in enumerate(pool.imap_unordered(jugar_partida, tareas, chunksize=1), 1):
            if r[0]:
                G += [partidas_ok] * len(r[0])
                partidas_ok += 1
            T += r[0]
            C += r[1]
            P += r[2]
            YB += r[3]
            YR += r[4]
            if k % 200 == 0:
                print(f"{k} partidas, {len(YR)} posiciones, {time.perf_counter() - t:.0f} s", flush=True)
    np.savez_compressed(
        o.salida, T=np.array(T, np.int8), C=np.array(C, np.int8), P=np.array(P, np.int8),
        YB=np.array(YB, np.float32), YR=np.array(YR, np.float32), G=np.array(G, np.int32),
    )
    print(f"guardado {o.salida}: {len(YR)} posiciones de {o.partidas} partidas")


if __name__ == "__main__":
    main()
