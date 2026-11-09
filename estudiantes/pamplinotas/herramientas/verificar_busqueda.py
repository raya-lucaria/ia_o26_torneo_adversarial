"""Comprueba que la búsqueda podada (alfa-beta + Star1 + TT) da exactamente lo mismo que un
expectiminimax sin poda, y que las reglas del motor rápido coinciden con el motor oficial.

    PYTHONPATH=sistema:estudiantes/pamplinotas python estudiantes/pamplinotas/herramientas/verificar_busqueda.py

La referencia es la definición de la unidad, escrita en Python puro: MAX elige, AZAR promedia con
p y 1−p, finales ±(M − ply) antes de profundidad 0 y EVAL en las hojas. Se compara sin quietud
(prof_quietud=0) porque la referencia no la tiene.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
from motor import Estado

import buscador

CARPETA = Path(__file__).resolve().parent.parent


def expectiminimax(tab, color, pts, prof, ply, w):
    if pts[1 - color] >= buscador.PUNTOS_PARA_GANAR:
        return -(buscador.M - ply)
    jugadas = np.zeros(buscador.MAX_JUGADAS, np.int32)
    n = buscador.generar(tab, color, jugadas)
    if n == 0:
        return -(buscador.M - ply)
    if prof <= 0:
        return buscador.evaluar(tab, color, pts, w)
    rival = 1 - color
    pieza = color + 1
    meta = buscador.LADO - 1 if color == 0 else 0
    mejor = -np.inf
    for i in range(n):
        o, d = divmod(int(jugadas[i]), buscador.CASILLAS)
        anota = d // buscador.LADO == meta
        if tab[d] == 0:
            tab[o] = 0
            if anota:
                pts[color] += 1
            else:
                tab[d] = pieza
            v = -expectiminimax(tab, rival, pts, prof - 1, ply + 1, w)
            if anota:
                pts[color] -= 1
            else:
                tab[d] = 0
            tab[o] = pieza
        else:
            p = buscador.prob_combate(buscador.apoyo(tab, o, color), buscador.apoyo(tab, d, rival))
            tab[o] = 0
            tab[d] = 0
            if anota:
                pts[color] += 1
            else:
                tab[d] = pieza
            gana = -expectiminimax(tab, rival, pts, prof - 1, ply + 1, w)
            if anota:
                pts[color] -= 1
            tab[d] = rival + 1
            pierde = -expectiminimax(tab, rival, pts, prof - 1, ply + 1, w)
            tab[o] = pieza
            v = p * gana + (1 - p) * pierde
        mejor = max(mejor, v)
    return mejor


def posiciones_al_azar(cuantas, rng):
    """Posiciones de partidas al azar de largo variable (nunca terminadas)."""
    out = []
    while len(out) < cuantas:
        e = Estado.inicial()
        for _ in range(rng.randrange(0, 45)):
            if e.terminado():
                break
            j = rng.choice(e.jugadas())
            e = e.aplicar(j, gana_ataque=rng.random() < e.prob_ataque(j)) if e.es_ataque(j) else e.aplicar(j)
        if not e.terminado():
            out.append(e)
    return out


def finales_con_puntos(cuantas, rng):
    """Posiciones de final: pocos peones y alguien ya con puntos, donde un tercer punto o quedarse
    sin peones cabe en el horizonte y la cota de Star1 tiene que ser exacta."""
    out = []
    while len(out) < cuantas:
        e = Estado.inicial()
        while not e.terminado():
            j = rng.choice(e.jugadas())
            e = e.aplicar(j, gana_ataque=rng.random() < e.prob_ataque(j)) if e.es_ataque(j) else e.aplicar(j)
            tab, _, pts = buscador.tablero_desde_estado(e)
            if (pts[0] + pts[1] >= 1) and int((tab != 0).sum()) <= 10 and not e.terminado():
                out.append(e)
                break
    return out


def comparar(estados, profundidades, w, rng_nombre):
    total = difs = 0
    t = time.perf_counter()
    for e in estados:
        tab, c, pts = buscador.tablero_desde_estado(e)
        # Sin quietud ni LMR: la referencia es el expectiminimax exacto, y LMR cambia valores a
        # propósito (se mide con partidas, no aquí).
        b = buscador.Buscador(w, prof_quietud=0, lmr=0)
        for prof in profundidades:
            _, v, _ = b.buscar(tab.copy(), c, pts.copy(), prof, time.perf_counter() + 60)
            r = expectiminimax(tab.copy(), c, pts.copy(), prof, 0, w)
            total += 1
            if abs(v - r) > 1e-6:
                difs += 1
                print(f"  DIF ({rng_nombre}) prof {prof}: podada {v:.4f}, referencia {r:.4f}")
    print(f"búsqueda ({rng_nombre}): {total} comparaciones, {difs} diferencias ({time.perf_counter() - t:.0f} s)")


def main():
    with open(CARPETA / "data" / "pesos.json", encoding="utf-8") as f:
        w = buscador.pesos_desde_json(json.load(f))
    rng = random.Random(2026)
    estados = posiciones_al_azar(40, rng)

    # Reglas: jugadas legales, p de cada ataque y clave Zobrist, motor rápido vs oficial.
    difs_reglas = 0
    for e in estados:
        tab, c, pts = buscador.tablero_desde_estado(e)
        jugadas = np.zeros(buscador.MAX_JUGADAS, np.int32)
        n = buscador.generar(tab, c, jugadas)
        propias = {buscador.jugada_a_texto(int(j)) for j in jugadas[:n]}
        oficiales = {str(j) for j in e.jugadas()}
        if propias != oficiales or int(buscador.clave_zobrist(tab, c, pts)) != e.clave:
            difs_reglas += 1
            continue
        for j in e.jugadas():
            if e.es_ataque(j):
                p = buscador.prob_combate(buscador.apoyo(tab, j.origen, c), buscador.apoyo(tab, j.destino, 1 - c))
                if abs(p - e.prob_ataque(j)) > 1e-12:
                    difs_reglas += 1
    print(f"reglas: {len(estados)} posiciones, {difs_reglas} diferencias con el motor oficial")

    # Búsqueda: valor en la raíz, podada vs referencia. Profundización iterativa con la misma
    # tabla de transposición, killers e historial entre iteraciones, como en la partida.
    comparar(estados[:20], (1, 2, 3, 4), w, "medio juego")
    comparar(finales_con_puntos(60, rng), (1, 2, 3, 4, 5, 6), w, "finales con puntos")


if __name__ == "__main__":
    main()
