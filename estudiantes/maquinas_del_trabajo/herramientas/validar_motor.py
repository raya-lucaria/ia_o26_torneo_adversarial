"""Fuzz diferencial: nuestro motor rápido contra el motor oficial.

Uso (desde estudiantes/maquinas_del_trabajo/):
    ../../.venv/bin/python -m herramientas.validar_motor --posiciones 1000000 --procesos 8

Para cada posición compara jugadas legales (y su orden), probabilidad de cada ataque, las
posiciones que resultan de cada jugada (gane o pierda el ataque) y las claves Zobrist.
Sale con código 0 solo si no hay ninguna discrepancia.
"""

import argparse
import multiprocessing
import os
import sys

import numpy as np

import motor_rapido as mr
from herramientas import desde_python
from herramientas.posiciones import muestrear

LOTE = 5000  # posiciones por tarea de cada proceso


def discrepancias(estado) -> list[str]:
    """Diferencias entre nuestro motor y el oficial en `estado` (no terminal). Vacía si coinciden."""
    fallas = []
    tab, pts, color = mr.desde_estado(estado)
    # numba regresa la clave como int de Python; para volver a pasarla debe ser uint64.
    h = np.uint64(mr.clave(tab, color, pts))
    if int(h) != estado.clave:
        fallas.append(f"clave {int(h)} != {estado.clave}")
    origenes = np.empty(mr.MAX_JUGADAS, np.int64)
    destinos = np.empty(mr.MAX_JUGADAS, np.int64)
    n = mr.generar(tab, color, origenes, destinos)
    nuestras = [(int(o), int(d)) for o, d in zip(origenes[:n], destinos[:n], strict=True)]
    oficiales = [(j.origen, j.destino) for j in estado.jugadas()]
    if nuestras != oficiales:
        return fallas + [f"jugadas {nuestras} != {oficiales}"]
    tab_hijo = np.empty(mr.CASILLAS, np.int8)
    pts_hijo = np.empty(2, np.int64)
    for j in estado.jugadas():
        p = desde_python.prob_jugada(tab, color, j.origen, j.destino)
        if p != estado.prob_ataque(j):
            fallas.append(f"{j}: p {p} != {estado.prob_ataque(j)}")
        resultados = (True, False) if estado.es_ataque(j) else (None,)
        for gana in resultados:
            hijo = estado.aplicar(j, gana)
            h_hijo = desde_python.hacer(
                tab, pts, color, j.origen, j.destino, gana is not False, h, tab_hijo, pts_hijo
            )
            if not np.array_equal(tab_hijo, hijo._tablero):
                fallas.append(f"{j} (gana={gana}): tablero distinto")
            if not np.array_equal(pts_hijo, hijo._puntos):
                fallas.append(f"{j} (gana={gana}): puntos {pts_hijo} != {hijo._puntos}")
            if int(h_hijo) != hijo.clave:
                fallas.append(f"{j} (gana={gana}): clave del hijo distinta")
    if not np.array_equal(tab, estado._tablero) or not np.array_equal(pts, estado._puntos):
        fallas.append("hacer cambió la posición original")
    return fallas


def _revisar_lote(argumentos: tuple[int, int]) -> tuple[int, int, list]:
    semilla, cuantas = argumentos
    malas = 0
    ejemplos = []
    for estado in muestrear(cuantas, semilla):
        encontradas = discrepancias(estado)
        if encontradas:
            malas += 1
            if len(ejemplos) < 3:
                ejemplos.append((estado.a_dict(), encontradas[:3]))
    return cuantas, malas, ejemplos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--posiciones", type=int, default=100_000)
    parser.add_argument("--procesos", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--semilla", type=int, default=7)
    args = parser.parse_args(argv)

    lotes = []
    faltan = args.posiciones
    while faltan > 0:
        cuantas = min(LOTE, faltan)
        lotes.append((args.semilla * 1_000_003 + len(lotes), cuantas))
        faltan -= cuantas

    revisadas = malas = 0
    ejemplos = []
    with multiprocessing.Pool(args.procesos) as pool:
        for cuantas, malas_lote, ejemplos_lote in pool.imap_unordered(_revisar_lote, lotes):
            revisadas += cuantas
            malas += malas_lote
            ejemplos.extend(ejemplos_lote)
            print(f"  {revisadas}/{args.posiciones} revisadas, {malas} con discrepancias", flush=True)
    for posicion, fallas in ejemplos[:5]:
        print("Ejemplo:", posicion, fallas)
    print(f"{malas} discrepancias en {revisadas} posiciones")
    return 0 if malas == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
