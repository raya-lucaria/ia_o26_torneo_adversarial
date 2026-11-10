"""Cuántos nodos necesita la búsqueda para llegar a una profundidad fija (eficiencia de la poda).

Uso (desde estudiantes/maquinas_del_trabajo/):
    ../../.venv/bin/python -m herramientas.nodos_a_profundidad
    ../../.venv/bin/python -m herramientas.nodos_a_profundidad --prof 6 --opciones 1,1,0

Busca con profundización iterativa hasta `--prof` en posiciones fijas de medio juego (memoria
nueva en cada una, sin límite de tiempo) con los pesos de `data/pesos.npz`. El número de nodos
no depende de la máquina: si la búsqueda no cambia, sale idéntico en cualquier computadora. Sirve
para medir cambios de orden de jugadas y de poda: menos nodos a la misma profundidad = más
profundidad con el mismo reloj. El tiempo sí depende de la máquina (y del costo por nodo).

`--opciones` fija `memoria.opciones` (ver `busqueda.Memoria`): una lista de 0/1 separada por comas.
"""

import argparse
import sys
import time

from busqueda import crear_memoria, profundizar
from herramientas.posiciones import muestrear
from motor_rapido import desde_estado
from strategy import cargar_pesos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prof", type=int, default=6)
    parser.add_argument("--posiciones", type=int, default=30)
    parser.add_argument("--semilla", type=int, default=5)
    parser.add_argument("--opciones", help="valores de memoria.opciones, p. ej. 1,1,0")
    args = parser.parse_args(argv)

    pesos = cargar_pesos("data")
    posiciones = [
        desde_estado(e)
        for e in muestrear(args.posiciones, args.semilla, sinteticas=0, ply_min=20, ply_max=60)
    ]
    opciones = [int(x) for x in args.opciones.split(",")] if args.opciones else None
    # Una búsqueda corta antes de medir: así el tiempo de compilar no cuenta.
    m = crear_memoria(pesos=pesos)
    profundizar(m, *posiciones[0], 1e6, 1e6, prof_max=2)

    total_nodos, total_s, suma_valores = 0, 0.0, 0.0
    for tab, pts, color in posiciones:
        m = crear_memoria(pesos=pesos)
        if opciones is not None:
            m.opciones[: len(opciones)] = opciones
        inicio = time.perf_counter()
        r = profundizar(m, tab, pts, color, 1e6, 1e6, prof_max=args.prof)
        total_s += time.perf_counter() - inicio
        total_nodos += r.nodos
        suma_valores += r.valor
    print(
        f"profundidad {args.prof}, {len(posiciones)} posiciones (semilla {args.semilla}):"
        f" {total_nodos:,} nodos, {total_s:.1f} s, {total_nodos / total_s / 1e6:.2f} M nodos/s,"
        f" suma de valores {suma_valores:+.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
