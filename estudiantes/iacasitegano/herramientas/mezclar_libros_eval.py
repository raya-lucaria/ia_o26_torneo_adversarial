"""Construye una mezcla experimental del libro, separada por turno.

Cada entrada guarda origen y destino como índices `fila * 9 + columna`. Todo
peón avanza una fila: delta positivo identifica Blancas y negativo Negras.
Conserva las entradas blancas del libro nuevo y las negras del baseline.
Solo se usa offline para evaluación.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def color_de_entrada(jugada: list[int]) -> str:
    if len(jugada) != 2:
        raise ValueError(f"jugada inválida en el libro: {jugada!r}")
    delta = int(jugada[1]) - int(jugada[0])
    if delta == 0 or abs(delta) not in (8, 9, 10):
        raise ValueError(f"movimiento no es un avance legal de una fila: {jugada!r}")
    return "blancas" if delta > 0 else "negras"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nuevo", type=Path, required=True)
    ap.add_argument("--actual", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    libro_nuevo = json.loads(args.nuevo.read_text())
    libro_actual = json.loads(args.actual.read_text())
    turnos = {
        clave: color_de_entrada(jugada)
        for libro in (libro_nuevo, libro_actual)
        for clave, jugada in libro.items()
    }

    mixto = dict(libro_actual)
    blancas_nuevas = [k for k in libro_nuevo if turnos[k] == "blancas"]
    negras_actuales = [k for k in libro_actual if turnos[k] == "negras"]
    for clave in blancas_nuevas:
        mixto[clave] = libro_nuevo[clave]

    args.salida.write_text(json.dumps(mixto))
    print(
        f"Actual: {len(libro_actual)} entradas; nuevo: {len(libro_nuevo)}; "
        f"mixto: {len(mixto)}; blancas del nuevo: {len(blancas_nuevas)}; "
        f"negras del actual: {len(negras_actuales)}"
    )


if __name__ == "__main__":
    main()
