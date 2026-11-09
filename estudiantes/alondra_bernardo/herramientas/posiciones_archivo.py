"""Convierte las partidas archivadas en caminos.sqlite al formato de generar_posiciones.py.

Cada posición lleva como etiqueta y como resultado si el lado que mueve
terminó ganando esa partida (no hay valor de búsqueda para ellas).

python estudiantes/alondra_bernardo/herramientas/posiciones_archivo.py \
    --base .../caminos.sqlite --salida /ruta/archivo.npz
"""

import argparse
import json
import sqlite3
import zlib
from pathlib import Path

import numpy as np
from motor import Estado


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--omitir", type=int, default=2)
    a = p.parse_args()
    tabs, pts, cols, res = [], [], [], []
    with sqlite3.connect(a.base.resolve().as_uri() + "?mode=ro", uri=True) as con:
        for (registro,) in con.execute("SELECT registro FROM partidas WHERE valida=1"):
            datos = json.loads(zlib.decompress(registro))
            ganador = 0 if datos["ganador"] == "blancas" else 1
            e = Estado.inicial()
            for n, r in enumerate(datos["jugadas"]):
                if n >= a.omitir:
                    tabs.append(e._tablero.copy())
                    pts.append(e._puntos.copy())
                    cols.append(e._color)
                    res.append(float(e._color == ganador))
                combate = r.get("combate")
                e = (e.aplicar(r["jugada"]) if combate is None
                     else e.aplicar(r["jugada"], gana_ataque=combate["gano"]))
    res = np.array(res)
    np.savez_compressed(a.salida, tableros=np.array(tabs, np.int8), puntos=np.array(pts, np.int64),
                        colores=np.array(cols, np.int64), etiquetas=res, resultados=res)
    print(f"{len(res)} posiciones en {a.salida}")


if __name__ == "__main__":
    main()
