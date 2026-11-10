"""Reetiqueta posiciones ya guardadas con una búsqueda más profunda (offline, para entrenar).

Uso (desde estudiantes/maquinas_del_trabajo/; los datos van a `datos_entrenamiento/`, fuera de
git, como los de herramientas/datos.py):
    ../../.venv/bin/python -m herramientas.reetiquetar datos_entrenamiento/r2_autojuego.npz \
        --muestra 150000 --nodos 500000 --procesos 7 --carpeta datos_entrenamiento/reet_autojuego
    ../../.venv/bin/python -m herramientas.reetiquetar datos_entrenamiento/r2_autojuego.npz \
        --muestra 150000 --carpeta datos_entrenamiento/reet_autojuego --juntar \
        --salida datos_entrenamiento/r6_autojuego_500k.npz

Los datos de `herramientas.datos` traen el valor de una búsqueda de 20 000 nodos por posición; la
arena mostró que a esa profundidad las diferencias de evaluación casi no se ven. Esta herramienta
toma una muestra fija de esas posiciones (las de `--semilla`) y busca cada una con `--nodos`
nodos y los pesos de `data/pesos.npz`, con memoria nueva en cada posición: el resultado no
depende del orden ni de cuántos procesos se usen.

Se guarda por bloques de 1 000 posiciones en `--carpeta`: si se interrumpe, el mismo comando
sigue donde se quedó. Con `--juntar` se arma un `.npz` como los de `herramientas.datos` (con
`partida`, `y`, `tabs`, …) donde `v` es el valor profundo; el valor anterior queda en
`v_superficial`. Si una posición tiene una sola jugada, su `v` es NaN (no hay nada que buscar) y
`entrenar` usa el resultado de la partida.
"""

import argparse
import sys
import time
from functools import cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from busqueda import crear_memoria, profundizar
from strategy import cargar_pesos

CARPETA = Path(__file__).resolve().parent.parent
POR_BLOQUE = 1000
BITS_TT = 20  # 1 M lugares: de sobra para 500 000 nodos por posición


def muestra(n_posiciones: int, cuantas: int, semilla: int) -> np.ndarray:
    """Índices (ordenados) de las posiciones a reetiquetar: siempre los mismos para una semilla."""
    cuantas = min(cuantas, n_posiciones)
    return np.sort(np.random.default_rng(semilla).choice(n_posiciones, cuantas, replace=False))


@cache
def _datos(archivo: str):
    """Se lee una vez por proceso."""
    with np.load(archivo) as d:
        return d["tabs"], d["ptss"].astype(np.int64), d["colores"]


@cache
def _pesos():
    return cargar_pesos(CARPETA / "data")


def valor_profundo(tab, pts, color: int, nodos: int, pesos) -> tuple[float, int]:
    """(valor para el bando que mueve, profundidad completa alcanzada); NaN si hay una sola jugada."""
    r = profundizar(crear_memoria(bits_tt=BITS_TT, pesos=pesos), tab, pts, color, 1e6, 1e6, max_nodos=nodos)
    if r.prof == 0:
        return float("nan"), 0
    return float(r.valor), int(r.prof)


def _bloque(trabajo):
    archivo, indices, nodos = trabajo
    tabs, ptss, colores = _datos(archivo)
    valores = np.empty(len(indices), np.float32)
    profs = np.empty(len(indices), np.int8)
    for k, i in enumerate(indices):
        valores[k], profs[k] = valor_profundo(tabs[i], ptss[i], int(colores[i]), nodos, _pesos())
    return indices, valores, profs


def _archivo_bloque(carpeta: Path, k: int) -> Path:
    return carpeta / f"bloque_{k:04d}.npz"


def reetiquetar(archivo: Path, carpeta: Path, cuantas: int, semilla: int, nodos: int, procesos: int) -> None:
    with np.load(archivo) as d:
        n_posiciones = len(d["y"])
    elegidas = muestra(n_posiciones, cuantas, semilla)
    bloques = [elegidas[k : k + POR_BLOQUE] for k in range(0, len(elegidas), POR_BLOQUE)]
    carpeta.mkdir(parents=True, exist_ok=True)
    faltan = [k for k in range(len(bloques)) if not _archivo_bloque(carpeta, k).exists()]
    print(f"{len(elegidas)} posiciones en {len(bloques)} bloques; faltan {len(faltan)}", flush=True)
    inicio = time.perf_counter()
    trabajos = [(str(archivo), bloques[k], nodos) for k in faltan]
    with Pool(procesos) as pool:
        for hechos, (indices, valores, profs) in enumerate(pool.imap_unordered(_bloque, trabajos), 1):
            k = int(np.searchsorted(elegidas, indices[0])) // POR_BLOQUE
            temporal = carpeta / f"bloque_{k:04d}.tmp.npz"
            np.savez(temporal, indices=indices, v=valores, prof=profs)
            temporal.replace(_archivo_bloque(carpeta, k))  # el bloque aparece completo o no aparece
            segundos = time.perf_counter() - inicio
            resta = segundos / hechos * (len(faltan) - hechos)
            print(
                f"  bloque {k:4d} listo ({hechos}/{len(faltan)}); prof. mediana {int(np.median(profs))};"
                f" {segundos / 60:.0f} min, faltan ~{resta / 60:.0f} min",
                flush=True,
            )
    print(f"terminado: {len(bloques)} bloques en {carpeta} ({(time.perf_counter() - inicio) / 60:.0f} min)")


def juntar(archivo: Path, carpeta: Path, cuantas: int, semilla: int, salida: Path) -> None:
    with np.load(archivo) as d:
        datos = {campo: d[campo] for campo in d.files}
    elegidas = muestra(len(datos["y"]), cuantas, semilla)
    n_bloques = -(-len(elegidas) // POR_BLOQUE)
    faltan = [k for k in range(n_bloques) if not _archivo_bloque(carpeta, k).exists()]
    if faltan:
        sys.exit(f"faltan {len(faltan)} bloques (por ejemplo el {faltan[0]}): corre primero sin --juntar")
    valores, profs = [], []
    for k in range(n_bloques):
        with np.load(_archivo_bloque(carpeta, k)) as b:
            if not np.array_equal(b["indices"], elegidas[k * POR_BLOQUE : (k + 1) * POR_BLOQUE]):
                sys.exit(f"el bloque {k} no corresponde a esta muestra (¿otra --semilla o --muestra?)")
            valores.append(b["v"])
            profs.append(b["prof"])
    salida_datos = {campo: valores_campo[elegidas] for campo, valores_campo in datos.items()}
    salida_datos["v_superficial"] = salida_datos["v"]
    salida_datos["v"] = np.concatenate(valores)
    salida_datos["prof"] = np.concatenate(profs)
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida, **salida_datos)
    v = salida_datos["v"]
    print(
        f"{len(v)} posiciones → {salida}; sin valor (una sola jugada): {int(np.isnan(v).sum())};"
        f" profundidad mediana {int(np.median(salida_datos['prof']))}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datos", type=Path, help=".npz de herramientas.datos (con tabs, ptss, colores)")
    parser.add_argument("--carpeta", type=Path, required=True, help="donde se guardan los bloques")
    parser.add_argument("--muestra", type=int, default=150_000, help="cuántas posiciones")
    parser.add_argument("--semilla", type=int, default=6, help="qué posiciones (la muestra)")
    parser.add_argument("--nodos", type=int, default=500_000, help="nodos por posición")
    parser.add_argument("--procesos", type=int, default=6)
    parser.add_argument("--juntar", action="store_true", help="arma el .npz final con los bloques")
    parser.add_argument("--salida", type=Path, help="el .npz final (con --juntar)")
    args = parser.parse_args(argv)
    if args.juntar:
        if args.salida is None:
            parser.error("--juntar necesita --salida")
        juntar(args.datos, args.carpeta, args.muestra, args.semilla, args.salida)
    else:
        reetiquetar(args.datos, args.carpeta, args.muestra, args.semilla, args.nodos, args.procesos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
