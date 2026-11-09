"""Genera un libro de apertura: para las primeras `PLIES_LIBRO` jugadas NUESTRAS,
qué jugada ganó más seguido en self-play, con cuántas muestras la vieron. Guarda
`data/libro.json`: `{"<clave del tablero>": [origen, destino]}`. `strategy.preparar()`
lo carga si existe; `jugar()` lo consulta antes de buscar, solo mientras la posición
esté ahí (si no, busca en vivo como siempre — el libro nunca es obligatorio).

Por qué al final (ver el plan): si la eval o la búsqueda todavía tuvieran un bug, un
libro construido antes de corregirlo lo congelaría para siempre. Este se regenera con
la versión actual de `busqueda_numba` + `eval.PESOS`, así que ya refleja los pesos
entrenados, no los de mano.

No corre en la liga.

Uso:
    cd estudiantes/iacasitegano
    python herramientas/libro_apertura.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

CARPETA_EQUIPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARPETA_EQUIPO))
sys.path.insert(0, str(CARPETA_EQUIPO.parent.parent / "sistema"))

from motor import Estado, Jugada  # noqa: E402

import busqueda_numba as bn  # noqa: E402
import eval as ev  # noqa: E402

PARTIDAS_POR_SEMILLA = 100
PROFUNDIDAD = 4  # barata: solo nos importa que todas las partidas vean las MISMAS aperturas
PLIES_LIBRO = 4  # cuántas de nuestras jugadas cubre el libro (las primeras, las más repetidas)
MUESTRAS_MINIMAS = 10  # no metemos una entrada sin haberla visto al menos esta cantidad de veces
VENTAJA_MINIMA = 0.08  # solo si la mejor jugada ganó claramente más que la segunda: si no, que
# decida la búsqueda en vivo (no vale la pena congelar una apertura dudosa)
SEMILLAS = (4242, 5151, 6262)


def _jugar_partida(rng: random.Random, pesos: np.ndarray, ganancia: float):
    """Self-play (mismos pesos en ambos lados). Regresa [(clave, origen, destino), ...] de
    nuestras primeras `PLIES_LIBRO` jugadas, y si ganamos esa partida."""
    estado = Estado.inicial()
    nuestro_color = rng.choice(["blancas", "negras"])
    posiciones: list[tuple[int, int, int]] = []
    ply_propio = 0
    while not estado.terminado():
        color = estado.turno
        tablero, puntos, color_idx = bn._tablero_desde_estado(estado)
        o, d, _ = bn._mejor_jugada_numba(
            tablero, puntos, color_idx, PROFUNDIDAD, pesos, ganancia, -1, -1
        )
        if color == nuestro_color and ply_propio < PLIES_LIBRO:
            posiciones.append((estado.clave, int(o), int(d)))
            ply_propio += 1
        jugada = Jugada(int(o), int(d))
        probs, hijos = zip(*estado.siguiente(jugada), strict=True)
        estado = rng.choices(hijos, weights=probs)[0]
    gano = estado.ganador() == nuestro_color
    return posiciones, gano


def generar_libro(partidas_por_semilla: int, semillas: tuple[int, ...]) -> dict[str, list[int]]:
    ruta_pesos_actuales = CARPETA_EQUIPO / "data" / "pesos.npy"
    if ruta_pesos_actuales.exists():  # los pesos entrenados, no los de mano (ver docstring)
        ev.cargar_pesos(np.load(ruta_pesos_actuales))
    ganancia = ev.cota() + 100.0
    pesos = np.ascontiguousarray(ev.PESOS, dtype=np.float64)
    # conteo[clave][(origen, destino)] = [victorias, total]
    conteo: dict[int, dict[tuple[int, int], list[int]]] = {}
    inicio = time.perf_counter()
    total = partidas_por_semilla * len(semillas)
    hechas = 0
    for semilla in semillas:
        rng = random.Random(semilla)
        for _ in range(partidas_por_semilla):
            posiciones, gano = _jugar_partida(rng, pesos, ganancia)
            for clave, o, d in posiciones:
                bucket = conteo.setdefault(clave, {})
                registro = bucket.setdefault((o, d), [0, 0])
                registro[1] += 1
                if gano:
                    registro[0] += 1
            hechas += 1
            if hechas % 10 == 0 or hechas == total:
                print(f"  {hechas}/{total} partidas, {time.perf_counter() - inicio:.0f}s")

    libro: dict[str, list[int]] = {}
    for clave, movimientos in conteo.items():
        candidatos = [
            (o, d, v[0] / v[1]) for (o, d), v in movimientos.items() if v[1] >= MUESTRAS_MINIMAS
        ]
        if not candidatos:
            continue
        candidatos.sort(key=lambda c: c[2], reverse=True)
        mejor = candidatos[0]
        if len(candidatos) > 1 and mejor[2] - candidatos[1][2] < VENTAJA_MINIMA:
            continue  # no está claro cuál es mejor: que decida la búsqueda en vivo
        libro[str(clave)] = [mejor[0], mejor[1]]
    return libro


def main() -> None:
    total = PARTIDAS_POR_SEMILLA * len(SEMILLAS)
    print(f"Generando {total} partidas de self-play (semillas: {SEMILLAS}; "
          f"profundidad {PROFUNDIDAD}, primeras {PLIES_LIBRO} jugadas propias)...")
    libro = generar_libro(PARTIDAS_POR_SEMILLA, SEMILLAS)
    print(f"{len(libro)} posiciones con jugada clara en el libro "
          f"(mínimo {MUESTRAS_MINIMAS} muestras, ventaja mínima {VENTAJA_MINIMA}).")
    destino = CARPETA_EQUIPO / "data" / "libro.json"
    destino.write_text(json.dumps(libro))
    print(f"Guardado en {destino}")


if __name__ == "__main__":
    main()
