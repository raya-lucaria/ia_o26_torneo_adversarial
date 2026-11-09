"""Genera el libro de aperturas `data/libro.json` (OFFLINE; no se ejecuta en la liga).

Para cada posición de apertura donde nos toca jugar, corre la misma búsqueda del bot
(`strategy.analizar`: bandido UCB1 con azar exacto) con mucho más tiempo que en partida
y guarda la jugada elegida. Cubre TODAS las respuestas del rival:

- Blancas: nuestra jugada 1, 2 y 3 (1 + ~25 + ~625 posiciones).
- Negras: nuestra jugada 1 y 2 (~25 + ~625 posiciones).

Cada ronda parte de la jugada del libro de la ronda anterior, aplica cada desenlace de
esa jugada y luego cada respuesta del rival (y sus desenlaces, si ataca).

Uso (desde la raíz del repo, con .venv activo; ~75 min con 3 procesos):
  python estudiantes/frame_perfect_push/herramientas/generar_libro.py --procesos 3
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, current_process
from pathlib import Path

EQUIPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EQUIPO))

import strategy  # noqa: E402
from motor import Estado  # noqa: E402

RONDAS = {"blancas": 3, "negras": 2}  # jugadas propias cubiertas por color
SEGUNDOS = [30.0, 15.0, 10.0]  # por posición, según la ronda
SEGUNDOS_RAIZ = 60.0  # la primera jugada de Blancas (una sola posición)


def _iniciar(nucleos):
    if nucleos and hasattr(os, "sched_setaffinity"):
        indice = (current_process()._identity or (1,))[0] - 1
        os.sched_setaffinity(0, {nucleos[indice % len(nucleos)]})
    strategy.simular_al_azar(Estado.inicial(), 1, 0, para="blancas")  # calentar numba


def _analizar(tarea):
    estado_dict, color, segundos, semilla = tarea
    estado = Estado.desde_dict(estado_dict)
    jugada, valor, tiradas = strategy.analizar(estado, segundos, color, semilla)
    return str(estado.clave), str(jugada), round(valor, 4), tiradas


def _desenlaces(estado, jugada):
    return [s for _, s in estado.siguiente(jugada) if s.ganador() is None]


def _siguiente_frontera(frontera, libro):
    """Posiciones (nos toca) tras la jugada del libro y cada respuesta del rival."""
    nueva = {}
    for estado, color in frontera:
        texto = libro.get(str(estado.clave))
        if texto is None:
            continue
        for s in _desenlaces(estado, estado._legal(texto)):
            for r in s.jugadas():
                for t in _desenlaces(s, r):
                    nueva.setdefault(t.clave, (t, color))
    return list(nueva.values())


def _guardar(libro, info, ruta):
    datos = {"version": 1, "info": info, "jugadas": libro}
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, indent=0, sort_keys=True), encoding="utf-8")
    tmp.replace(ruta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--procesos", type=int, default=3)
    p.add_argument("--escala", type=float, default=1.0, help="multiplica los segundos (prueba rápida: 0.01)")
    p.add_argument("--salida", type=Path, default=EQUIPO / "data" / "libro.json")
    p.add_argument("--semilla", type=int, default=20261108)
    args = p.parse_args()
    salida = args.salida.resolve()
    if not salida.is_relative_to(EQUIPO.resolve()):
        p.error("la salida debe quedar dentro de la carpeta del equipo")

    nucleos = sorted(os.sched_getaffinity(0))[: args.procesos] if hasattr(os, "sched_getaffinity") else []
    libro = {}
    info = {"segundos": SEGUNDOS, "segundos_raiz": SEGUNDOS_RAIZ, "rondas": RONDAS, "semilla": args.semilla}
    inicial = Estado.inicial()
    fronteras = {
        "blancas": [(inicial, "blancas")],
        "negras": [(s, "negras") for j in inicial.jugadas() for s in _desenlaces(inicial, j)],
    }
    with Pool(args.procesos, initializer=_iniciar, initargs=(nucleos,)) as pool:
        for ronda in range(max(RONDAS.values())):
            tareas = []
            for color, frontera in fronteras.items():
                if ronda >= RONDAS[color]:
                    continue
                for k, (estado, _) in enumerate(frontera):
                    seg = SEGUNDOS_RAIZ if (ronda == 0 and color == "blancas") else SEGUNDOS[ronda]
                    tareas.append((estado.a_dict(), color, seg * args.escala, args.semilla + ronda * 100_000 + k))
            t = time.perf_counter()
            print(f"ronda {ronda}: {len(tareas)} posiciones", flush=True)
            for clave, jugada, valor, tiradas in pool.imap_unordered(_analizar, tareas):
                libro[clave] = jugada
            _guardar(libro, info, salida)
            print(f"ronda {ronda}: lista en {time.perf_counter() - t:.0f} s; libro con {len(libro)}", flush=True)
            for color in fronteras:
                if ronda + 1 < RONDAS[color]:
                    fronteras[color] = _siguiente_frontera(fronteras[color], libro)
    print(f"Libro: {salida} ({len(libro)} posiciones)")


if __name__ == "__main__":
    main()
