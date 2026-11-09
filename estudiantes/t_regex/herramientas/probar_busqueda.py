"""Pruebas de la búsqueda (no se ejecuta en la liga).

1. Mide cuánto tarda compilar numba (debe caber holgado en los 10 s de `preparar`).
2. Verifica que negamax con poda Star1 + tabla da el mismo valor que expectiminimax puro.
3. Mide nodos por segundo desde la posición inicial.

Uso (con los límites de la liga, 1 núcleo):
    docker run --rm --cpus=1 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/probar_busqueda.py
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import busqueda as bq  # noqa: E402
import strategy  # noqa: E402
from motor import Estado  # noqa: E402


def posicion_al_azar(rng, jugadas):
    e = Estado.inicial()
    for _ in range(jugadas):
        js = e.jugadas()
        if not js or e.terminado():
            break
        j = rng.choice(js)
        e = e.aplicar(j, gana_ataque=rng.random() < 0.5) if e.es_ataque(j) else e.aplicar(j)
    return e


class Config:
    color, semilla, reloj_s = "blancas", 1, 30.0
    carpeta_datos = Path("data")


def main():
    t = time.perf_counter()
    strategy.preparar(Config())
    print(f"preparar (compila numba): {time.perf_counter() - t:.2f} s")
    t = time.perf_counter()
    bq.expectimax_puro(*strategy.a_arreglos(Estado.inicial()), 0, 1, 0, bq.nuevos_buferes())
    print(f"compilar expectimax_puro: {time.perf_counter() - t:.2f} s")

    rng = random.Random(7)
    malos = 0
    for k in range(60):
        e = posicion_al_azar(rng, rng.randrange(10, 70))
        if e.terminado():
            continue
        tab, pts = strategy.a_arreglos(e)
        color = 0 if e.turno == "blancas" else 1
        prof = 3 if k % 2 else 2
        esperado = bq.expectimax_puro(tab.copy(), pts.copy(), color, prof, 0, bq.nuevos_buferes())
        _, v, _, _, _ = bq.buscar(bq.nuevo_contexto(), tab, pts, color, prof, -1, 1 << 62)
        if abs(v - esperado) > 1e-9:
            malos += 1
            print(f"DIFERENTE prof={prof}: poda={v:.6f} puro={esperado:.6f}\n{e}")
    print(f"comparación con expectimax puro: {'OK' if malos == 0 else f'{malos} diferencias'}")

    tab, pts = strategy.a_arreglos(Estado.inicial())
    ctx = bq.nuevo_contexto()
    for prof in range(1, 10):
        t = time.perf_counter()
        j, v, _, nodos, _ = bq.buscar(ctx, tab, pts, 0, prof, -1, 1 << 62)
        dt = time.perf_counter() - t
        print(f"prof {prof}: {nodos:>9} nodos  {dt:6.2f} s  "
              f"{nodos / max(dt, 1e-9):>9.0f} nodos/s  valor {v:+.3f}")
        if dt > 5:
            break


if __name__ == "__main__":
    main()
