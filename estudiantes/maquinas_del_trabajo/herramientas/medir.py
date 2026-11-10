"""Mediciones del bot: compilación, velocidad y fuzz de legalidad.

Uso (desde estudiantes/maquinas_del_trabajo/):
    ../../.venv/bin/python -m herramientas.medir
    NUMBA_BOUNDSCHECK=1 ../../.venv/bin/python -m herramientas.medir --fuzz 100000

- Compilación: en un proceso nuevo, calienta el motor oficial como lo hace el runner y mide
  `import strategy` + `strategy.preparar(config)`, que es lo que cuenta contra los 20 s
  (el validador avisa arriba de 10 s).
- Velocidad: nodos por segundo y profundidad alcanzada pensando 0.7 s (corte duro 2.1 s) en
  posiciones de medio juego, cada una con memoria nueva (como la primera jugada de una partida).
- `--fuzz N`: búsqueda a profundidad 2 en N posiciones (misma memoria para todas) y `jugar` con
  0.05 s en 10 000 posiciones sintéticas. Con NUMBA_BOUNDSCHECK=1, un índice fuera de rango
  truena con IndexError en vez de corromper memoria.

Las cifras oficiales son las de Docker y la prueba de GitHub; las de esta máquina son referencia.
"""

import argparse
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from busqueda import buscar, crear_memoria, profundizar
from herramientas.posiciones import muestrear
from motor_rapido import desde_estado, texto_jugada

CARPETA = Path(__file__).resolve().parent.parent
CONFIG = SimpleNamespace(color="blancas", semilla=1, carpeta_datos=CARPETA / "data", reloj_s=30.0)

# Lo que corre el proceso nuevo: primero lo mismo que el runner hace fuera del reloj.
MEDIR_PREPARAR = """
import time
from pathlib import Path
from types import SimpleNamespace
from motor import Estado, simular_al_azar
estado = Estado.inicial()
estado.siguiente(estado.jugadas()[0])
simular_al_azar(estado, 1, 0, "blancas")
inicio = time.perf_counter()
import strategy
strategy.preparar(SimpleNamespace(color="blancas", semilla=1, carpeta_datos=Path("data"), reloj_s=30.0))
print(time.perf_counter() - inicio)
"""


def medir_compilacion() -> float:
    """Segundos de `import strategy` + `preparar` en un proceso nuevo."""
    salida = subprocess.run(
        [sys.executable, "-c", MEDIR_PREPARAR], cwd=CARPETA, capture_output=True, text=True, check=True
    )
    return float(salida.stdout.strip().splitlines()[-1])


def medir_velocidad(cuantas: int = 30) -> tuple[float, list[int], float]:
    """(nodos/s, profundidades alcanzadas, segundos promedio por jugada) pensando 0.7 s."""
    posiciones = muestrear(cuantas, 21, sinteticas=0, ply_min=20, ply_max=60)
    tab, pts, color = desde_estado(posiciones[0])
    profundizar(crear_memoria(bits_tt=16), tab, pts, color, 0.05, 0.1)  # compila
    profundidades, nodos, segundos = [], 0, 0.0
    for estado in posiciones:
        tab, pts, color = desde_estado(estado)
        r = profundizar(crear_memoria(), tab, pts, color, 0.7, 2.1)
        profundidades.append(r.prof)
        nodos += r.nodos
        segundos += r.segundos
    return nodos / segundos, profundidades, segundos / cuantas


def fuzz(cuantas: int) -> int:
    """Cuántas jugadas ilegales salieron (0 es lo esperado)."""
    import strategy

    malas = 0
    memoria = crear_memoria(bits_tt=18)
    for i, estado in enumerate(muestrear(cuantas, 31), 1):
        tab, pts, color = desde_estado(estado)
        _, cod = buscar(memoria, tab, pts, color, 2)
        if texto_jugada(cod) not in {str(j) for j in estado.jugadas()}:
            malas += 1
            print("Ilegal (buscar):", estado.a_dict(), texto_jugada(cod))
        if i % 20_000 == 0:
            print(f"  buscar: {i}/{cuantas}", flush=True)
    strategy.preparar(CONFIG)
    for estado in muestrear(10_000, 32, sinteticas=1.0):
        jugada = strategy.jugar(estado, 0.05)
        if jugada not in {str(j) for j in estado.jugadas()}:
            malas += 1
            print("Ilegal (jugar):", estado.a_dict(), jugada)
    return malas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fuzz", type=int, default=0, help="posiciones para el fuzz (0 = no)")
    args = parser.parse_args(argv)

    if args.fuzz:
        malas = fuzz(args.fuzz)
        print(f"fuzz: {malas} jugadas ilegales en {args.fuzz} búsquedas + 10000 jugadas")
        return 0 if malas == 0 else 1

    compilacion = medir_compilacion()
    velocidad, profundidades, por_jugada = medir_velocidad()
    mediana = statistics.median(profundidades)
    puertas = {
        "compilación ≤ 3 s": compilacion <= 3.0,
        "≥ 1 M nodos/s": velocidad >= 1e6,
        "profundidad mediana ≥ 6": mediana >= 6,
    }
    print(f"compilación (import + preparar): {compilacion:.2f} s")
    print(f"velocidad: {velocidad / 1e6:.2f} M nodos/s, {por_jugada:.2f} s por jugada")
    print(
        f"profundidad: mediana {mediana}, mínima {min(profundidades)}, máxima {max(profundidades)}"
    )
    for nombre, paso in puertas.items():
        print(f"  {'✓' if paso else '✗'} {nombre}")
    return 0 if all(puertas.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
