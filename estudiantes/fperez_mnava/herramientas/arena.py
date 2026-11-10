"""Arena: partidas entre dos bots, con los dados y el reloj del torneo, sin Docker.

Sirve para medir versiones de tu bot (A) contra rivales de prueba o contra tu versión anterior (B).
Usa todos los núcleos (offline se vale). NO es el árbitro oficial: la prueba final es `./torneo`.

Uso, desde la raíz del repo con el entorno activo (`source .venv/bin/activate`):

    python estudiantes/<equipo>/herramientas/arena.py estudiantes/<equipo> estudiantes/montecarlo -n 25
    python estudiantes/<equipo>/herramientas/arena.py estudiantes/<equipo> \
        estudiantes/<equipo>/herramientas/rivales/material2.py -n 25 --reloj 5 --jobs 4

Cada bot es una carpeta con `strategy.py` o un archivo `.py` con `preparar` y `jugar`.
`-n` es el número de juegos de dados: cada uno se juega con los dos colores (2n partidas),
igual que en la liga, así los dos bots reciben la misma suerte con cada color.

Limitación: si A y B importan un módulo auxiliar con el mismo nombre (p. ej. `mi_motor`),
comparten ese módulo en el mismo proceso. Para comparar dos versiones tuyas, dale a cada una
su propio nombre de auxiliar.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from motor import Estado, Jugada

COLORES = ("blancas", "negras")
PREPARAR_S = 10.0  # mismo límite que la liga
MAX_JUGADAS = 400  # red de seguridad; una partida real termina mucho antes

_cache: dict[tuple[str, str], SimpleNamespace] = {}  # (ruta, ranura) -> módulo del bot


def _cargar(ruta: str, ranura: str) -> SimpleNamespace:
    """Importa el bot una vez por proceso y ranura (A o B), para que un espejo no comparta globales."""
    clave = (ruta, ranura)
    if clave not in _cache:
        p = Path(ruta)
        archivo = p if p.suffix == ".py" else p / "strategy.py"
        carpeta = archivo.parent
        sys.path.insert(0, str(carpeta))
        # Carga dinámica a propósito: la arena tiene que importar los bots que va a enfrentar.
        # El validador la marca para revisión manual; es esperable (solo vive en herramientas/).
        try:
            spec = importlib.util.spec_from_file_location(f"bot_{ranura}_{abs(hash(ruta))}", archivo)
            modulo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo)
        finally:
            sys.path.remove(str(carpeta))
        _cache[clave] = SimpleNamespace(modulo=modulo, carpeta=carpeta)
    return _cache[clave]


class Dados:
    """Un generador por color. El ataque k de un color usa el número k de su generador (como el árbitro)."""

    def __init__(self, semilla: int, pareja: int, juego: int) -> None:
        self._g = [
            np.random.Generator(np.random.PCG64(np.random.SeedSequence([semilla, pareja, juego, c])))
            for c in range(2)
        ]

    def tirar(self, color: str) -> float:
        return float(self._g[COLORES.index(color)].random())


def jugar_partida(t: dict) -> dict:
    """Una partida. `t` trae rutas por color, semilla de dados y reloj. Regresa quién ganó y por qué."""
    bots = {
        "blancas": _cargar(t["ruta_blancas"], t["ranura_blancas"]),
        "negras": _cargar(t["ruta_negras"], t["ranura_negras"]),
    }
    dados = Dados(t["semilla"], t["pareja"], 0)
    reloj = t["reloj"]
    usado = dict.fromkeys(COLORES, 0.0)
    perdedor = motivo = None

    for c in COLORES:  # preparar: 10 s aparte del reloj
        config = SimpleNamespace(
            color=c,
            semilla=int(np.random.SeedSequence([t["semilla"], t["pareja"], COLORES.index(c), 7]).generate_state(1)[0]),
            carpeta_datos=bots[c].carpeta / "data",
            reloj_s=reloj,
        )
        t0 = time.perf_counter()
        try:
            bots[c].modulo.preparar(config)
        except Exception as e:  # noqa: BLE001
            perdedor, motivo = c, f"error en preparar: {e!r}"
            break
        if time.perf_counter() - t0 > PREPARAR_S:
            perdedor, motivo = c, "preparar > 10 s"
            break

    estado = Estado.inicial()
    n = 0
    while perdedor is None and estado.ganador() is None and n < MAX_JUGADAS:
        c = estado.turno
        t0 = time.perf_counter()
        try:
            resp = bots[c].modulo.jugar(estado, max(0.0, reloj - usado[c]))
        except Exception as e:  # noqa: BLE001
            perdedor, motivo = c, f"excepción: {e!r}"
            break
        usado[c] += time.perf_counter() - t0
        if usado[c] > reloj:
            perdedor, motivo = c, "reloj"
            break
        texto = str(resp)
        if texto not in {str(j) for j in estado.jugadas()}:
            perdedor, motivo = c, f"jugada ilegal {texto!r}"
            break
        j = Jugada.desde_texto(texto)
        if estado.es_ataque(j):
            estado = estado.aplicar(j, gana_ataque=dados.tirar(c) < estado.prob_ataque(j))
        else:
            estado = estado.aplicar(j)
        n += 1

    if perdedor is not None:
        ganador = COLORES[1 - COLORES.index(perdedor)]
    else:
        ganador = estado.ganador()
        motivo = "puntos" if max(estado.puntos.values()) >= 3 else "sin jugadas"
    return {**t, "ganador": ganador, "motivo": motivo, "jugadas": n, "usado": usado}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson al 95 % para una proporción."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centro = p + z * z / (2 * n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centro - margen) / d, (centro + margen) / d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="bot A (carpeta con strategy.py o archivo .py)")
    ap.add_argument("b", help="bot B")
    ap.add_argument("-n", type=int, default=10, help="juegos de dados (cada uno con los 2 colores)")
    ap.add_argument("--reloj", type=float, default=30.0, help="reloj por partida en segundos (liga: 30)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="procesos en paralelo")
    ap.add_argument("--semilla", type=int, default=2026, help="semilla maestra de los dados")
    args = ap.parse_args()

    tareas = []
    for i in range(args.n):
        for a_blancas in (True, False):
            ra, rb = args.a, args.b
            tareas.append(
                {
                    "ruta_blancas": ra if a_blancas else rb,
                    "ranura_blancas": "A" if a_blancas else "B",
                    "ruta_negras": rb if a_blancas else ra,
                    "ranura_negras": "B" if a_blancas else "A",
                    "semilla": args.semilla,
                    "pareja": i,
                    "reloj": args.reloj,
                    "a_color": "blancas" if a_blancas else "negras",
                }
            )

    t0 = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            res = list(ex.map(jugar_partida, tareas))
    else:
        res = [jugar_partida(t) for t in tareas]

    ganadas = sum(r["ganador"] == r["a_color"] for r in res)
    total = len(res)
    lo, hi = wilson(ganadas, total)
    print(f"\nA = {args.a}\nB = {args.b}")
    print(f"A gana {ganadas} de {total} ({ganadas / total:.0%}); IC 95 %: {lo:.0%} a {hi:.0%}")
    for c in COLORES:
        sub = [r for r in res if r["a_color"] == c]
        g = sum(r["ganador"] == c for r in sub)
        print(f"  A con {c}: {g} de {len(sub)}")
    motivos: dict[str, int] = {}
    for r in res:
        if r["ganador"] != r["a_color"]:
            motivos[r["motivo"]] = motivos.get(r["motivo"], 0) + 1
    if motivos:
        print("  Derrotas de A por motivo:", motivos)
    maximo = max(r["usado"][r["a_color"]] for r in res)
    print(f"  Reloj máx. usado por A: {maximo:.2f} s de {args.reloj:.0f} s")
    print(f"  Jugadas por partida (media): {sum(r['jugadas'] for r in res) / total:.0f}")
    print(f"  Tiempo total: {time.perf_counter() - t0:.0f} s")
    if hi - lo > 0.25:
        print("  Aviso: el intervalo es muy ancho; sube -n antes de decidir entre versiones.")


if __name__ == "__main__":
    main()
