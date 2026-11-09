"""Arena offline: partidas rápidas en paralelo entre versiones del bot y los bots de referencia.

No se ejecuta en la liga. Sirve para medir (muchas partidas, todos los núcleos) y para ajustar
los pesos de la evaluación. Se corre desde la raíz del repo con el .venv activo:

    PYTHONPATH=sistema:estudiantes/pamplinotas python estudiantes/pamplinotas/herramientas/arena.py \
        --a buscador --b montecarlo --partidas 40 --reloj 30

Jugadores: `buscador` (nuestro bot con los pesos de data/; `--pesos-a archivo.json` para otros,
`--pesos-a ""` para los pesos a mano, `--prof-a N` para profundidad fija
en vez de reloj), `montecarlo` (el mismo algoritmo que estudiantes/montecarlo, con el reloj que
se le dé) y `random`.
"""

import argparse
import itertools
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

from motor import Estado, Jugada, simular_al_azar

import buscador
import strategy

PESOS_ENTREGADOS = str(Path(__file__).resolve().parent.parent / "data" / "pesos.json")

# --- Jugadores -------------------------------------------------------------------------------


class JugadorBuscador:
    def __init__(self, pesos=None, reloj_s=30.0, prof=None, quietud_ataques=True, reloj_cfg=None, lmr=None):
        self.b = buscador.Buscador(pesos, quietud_ataques=quietud_ataques, lmr=lmr)
        self.reloj = reloj_s
        cfg = reloj_cfg or {}
        self.reserva = min(strategy.RESERVA_MAX_S, cfg.get("reserva", strategy.RESERVA_FRACCION) * reloj_s)
        self.esperadas = cfg.get("esperadas", strategy.JUGADAS_PROPIAS_ESPERADAS)
        self.minimo = cfg.get("minimo", strategy.MIN_JUGADAS_RESTANTES)
        self.prof = prof
        self.hechas = 0

    def jugar(self, estado):
        tab, color, pts = buscador.tablero_desde_estado(estado)
        self.hechas += 1
        if self.prof is not None:
            j, v, _ = self.b.buscar(tab, color, pts, self.prof, time.perf_counter() + 60)
            return buscador.jugada_a_texto(j)
        inicio = time.perf_counter()
        objetivo = strategy.presupuesto(self.reloj, self.reserva, self.hechas, self.esperadas, self.minimo)
        mejor, _, _, _ = strategy.elegir(self.b, tab, color, pts, inicio, objetivo)
        self.reloj -= time.perf_counter() - inicio
        return buscador.jugada_a_texto(mejor) if mejor >= 0 else str(estado.jugadas()[0])


class JugadorMontecarlo:
    """Copia fiel de estudiantes/montecarlo/strategy.py, con reloj configurable."""

    SIMULACIONES = 20

    def __init__(self, color, semilla, reloj_s=30.0):
        self.color = color
        self.rng = random.Random(semilla)
        self.reserva = 0.3 * reloj_s
        self.reloj = reloj_s
        simular_al_azar(Estado.inicial(), 1, 0, para=color)

    def jugar(self, estado):
        inicio = time.perf_counter()
        disponible = self.reloj - self.reserva
        limite = inicio + disponible / 25
        jugadas = estado.jugadas()
        for j in jugadas:
            if all(hijo.ganador() == self.color for _, hijo in estado.siguiente(j)):
                return str(j)
        if disponible <= 0:
            return str(self.rng.choice(jugadas))
        total = {j: 0.0 for j in jugadas}
        muestras = {j: 0 for j in jugadas}
        for hechas, j in enumerate(itertools.cycle(jugadas), 1):
            probs, hijos = zip(*estado.siguiente(j), strict=True)
            hijo = self.rng.choices(hijos, weights=probs)[0]
            total[j] += simular_al_azar(hijo, self.SIMULACIONES, self.rng.randrange(2**31), para=self.color)
            muestras[j] += 1
            if hechas >= len(jugadas) and time.perf_counter() >= limite:
                break
        mejor = max(jugadas, key=lambda j: total[j] / muestras[j])
        self.reloj -= time.perf_counter() - inicio
        return str(mejor)


class JugadorRandom:
    def __init__(self, semilla):
        self.rng = random.Random(semilla)

    def jugar(self, estado):
        return str(self.rng.choice(estado.jugadas()))


def crear(spec, color, semilla):
    tipo = spec["tipo"]
    if tipo == "buscador":
        pesos = None
        if spec.get("pesos"):
            with open(spec["pesos"], encoding="utf-8") as f:
                pesos = buscador.pesos_desde_json(json.load(f))
        if pesos is not None and spec.get("lineal"):
            pesos = pesos.copy()
            pesos[buscador.I_ESCALA_LOGIT] = 0.0
        return JugadorBuscador(pesos, spec.get("reloj", 30.0), spec.get("prof"), spec.get("quietud_ataques", True), spec.get("reloj_cfg"), spec.get("lmr"))
    if tipo == "montecarlo":
        return JugadorMontecarlo(color, semilla, spec.get("reloj", 30.0))
    if tipo == "random":
        return JugadorRandom(semilla)
    raise ValueError(tipo)


# --- Partidas ---------------------------------------------------------------------------------


def partida(args):
    """Una partida; `args` = (spec_blancas, spec_negras, semilla). Regresa (ganador, puntos, jugadas)."""
    spec_b, spec_n, semilla = args
    dados = {0: random.Random(semilla * 2 + 1), 1: random.Random(semilla * 2 + 2)}
    jugadores = {"blancas": crear(spec_b, "blancas", semilla), "negras": crear(spec_n, "negras", semilla + 7)}
    e = Estado.inicial()
    n = 0
    while not e.terminado():
        color = e.turno
        texto = jugadores[color].jugar(e)
        j = Jugada.desde_texto(texto)
        if e.es_ataque(j):
            u = dados[0 if color == "blancas" else 1].random()
            e = e.aplicar(j, gana_ataque=u < e.prob_ataque(j))
        else:
            e = e.aplicar(j)
        n += 1
        if n > 400:
            break
    return e.ganador(), e.puntos, n


def enfrentar(spec_a, spec_b, partidas, semilla=0, procesos=None):
    """`partidas` juegos alternando colores. Regresa (victorias_a, victorias_b)."""
    tareas = []
    for i in range(partidas):
        s = semilla * 100_000 + i
        if i % 2 == 0:
            tareas.append((spec_a, spec_b, s))
        else:
            tareas.append((spec_b, spec_a, s))
    with mp.Pool(procesos) as pool:
        resultados = pool.map(partida, tareas, chunksize=1)
    va = vb = 0
    for i, (ganador, _, _) in enumerate(resultados):
        a_es = "blancas" if i % 2 == 0 else "negras"
        if ganador == a_es:
            va += 1
        elif ganador is not None:
            vb += 1  # una partida cortada por largo (sin ganador) no cuenta para nadie
    return va, vb


def reloj_cfg(o, lado):
    """Parámetros de reparto del reloj dados en la línea de comandos para el lado `lado`."""
    valores = {k: getattr(o, f"{k}_{lado}") for k in ("reserva", "esperadas", "minimo")}
    return {k: v for k, v in valores.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="buscador")
    ap.add_argument("--b", default="montecarlo")
    ap.add_argument("--pesos-a", default=PESOS_ENTREGADOS, help="json de pesos (por omisión, los de data/)")
    ap.add_argument("--pesos-b", default=PESOS_ENTREGADOS)
    ap.add_argument("--prof-a", type=int)
    ap.add_argument("--prof-b", type=int)
    ap.add_argument("--lineal-a", action="store_true", help="EVAL lineal (sin la escala logística)")
    ap.add_argument("--lineal-b", action="store_true")
    ap.add_argument("--sin-quietud-a", action="store_true", help="quietud solo con avances que anotan")
    ap.add_argument("--sin-quietud-b", action="store_true")
    ap.add_argument("--reloj", type=float, default=30.0)
    ap.add_argument("--reloj-b", type=float, help="reloj distinto para b (por omisión, el mismo)")
    for lado in "ab":
        ap.add_argument(f"--reserva-{lado}", type=float, help="fracción del reloj en reserva (RESERVA_FRACCION)")
        ap.add_argument(f"--esperadas-{lado}", type=int, help="jugadas propias esperadas (JUGADAS_PROPIAS_ESPERADAS)")
        ap.add_argument(f"--minimo-{lado}", type=int, help="mínimo de jugadas restantes supuestas (MIN_JUGADAS_RESTANTES)")
    ap.add_argument("--partidas", type=int, default=20)
    ap.add_argument("--lmr-a", type=int, help="LMR en a: desde qué jugada se reduce (0 = no)")
    ap.add_argument("--lmr-b", type=int)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--procesos", type=int)
    o = ap.parse_args()
    spec_a = {"tipo": o.a, "pesos": o.pesos_a, "prof": o.prof_a, "reloj": o.reloj, "quietud_ataques": not o.sin_quietud_a, "lineal": o.lineal_a, "reloj_cfg": reloj_cfg(o, "a"), "lmr": o.lmr_a}
    spec_b = {"tipo": o.b, "pesos": o.pesos_b, "prof": o.prof_b, "reloj": o.reloj_b or o.reloj, "quietud_ataques": not o.sin_quietud_b, "lineal": o.lineal_b, "reloj_cfg": reloj_cfg(o, "b"), "lmr": o.lmr_b}
    t = time.perf_counter()
    va, vb = enfrentar(spec_a, spec_b, o.partidas, o.semilla, o.procesos)
    print(f"{o.a} {va} – {vb} {o.b}  ({o.partidas} partidas, reloj {o.reloj} s, {time.perf_counter() - t:.0f} s)")


if __name__ == "__main__":
    main()
