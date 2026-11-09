"""Arena local: partidas en proceso y en paralelo, con reloj y dados propios.

Uso (desde estudiantes/maquinas_del_trabajo/):
    ../../.venv/bin/python -m herramientas.arena --partidas 40
    ../../.venv/bin/python -m herramientas.arena --peso APOYO=0.1 --rival nuestro
    ../../.venv/bin/python -m herramientas.arena --rival nuestro --pesos nuevos.npz \
        --nodos 500000 --sprt --partidas 1000
    ../../.venv/bin/python -m herramientas.arena --rival nuestro --nodos 20000 \
        --nodos-rival 80000 --partidas 200

Juega «nuestro» bot (con sus pesos de `data/pesos.npz`, o los a mano de `evaluacion.py` si no
hay; otros con --pesos o cambiados con --peso) contra el rival: `montecarlo` (copia del bot de
referencia), `nuestro` (con --pesos-rival o --peso-rival) o uno de los estilos de
`herramientas/rivales.py` (`agresivo`, `corredor`, `defensivo`, `equilibrado`). Las partidas van
en pares con la misma semilla y los colores invertidos, para quitar suerte.

- Con --nodos (y --nodos-rival), nuestro bot busca exactamente esos nodos en cada jugada en vez
  de usar el reloj: el resultado no depende de la máquina ni de su carga. Ojo con cuántos:
  las diferencias de evaluación solo aparecen a la profundidad de una partida real. Medido
  (tabla L2 1e-3 contra 1e-4): 20k y 100k nodos → 54–55 %; 500k → 68 %; reloj de 30 s → 72 %.
  Para decidir, 500k nodos (~16 min por 200 partidas); 20k (~1 min por 400) solo sirve para
  detectar errores gruesos. Los cambios que mueven la velocidad se confirman con reloj.
- Con --sprt se juega por pares hasta que la prueba secuencial decide (`herramientas/sprt.py`):
  «H1» = la nueva es mejor por al menos --elo1, «H0» = no lo es. --partidas es el máximo.
- --reloj-rival da al rival otro reloj (por ejemplo, un montecarlo con más tiempo).
- --guardar escribe las partidas que perdió nuestro (una por línea, en JSON: semilla, color,
  jugadas y dados) para revisarlas después.
- --opciones / --opciones-rival fijan las primeras `memoria.opciones` de cada bot (ver
  `busqueda.Memoria`), p. ej. `1,1`: así se compara una opción de la búsqueda encendida contra
  apagada.

Es para comparar versiones rápido; la verdad sobre reloj y memoria sigue siendo Docker
(`./torneo jugar`). Los combates se resuelven con un `random.Random` propio: nada que ver con
los dados del árbitro.
"""

import argparse
import json
import random
import sys
import time
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from motor import Estado

import busqueda
from busqueda import crear_memoria, profundizar
from evaluacion import NOMBRES, PESOS_F2
from herramientas import sprt
from herramientas.rival_montecarlo import Montecarlo
from herramientas.rivales import ESTILOS, Estilo
from motor_rapido import desde_estado, texto_jugada
from strategy import Bot, cargar_pesos

CARPETA = Path(__file__).resolve().parent.parent
SIN_LIMITE_S = 1e6  # con --nodos el reloj no corta la búsqueda: solo el número de nodos
# `jugadas`: [texto, ¿ganó el ataque?] de cada ply (None si no fue ataque); `puntos`: los finales.
Resultado = namedtuple("Resultado", ["ganador", "motivo", "plies", "reloj", "jugadas", "puntos"])
# Todo lo que define una serie (igual para cada par de partidas).
Serie = namedtuple(
    "Serie",
    ["rival", "cambios", "cambios_rival", "pesos", "pesos_rival", "nodos", "nodos_rival",
     "reloj", "reloj_rival", "factor", "opciones", "opciones_rival", "escala", "escala_rival"],
)


class Nuestro:
    """Nuestro bot con pesos dados (adaptador para la arena); con `nodos`, busca ese número
    de nodos por jugada en vez de usar el reloj; con `opciones`, las fija en su memoria."""

    def __init__(self, pesos, nodos=None, opciones=None, escala=None):
        self.pesos = pesos
        self.nodos = nodos
        self.opciones = opciones
        self.escala = escala

    def preparar(self, config):
        self.bot = Bot(pesos=self.pesos)
        if self.opciones is not None:
            self.bot.memoria.opciones[: len(self.opciones)] = self.opciones
        if self.escala is not None:
            self.bot.escala_tiempo = self.escala

    def jugar(self, estado, tiempo_restante):
        if self.nodos is None:
            return self.bot.jugar(estado, tiempo_restante)
        tab, pts, color = desde_estado(estado)
        r = profundizar(
            self.bot.memoria, tab, pts, color, SIN_LIMITE_S, SIN_LIMITE_S, max_nodos=self.nodos
        )
        return texto_jugada(r.jugada)


def crear_jugador(
    nombre: str, cambios: dict[str, float] | None = None, archivo=None, nodos=None, opciones=None,
    escala=None,
):
    """`montecarlo`, un estilo de `rivales.py` o `nuestro`: los pesos del bot (o los de
    `archivo` .npz) y los `cambios`."""
    if nombre == "montecarlo":
        return Montecarlo()
    if nombre in ESTILOS:
        return Estilo(nombre)
    pesos = np.load(archivo)["pesos"] if archivo is not None else cargar_pesos(CARPETA / "data")
    if pesos is None:
        pesos = PESOS_F2.copy()
    for clave, valor in (cambios or {}).items():
        if clave not in NOMBRES:
            raise ValueError(f"Peso desconocido: {clave}. Opciones: {', '.join(NOMBRES)}")
        pesos[NOMBRES.index(clave)] = valor
    return Nuestro(pesos, nodos, opciones, escala)


def partida(semilla: int, blancas, negras, reloj_s: float = 30.0, reloj_negras=None) -> Resultado:
    """Una partida completa. Pierde quien regresa una jugada ilegal o se pasa de su reloj.

    Cada bando tiene `reloj_s` segundos, salvo Negras si se da `reloj_negras`.
    """
    rng = random.Random(semilla)
    jugadores = {"blancas": blancas, "negras": negras}
    relojes = {"blancas": reloj_s, "negras": reloj_s if reloj_negras is None else reloj_negras}
    for color, jugador in jugadores.items():
        jugador.preparar(
            SimpleNamespace(
                color=color, semilla=semilla, carpeta_datos=CARPETA / "data",
                reloj_s=relojes[color],
            )
        )
    usado = {"blancas": 0.0, "negras": 0.0}
    estado = Estado.inicial()
    plies = 0
    jugadas = []
    while not estado.terminado():
        turno = estado.turno
        otro = "negras" if turno == "blancas" else "blancas"
        inicio = time.perf_counter()
        texto = str(jugadores[turno].jugar(estado, max(0.0, relojes[turno] - usado[turno])))
        usado[turno] += time.perf_counter() - inicio
        if usado[turno] > relojes[turno]:
            return Resultado(otro, "reloj", plies, usado, jugadas, estado.puntos)
        legales = {str(j): j for j in estado.jugadas()}
        if texto not in legales:
            return Resultado(otro, "ilegal", plies, usado, jugadas, estado.puntos)
        jugada = legales[texto]
        gana = rng.random() < estado.prob_ataque(jugada) if estado.es_ataque(jugada) else None
        estado = estado.aplicar(jugada, gana)
        jugadas.append([texto, gana])
        plies += 1
    return Resultado(estado.ganador(), "juego", plies, usado, jugadas, estado.puntos)


def _jugar_par(trabajo) -> list[tuple[bool, str, int, float, dict]]:
    """Las dos partidas de una semilla (nuestro con Blancas y con Negras): por cada una,
    (¿ganó nuestro?, motivo, plies, reloj que usó nuestro, registro de la partida)."""
    semilla, s = trabajo
    busqueda.FACTOR_ITERACION = s.factor  # en este proceso (también al rival, si es nuestro)
    resultados = []
    for color in ("blancas", "negras"):
        nuestro = crear_jugador("nuestro", s.cambios, s.pesos, s.nodos, s.opciones, s.escala)
        otro = crear_jugador(
            s.rival, s.cambios_rival, s.pesos_rival, s.nodos_rival, s.opciones_rival, s.escala_rival
        )
        if color == "blancas":
            r = partida(semilla, nuestro, otro, s.reloj, s.reloj_rival)
        else:
            r = partida(semilla, otro, nuestro, s.reloj_rival, s.reloj)
        registro = {
            "semilla": semilla, "nuestro": color, "rival": s.rival, "ganador": r.ganador,
            "motivo": r.motivo, "puntos": r.puntos, "jugadas": r.jugadas,
        }
        resultados.append((r.ganador == color, r.motivo, r.plies, r.reloj[color], registro))
    return resultados


def _opciones(texto: str | None) -> list[int] | None:
    return None if texto is None else [int(x) for x in texto.split(",")]


def _cambios(textos: list[str]) -> dict[str, float]:
    cambios = {}
    for texto in textos:
        nombre, _, valor = texto.partition("=")
        cambios[nombre] = float(valor)
    return cambios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partidas", type=int, default=40, help="total (máximo con --sprt)")
    parser.add_argument("--procesos", type=int, default=6)
    parser.add_argument("--semilla", type=int, default=1)
    parser.add_argument("--reloj", type=float, help="segundos por bando (30; sin límite con --nodos)")
    parser.add_argument("--reloj-rival", type=float, help="si es distinto del nuestro")
    parser.add_argument(
        "--rival", choices=["montecarlo", "nuestro", *ESTILOS], default="montecarlo"
    )
    parser.add_argument("--peso", action="append", default=[], help="NOMBRE=valor (nuestro)")
    parser.add_argument("--peso-rival", action="append", default=[], help="NOMBRE=valor (rival)")
    parser.add_argument("--factor-iteracion", type=float, default=busqueda.FACTOR_ITERACION)
    parser.add_argument("--pesos", type=Path, help="pesos de nuestro (.npz de herramientas.entrenar)")
    parser.add_argument("--pesos-rival", type=Path, help="pesos del rival nuestro (.npz)")
    parser.add_argument("--nodos", type=int, help="nodos por jugada de nuestro (en vez del reloj)")
    parser.add_argument("--nodos-rival", type=int, help="nodos por jugada del rival nuestro")
    parser.add_argument("--opciones", help="memoria.opciones de nuestro, p. ej. 1,1,0")
    parser.add_argument("--opciones-rival", help="memoria.opciones del rival nuestro")
    parser.add_argument("--escala-tiempo", type=float, help="reloj.ESCALA_TIEMPO de nuestro")
    parser.add_argument("--escala-tiempo-rival", type=float, help="la del rival nuestro")
    parser.add_argument("--guardar", type=Path, help="archivo .jsonl con las partidas perdidas")
    parser.add_argument("--sprt", action="store_true", help="parar en cuanto el SPRT decida")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=35.0)
    parser.add_argument("--alfa", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args(argv)

    if args.reloj is None:
        # Con nodos fijos el reloj no debe decidir partidas: si la máquina va cargada, el bando
        # que busca más nodos perdería por tiempo y el resultado saldría inflado.
        args.reloj = SIN_LIMITE_S if args.nodos is not None or args.nodos_rival is not None else 30.0
    rival_nuestro = (args.nodos_rival, args.opciones_rival, args.escala_tiempo_rival)
    if any(x is not None for x in rival_nuestro) and args.rival != "nuestro":
        parser.error(
            "--nodos-rival, --opciones-rival y --escala-tiempo-rival solo aplican con --rival nuestro"
        )
    serie = Serie(
        rival=args.rival, cambios=_cambios(args.peso), cambios_rival=_cambios(args.peso_rival),
        pesos=args.pesos, pesos_rival=args.pesos_rival, nodos=args.nodos,
        nodos_rival=args.nodos_rival, reloj=args.reloj,
        reloj_rival=args.reloj if args.reloj_rival is None else args.reloj_rival,
        factor=args.factor_iteracion, opciones=_opciones(args.opciones),
        opciones_rival=_opciones(args.opciones_rival), escala=args.escala_tiempo,
        escala_rival=args.escala_tiempo_rival,
    )
    n_opciones = len(crear_memoria(bits_tt=4).opciones)
    for opciones in (serie.opciones, serie.opciones_rival):
        if opciones is not None and len(opciones) > n_opciones:
            parser.error(f"hay {n_opciones} opciones en busqueda.Memoria")
    crear_jugador("nuestro", serie.cambios, serie.pesos)  # valida antes de lanzar procesos
    trabajos = [(args.semilla * 100_003 + k, serie) for k in range(args.partidas // 2)]
    inicio = time.perf_counter()
    pares, resultados, veredicto = [], [], ""
    with Pool(args.procesos) as pool:  # al salir del bloque se detienen los pares pendientes
        for par in pool.imap_unordered(_jugar_par, trabajos):
            resultados.extend(par)
            pares.append(sum(g for g, *_ in par) / 2)
            if args.sprt:
                veredicto = sprt.decision(
                    sprt.llr(pares, args.elo0, args.elo1), args.alfa, args.beta
                )
                if veredicto:
                    break
    ganadas = sum(g for g, *_ in resultados)
    por_color = [sum(g for g, *_ in resultados[k::2]) for k in (0, 1)]
    raras = [m for _, m, *_ in resultados if m != "juego"]
    print(
        f"nuestro {ganadas} – {len(resultados) - ganadas} {args.rival}"
        f" ({100 * ganadas / len(resultados):.0f} %; con blancas {por_color[0]},"
        f" con negras {por_color[1]}) · plies promedio"
        f" {sum(p for _, _, p, *_ in resultados) / len(resultados):.0f} · reloj nuestro máx"
        f" {max(r for _, _, _, r, _ in resultados):.1f} s · {time.perf_counter() - inicio:.0f} s"
    )
    elo, bajo, alto = sprt.intervalo_elo(pares)
    print(f"Elo {elo:+.0f} (95 %: {bajo:+.0f} a {alto:+.0f}) en {len(pares)} pares")
    if args.sprt:
        valor = sprt.llr(pares, args.elo0, args.elo1)
        inferior, superior = sprt.cotas(args.alfa, args.beta)
        texto = {"H1": "la nueva es mejor", "H0": "la nueva no es mejor", "": "sin decidir"}
        print(
            f"SPRT [{args.elo0:+.0f}, {args.elo1:+.0f}] LLR {valor:+.2f}"
            f" (cotas {inferior:+.2f}, {superior:+.2f}): {texto[veredicto]}"
        )
    perdidas = [registro for g, *_, registro in resultados if not g]
    if perdidas:
        # Por el juego se pierde porque el rival llega a 3 puntos o porque nos deja sin jugadas.
        por_juego = [r for r in perdidas if r["motivo"] == "juego"]
        por_puntos = sum(max(r["puntos"].values()) >= 3 for r in por_juego)
        sin_jugadas = len(por_juego) - por_puntos
        print(f"perdidas: {por_puntos} por puntos, {sin_jugadas} sin jugadas")
    if args.guardar is not None:
        with open(args.guardar, "w", encoding="utf-8") as archivo:
            for registro in perdidas:
                archivo.write(json.dumps(registro) + "\n")
        print(f"{len(perdidas)} partidas perdidas en {args.guardar}")
    if raras:
        print(f"⚠ partidas que no terminaron por el juego: {raras}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
