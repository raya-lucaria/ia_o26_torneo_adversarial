"""Lanza partidas pareadas localmente, fijando cada trabajador a un CPU."""

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


EQUIPO = Path(__file__).resolve().parents[1]
RAIZ = EQUIPO.parents[1]


def _nucleos():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oponente", help="por ejemplo: montecarlo o random")
    parser.add_argument("--bot", default="frame_perfect_push")
    parser.add_argument("--partidas", type=int, default=20, help="numero par; alterna colores")
    parser.add_argument("--semilla", type=int, default=20261107)
    parser.add_argument("--paralelo", type=int, help="trabajadores; maximo nucleos disponibles - 1")
    parser.add_argument("--resultados", type=Path)
    args = parser.parse_args()
    if args.partidas < 2 or args.partidas % 2:
        parser.error("--partidas debe ser un numero par de al menos 2")
    if args.paralelo is not None and args.paralelo < 1:
        parser.error("--paralelo debe ser al menos 1")

    taskset = shutil.which("taskset")
    if taskset is None:
        parser.error("se requiere taskset (Linux)")
    nucleos = _nucleos()
    if not nucleos:
        parser.error("no hay nucleos disponibles")
    max_trabajadores = max(1, len(nucleos) - 1)
    solicitados = args.paralelo if args.paralelo is not None else max_trabajadores
    trabajadores = min(args.partidas // 2, max_trabajadores, solicitados)
    base = args.resultados or EQUIPO / "resultados" / f"paralelo_{args.bot}_vs_{args.oponente}_{args.semilla}"
    base = base.resolve()
    if not base.is_relative_to(EQUIPO.resolve()):
        parser.error("la ruta de resultados debe quedar dentro de la carpeta del equipo")
    base.mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)

    lotes = [[] for _ in range(trabajadores)]
    for pareja in range(args.partidas // 2):
        semilla = args.semilla + pareja
        lotes[pareja % trabajadores].append((pareja, semilla))

    torneo = str(RAIZ / "torneo")

    def ejecutar_lote(indice, tareas):
        nucleo = nucleos[indice]
        resultados = []
        for pareja, semilla in tareas:
            carpeta = base / f"pareja_{pareja:03d}"
            carpeta.mkdir(exist_ok=True)
            log = base / "logs" / f"pareja_{pareja:03d}.txt"
            entorno = os.environ.copy()
            entorno["TORNEO_RESULTADOS"] = str(carpeta / "registros")
            entorno["NUMBA_CACHE_DIR"] = str(base / ".cache" / "numba")
            comando = [
                taskset,
                "-c",
                str(nucleo),
                torneo,
                "--sin-docker",
                "jugar",
                args.bot,
                args.oponente,
                "--partidas",
                "2",
                "--semilla",
                str(semilla),
            ]
            with log.open("w", encoding="utf-8") as salida:
                proceso = subprocess.run(
                    comando,
                    cwd=RAIZ,
                    env=entorno,
                    stdout=salida,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            resultados.append((pareja, proceso.returncode))
        return resultados

    with ThreadPoolExecutor(max_workers=trabajadores) as pool:
        futuros = [
            pool.submit(ejecutar_lote, indice, tareas)
            for indice, tareas in enumerate(lotes)
        ]
        resultados = [futuro.result() for futuro in futuros]
    fallidas = [(pareja, codigo) for lote in resultados for pareja, codigo in lote if codigo]
    print(f"Partidas: {args.partidas}; parejas: {args.partidas // 2}; trabajadores: {trabajadores}")
    print(f"Resultados: {base}")
    print(f"Fallos de comando: {len(fallidas)}")
    if fallidas:
        print("Parejas fallidas: " + ", ".join(f"{p} (exit {c})" for p, c in fallidas))
        return 1
    print(f"Resumen: python estudiantes/frame_perfect_push/herramientas/resumir_resultados.py {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())