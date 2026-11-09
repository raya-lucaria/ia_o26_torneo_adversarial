"""Afinado automático de los pesos de la evaluación por autojuego (no corre en la liga).

Búsqueda local, peso por peso: el candidato es la versión vigente con UN peso movido un paso
(arriba o abajo). Juega contra la vigente en la arena con los mismos dados y ambos colores:
- filtro: 200 partidas; si gana ≥ 55 % pasa a confirmación;
- confirmación: 400 partidas más; si en las 600 gana ≥ 53.5 %, pasa al ancla;
- ancla: 400 partidas contra una versión fija ya validada (`--ancla`); si no pierde (≥ 50 %),
  se adopta. Sin el ancla, una cadena de cambios que le ganan cada uno a su predecesor puede
  terminar peor que el inicio (nos pasó: 4 adopciones seguidas perdieron 41 % contra el inicio).
La adoptada pasa a ser la vigente (la «versión anterior» del siguiente intento). Si en una
vuelta completa no se adopta nada, el paso se reduce a la mitad.

Todo queda en `--carpeta` (por omisión resultados/afinado/): estado.json para retomar si se
interrumpe, historial.txt y pesos_vN.npy de cada versión adoptada.

Uso (desde la raíz del repo; corre durante horas):
    docker run --rm --cpus=8 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/afinar.py
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arena  # noqa: E402
import busqueda as bq  # noqa: E402

# Orden en que se prueban los pesos (los que creemos más influyentes primero).
ORDEN = [9, 12, 13, 1, 10, 11, 0, 14, 15, 2, 3, 4, 5, 6, 7, 8]
PASO_RELATIVO = 0.25  # paso inicial: 25 % del peso (mínimo PASO_MINIMO)
PASO_MINIMO = 0.05
FILTRO_PARES, FILTRO_UMBRAL = 100, 0.55
CONFIRMA_PARES, CONFIRMA_UMBRAL = 200, 0.535
ANCLA_PARES, ANCLA_UMBRAL = 200, 0.50


def cambios(pesos):
    """Texto de --pa/--pb que fija TODOS los pesos (así no dependemos de data/pesos.npy)."""
    return ",".join(f"PESOS[{i}]={w:.6f}" for i, w in enumerate(pesos))


def jugar(procesos, candidato, vigente, pares, semilla, nodos):
    """Partidas del candidato (A) contra la vigente (B). Regresa victorias de A y total.

    Cada enfrentamiento usa procesos nuevos: cada combinación de pesos compila su propia
    copia del módulo y, si reusáramos los procesos durante horas, se acumularían en memoria.
    """
    a, b = (".", cambios(candidato)), (".", cambios(vigente))
    trabajos = []
    for k in range(pares):
        trabajos.append((a, b, semilla + k, nodos))
        trabajos.append((b, a, semilla + k, nodos))
    victorias = 0
    with Pool(procesos) as pool:
        for i, (ganador, *_) in enumerate(pool.imap(arena.partida, trabajos)):
            a_es_blancas = i % 2 == 0
            victorias += (ganador == 0) == a_es_blancas
    return victorias, len(trabajos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default="/torneo/resultados/afinado")
    ap.add_argument("--nodos", type=int, default=100_000)
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--ancla", help="pesos .npy de una versión validada contra la que no perder")
    args = ap.parse_args()
    ancla = [float(w) for w in np.load(args.ancla)] if args.ancla else None
    carpeta = Path(args.carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo_estado = carpeta / "estado.json"
    if archivo_estado.exists():
        estado = json.loads(archivo_estado.read_text())
    else:
        pesos = [float(w) for w in bq.PESOS]
        estado = {
            "pesos": pesos,
            "pasos": [max(abs(w) * PASO_RELATIVO, PASO_MINIMO) for w in pesos],
            "posicion": 0,  # índice en ORDEN×{+,-} del siguiente intento
            "version": 0,
            "adoptados_en_vuelta": 0,
            "semilla": 50_000,
        }
        np.save(carpeta / "pesos_v0.npy", np.array(pesos))

    def guardar():
        archivo_estado.write_text(json.dumps(estado, indent=1))

    def anotar(texto):
        linea = f"{time.strftime('%Y-%m-%d %H:%M')}  {texto}"
        print(linea, flush=True)
        with open(carpeta / "historial.txt", "a") as f:
            f.write(linea + "\n")

    anotar(f"inicio: versión {estado['version']}, posición {estado['posicion']}")
    while True:
        i = ORDEN[estado["posicion"] // 2]
        signo = 1 if estado["posicion"] % 2 == 0 else -1
        vigente = estado["pesos"]
        candidato = list(vigente)
        candidato[i] += signo * estado["pasos"][i]
        nombre = f"{bq.NOMBRES_RASGOS[i]} {vigente[i]:+.3f} → {candidato[i]:+.3f}"

        estado["semilla"] += 1000
        v, n = jugar(
            args.procesos, candidato, vigente, FILTRO_PARES, estado["semilla"], args.nodos
        )
        adoptado = False
        if v / n >= FILTRO_UMBRAL:
            estado["semilla"] += 1000
            v2, n2 = jugar(args.procesos, candidato, vigente, CONFIRMA_PARES, estado["semilla"],
                           args.nodos)
            v, n = v + v2, n + n2
            adoptado = v / n >= CONFIRMA_UMBRAL
        texto = f"{nombre}: {v}/{n} = {v / n:.1%}"
        if adoptado and ancla is not None:
            estado["semilla"] += 1000
            va, na = jugar(args.procesos, candidato, ancla, ANCLA_PARES, estado["semilla"],
                           args.nodos)
            adoptado = va / na >= ANCLA_UMBRAL
            texto += f", contra el ancla {va}/{na} = {va / na:.1%}"
        anotar(texto + ("  ← ADOPTADO" if adoptado else ""))

        if adoptado:
            estado["pesos"] = candidato
            estado["version"] += 1
            estado["adoptados_en_vuelta"] += 1
            np.save(carpeta / f"pesos_v{estado['version']}.npy", np.array(candidato))
            np.save(carpeta / "pesos_mejores.npy", np.array(candidato))
            # Si subir ayudó, no probamos bajar el mismo peso: seguimos con el siguiente.
            if signo == 1:
                estado["posicion"] += 1
        estado["posicion"] += 1
        if estado["posicion"] >= 2 * len(ORDEN):
            if estado["adoptados_en_vuelta"] == 0:
                estado["pasos"] = [max(p / 2, PASO_MINIMO / 2) for p in estado["pasos"]]
                anotar("vuelta sin mejoras: paso a la mitad")
            estado["posicion"] = 0
            estado["adoptados_en_vuelta"] = 0
        guardar()


if __name__ == "__main__":
    main()
