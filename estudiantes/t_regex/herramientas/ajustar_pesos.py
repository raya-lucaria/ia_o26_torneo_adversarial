"""Ajusta los pesos de la evaluación con regresión logística (no corre en la liga).

Modelo: P(gana quien está en turno) = σ(PESOS · rasgos). Lo ajustamos por máxima verosimilitud
con una penalización L2 pequeña (método de Newton, solo numpy). Separamos las posiciones por
PARTIDA (no por posición) en entrenamiento y validación, porque las posiciones de una misma
partida se parecen mucho.

Uso (desde la raíz del repo):
    docker run --rm -v "$PWD":/torneo -w /torneo/estudiantes/t_regex torneo-bot-base:local \
        python herramientas/ajustar_pesos.py /torneo/resultados/entrenamiento/partidas_v3.npz \
        --salida data/pesos.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import busqueda as bq  # noqa: E402

L2 = 1e-3


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def ajustar(x, y):
    w = np.zeros(x.shape[1])
    for _ in range(50):
        p = sigmoide(x @ w)
        gradiente = x.T @ (p - y) / len(y) + L2 * w
        hessiana = (x.T * (p * (1 - p))) @ x / len(y) + L2 * np.eye(len(w))
        paso = np.linalg.solve(hessiana, gradiente)
        w -= paso
        if np.abs(paso).max() < 1e-8:
            break
    return w


def medir(x, y, w):
    p = np.clip(sigmoide(x @ w), 1e-9, 1 - 1e-9)
    perdida = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    acierto = np.mean((p > 0.5) == (y > 0.5))
    return perdida, acierto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datos", nargs="+")
    ap.add_argument("--salida")
    args = ap.parse_args()
    partes = [np.load(f) for f in args.datos]
    x = np.concatenate([d["x"] for d in partes])
    y = np.concatenate([d["y"] for d in partes])
    partida = np.concatenate([d["partida"] + 10**7 * i for i, d in enumerate(partes)])

    validacion = (partida % 5) == 0  # 20 % de las partidas
    w = ajustar(x[~validacion], y[~validacion])
    print(f"{len(y)} posiciones ({validacion.sum()} de validación)")
    for nombre, pesos in (("a mano", bq.PESOS_A_MANO), ("ajustados", w)):
        perdida, acierto = medir(x[validacion], y[validacion], pesos)
        print(f"pesos {nombre:9}: log-loss {perdida:.4f}  acierto {acierto:.1%}  (validación)")
    print()
    print(f"{'rasgo':28} {'a mano':>8} {'ajustado':>9}")
    for nombre, a, b in zip(bq.NOMBRES_RASGOS, bq.PESOS_A_MANO, w, strict=True):
        print(f"{nombre:28} {a:8.3f} {b:9.3f}")

    if args.salida:
        w_final = ajustar(x, y)  # con todas las partidas
        np.save(args.salida, w_final)
        print(f"\nPesos (con todas las partidas) → {args.salida}")


if __name__ == "__main__":
    main()
