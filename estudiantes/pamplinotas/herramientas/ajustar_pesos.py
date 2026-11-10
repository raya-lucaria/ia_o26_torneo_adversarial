"""Ajusta los pesos de EVAL por regresión logística sobre posiciones guardadas (no corre en la liga).

Modelo: P(gana quien mueve) = σ(k · Σ w_i f_i). La etiqueta de cada posición mezcla la de búsqueda
y el resultado real: y = m·YB + (1 − m)·YR (ver `generar_posiciones.py`). En los archivos que traen
la partida de cada fila (`G`) se puede usar además la etiqueta TD(λ), que va hacia atrás desde el
resultado y en cada paso mezcla la estimación de la búsqueda en la posición siguiente:
y = t·TD(λ) + (1 − t)·y. Los rasgos se calculan
aquí desde los tableros con `buscador.rasgos`. Las posiciones de partidas contra otro bot (archivos
cuyo nombre contiene «vs_») cuentan `--repetir-rival` veces. Se minimiza la log-pérdida con
regularización L2 (scipy L-BFGS); los pesos se reescalan para que la tabla peón-casilla valga en
promedio 100 y se guardan en data/pesos.json, que `strategy.py` carga en `preparar`.

    H=estudiantes/pamplinotas/herramientas
    PYTHONPATH=sistema:estudiantes/pamplinotas python $H/ajustar_pesos.py $H/posiciones/*.npz \\
        --mezcla 0.8 --salida estudiantes/pamplinotas/data/pesos.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import buscador


def perdida(w, X, y, lam):
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-9
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    grad = X.T @ (p - y) / len(y)
    return ll + lam * np.dot(w, w), grad + 2 * lam * w


def objetivo_td(YB, YR, G, lam):
    """TD(λ) desde quien mueve, hacia atrás dentro de cada partida: en la última posición vale el
    resultado; antes, T_t = 1 − [(1 − λ)·p_{t+1} + λ·T_{t+1}] (la posición siguiente es del rival)."""
    T = np.empty(len(YB))
    for i in range(len(YB) - 1, -1, -1):
        if i == len(YB) - 1 or G[i + 1] != G[i]:
            T[i] = YR[i]
        else:
            T[i] = 1.0 - ((1 - lam) * YB[i + 1] + lam * T[i + 1])
    return T


def rasgos_de(T, C, P):
    X = np.zeros((len(T), buscador.N_RASGOS))
    f = np.zeros(buscador.N_RASGOS)
    for i in range(len(T)):
        buscador.rasgos(T[i], int(C[i]), P[i], f)
        X[i] = f
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("--mezcla", type=float, required=True, help="peso de la etiqueta de búsqueda (1 = sólo búsqueda)")
    ap.add_argument("--td", type=float, default=0.0, help="peso de la etiqueta TD(λ) en los archivos con G")
    ap.add_argument("--lam-td", type=float, default=0.7, help="λ de la etiqueta TD")
    ap.add_argument("--repetir-rival", type=int, default=2)
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-5)
    ap.add_argument("--salida", required=True)
    o = ap.parse_args()
    Xs, ys = [], []
    for a in o.archivos:
        d = np.load(a)
        X = rasgos_de(d["T"], d["C"], d["P"])
        y = o.mezcla * d["YB"] + (1 - o.mezcla) * d["YR"]
        if "G" in d.files and o.td > 0:
            td = objetivo_td(d["YB"].astype(np.float64), d["YR"].astype(np.float64), d["G"], o.lam_td)
            y = o.td * td + (1 - o.td) * y
        for _ in range(o.repetir_rival if "vs_" in Path(a).name else 1):
            Xs.append(X)
            ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(np.float64)
    print(f"{len(y)} posiciones, {X.shape[1]} rasgos, mezcla {o.mezcla}")
    escala = X.std(axis=0)
    escala[escala == 0] = 1.0
    res = minimize(perdida, np.zeros(X.shape[1]), args=(X / escala, y, o.lam), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    w = res.x / escala
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    print(f"log-pérdida {res.fun:.4f}, acierto {np.mean((p > 0.5) == (y > 0.5)):.3f}")
    base = np.mean(np.abs(w[buscador.F_TABLA : buscador.F_TABLA + 35]))
    w = w * (100.0 / base)
    k = base / 100.0  # P(gana) = σ(k · Σ wᵢfᵢ) con los pesos reescalados
    with open(o.salida, "w", encoding="utf-8") as f:
        json.dump({"pesos": [round(float(x), 3) for x in w], "logit_por_punto": k}, f)
    print(f"guardado {o.salida}")


if __name__ == "__main__":
    main()
