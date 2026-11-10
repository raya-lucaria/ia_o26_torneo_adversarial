"""Ajusta los pesos de la evaluación con regresión logística L2 (offline, no corre en la liga).

    P(gana quien mueve) = 1 / (1 + exp(-w · rasgos))

El bot usa tanh(z/2) = 2·P - 1 como valor de cada hoja, así que los promedios en los nodos de
azar (expectiminimax) son promedios de probabilidad de ganar.

Uso (desde estudiantes/<equipo>/):
    python herramientas/ajustar.py datos_r1_w0.npz datos_r1_w1.npz ... --l2 1e-5

Imprime log-loss y exactitud en validación (10 % de las partidas, separadas por partida) y los
pesos. Con --escribir data/pesos.json los guarda como "pesos_autojuego" (el punto de partida de
herramientas/ajustar_vs_rival.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

CARPETA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARPETA))


def perdida(w, X, y, l2, reg):
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    g = X.T @ (p - y) / len(y)
    return ll + 0.5 * l2 * np.sum((w * reg) ** 2), g + l2 * w * reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datos", nargs="+")
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--escribir", default=None, help="ruta de data/pesos.json a actualizar")
    a = ap.parse_args()
    import strategy as bot

    X, y, G = [], [], []
    for i, ruta in enumerate(a.datos):
        z = np.load(ruta)
        X.append(z["X"]); y.append(z["Y"]); G.append(z["G"] + i * 10**9)
    X, y, G = np.concatenate(X), np.concatenate(y), np.concatenate(G)
    juegos = np.unique(G)
    val = set(np.random.default_rng(0).choice(juegos, size=max(1, len(juegos) // 10), replace=False).tolist())
    es_val = np.array([g in val for g in G])
    reg = np.ones(X.shape[1]); reg[0] = 0.0          # el sesgo no se regulariza
    r = minimize(perdida, np.zeros(X.shape[1]), args=(X[~es_val], y[~es_val], a.l2, reg),
                 jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    w = r.x
    for nombre, m in (("entrenamiento", ~es_val), ("validación", es_val)):
        if not m.any():
            continue
        p = 1 / (1 + np.exp(-(X[m] @ w)))
        ll = -np.mean(y[m] * np.log(p + 1e-12) + (1 - y[m]) * np.log(1 - p + 1e-12))
        print(f"{nombre}: {m.sum()} posiciones, log-loss {ll:.4f}, exactitud {np.mean((p > .5) == (y[m] > .5)):.3f}")
    for n, v in zip(bot.NOMBRES_RASGOS, w):
        print(f"  {n:26s} {v:+.4f}")
    print("PESOS_AJUSTADOS = [" + ", ".join(f"{v:.5f}" for v in w) + "]")
    if a.escribir:
        ruta = Path(a.escribir)
        d = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
        d["rasgos"] = bot.NOMBRES_RASGOS
        d["pesos_autojuego"] = [round(float(v), 6) for v in w]
        d.setdefault("pesos", d["pesos_autojuego"])
        d.setdefault("ventaja", 1.0)
        ruta.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"guardado en {ruta} (pesos_autojuego)")


if __name__ == "__main__":
    main()
