"""Ajusta los pesos de la evaluación por regresión logística (offline).

Minimiza la entropía cruzada entre σ(w·f) y la etiqueta de cada posición
(probabilidad de ganar que dio la búsqueda), con regularización L2 pequeña.
Separa 10 % de las partidas para validar.

python estudiantes/alondra_bernardo/herramientas/ajustar_pesos.py \
    a.npz b.npz ... --salida estudiantes/alondra_bernardo/data/pesos.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

EQUIPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EQUIPO))
import rapido


def matriz(archivos, campo, mezcla):
    X, y = [], []
    f = np.zeros(rapido.RASGOS)
    for ruta in archivos:
        d = np.load(ruta)
        objetivo = d[campo]
        if mezcla > 0:
            objetivo = (1 - mezcla) * objetivo + mezcla * d["resultados"]
        for tab, pts, c, e in zip(d["tableros"], d["puntos"], d["colores"], objetivo):
            rapido.rasgos(tab, pts, int(c), f)
            X.append(f.copy())
            y.append(e)
    return np.array(X), np.clip(np.array(y), 1e-4, 1 - 1e-4)


def perdida(w, X, y, lam):
    z = X @ w
    # log(1 + e^z) estable
    l = np.logaddexp(0, z) - y * z
    g = X.T @ (1 / (1 + np.exp(-z)) - y) / len(y) + 2 * lam * w
    return l.mean() + lam * w @ w, g


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archivos", nargs="+", type=Path)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--lam", type=float, default=1e-5)
    p.add_argument("--etiqueta", choices=("etiquetas", "resultados"), default="etiquetas",
                   help="valor de la búsqueda (etiquetas) o quién ganó la partida (resultados)")
    p.add_argument("--mezcla", type=float, default=0.0,
                   help="fracción del resultado de la partida que se mezcla con la etiqueta")
    a = p.parse_args()
    X, y = matriz(a.archivos, a.etiqueta, a.mezcla)
    rng = np.random.default_rng(0)
    valid = rng.random(len(y)) < 0.1
    w0 = rapido.pesos_iniciales()
    r = minimize(perdida, w0, args=(X[~valid], y[~valid], a.lam), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=2000))
    w = r.x
    for nombre, m in (("entrenamiento", ~valid), ("validación", valid)):
        z = X[m] @ w
        pred = 1 / (1 + np.exp(-z))
        z0 = X[m] @ w0
        pred0 = 1 / (1 + np.exp(-z0))
        print(f"{nombre}: {m.sum()} posiciones, error abs medio {np.abs(pred - y[m]).mean():.4f} "
              f"(pesos iniciales {np.abs(pred0 - y[m]).mean():.4f}), "
              f"acuerdo de signo {((pred > .5) == (y[m] > .5)).mean():.3f}")
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps({"pesos": w.tolist(), "rasgos": rapido.RASGOS,
                                    "posiciones": int(len(y))}, indent=1) + "\n")
    print("Pesos guardados en", a.salida)


if __name__ == "__main__":
    main()
