"""Ajusta los pesos de la evaluación con regresión logística («Texel tuning»), offline.

Modelo: P(gana el que mueve) = 1 / (1 + exp(-x·w)), con x = características de alfabeta.py.
Se ajusta w por máxima verosimilitud con regularización L2 (método de Newton). Se separa un 20 %
de las partidas para validar (por partida, no por posición, para no mezclar posiciones parecidas).
Los pesos se guardan como enteros en milésimas de logit en data/pesos.json.

Uso (desde la raíz del repo, con .venv activado):
    python estudiantes/liszt_breakthrough/herramientas/ajustar_pesos.py datos_ronda1.npz [más.npz ...]
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parents[1]
RAIZ = CARPETA.parents[1]
sys.path[:0] = [str(RAIZ / "sistema"), str(CARPETA)]

import alfabeta as ab  # noqa: E402

L2 = 5.0  # fuerza de la regularización (sobre la suma de log-verosimilitudes, características normalizadas)


def logloss(X, y, w):
    z = np.clip(X @ w, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def _objetivo(Xs, y, v, reg):
    z = Xs @ v
    # log(1 + e^z) − y·z, estable numéricamente
    return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(reg * v * v))


def ajustar(X, y, w0, iters=100):
    """Newton con búsqueda de paso, sobre características normalizadas (la constante no se toca)."""
    n, k = X.shape
    escala = X.std(axis=0)
    escala[escala < 1e-9] = 1.0
    escala[-1] = 1.0
    Xs = X / escala
    v = w0 * escala  # mismos logits con las características normalizadas
    reg = L2 * np.ones(k)
    reg[-1] = 0.0  # la constante no se regulariza
    f = _objetivo(Xs, y, v, reg)
    for _ in range(iters):
        z = np.clip(Xs @ v, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xs.T @ (p - y) + reg * v
        hess = (Xs * (p * (1 - p))[:, None]).T @ Xs + np.diag(reg) + 1e-6 * np.eye(k)
        paso = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-6:
            nuevo = v - t * paso
            fn = _objetivo(Xs, y, nuevo, reg)
            if fn <= f:
                break
            t *= 0.5
        if t <= 1e-6:
            break
        mejora = f - fn
        v, f = nuevo, fn
        if mejora < 1e-8 * max(1.0, abs(f)):
            break
    return v / escala


def main():
    archivos = sys.argv[1:]
    if not archivos:
        print(__doc__)
        sys.exit(1)
    Xs, ys, gs = [], [], []
    desfase = 0
    for a in archivos:
        d = np.load(a)
        Xs.append(d["X"].astype(np.float64))
        ys.append(d["y"].astype(np.float64))
        gs.append(d["g"].astype(np.int64) + desfase)
        desfase = int(gs[-1].max()) + 1
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    rng = np.random.default_rng(0)
    juegos = np.unique(g)
    val = set(rng.choice(juegos, size=max(1, len(juegos) // 5), replace=False).tolist())
    es_val = np.array([gi in val for gi in g])
    Xt, yt, Xv, yv = X[~es_val], y[~es_val], X[es_val], y[es_val]

    w_mano = ab.PESOS.astype(np.float64) / 1000.0
    w = ajustar(Xt, yt, w_mano)
    print(f"Posiciones: {len(y)} ({len(juegos)} partidas) · entrenamiento {len(yt)} · validación {len(yv)}")
    print(f"log-loss validación  pesos a mano: {logloss(Xv, yv, w_mano):.4f}   "
          f"ajustados: {logloss(Xv, yv, w):.4f}   (siempre 0.5: {np.log(2):.4f})")
    w = ajustar(X, y, w)  # ajuste final con todo
    enteros = [int(round(v * 1000)) for v in w]
    nombres = (["peones"] + [f"avance_{i}" for i in range(1, 8)] +
               ["apoyos", "libres", "avance_libres", "puntos", "carrera", "anoto_ya",
                "rival_anota", "mejor_ataque_mio", "mejor_ataque_rival",
                "ataques_buenos_mios", "ataques_buenos_rival", "turno"])
    for nm, a, b in zip(nombres, ab.PESOS.tolist(), enteros):
        print(f"  {nm:22s} {a:7d} -> {b:7d}")
    salida = CARPETA / "data" / "pesos.json"
    salida.parent.mkdir(exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump({"pesos": enteros, "nombres": nombres, "posiciones": int(len(y)),
                   "partidas": int(len(juegos)), "logloss_validacion": round(logloss(Xv, yv, w), 4),
                   "fecha": date.today().isoformat(), "archivos": [Path(a).name for a in archivos]},
                  f, ensure_ascii=False, indent=1)
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
