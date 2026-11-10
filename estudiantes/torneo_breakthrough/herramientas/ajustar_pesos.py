"""Ajusta los pesos de la evaluación lineal (rasgos.py) con regresión logística: método de Texel.

Minimiza la log-pérdida de P(ganan Blancas) = σ(w·φ(s)) contra el resultado real de la partida,
con regularización L2. Separa partidas de validación (no posiciones) y compara contra los 46 pesos
vigentes (data/pesos_46.npy) y contra la evaluación v2 calibrada con un solo factor de escala.
`--validar-con` agrega otros conjuntos solo para medir (por ejemplo, partidas contra montecarlo).
Escribe data/pesos_46.npy y data/pesos_46.json con `--guardar` (u otra ruta con `--guardar-en`).
Esos 46 pesos son el punto de partida de ajustar_marcador.py, que hace los que juega el bot.

Uso (desde la raíz del repo):
  python estudiantes/torneo_breakthrough/herramientas/ajustar_pesos.py \\
      laboratorio/resultados/texel/partidas_v2.npz [--l2 1e-4] [--guardar]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))

import evaluacion as ev  # noqa: E402
import rasgos as rg  # noqa: E402
import tablero_bits as tb  # noqa: E402


@njit(cache=False)
def matriz_rasgos(bbs, ptss, turnos, X, v2, anota):
    """Rasgos de cada posición, la evaluación v2 desde Blancas y si quien mueve puede anotar ya."""
    phi = np.zeros(rg.K, np.float64)
    lista = np.zeros(64, np.int64)
    for i in range(bbs.shape[0]):
        bb = bbs[i].astype(np.int64)
        pts = ptss[i].astype(np.int64)
        c = turnos[i]
        rg.rasgos(bb, pts, c, phi)
        X[i, :] = phi
        v = ev.evaluar(bb, pts, c)
        v2[i] = v if c == 0 else -v
        meta = tb.LADO - 1 if c == 0 else 0
        n = tb.generar_tacticas(bb, c, lista)
        anota[i] = 0
        for k in range(n):
            if (lista[k] % tb.CASILLAS) // tb.LADO == meta:
                anota[i] = 1
                break


def cargar(rutas, min_ply):
    """Posiciones de uno o más .npz: rasgos, v2, resultado y partida (con ids únicos por archivo)."""
    partes = [np.load(f) for f in rutas]
    d = {k: np.concatenate([p[k] for p in partes]) for k in partes[0].files}
    d["partida"] = np.concatenate([p["partida"] + 10_000_000 * i for i, p in enumerate(partes)])
    n = len(d["resultado"])
    X = np.zeros((n, rg.K), np.float64)
    v2 = np.zeros(n, np.float64)
    anota = np.zeros(n, np.int64)
    matriz_rasgos(d["bb"], d["pts"], d["turno"].astype(np.int64), X, v2, anota)
    usar = (d["ply"] >= min_ply) & (anota == 0)  # fuera aperturas y posiciones de anotar ya
    return X, v2, d["resultado"].astype(np.float64), d["partida"], usar


def perdida(w, X, y, l2):
    z = X @ w
    # log(1 + e^{-z}) para y = 1 y log(1 + e^{z}) para y = 0, estable numéricamente
    m = np.where(y == 1, np.logaddexp(0.0, -z), np.logaddexp(0.0, z)).mean()
    p = 0.5 * (1.0 + np.tanh(0.5 * z))
    g = X.T @ (p - y) / len(y)
    return m + 0.5 * l2 * (w @ w), g + l2 * w


def logloss(z, y):
    return float(np.where(y == 1, np.logaddexp(0.0, -z), np.logaddexp(0.0, z)).mean())


def calibracion(z, y, bins=10):
    p = 0.5 * (1.0 + np.tanh(0.5 * z))
    filas = []
    for k in range(bins):
        sel = (p >= k / bins) & (p < (k + 1) / bins)
        if sel.sum() >= 50:
            filas.append((k / bins, (k + 1) / bins, p[sel].mean(), y[sel].mean(), int(sel.sum())))
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datos", nargs="+")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--validacion", type=float, default=0.1, help="fracción de partidas apartadas")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--excluir", default="", help="prefijos de rasgos que se fijan en 0 (con comas)")
    ap.add_argument("--validar-con", nargs="*", default=[], help="otros .npz solo para medir")
    ap.add_argument("--guardar", action="store_true", help="escribe data/pesos_46.npy y pesos_46.json")
    ap.add_argument("--guardar-en", default="", help="otra ruta .npy (para probar antes de adoptar)")
    a = ap.parse_args()
    X, v2, y, partida, usar = cargar(a.datos, a.min_ply)
    partidas = np.unique(partida)
    rng = np.random.default_rng(0)
    n_val = max(1, int(a.validacion * len(partidas)))
    es_val = np.isin(partida, rng.choice(partidas, size=n_val, replace=False))
    ent, val = usar & ~es_val, usar & es_val
    print(f"posiciones: {len(y)} | usadas {usar.sum()} (entrenamiento {ent.sum()}, validación "
          f"{val.sum()}) | partidas {len(partidas)} | gana Blancas {100 * y[usar].mean():.1f}%")
    prefijos = [p for p in a.excluir.split(",") if p]
    libres = np.array([not any(n.startswith(p) for p in prefijos) for n in rg.NOMBRES])
    r = minimize(perdida, np.zeros(libres.sum()), args=(X[ent][:, libres], y[ent], a.l2), jac=True,
                 method="L-BFGS-B")
    w = np.zeros(rg.K)
    w[libres] = r.x
    vigentes = np.load(BOT / "data" / "pesos_46.npy")
    if len(vigentes) != rg.K:
        vigentes = None
    # v2 calibrada con un solo factor: P = σ(s·atanh(v2))
    t = np.arctanh(np.clip(v2, -0.999999, 0.999999))

    def perdida_escala(s):
        p = 0.5 * (1 + np.tanh(0.5 * s[0] * t[ent]))
        return logloss(s[0] * t[ent], y[ent]), np.array([np.mean((p - y[ent]) * t[ent])])

    s = minimize(perdida_escala, np.array([2.0]), jac=True, method="L-BFGS-B").x[0]

    def reporte(nombre, Xs, ts, ys):
        partes = [f"sin información {logloss(np.zeros(len(ys)), ys):.4f}",
                  f"v2 calibrada {logloss(s * ts, ys):.4f}"]
        if vigentes is not None:
            partes.append(f"pesos vigentes {logloss(Xs @ vigentes, ys):.4f}")
        partes.append(f"**nuevos {logloss(Xs @ w, ys):.4f}**")
        print(f"log-pérdida {nombre} ({len(ys)} posiciones): " + " | ".join(partes))

    reporte("de validación", X[val], t[val], y[val])
    for ruta in a.validar_con:
        Xo, v2o, yo, _, usar_o = cargar([ruta], a.min_ply)
        reporte(Path(ruta).name, Xo[usar_o], np.arctanh(np.clip(v2o[usar_o], -0.999999, 0.999999)),
                yo[usar_o])
    print("calibración (validación, nuevos): [p_desde, p_hasta) predicha → observada (n)")
    for lo, hi, pm, om, cnt in calibracion(X[val] @ w, y[val]):
        print(f"  [{lo:.1f}, {hi:.1f}) {pm:.3f} → {om:.3f} ({cnt})")
    print("pesos:" + (f" (excluidos: {', '.join(prefijos)})" if prefijos else ""))
    for k, (nombre, wk) in enumerate(zip(rg.NOMBRES, w, strict=True)):
        previo = f"  (vigente {vigentes[k]:+.4f})" if vigentes is not None else ""
        print(f"  {nombre:24s} {wk:+.4f}{previo}")
    if a.guardar or a.guardar_en:
        ruta = Path(a.guardar_en) if a.guardar_en else BOT / "data" / "pesos_46.npy"
        np.save(ruta, w.astype(np.float64))
        ruta.with_suffix(".json").write_text(json.dumps(
            {"nombres": rg.NOMBRES, "pesos": [float(x) for x in w], "l2": a.l2,
             "excluidos": prefijos, "fuentes": [Path(f).name for f in a.datos],
             "posiciones": int(ent.sum())},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"guardado en {ruta} y {ruta.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
