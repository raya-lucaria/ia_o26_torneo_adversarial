"""Texel con etiquetas de búsqueda: ajustar la evaluación a una mezcla de resultado y búsqueda.

En vez de la etiqueta 0/1 del resultado (ruidosa por los dados que faltan), cada posición lleva
t = λ·resultado + (1 − λ)·P_búsqueda, donde P_búsqueda = (1 + v)/2 es el valor de una búsqueda de
N nodos con la evaluación vigente (idea de TreeStrap; Veness et al. 2009). Se ajusta la misma
regresión logística con etiquetas suaves y se reporta la log-pérdida contra el resultado en un
conjunto externo. El criterio para adoptarla es el SPRT, no esta log-pérdida.

Uso (desde la raíz del repo, máquina libre). Así salieron los data/pesos_46.npy (λ = 0.3,
maestro con los pesos de la 2.ª iteración; los .npz son de generar_partidas.py, INFORME §5):
  python estudiantes/torneo_breakthrough/herramientas/etiquetas_busqueda.py --nodos 50000 \\
      --posiciones 400000 --lam 0.3 --procesos 10 --pesos-busqueda pesos_iter2.npy \\
      --validar partidas_iter2_a.npz --guardar-en pesos_busqueda.npy \\
      partidas_lineal_16k.npz partidas_iter2_b.npz partidas_v2_16k.npz
  cp pesos_busqueda_lam0.30.npy estudiantes/torneo_breakthrough/data/pesos_46.npy
El orden de los .npz, --semilla y --procesos fijan qué posiciones se buscan y en qué orden.
Con --guardar-en también guarda las etiquetas (<prefijo>_etiquetas.npz: posiciones, resultado y
valor de la búsqueda desde Blancas), para ajustar otros modelos sin repetir las búsquedas.
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))


def buscar(args):
    """Valor (desde Blancas) de una búsqueda de `nodos` nodos para cada posición del trozo."""
    bbs, ptss, turnos, nodos, pesos = args
    import strategy

    m = strategy.Motor(np.load(pesos))
    salida = np.zeros(len(bbs))
    for i in range(len(bbs)):
        bb = bbs[i].astype(np.int64)
        pts = ptss[i].astype(np.int64)
        c = int(turnos[i])
        m.tt_k[:] = 0
        _, v, _, _ = m.pensar(bb, pts, c, 0.0, max_nodos=nodos)
        salida[i] = v if c == 0 else -v
    return salida


def main():
    import ajustar_pesos as aj

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datos", nargs="+")
    ap.add_argument("--nodos", type=int, default=50_000)
    ap.add_argument("--posiciones", type=int, default=400_000)
    ap.add_argument("--lam", type=float, nargs="+", default=[0.5])
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--procesos", type=int, default=10)
    ap.add_argument("--pesos-busqueda", default=str(BOT / "data" / "pesos.npy"),
                    help="pesos de la evaluación con la que busca el «maestro»")
    ap.add_argument("--semilla", type=int, default=3, help="muestreo de posiciones")
    ap.add_argument("--validar", required=True, help=".npz externo (otra semilla)")
    ap.add_argument("--guardar-en", default="", help="prefijo .npy para los pesos de cada λ")
    a = ap.parse_args()
    X, _, y, partida, usar = aj.cargar(a.datos, 8)
    partes = [np.load(f) for f in a.datos]
    bbs = np.concatenate([p["bb"] for p in partes])
    ptss = np.concatenate([p["pts"] for p in partes])
    turnos = np.concatenate([p["turno"] for p in partes])
    rng = np.random.default_rng(a.semilla)
    idx = rng.choice(np.where(usar)[0], size=min(a.posiciones, int(usar.sum())), replace=False)
    t0 = time.time()
    trozos = np.array_split(idx, a.procesos * 8)
    with mp.get_context("spawn").Pool(a.procesos) as pool:
        valores = pool.map(buscar, [(bbs[t], ptss[t], turnos[t], a.nodos, a.pesos_busqueda)
                                    for t in trozos])
    v = np.concatenate(valores)
    print(f"{len(idx)} posiciones buscadas a {a.nodos:,} nodos en {time.time() - t0:.0f} s")
    if a.guardar_en:  # las etiquetas, para entrenar otros modelos sin repetir las búsquedas
        ruta = Path(a.guardar_en)
        np.savez_compressed(ruta.with_name(ruta.stem + "_etiquetas.npz"), bb=bbs[idx], pts=ptss[idx],
                            turno=turnos[idx], partida=partida[idx], resultado=y[idx], valor=v)
    p_busq = 0.5 * (1.0 + np.clip(v, -0.999, 0.999))
    Xv, _, yv, _, usar_v = aj.cargar([a.validar], 8)
    Xv, yv = Xv[usar_v], yv[usar_v]
    vigentes = np.load(a.pesos_busqueda)
    print(f"log-pérdida del resultado en {Path(a.validar).name}: pesos del maestro "
          f"{aj.logloss(Xv @ vigentes, yv):.4f}")
    for lam in a.lam:
        t = lam * y[idx] + (1.0 - lam) * p_busq

        def f(w, X=X[idx], t=t):
            z = X @ w
            p = 0.5 * (1.0 + np.tanh(0.5 * z))
            perdida = np.mean(np.logaddexp(0.0, z) - t * z)  # −t·log σ − (1−t)·log(1−σ)
            return perdida + 0.5 * a.l2 * (w @ w), X.T @ (p - t) / len(t) + a.l2 * w

        w = minimize(f, vigentes.copy(), jac=True, method="L-BFGS-B").x
        escala = float(w @ vigentes / (vigentes @ vigentes))
        print(f"λ = {lam:.2f}: log-pérdida del resultado {aj.logloss(Xv @ w, yv):.4f} | escala sobre "
              f"los del maestro {escala:.3f}")
        if a.guardar_en:
            ruta = Path(a.guardar_en.replace(".npy", f"_lam{lam:.2f}.npy"))
            np.save(ruta, w)
            print(f"  guardado {ruta}")


if __name__ == "__main__":
    main()
