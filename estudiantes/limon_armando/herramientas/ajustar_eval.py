"""Ajusta la evaluación de posiciones (regresión logística) y la guarda en data/eval_posicion.npz.

Uso (desde la raíz del repo, con .venv activo):

    python estudiantes/limon_armando/herramientas/ajustar_eval.py <corrida_ajuste> -- <corrida_prueba>

Reproduce las partidas guardadas con el motor y, para cada posición, calcula los rasgos de
`simulacion.rasgos_posicion` y si al final ganaron Blancas. Ajusta
    P(gana Blancas | posición) = 1 / (1 + exp(-w · rasgos))
por máxima verosimilitud con un poco de L2 (Newton). Solo usa partidas entre nuestros bots y
`montecarlo` (el bot del curso), jugadas en nuestra máquina. Imprime el error de Brier en las
partidas de prueba, para compararlo con el de estimar con simulaciones al azar.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from motor import Estado, Jugada  # noqa: E402

import modelo_oponente as mo  # noqa: E402
import simulacion as sim  # noqa: E402
from ajustar_prior import registros  # noqa: E402

FUERTES = {"limon_armando", "limon_exacto", "limon_a", "montecarlo"}
L2 = 1.0
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "eval_posicion.npz")


def datos(rutas):
    """(X, y, partida) de todas las posiciones no terminadas de partidas entre bots fuertes."""
    X, y, g = [], [], []
    f = np.zeros(sim.D_EVAL)
    for n, r in enumerate(registros(rutas)):
        meta = r["meta"]
        if not {meta.get("blancas"), meta.get("negras")} <= FUERTES:
            continue
        ganador = r.get("ganador")
        if ganador not in ("blancas", "negras"):
            continue
        estado = Estado.inicial()
        for jug in r["jugadas"]:
            if estado.ganador() is None:
                tab, pts, turno = mo.tablero_de(estado)
                sim.rasgos_posicion(tab, pts, turno, f)
                X.append(f.copy())
                y.append(1.0 if ganador == "blancas" else 0.0)
                g.append(n)
            combate = jug.get("combate")
            estado = estado.aplicar(Jugada.desde_texto(jug["jugada"]),
                                    gana_ataque=combate["gano"] if combate else None)
    return np.array(X), np.array(y), np.array(g)


def ajustar(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - L2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + L2 * np.eye(len(w))
        paso = np.linalg.solve(H, g)
        w += paso
        if np.abs(paso).max() < 1e-8:
            break
    return w


def main(ajuste, prueba):
    X, y, _ = datos(ajuste)
    w = ajustar(X, y)
    print(f"ajuste: {len(y)} posiciones")
    print("pesos: " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(sim.NOMBRES_EVAL, w)))
    if prueba:
        Xp, yp, _ = datos(prueba)
        p = 1 / (1 + np.exp(-Xp @ w))
        print(f"prueba: {len(yp)} posiciones  Brier {np.mean((p - yp) ** 2):.4f}  "
              f"(constante {np.mean((yp.mean() - yp) ** 2):.4f})")
    np.savez(SALIDA, w=w, nombres=np.array(sim.NOMBRES_EVAL))
    print(f"Guardado {os.path.normpath(SALIDA)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    corte = args.index("--") if "--" in args else len(args)
    main(args[:corte], args[corte + 1:])
