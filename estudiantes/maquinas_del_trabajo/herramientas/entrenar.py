"""Ajusta los pesos de la evaluación con regresión logística (modelo lineal, nada de redes).

Uso (desde estudiantes/maquinas_del_trabajo/). Así salieron los pesos actuales, `data/pesos.npz`
(con los datos de herramientas/datos.py; ojo: `--l2-patrones 1e-3` y `--lambda 0.5` no son los
valores por defecto, y sin ellos no se reproducen):
    ../../.venv/bin/python -m herramientas.entrenar datos_entrenamiento/r2_autojuego.npz \
        datos_entrenamiento/r2_montecarlo.npz --lambda 0.5 --l2-patrones 1e-3 \
        --salida datos_entrenamiento/r2_pesos_l0.5.npz
    cp datos_entrenamiento/r2_pesos_l0.5.npz data/pesos.npz

El modelo es el mismo que usa `evaluacion.evaluar`: P(gana el bando que mueve) = σ(pesos ·
rasgos). Se minimiza la entropía cruzada (con una L2 pequeña, L-BFGS de scipy) contra una
etiqueta que mezcla quién ganó la partida con el valor que le dio nuestra búsqueda:
`λ·(valor + 1)/2 + (1 − λ)·ganó` (el resultado solo es muy ruidoso por los dados). Se aparta el 20 % de las PARTIDAS (no de las posiciones) para validar contra el
resultado real, y se imprime la pérdida y el acierto de los pesos actuales y de los ajustados.

Si los datos traen las posiciones (`tabs`, `ptss`, `colores`), los rasgos y patrones se
recalculan con el código actual de `evaluacion.py`: así se entrena con rasgos nuevos sin volver
a jugar las partidas.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from evaluacion import (
    I_PATRONES,
    MAX_PEONES,
    N_BASE,
    N_PATRONES,
    N_PESOS,
    NOMBRES,
    PESOS_F2,
    patrones,
    rasgos,
)

# Punto de partida: los pesos a mano con la tabla de patrones en cero.
INICIALES = np.concatenate([PESOS_F2, np.zeros(N_PATRONES)])


def matriz(X: np.ndarray, indices: np.ndarray, signos: np.ndarray) -> sparse.csr_matrix:
    """Matriz dispersa [rasgos | tabla de patrones]: una fila por posición, N_PESOS columnas.

    En la parte de patrones, cada peón suma su signo en la columna de su patrón.
    """
    filas, peones = np.nonzero(signos)
    patrones = sparse.csr_matrix(
        (signos[filas, peones].astype(np.float64), (filas, indices[filas, peones])),
        shape=(len(X), N_PESOS - I_PATRONES),
    )
    return sparse.hstack([sparse.csr_matrix(np.asarray(X, np.float64)), patrones], format="csr")


def recalcular(tabs, ptss, colores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, indices, signos) de cada posición con el código actual de la evaluación."""
    n = len(tabs)
    X = np.empty((n, N_BASE), np.float32)
    indices = np.full((n, MAX_PEONES), -1, np.int32)
    signos = np.zeros((n, MAX_PEONES), np.int8)
    salida = np.empty(N_BASE)
    indices_peon = np.empty(MAX_PEONES, np.int64)
    signos_peon = np.empty(MAX_PEONES, np.int64)
    tabs = np.asarray(tabs, np.int8)
    ptss = np.asarray(ptss, np.int64)
    for k in range(n):
        color = int(colores[k])
        rasgos(tabs[k], ptss[k], color, salida)
        X[k] = salida
        m = patrones(tabs[k], color, indices_peon, signos_peon)
        indices[k, :m] = indices_peon[:m]
        signos[k, :m] = signos_peon[:m]
    return X, indices, signos


def perdida(X, y: np.ndarray, pesos: np.ndarray, l2=0.0) -> float:
    """Entropía cruzada promedio de σ(X · pesos) contra `y` (0/1 o probabilidades), más L2.

    `X` puede ser densa o dispersa; `l2` un número o un valor por peso.
    """
    x = X @ pesos
    return float(np.mean(np.logaddexp(0.0, x) - y * x) + np.sum(l2 * pesos**2))


def _perdida_y_gradiente(pesos, X, y, l2):
    x = X @ pesos
    valor = np.mean(np.logaddexp(0.0, x) - y * x) + np.sum(l2 * pesos**2)
    gradiente = X.T @ (1.0 / (1.0 + np.exp(-x)) - y) / len(y) + 2 * l2 * pesos
    return valor, gradiente


def ajustar(X, y: np.ndarray, iniciales: np.ndarray, l2=1e-4) -> np.ndarray:
    """Pesos que minimizan la pérdida logística, empezando de `iniciales`."""
    if not sparse.issparse(X):
        X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    resultado = minimize(
        _perdida_y_gradiente, np.asarray(iniciales, dtype=np.float64), args=(X, y, l2),
        jac=True, method="L-BFGS-B",
    )
    return resultado.x


def _acierto(X, y, pesos) -> float:
    return float(np.mean((X @ pesos > 0) == (y == 1)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datos", type=Path, nargs="+", help="archivos .npz de herramientas.datos")
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 de los pesos de los rasgos")
    parser.add_argument("--l2-patrones", type=float, default=1e-4, help="L2 de la tabla de patrones")
    parser.add_argument("--sin-patrones", action="store_true", help="solo los rasgos")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.7,
                        help="peso del valor de búsqueda en la etiqueta (0 = solo el resultado)")
    parser.add_argument("--validacion", type=float, default=0.2)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args(argv)

    Xs, ys, vs, ids, indices, signos = [], [], [], [], [], []
    desplazamiento = 0
    for archivo in args.datos:
        datos = np.load(archivo)
        if "tabs" in datos:
            X_archivo, indices_archivo, signos_archivo = recalcular(
                datos["tabs"], datos["ptss"], datos["colores"]
            )
        elif datos["X"].shape[1] == N_BASE:
            X_archivo, indices_archivo, signos_archivo = datos["X"], datos["indices"], datos["signos"]
        else:
            sys.exit(f"{archivo}: tiene {datos['X'].shape[1]} rasgos y no trae posiciones")
        Xs.append(X_archivo.astype(np.float64))
        ys.append(datos["y"].astype(np.float64))
        vs.append(datos["v"].astype(np.float64) if "v" in datos else np.full(len(datos["y"]), np.nan))
        ids.append(datos["partida"] + desplazamiento)
        indices.append(indices_archivo)
        signos.append(signos_archivo * (not args.sin_patrones))
        desplazamiento = ids[-1].max() + 1
    X = matriz(np.concatenate(Xs), np.concatenate(indices), np.concatenate(signos))
    y, ids = np.concatenate(ys), np.concatenate(ids)
    v = np.concatenate(vs)
    etiqueta = np.where(np.isnan(v), y, args.lam * (v + 1) / 2 + (1 - args.lam) * y)
    partidas = np.unique(ids)
    rng = np.random.default_rng(args.semilla)
    de_validacion = rng.choice(partidas, int(len(partidas) * args.validacion), replace=False)
    es_validacion = np.isin(ids, de_validacion)
    X_e, t_e = X[~es_validacion], etiqueta[~es_validacion]
    X_v, y_v = X[es_validacion], y[es_validacion]  # se valida contra el resultado real

    l2 = np.full(N_PESOS, args.l2_patrones)
    l2[:N_BASE] = args.l2
    pesos = ajustar(X_e, t_e, INICIALES, l2)
    print(f"{len(partidas)} partidas, {len(y)} posiciones ({int(es_validacion.sum())} de validación)")
    for nombre, actuales in (("actuales", INICIALES), ("ajustados", pesos)):
        print(
            f"  {nombre:9s}: pérdida {perdida(X_v, y_v, actuales):.4f},"
            f" acierto {100 * _acierto(X_v, y_v, actuales):.1f} %"
        )
    for nombre, antes, ahora in zip(NOMBRES, PESOS_F2[:N_BASE], pesos[:N_BASE], strict=True):
        print(f"  {nombre:10s} {antes:7.3f} → {ahora:7.3f}")
    tabla = pesos[I_PATRONES:]
    usados = np.count_nonzero(np.asarray(abs(X[:, I_PATRONES:]).sum(axis=0)).ravel())
    print(
        f"  patrones: {usados} de {len(tabla)} aparecen en los datos;"
        f" valor típico {np.std(tabla[tabla != 0]) if usados else 0:.3f}"
    )
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.salida, pesos=pesos)
    print(f"pesos → {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
