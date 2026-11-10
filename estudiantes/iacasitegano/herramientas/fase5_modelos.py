"""Ajustes offline y GA continuo; ninguna funcion modifica el modelo activo."""
from __future__ import annotations

import numpy as np
from fase5_features import probabilidades


def pesos_grupos(grupos):
    _, i, n = np.unique(grupos, return_inverse=True, return_counts=True)
    w = 1.0 / n[i]
    return w * len(w) / w.sum()


def escala(X):
    s = X.std(axis=0)
    s[s < 1e-12] = 1
    return s


def ridge(X, y, lam, peso=None):
    if lam <= 0:
        raise ValueError("Lambda debe ser positiva")
    peso = np.ones(len(y)) if peso is None else np.asarray(peso, float)
    s = escala(X)
    A = X / s
    w = np.linalg.solve(A.T @ (A * peso[:, None]) + lam * np.eye(X.shape[1]), A.T @ (peso * y))
    return w / s


def logistica(X, y, lam, peso=None, iteraciones=50):
    if lam <= 0 or np.any((y < 0) | (y > 1)):
        raise ValueError("Logistica necesita lambda positiva y objetivos en [0,1]")
    peso = np.ones(len(y)) if peso is None else np.asarray(peso, float)
    s = escala(X)
    A = X / s
    w = np.zeros(A.shape[1])
    I = np.eye(A.shape[1])

    def objetivo(v):
        z = A @ v
        return float(peso @ (np.logaddexp(0, z) - y*z) + .5*lam*(v@v))

    for _ in range(iteraciones):
        p = probabilidades(A @ w)
        H = A.T @ (A * (peso*np.maximum(p*(1-p),1e-8))[:,None]) + lam*I
        g = A.T @ (peso*(y-p)) - lam*w
        paso = np.linalg.solve(H, g)
        antes = objetivo(w)
        factor = 1.0
        while factor > 1e-6 and objetivo(w + factor*paso) > antes:
            factor *= .5
        w += factor * paso
        if np.max(np.abs(factor*paso)) < 1e-8:
            break
    return w / s


def prob_modelo(X, w, modelo):
    z = X @ w
    if modelo == "logistica":
        return probabilidades(z)
    if modelo == "ridge":
        return .5 * (np.clip(z, -1, 1) + 1)
    raise ValueError(modelo)


def contribuciones(X, w):
    """Puntos, filas, marcador ajustables; patrones fijos como referencia."""
    grupos = ((0,1),(244,253),(263,266),(1,244))
    return np.column_stack([X[:,a:b] @ w[a:b] for a,b in grupos])


def aplicar_genoma(w, genoma):
    copia = np.array(w, dtype=float, copy=True)
    for factor,(a,b) in zip(genoma,((0,1),(244,253),(263,266)),strict=True):
        copia[a:b] *= factor
    return copia


def fitness_genoma(genoma, Z, y, peso, modelo):
    z = Z[:,:3] @ genoma + Z[:,3]
    p = probabilidades(z) if modelo == "logistica" else .5*(np.clip(z,-1,1)+1)
    return float(np.average((p-y)**2,weights=peso))


def genetico(Z, y, peso, modelo="logistica", semilla=20261028, poblacion=12, generaciones=5):
    """GA: torneo, cruce aritmetico, mutacion gaussiana y dos elites.

    Presupuesto incluye poblacion inicial. Salida elegida por fitness de ajuste;
    seleccion/prueba se evalua fuera de esta funcion.
    """
    if poblacion < 4 or generaciones < 1:
        raise ValueError("Poblacion >=4 y generaciones >=1")
    rng = np.random.default_rng(semilla)
    pop = rng.uniform(.75,1.25,size=(poblacion,3))
    pop[0] = 1
    historia = []
    mejor, mejor_f = np.ones(3), float("inf")
    evaluaciones = 0
    for generacion in range(generaciones+1):
        valores = np.array([fitness_genoma(v,Z,y,peso,modelo) for v in pop])
        evaluaciones += len(pop)
        orden = np.argsort(valores,kind="stable")
        if valores[orden[0]] < mejor_f:
            mejor, mejor_f = pop[orden[0]].copy(), float(valores[orden[0]])
        historia.append({"generacion":generacion,"genoma":mejor.tolist(),"fitness":mejor_f,
                         "evaluaciones":evaluaciones})
        if generacion == generaciones:
            break

        def padre():
            ids = rng.integers(0,poblacion,size=3)
            return pop[ids[np.argmin(valores[ids])]]

        hijos = [pop[i].copy() for i in orden[:2]]
        while len(hijos) < poblacion:
            mezcla = rng.random(3)
            hijo = mezcla*padre() + (1-mezcla)*padre()
            hijo += (rng.random(3)<(1/3))*rng.normal(0,.05,3)
            hijos.append(np.clip(hijo,.75,1.25))
        pop = np.array(hijos)
    # Referencia aleatoria con exactamente el mismo presupuesto.
    rng = np.random.default_rng(semilla+1)
    candidatos = rng.uniform(.75,1.25,size=(evaluaciones,3))
    candidatos[0] = 1
    vals = [fitness_genoma(v,Z,y,peso,modelo) for v in candidatos]
    aleatorio = candidatos[int(np.argmin(vals))]
    return {"mejor_genoma":mejor.tolist(),"fitness":mejor_f,"historia":historia,
            "evaluaciones":evaluaciones,"semilla":semilla,
            "aleatorio_genoma":aleatorio.tolist(),"aleatorio_fitness":float(min(vals))}
