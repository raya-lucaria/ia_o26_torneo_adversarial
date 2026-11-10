"""Features ligeras offline de eval3, vectorizadas, sin compilar ni importar estrategia.

Solo puntos, patrones, filas y marcador. El resto de las 267 columnas es cero.
La equivalencia se contrasta con eval3.features_anchas_py en pruebas separadas.
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 267
PATRONES = slice(1, 244)
FILAS = slice(244, 253)
MARCADOR = slice(263, 266)


def features(tableros, puntos, turnos):
    tableros = np.asarray(tableros, dtype=np.int8)
    puntos = np.asarray(puntos)
    turnos = np.asarray(turnos, dtype=np.int64)
    n = len(tableros)
    if tableros.shape != (n, 81) or puntos.shape != (n, 2) or turnos.shape != (n,):
        raise ValueError("Dimensiones esperadas: tablero[n,81], puntos[n,2], turno[n]")
    if np.any((turnos < 0) | (turnos > 1)) or np.any((puntos < 0) | (puntos > 2)):
        raise ValueError("Features de posiciones no terminales con turno 0/1 y puntos 0..2")
    X = np.zeros((n, N_FEATURES), dtype=np.float64)
    filas = np.arange(n)
    X[:, 0] = puntos[filas, turnos] - puntos[filas, 1 - turnos]
    for color in (0, 1):
        i, casilla = np.nonzero(tableros == color + 1)
        f, c = np.divmod(casilla, 9)
        d = 1 if color == 0 else -1
        codigo = np.zeros(len(i), np.int64)
        for df, dc, peso in ((d, -1, 1), (d, 0, 3), (d, 1, 9), (-d, -1, 27), (-d, 1, 81)):
            f2, c2 = f + df, c + dc
            valido = (f2 >= 0) & (f2 < 9) & (c2 >= 0) & (c2 < 9)
            vecinos = np.zeros(len(i), np.int8)
            vecinos[valido] = tableros[i[valido], f2[valido] * 9 + c2[valido]]
            codigo += peso * np.where(vecinos == 0, 0, np.where(vecinos == color + 1, 1, 2))
        signo = np.where(turnos[i] == color, 1.0, -1.0)
        np.add.at(X, (i, 1 + codigo), signo)
        avance = f if color == 0 else 8 - f
        np.add.at(X, (i, 244 + avance), signo)
    propios, rivales = puntos[filas, turnos], puntos[filas, 1 - turnos]
    for col, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        # eval3: el marcador favorable suma positivo.
        X[:, 263 + col] = ((propios == b) & (rivales == a)).astype(float) - ((propios == a) & (rivales == b))
    return X


def particion_hash(ids, fraccion=0.2, sal=""):
    import hashlib
    if not 0 < fraccion < 1:
        raise ValueError("Fraccion debe estar entre 0 y 1")
    corte = int(fraccion * 2**32)
    return np.array([int(hashlib.sha256((str(x) + sal).encode()).hexdigest()[:8], 16) < corte for x in ids])


def probabilidades(z):
    z = np.asarray(z, float)
    return np.exp(-np.logaddexp(0.0, -z))


def metricas(y, prob, grupos):
    y, prob = np.asarray(y, float), np.clip(np.asarray(prob, float), 1e-12, 1 - 1e-12)
    if not len(y):
        return {"n": 0, "grupos": 0, "logloss": None, "brier": None}
    _, indices, cuentas = np.unique(grupos, return_inverse=True, return_counts=True)
    peso = 1.0 / cuentas[indices]
    peso /= peso.sum()
    ll = -(y * np.log(prob) + (1 - y) * np.log1p(-prob))
    return {"n": len(y), "grupos": len(cuentas), "logloss": float(np.mean(ll)),
            "brier": float(np.mean((y - prob)**2)),
            "logloss_igual_peso_grupo": float(peso @ ll),
            "brier_igual_peso_grupo": float(peso @ ((y - prob)**2))}
