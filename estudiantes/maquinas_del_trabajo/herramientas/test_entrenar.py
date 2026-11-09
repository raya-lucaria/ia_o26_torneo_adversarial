"""Pruebas del entrenamiento: datos de partidas y ajuste de pesos por regresión logística.

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_entrenar -v
"""

import unittest

import numpy as np

from evaluacion import I_PATRONES, MAX_PEONES, N_BASE, N_PESOS, rasgos
from herramientas.datos import partida_para_datos
from herramientas.entrenar import INICIALES, ajustar, matriz, perdida


class PruebasDatos(unittest.TestCase):
    def test_autojuego_da_rasgos_y_etiquetas(self):
        d = partida_para_datos(semilla=1, rival="nuestro", nodos=2000)
        X, y, v = d.X, d.y, d.v
        self.assertEqual(X.shape[1], N_BASE)
        self.assertEqual(d.indices.shape, (len(y), MAX_PEONES))
        self.assertTrue(set(np.unique(d.signos)) <= {-1, 0, 1})
        # Con signo 0 (relleno) el índice es −1; con peón, un lugar válido de la tabla.
        self.assertTrue(np.all((d.signos == 0) == (d.indices == -1)))
        self.assertTrue(np.all(d.indices < N_PESOS - I_PATRONES))
        self.assertEqual(len(v), len(y))
        self.assertTrue(np.all(np.abs(v) <= 1.0))  # en autojuego toda posición tiene valor
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(y), 10)
        self.assertTrue(set(np.unique(y)) <= {0, 1})
        self.assertTrue(np.all(X[:, 4] == 1.0))  # el rasgo de turno siempre vale 1
        # Las posiciones alternan de turno: si una es del ganador, la siguiente no.
        self.assertTrue(np.all(y[1:] != y[:-1]))
        # La posición guardada reproduce sus rasgos (así se pueden recalcular si cambian).
        self.assertEqual((d.tabs.shape, d.ptss.shape, d.colores.shape), ((len(y), 81), (len(y), 2), (len(y),)))
        salida = np.empty(N_BASE)
        for k in range(0, len(y), 7):
            rasgos(d.tabs[k], d.ptss[k].astype(np.int64), int(d.colores[k]), salida)
            self.assertTrue(np.allclose(salida, X[k]), k)

    def test_autojuego_es_determinista(self):
        a = partida_para_datos(semilla=2, rival="nuestro", nodos=2000)
        b = partida_para_datos(semilla=2, rival="nuestro", nodos=2000)
        self.assertTrue(all(np.array_equal(p, q) for p, q in zip(a, b, strict=True)))

    def test_contra_montecarlo(self):
        d = partida_para_datos(semilla=3, rival="montecarlo", nodos=2000, reloj_rival=2.0)
        X, y, v = d.X, d.y, d.v
        self.assertEqual(X.shape[1], N_BASE)
        self.assertGreater(len(y), 10)
        # Las posiciones donde movió montecarlo no tienen valor de búsqueda.
        self.assertTrue(np.isnan(v).any() and (~np.isnan(v)).any())


class PruebasAjuste(unittest.TestCase):
    def test_recupera_pesos_conocidos(self):
        rng = np.random.default_rng(0)
        verdaderos = rng.normal(0, 0.5, N_BASE)
        X = rng.normal(0, 1, (40_000, N_BASE))
        y = (rng.random(40_000) < 1 / (1 + np.exp(-X @ verdaderos))).astype(np.uint8)
        ajustados = ajustar(X, y, np.zeros(N_BASE), l2=0.0)
        self.assertLess(np.max(np.abs(ajustados - verdaderos)), 0.08, (ajustados, verdaderos))

    def test_recupera_patrones_conocidos(self):
        # 30 patrones con valor conocido, cada posición con 3 peones de esos patrones.
        rng = np.random.default_rng(2)
        n = 60_000
        usados = rng.choice(N_PESOS - I_PATRONES, 30, replace=False)
        valor_patron = rng.normal(0, 0.6, 30)
        X = rng.normal(0, 1, (n, N_BASE))
        indices = np.full((n, MAX_PEONES), -1)
        signos = np.zeros((n, MAX_PEONES), np.int64)
        elegidos = rng.integers(0, 30, (n, 3))
        indices[:, :3] = usados[elegidos]
        signos[:, :3] = rng.choice([-1, 1], (n, 3))
        verdaderos = np.zeros(N_PESOS)
        verdaderos[:N_BASE] = rng.normal(0, 0.3, N_BASE)
        verdaderos[I_PATRONES + usados] = valor_patron
        A = matriz(X, indices, signos)
        y = (rng.random(n) < 1 / (1 + np.exp(-(A @ verdaderos)))).astype(np.uint8)
        ajustados = ajustar(A, y, np.zeros(N_PESOS), l2=0.0)
        self.assertLess(np.max(np.abs(ajustados[I_PATRONES + usados] - valor_patron)), 0.1)
        self.assertLess(np.max(np.abs(ajustados[:N_BASE] - verdaderos[:N_BASE])), 0.1)

    def test_etiquetas_suaves_exactas(self):
        # Si la etiqueta ya es la probabilidad del modelo verdadero, el ajuste lo recupera.
        rng = np.random.default_rng(1)
        verdaderos = rng.normal(0, 0.5, N_BASE)
        X = rng.normal(0, 1, (5000, N_BASE))
        probabilidades = 1 / (1 + np.exp(-X @ verdaderos))
        ajustados = ajustar(X, probabilidades, np.zeros(N_BASE), l2=0.0)
        self.assertLess(np.max(np.abs(ajustados - verdaderos)), 1e-3)

    def test_no_empeora_la_perdida(self):
        d = partida_para_datos(semilla=4, rival="nuestro", nodos=2000)
        A = matriz(d.X, d.indices, d.signos)
        ajustados = ajustar(A, d.y, INICIALES, l2=1e-3)
        self.assertLessEqual(perdida(A, d.y, ajustados, l2=1e-3), perdida(A, d.y, INICIALES, l2=1e-3))


if __name__ == "__main__":
    unittest.main()
