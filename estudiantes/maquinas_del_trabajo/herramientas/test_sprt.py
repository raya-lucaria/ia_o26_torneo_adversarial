"""Pruebas del SPRT por pares de partidas.

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_sprt -v
"""

import math
import random
import unittest

from herramientas import sprt


class PruebasSprt(unittest.TestCase):
    def test_elo_y_puntaje_son_inversos(self):
        self.assertAlmostEqual(sprt.puntaje_esperado(0.0), 0.5)
        self.assertAlmostEqual(sprt.puntaje_esperado(400.0), 10 / 11)
        for elo in (-200.0, -35.0, 0.0, 35.0, 300.0):
            self.assertAlmostEqual(sprt.elo_de(sprt.puntaje_esperado(elo)), elo, places=9)
        self.assertEqual(sprt.elo_de(1.0), math.inf)

    def test_llr_a_mano(self):
        # 1, 1, 0.5, 0 repetidos → media 0.625, varianza 0.171875.
        pares = [1.0, 1.0, 0.5, 0.0] * 10
        s0, s1 = 0.5, sprt.puntaje_esperado(35.0)
        esperado = 40 * (s1 - s0) * (2 * 0.625 - s0 - s1) / (2 * 0.171875)
        self.assertAlmostEqual(sprt.llr(pares), esperado, places=12)

    def test_no_decide_con_pocos_pares(self):
        # Dos pares iguales tienen varianza 0: sin el mínimo, el LLR sería enorme.
        self.assertEqual(sprt.llr([0.5, 0.5]), 0.0)
        self.assertEqual(sprt.llr([1.0] * (sprt.MIN_PARES - 1)), 0.0)

    def test_cotas_de_wald(self):
        inferior, superior = sprt.cotas(0.1, 0.1)
        self.assertAlmostEqual(inferior, math.log(0.1 / 0.9))
        self.assertAlmostEqual(superior, math.log(0.9 / 0.1))
        self.assertEqual(sprt.decision(superior), "H1")
        self.assertEqual(sprt.decision(inferior), "H0")
        self.assertEqual(sprt.decision(0.0), "")

    def test_decide_bien_con_monedas_conocidas(self):
        # Pares simulados: cada partida se gana con probabilidad p. Con +100 Elo debe aceptar H1;
        # con −50 Elo, H0; casi siempre antes de 2000 pares.
        for elo, esperado in ((100.0, "H1"), (-50.0, "H0")):
            p = sprt.puntaje_esperado(elo)
            aciertos = 0
            for intento in range(20):
                rng = random.Random(intento)
                pares, veredicto = [], ""
                while not veredicto and len(pares) < 2000:
                    pares.append(((rng.random() < p) + (rng.random() < p)) / 2)
                    veredicto = sprt.decision(sprt.llr(pares))
                aciertos += veredicto == esperado
            self.assertGreaterEqual(aciertos, 18, elo)

    def test_intervalo_contiene_la_estimacion(self):
        pares = [1.0, 0.5, 0.5, 1.0, 0.0, 0.5, 1.0, 0.5]
        elo, bajo, alto = sprt.intervalo_elo(pares)
        self.assertLess(bajo, elo)
        self.assertLess(elo, alto)
        self.assertAlmostEqual(elo, sprt.elo_de(0.625))


if __name__ == "__main__":
    unittest.main()
