"""Pruebas del reparto del reloj.

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_reloj -v
"""

import unittest

from reloj import RESERVA_S, jugadas_restantes, presupuesto


class PruebasReloj(unittest.TestCase):
    def test_inicio(self):
        # 18 peones y 0 puntos: 1.5·18 + 6 = 33 jugadas por delante.
        blando, duro = presupuesto(30.0, 18, 0)
        esperado = (30 - 3.5 - 0.015 * 33) / 33
        self.assertAlmostEqual(blando, esperado, delta=1e-12)
        self.assertAlmostEqual(duro, 3 * esperado, delta=1e-12)

    def test_restantes_a_mano(self):
        self.assertEqual(jugadas_restantes(18, 0), 33)
        self.assertEqual(jugadas_restantes(10, 2), 17)  # 15 + 6 − 4
        self.assertEqual(jugadas_restantes(2, 4), 8)  # 3 + 6 − 8 = 1 → mínimo 8

    def test_reserva(self):
        self.assertEqual(presupuesto(2.4, 10, 0), (0.0, 0.0))

    def test_poco_reloj(self):
        # Apenas arriba de la reserva: se piensa poco y nunca se toca la reserva.
        blando, duro = presupuesto(3.75, 0, 0)  # n = 8: útil = 0.25 − 0.12 = 0.13
        self.assertAlmostEqual(blando, 0.13 / 8, delta=1e-12)
        self.assertLessEqual(duro, 3.75 - RESERVA_S)

    def test_nunca_pasa_del_reloj(self):
        for decimas in range(0, 301):
            tiempo = decimas / 10
            for peones in range(0, 19):
                blando, duro = presupuesto(tiempo, peones, 0)
                self.assertTrue(
                    0 <= blando <= duro <= max(0.0, tiempo - RESERVA_S), (tiempo, peones, blando, duro)
                )

    def test_partida_larga_no_pasa_de_27_s(self):
        # Ritmo medido en Docker: cada jugada piensa ~1.3·blando (sin pasar de duro) y el
        # protocolo cobra ~15 ms aunque contestemos al instante. Con más de 27 s usados el
        # torneo marca ⚠ (90 % del reloj).
        for jugadas in (40, 60, 80):
            restante = 30.0
            for k in range(jugadas):
                peones = 18 - (14 * k) // jugadas  # de 18 a 4 peones
                puntos = (4 * k) // jugadas  # de 0 a 3 puntos en total
                blando, duro = presupuesto(restante, peones, puntos)
                restante -= min(1.3 * blando, duro) + 0.015
            self.assertLessEqual(30.0 - restante, 27.0, f"{jugadas} jugadas")

    def test_escala_multiplica_blando_sin_pasar_del_reloj(self):
        base, _ = presupuesto(30.0, 18, 0, escala=1.0)
        self.assertAlmostEqual(presupuesto(30.0, 18, 0, escala=1.25)[0], 1.25 * base, delta=1e-12)
        for decimas in range(0, 301):
            tiempo = decimas / 10
            for escala in (1.25, 1.5, 3.0):
                blando, duro = presupuesto(tiempo, 0, 4, escala)  # n = 8: la escala más agresiva
                self.assertTrue(0 <= blando <= duro <= max(0.0, tiempo - RESERVA_S), (tiempo, escala))

    def test_restantes_acotadas(self):
        for peones in range(0, 19):
            for puntos in range(0, 5):
                self.assertTrue(8 <= jugadas_restantes(peones, puntos) <= 70)


if __name__ == "__main__":
    unittest.main()
