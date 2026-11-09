import types
import unittest
from unittest.mock import patch

import strategy
from motor import Estado, Jugada, prob_combate


class ReglasMotorTest(unittest.TestCase):
    def test_ataque_ganado_quita_defensor(self):
        estado = Estado.desde_dict({"blancas": ["e4"], "negras": ["f5"], "turno": "blancas"})
        siguiente = estado.aplicar(Jugada.desde_texto("e4-f5"), gana_ataque=True)
        self.assertEqual(siguiente.piezas("blancas"), ["f5"])
        self.assertEqual(siguiente.piezas("negras"), [])

    def test_ataque_perdido_quita_atacante(self):
        estado = Estado.desde_dict({"blancas": ["e4"], "negras": ["f5"], "turno": "blancas"})
        siguiente = estado.aplicar(Jugada.desde_texto("e4-f5"), gana_ataque=False)
        self.assertEqual(siguiente.piezas("blancas"), [])
        self.assertEqual(siguiente.piezas("negras"), ["f5"])

    def test_no_se_ataca_recto(self):
        estado = Estado.desde_dict({"blancas": ["e4"], "negras": ["e5"], "turno": "blancas"})
        self.assertNotIn("e4-e5", {str(jugada) for jugada in estado.jugadas()})

    def test_anotar_suma_y_quita_peon(self):
        estado = Estado.desde_dict({"blancas": ["e8"], "negras": [], "turno": "blancas"})
        siguiente = estado.aplicar("e8-e9")
        self.assertEqual(siguiente.puntos["blancas"], 1)
        self.assertEqual(siguiente.piezas("blancas"), [])
        self.assertNotIn("e9", siguiente.piezas("blancas"))

    def test_tercer_punto_termina(self):
        estado = Estado.desde_dict(
            {"blancas": ["e8"], "negras": [], "turno": "blancas", "puntos": {"blancas": 2}}
        )
        siguiente = estado.aplicar("e8-e9")
        self.assertEqual(siguiente.puntos["blancas"], 3)
        self.assertTrue(siguiente.terminado())
        self.assertEqual(siguiente.ganador(), "blancas")

    def test_sin_jugadas_pierde_el_turno(self):
        estado = Estado.desde_dict({"blancas": [], "negras": ["a5"], "turno": "blancas"})
        self.assertEqual(estado.jugadas(), [])
        self.assertEqual(estado.ganador(), "negras")

    def test_apoyo_blanco_ejemplo_de_reglas(self):
        # docs/reglas.md: e4 con d3/f3 (apoyo 2) ataca f5 con g6 (apoyo 1): p = 0.65
        estado = Estado.desde_dict(
            {"blancas": ["e4", "d3", "f3"], "negras": ["f5", "g6"], "turno": "blancas"}
        )
        self.assertAlmostEqual(estado.prob_ataque(Jugada.desde_texto("e4-f5")), 0.65)
        self.assertAlmostEqual(prob_combate(2, 1), 0.65)

    def test_apoyo_negro_ejemplo_de_reglas(self):
        # d6 con c7 (apoyo 1) ataca e5 con d4/f4 (apoyo 2): p = 0.35
        estado = Estado.desde_dict(
            {"blancas": ["e5", "d4", "f4"], "negras": ["d6", "c7"], "turno": "negras"}
        )
        self.assertAlmostEqual(estado.prob_ataque(Jugada.desde_texto("d6-e5")), 0.35)


def _preparar(color="blancas", reloj=30.0):
    strategy.preparar(
        types.SimpleNamespace(color=color, semilla=7, carpeta_datos=None, reloj_s=reloj)
    )


class EstrategiaTest(unittest.TestCase):
    def test_jugada_legal_desde_el_inicio(self):
        _preparar()
        estado = Estado.inicial()
        jugada = strategy.jugar(estado, 3.5)  # presupuesto mínimo: rápido
        self.assertIn(jugada, estado.jugadas())

    def test_reloj_agotado_responde_rapido_y_legal(self):
        _preparar()
        estado = Estado.inicial()
        import time

        t = time.perf_counter()
        jugada = strategy.jugar(estado, 0.5)
        self.assertLess(time.perf_counter() - t, 0.5)
        self.assertIn(jugada, estado.jugadas())

    def test_respaldo_legal_ante_excepcion(self):
        _preparar()
        estado = Estado.inicial()
        with patch.object(strategy, "_decidir", side_effect=RuntimeError("falla")):
            self.assertEqual(strategy.jugar(estado, 30.0), estado.jugadas()[0])

    def test_toma_victoria_segura(self):
        _preparar()
        estado = Estado.desde_dict(
            {
                "blancas": ["a2", "e8"],
                "negras": ["i9", "a8"],
                "turno": "blancas",
                "puntos": {"blancas": 2},
            }
        )
        self.assertTrue(str(strategy.jugar(estado, 30.0)).endswith("9"))  # e8 anota

    def test_negras_toman_victoria_segura(self):
        _preparar("negras")
        estado = Estado.desde_dict(
            {
                "blancas": ["a1", "i5"],
                "negras": ["c2", "i9"],
                "turno": "negras",
                "puntos": {"negras": 2},
            }
        )
        self.assertTrue(str(strategy.jugar(estado, 30.0)).startswith("c2-"))  # c2 anota

    def test_desenlace_que_deja_ganar_al_rival_vale_cero(self):
        _preparar()
        # Negras con 2 puntos y un peón en d2 con d1 libre: ganan seguro en su turno.
        s = Estado.desde_dict(
            {
                "blancas": ["a5"],
                "negras": ["d2", "i9"],
                "turno": "negras",
                "puntos": {"negras": 2},
            }
        )
        self.assertEqual(strategy._valor_fijo(s), 0.0)

    def test_anotar_atacando_no_es_victoria_segura(self):
        _preparar()
        # Negras con 2 puntos; su único avance a la fila 1 es un ataque (puede fallar).
        s = Estado.desde_dict(
            {
                "blancas": ["c1", "e1", "d1", "a5"],
                "negras": ["d2", "i9"],
                "turno": "negras",
                "puntos": {"negras": 2},
            }
        )
        self.assertFalse(strategy._gana_seguro(s, "negras"))
        self.assertIsNone(strategy._valor_fijo(s))

    def test_amenaza_en_dos_sin_defensa_vale_cero(self):
        _preparar()
        # Negras (2 puntos) avanzan d3-d2 y luego anotan; Blancas no pueden capturar.
        s = Estado.desde_dict(
            {
                "blancas": ["i5"],
                "negras": ["d3", "a9"],
                "turno": "negras",
                "puntos": {"negras": 2},
            }
        )
        self.assertEqual(strategy._supervivencia(s), 0.0)

    def test_amenaza_en_dos_con_captura_es_su_probabilidad(self):
        _preparar()
        # Negras pueden ir a c2, d2 o e2; Blancas en c1/d1/e1 atacan cualquiera de las
        # tres: sobrevivir cuesta ganar ese combate. La cota debe estar entre 0 y 1.
        s = Estado.desde_dict(
            {
                "blancas": ["c1", "d1", "e1", "i5"],
                "negras": ["d3", "a9"],
                "turno": "negras",
                "puntos": {"negras": 2},
            }
        )
        tope = strategy._supervivencia(s)
        self.assertGreater(tope, 0.0)
        self.assertLess(tope, 1.0)

    def test_anota_seguro_detecta_casilla_libre(self):
        s = Estado.desde_dict({"blancas": ["e8"], "negras": ["e9", "d9", "f9"], "turno": "blancas"})
        self.assertFalse(strategy._anota_seguro(s._tablero, "blancas"))  # solo ataques
        s = Estado.desde_dict({"blancas": ["e8"], "negras": ["e9", "d9"], "turno": "blancas"})
        self.assertTrue(strategy._anota_seguro(s._tablero, "blancas"))  # f9 libre

    def test_libro_se_usa_si_la_jugada_es_legal(self):
        _preparar()
        estado = Estado.inicial()
        with patch.dict(strategy.libro, {str(estado.clave): "e2-e3"}, clear=True):
            self.assertEqual(str(strategy.jugar(estado, 30.0)), "e2-e3")
        with patch.dict(strategy.libro, {str(estado.clave): "e2-e5"}, clear=True):
            self.assertIn(strategy.jugar(estado, 3.5), estado.jugadas())  # ilegal: se ignora

    def test_desenlace_terminal_tiene_valor_exacto(self):
        _preparar()
        s = Estado.desde_dict(
            {"blancas": ["a5"], "negras": ["i9"], "turno": "negras", "puntos": {"blancas": 3}}
        )
        self.assertEqual(strategy._valor_fijo(s), 1.0)

    def test_azar_estratificado_respeta_probabilidades(self):
        _preparar()
        estado = Estado.desde_dict(
            {"blancas": ["e4", "d3", "f3"], "negras": ["f5", "g6", "a9"], "turno": "blancas"}
        )
        brazo = strategy._Brazo(estado, Jugada.desde_texto("e4-f5"))
        with patch.object(strategy, "simular_al_azar", return_value=0.5):
            for _ in range(100):
                brazo.tirar()
        visitas = [r[3] for r in brazo.resultados]
        self.assertEqual(sum(visitas), 100)
        self.assertLessEqual(abs(visitas[0] - 65), 1)  # p = 0.65 exacto, sin dado

    def test_evita_dejar_ganar_al_rival(self):
        _preparar()
        # Negras tienen 2 puntos y un peón en c2. Blancas pueden capturarlo desde b1
        # (b1-c2) o jugar otra cosa; si no lo capturan, Negras ganan seguro (c2-c1 libre).
        estado = Estado.desde_dict(
            {
                "blancas": ["b1", "h5"],
                "negras": ["c2", "a9"],
                "turno": "blancas",
                "puntos": {"negras": 2},
            }
        )
        self.assertEqual(str(strategy.jugar(estado, 30.0)), "b1-c2")


if __name__ == "__main__":
    unittest.main()
