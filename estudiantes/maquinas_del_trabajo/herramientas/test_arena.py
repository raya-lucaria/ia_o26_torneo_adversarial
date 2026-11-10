"""Pruebas de la arena local (partidas en proceso, con reloj y dados propios).

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_arena -v
"""

import time
import unittest

from herramientas.arena import Serie, _jugar_par, crear_jugador, partida


class SiempreIlegal:
    def preparar(self, config):
        pass

    def jugar(self, estado, tiempo_restante):
        return "a1-a1"


class Lento:
    """Juega la primera jugada legal, pero tarda más que todo su reloj."""

    def preparar(self, config):
        pass

    def jugar(self, estado, tiempo_restante):
        time.sleep(tiempo_restante + 0.01)
        return estado.jugadas()[0]


class PruebasArena(unittest.TestCase):
    def test_partida_contra_montecarlo_termina(self):
        r = partida(3, crear_jugador("nuestro"), crear_jugador("montecarlo"), reloj_s=4.0)
        self.assertIn(r.ganador, ("blancas", "negras"))
        self.assertEqual(r.motivo, "juego")
        self.assertGreater(r.plies, 10)

    def test_autojuego_con_pesos_distintos(self):
        a = crear_jugador("nuestro", {"APOYO": 0.2})
        b = crear_jugador("nuestro")
        r = partida(4, a, b, reloj_s=4.0)
        self.assertEqual(r.motivo, "juego")

    def test_jugada_ilegal_pierde(self):
        r = partida(5, SiempreIlegal(), crear_jugador("montecarlo"), reloj_s=4.0)
        self.assertEqual((r.ganador, r.motivo, r.plies), ("negras", "ilegal", 0))

    def test_pasarse_del_reloj_pierde(self):
        r = partida(6, crear_jugador("montecarlo"), Lento(), reloj_s=0.05)
        self.assertEqual((r.ganador, r.motivo), ("blancas", "reloj"))

    def test_pesos_desde_archivo(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "pesos.npz"
            from evaluacion import N_PESOS

            pesos = np.arange(N_PESOS, dtype=np.float64) / 1e6
            np.savez(archivo, pesos=pesos)
            jugador = crear_jugador("nuestro", {"APOYO": 9.0}, archivo)
            esperado = pesos.copy()
            esperado[3] = 9.0  # los cambios por nombre se aplican encima del archivo
            self.assertEqual(jugador.pesos.tolist(), esperado.tolist())

    def test_peso_desconocido_es_error(self):
        with self.assertRaises(ValueError):
            crear_jugador("nuestro", {"NO_EXISTE": 1.0})

    def test_por_nodos_es_repetible(self):
        # Con nodos fijos el resultado no depende del reloj: la misma semilla da la misma partida.
        a = partida(7, crear_jugador("nuestro", nodos=3000), crear_jugador("nuestro", nodos=6000))
        b = partida(7, crear_jugador("nuestro", nodos=3000), crear_jugador("nuestro", nodos=6000))
        self.assertEqual((a.ganador, a.motivo, a.plies), (b.ganador, b.motivo, b.plies))
        self.assertEqual(a.motivo, "juego")

    def test_reloj_por_jugador(self):
        # Negras con un reloj de 0.05 s se pasa; Blancas, con 30 s, no.
        r = partida(8, crear_jugador("montecarlo"), Lento(), reloj_s=30.0, reloj_negras=0.05)
        self.assertEqual((r.ganador, r.motivo), ("blancas", "reloj"))

    def test_opciones_por_jugador(self):
        # Sin TT ni orden, el mismo bot con los mismos nodos juega distinto que con ellos.
        con = crear_jugador("nuestro", nodos=3000)
        sin = crear_jugador("nuestro", nodos=3000, opciones=[0, 0])
        con.preparar(None)
        sin.preparar(None)
        self.assertEqual(sin.bot.memoria.opciones[:2].tolist(), [0, 0])
        self.assertEqual(con.bot.memoria.opciones[:2].tolist(), [1, 1])
        a = partida(10, crear_jugador("nuestro", nodos=3000), crear_jugador("nuestro", nodos=3000))
        b = partida(10, crear_jugador("nuestro", nodos=3000, opciones=[0, 0]),
                    crear_jugador("nuestro", nodos=3000))
        self.assertNotEqual((a.ganador, a.plies), (b.ganador, b.plies))

    def test_par_juega_ambos_colores(self):
        serie = Serie(
            rival="nuestro", cambios={}, cambios_rival={}, pesos=None, pesos_rival=None,
            nodos=2000, nodos_rival=2000, reloj=30.0, reloj_rival=30.0, factor=4.0,
            opciones=None, opciones_rival=None, escala=None, escala_rival=None,
        )
        par = _jugar_par((9, serie))
        self.assertEqual(len(par), 2)
        # Mismos pesos, mismos nodos y mismos dados: con cada color gana el mismo bando, así que
        # nuestro gana exactamente una de las dos.
        self.assertEqual(sum(g for g, *_ in par), 1)
        self.assertTrue(all(motivo == "juego" for _, motivo, *_ in par))


if __name__ == "__main__":
    unittest.main()
