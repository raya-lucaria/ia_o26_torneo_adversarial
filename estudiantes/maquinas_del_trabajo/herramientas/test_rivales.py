"""Pruebas de los rivales de otros estilos (`herramientas/rivales.py`).

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_rivales -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from motor import Estado

from herramientas.arena import Serie, _jugar_par, crear_jugador, main, partida
from herramientas.rivales import ESTILOS, Estilo, _rasgos, evaluar
from motor_rapido import LADO, desde_estado


def _jugada(nombre, estado, tiempo_restante=30.0):
    bot = Estilo(nombre)
    bot.preparar(SimpleNamespace(color=estado.turno, semilla=1))
    return str(bot.jugar(estado, tiempo_restante))


class PruebasRivales(unittest.TestCase):
    def test_rasgos_a_mano(self):
        # Ejemplo 3 de docs/reglas.md: Blancas e4, d3, f3; Negras f5, g6.
        e = Estado.desde_dict(
            {"blancas": ["e4", "d3", "f3"], "negras": ["f5", "g6"], "turno": "blancas"}
        )
        tablero = desde_estado(e)[0].reshape(LADO, LADO)
        # e4 avanzó 3 filas y lo apoyan d3 y f3; ninguno está en su fila de casa.
        self.assertEqual(tuple(_rasgos(tablero, 0)), (3, 3 + 2 + 2, 3, 1, 0))
        # Para Negras, f5 avanzó 4 filas (desde la 9) y lo apoya g6 (atrás de Negras es la fila 6).
        self.assertEqual(tuple(_rasgos(tablero, 1)), (2, 4 + 3, 4, 1, 0))

    def test_posicion_inicial_pareja(self):
        tab, pts, _ = desde_estado(Estado.inicial())
        for color in (0, 1):
            self.assertEqual(evaluar(tab, pts, color, ESTILOS["equilibrado"]), 0.0)

    def test_toma_el_tercer_punto(self):
        e = Estado.desde_dict({
            "blancas": ["e8", "a2"], "negras": ["a9", "i9"], "turno": "blancas",
            "puntos": {"blancas": 2, "negras": 0},
        })
        for nombre in ESTILOS:
            self.assertTrue(_jugada(nombre, e).startswith("e8-"), nombre)

    def test_ataca_al_peon_que_va_a_anotar(self):
        # Negras tienen 2 puntos y e2 anota en su turno: lo único que no pierde seguro es atacarlo.
        e = Estado.desde_dict({
            "blancas": ["d1", "a2"], "negras": ["e2", "i9"], "turno": "blancas",
            "puntos": {"blancas": 0, "negras": 2},
        })
        for nombre in ESTILOS:
            self.assertEqual(_jugada(nombre, e), "d1-e2", nombre)

    def test_sin_tiempo_juega_igual(self):
        # Con el reloj casi en la reserva solo piensa 2 plies, pero contesta una jugada legal.
        e = Estado.inicial()
        self.assertIn(_jugada("corredor", e, tiempo_restante=0.5), {str(j) for j in e.jugadas()})

    def test_partida_contra_nuestro(self):
        nuestro = crear_jugador("nuestro", nodos=2000)
        r = partida(3, nuestro, crear_jugador("defensivo"), reloj_s=4.0)
        self.assertEqual(r.motivo, "juego")
        self.assertEqual(len(r.jugadas), r.plies)
        self.assertGreaterEqual(max(r.puntos.values()), 0)

    def test_par_guarda_el_registro(self):
        serie = Serie(
            rival="corredor", cambios={}, cambios_rival={}, pesos=None, pesos_rival=None,
            nodos=2000, nodos_rival=None, reloj=4.0, reloj_rival=4.0, factor=4.0,
            opciones=None, opciones_rival=None, escala=None, escala_rival=None,
        )
        par = _jugar_par((11, serie))
        self.assertEqual([registro["nuestro"] for *_, registro in par], ["blancas", "negras"])
        for gano, motivo, plies, _, registro in par:
            self.assertEqual((registro["rival"], registro["motivo"]), ("corredor", motivo))
            self.assertEqual(len(registro["jugadas"]), plies)
            self.assertEqual(gano, registro["ganador"] == registro["nuestro"])

    def test_guardar_perdidas(self):
        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "perdidas.jsonl"
            argumentos = [
                "--rival", "agresivo", "--nodos", "500", "--reloj", "4", "--partidas", "2",
                "--procesos", "1", "--guardar", str(archivo),
            ]
            self.assertEqual(main(argumentos), 0)
            for linea in archivo.read_text(encoding="utf-8").splitlines():
                registro = json.loads(linea)
                self.assertNotEqual(registro["ganador"], registro["nuestro"])  # solo las perdidas


if __name__ == "__main__":
    unittest.main()
