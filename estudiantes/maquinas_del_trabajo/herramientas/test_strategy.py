"""Pruebas del bot completo (`strategy.py`): siempre una jugada legal, a tiempo, pase lo que pase.

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_strategy -v
"""

import io
import random
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from motor import Estado

import strategy
from herramientas.posiciones import muestrear, posicion

CONFIG = SimpleNamespace(color="blancas", semilla=1, carpeta_datos=Path("data"), reloj_s=30.0)

# En un proceso nuevo, como en la liga: preparar y luego contar cuántas veces se compiló cada
# función numba. Cada compilación extra cuesta tiempo de `preparar` (límite de 10 s).
CONTAR_COMPILACIONES = """
from pathlib import Path
from types import SimpleNamespace
from numba.core.registry import CPUDispatcher
import busqueda, evaluacion, motor_rapido, strategy
strategy.preparar(SimpleNamespace(color="blancas", semilla=1, carpeta_datos=Path("data"), reloj_s=30.0))
for modulo in (motor_rapido, evaluacion, busqueda):
    for nombre, f in vars(modulo).items():
        if isinstance(f, CPUDispatcher) and f.__module__ == modulo.__name__:
            print(nombre, len(f.signatures))
"""


class PruebasStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy.preparar(CONFIG)
        cls.medio_juego = muestrear(20, 10, sinteticas=0, ply_min=20, ply_max=60)

    def assertLegal(self, jugada, estado):
        self.assertIsInstance(jugada, str)
        self.assertIn(jugada, [str(j) for j in estado.jugadas()], estado.a_dict())

    def test_jugadas_legales_ambos_colores(self):
        turnos = set()
        for e in muestrear(300, 9):
            self.assertLegal(strategy.jugar(e, 0.3), e)
            turnos.add(e.turno)
        self.assertEqual(turnos, {"blancas", "negras"})
        for e in self.medio_juego[:6]:
            self.assertLegal(strategy.jugar(e, 30.0), e)

    def test_poco_tiempo_responde_rapido(self):
        for e in self.medio_juego:
            inicio = time.perf_counter()
            jugada = strategy.jugar(e, 0.05)
            self.assertLess(time.perf_counter() - inicio, 0.05)
            self.assertLegal(jugada, e)

    def test_una_sola_jugada(self):
        e = posicion(["a2"], ["a3", "h8"], "blancas")  # a2 solo puede ir a b3
        inicio = time.perf_counter()
        jugada = strategy.jugar(e, 30.0)
        self.assertLess(time.perf_counter() - inicio, 0.005)
        self.assertEqual(jugada, "a2-b3")

    def test_respaldo_si_falla_la_busqueda(self):
        def falla(*args, **kwargs):
            raise RuntimeError("falla de prueba")

        with mock.patch.object(strategy, "profundizar", falla), mock.patch("sys.stderr", io.StringIO()):
            for e in muestrear(20, 14):
                self.assertLegal(strategy.jugar(e, 10.0), e)

    def test_respaldo_prefiere_anotar(self):
        # e8-e9 anota sin combate; d9 y f9 son ataques; la primera legal sería a2-a3.
        e = posicion(["a2", "e8"], ["d9", "f9"], "blancas")
        self.assertEqual(strategy.respaldo(e, [str(j) for j in e.jugadas()]), "e8-e9")

    def test_partida_completa(self):
        # Contra jugadas al azar, con la misma memoria toda la partida (como en la liga).
        strategy.preparar(CONFIG)
        rng = random.Random(11)
        estado = Estado.inicial()
        for _ in range(300):
            if estado.terminado():
                break
            if estado.turno == "blancas":
                jugada = strategy.jugar(estado, 6.0)
                self.assertLegal(jugada, estado)
            else:
                jugada = str(rng.choice(estado.jugadas()))
            gana = rng.random() < estado.prob_ataque(jugada) if estado.es_ataque(jugada) else None
            estado = estado.aplicar(jugada, gana)
        self.assertTrue(estado.terminado(), estado.a_dict())

    def test_preparar_compila_cada_funcion_una_vez(self):
        carpeta = Path(__file__).resolve().parent.parent
        salida = subprocess.run(
            [sys.executable, "-c", CONTAR_COMPILACIONES],
            cwd=carpeta, capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        compilaciones = dict(linea.split() for linea in salida if linea.strip())
        self.assertIn("negamax", compilaciones)
        repetidas = {nombre: n for nombre, n in compilaciones.items() if int(n) > 1}
        self.assertEqual(repetidas, {})

    def test_cargar_pesos(self):
        import tempfile

        import numpy as np

        from evaluacion import N_BASE, N_PESOS

        with tempfile.TemporaryDirectory() as carpeta:
            self.assertIsNone(strategy.cargar_pesos(carpeta))  # sin archivo: pesos a mano
            ruta = Path(carpeta) / strategy.ARCHIVO_PESOS
            np.savez(ruta, pesos=np.arange(N_PESOS, dtype=np.float32))
            self.assertEqual(strategy.cargar_pesos(carpeta).tolist(), list(range(N_PESOS)))
            np.savez(ruta, pesos=np.zeros(7))
            with self.assertRaises(ValueError):
                strategy.cargar_pesos(carpeta)
            # Un archivo malo no tumba preparar: se juega con los pesos a mano.
            with mock.patch("traceback.print_exc"):
                strategy.preparar(SimpleNamespace(**{**vars(CONFIG), "carpeta_datos": Path(carpeta)}))
            self.assertEqual(len(strategy.bot.memoria.pesos), N_BASE)
        strategy.preparar(CONFIG)  # deja el bot de verdad para las demás pruebas
        self.assertEqual(len(strategy.bot.memoria.pesos), N_PESOS)  # con su tabla de data/

    def test_posiciones_raras(self):
        for e in muestrear(1000, 12, sinteticas=1.0):
            self.assertLegal(strategy.jugar(e, 0.05), e)


if __name__ == "__main__":
    unittest.main()
