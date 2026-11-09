"""Pruebas del motor rápido: debe coincidir en todo con el motor oficial.

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_motor_rapido -v
"""

import unittest

import numpy as np
from motor import nucleo

import motor_rapido as mr
from herramientas import desde_python
from herramientas.posiciones import muestrear
from herramientas.validar_motor import discrepancias


class SoloPublico:
    """Un estado con solo la API pública: sin `_tablero`, `_puntos` ni `_color`."""

    def __init__(self, estado):
        self.turno = estado.turno
        self.puntos = estado.puntos
        self._listas = {"blancas": estado.piezas("blancas"), "negras": estado.piezas("negras")}

    def piezas(self, color):
        return list(self._listas[color])


class PruebasMotorRapido(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.posiciones = muestrear(3000, semilla=1)

    def test_zobrist_igual_al_oficial(self):
        self.assertTrue(np.array_equal(mr.Z_PIEZAS, nucleo.Z_PIEZAS))
        self.assertEqual(int(mr.Z_TURNO), int(nucleo.Z_TURNO))
        self.assertTrue(np.array_equal(mr.Z_PUNTOS, nucleo.Z_PUNTOS))

    def test_diferencial(self):
        fallas = []
        for e in self.posiciones:
            encontradas = discrepancias(e)
            if encontradas:
                fallas.append((e.a_dict(), encontradas))
        self.assertEqual(fallas[:3], [], f"{len(fallas)} posiciones no coinciden")

    def test_tiene_jugada(self):
        origenes = np.empty(mr.MAX_JUGADAS, np.int64)
        destinos = np.empty(mr.MAX_JUGADAS, np.int64)
        vistos = set()
        for e in self.posiciones:
            tab, _, _ = mr.desde_estado(e)
            for color in (0, 1):
                hay = nucleo.generar(tab, color, origenes, destinos) > 0
                self.assertEqual(desde_python.tiene_jugada(tab, color), hay, e.a_dict())
                vistos.add(hay)
        self.assertEqual(vistos, {True, False}, "la muestra debe tener bandos sin jugadas")

    def test_desde_estado_sin_atributos(self):
        for e in self.posiciones[:300]:
            tab, pts, color = mr.desde_estado(SoloPublico(e))
            self.assertEqual(tab.dtype, np.int8)
            self.assertEqual(pts.dtype, np.int64)
            self.assertTrue(np.array_equal(tab, e._tablero))
            self.assertTrue(np.array_equal(pts, e._puntos))
            self.assertEqual(color, e._color)

    def test_desde_estado_regresa_copias(self):
        e = self.posiciones[0]
        antes = e._tablero.copy()
        tab, pts, _ = mr.desde_estado(e)
        tab[:] = 0  # el estado del árbitro es de solo lectura: necesitamos copias propias
        pts[:] = 0
        self.assertTrue(np.array_equal(e._tablero, antes))

    def test_texto_jugada(self):
        self.assertEqual(mr.texto_jugada(mr.codigo(13, 22)), "e2-e3")
        self.assertEqual(mr.texto_jugada(mr.codigo(67, 58)), "e8-e7")
        self.assertEqual(mr.texto_jugada(mr.codigo(0, 10)), "a1-b2")


if __name__ == "__main__":
    unittest.main()
