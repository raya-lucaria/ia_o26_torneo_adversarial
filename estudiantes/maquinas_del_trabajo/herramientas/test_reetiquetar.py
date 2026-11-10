"""Pruebas del reetiquetado con búsqueda profunda.

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_reetiquetar -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from herramientas.datos import partida_para_datos
from herramientas.reetiquetar import juntar, muestra, reetiquetar, valor_profundo
from strategy import cargar_pesos

CARPETA = Path(__file__).resolve().parent.parent


class PruebasReetiquetar(unittest.TestCase):
    def test_muestra_fija_y_sin_repetir(self):
        a = muestra(10_000, 2500, semilla=3)
        self.assertTrue(np.array_equal(a, muestra(10_000, 2500, semilla=3)))
        self.assertEqual(len(np.unique(a)), 2500)
        self.assertTrue(np.all(np.diff(a) > 0))
        self.assertEqual(len(muestra(10, 50, semilla=3)), 10)  # no pide más de las que hay

    @mock.patch("herramientas.reetiquetar.POR_BLOQUE", 50)  # varios bloques con pocos datos
    def test_reetiqueta_retoma_y_junta(self):
        partidas = [partida_para_datos(semilla=s, rival="nuestro", nodos=1500) for s in (1, 2, 3)]
        datos = {campo: np.concatenate([getattr(d, campo) for d in partidas]) for campo in partidas[0]._fields}
        datos["partida"] = np.concatenate([np.full(len(d.y), k, np.int32) for k, d in enumerate(partidas)])
        cuantas = 130  # tres bloques de 50, el último incompleto
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archivo = tmp / "datos.npz"
            np.savez(archivo, **datos)
            bloques = tmp / "bloques"
            reetiquetar(archivo, bloques, cuantas, semilla=1, nodos=800, procesos=2)
            primeros = sorted(bloques.glob("bloque_*.npz"))
            self.assertEqual(len(primeros), 3)
            fechas = [b.stat().st_mtime_ns for b in primeros]
            reetiquetar(archivo, bloques, cuantas, semilla=1, nodos=800, procesos=2)  # nada que hacer
            self.assertEqual([b.stat().st_mtime_ns for b in primeros], fechas)

            salida = tmp / "salida.npz"
            juntar(archivo, bloques, cuantas, semilla=1, salida=salida)
            with np.load(salida) as r:
                elegidas = muestra(len(datos["y"]), cuantas, semilla=1)
                self.assertEqual(len(r["v"]), cuantas)
                for campo in ("partida", "y", "tabs", "ptss", "colores"):
                    self.assertTrue(np.array_equal(r[campo], datos[campo][elegidas]), campo)
                self.assertTrue(np.array_equal(r["v_superficial"], datos["v"][elegidas]))
                v = r["v"]
                self.assertTrue(np.all(np.isnan(v) | (np.abs(v) <= 1.0)))
                # El valor guardado es el de buscar esa posición sola, con memoria nueva.
                pesos = cargar_pesos(CARPETA / "data")
                for k in (0, len(v) // 2, len(v) - 1):
                    esperado, _ = valor_profundo(
                        r["tabs"][k], r["ptss"][k].astype(np.int64), int(r["colores"][k]), 800, pesos
                    )
                    self.assertTrue(np.isclose(v[k], esperado, equal_nan=True), k)


if __name__ == "__main__":
    unittest.main()
