"""Pruebas de la búsqueda (negamax con alfa-beta y Star1).

La referencia es `posiciones.expectimax`: un expectimax lento, sin podas, con el motor
oficial. Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_busqueda -v
"""

import random
import time
import unittest

import numpy as np

from busqueda import (
    EXACTO,
    INFERIOR,
    MAX_NODOS,
    PROF_MAX,
    SUPERIOR,
    buscar,
    crear_memoria,
    negamax,
    profundizar,
)
from herramientas.desde_python import tt_buscar, tt_guardar, ultimo_punto
from herramientas.posiciones import expectimax, muestrear, posicion
from motor_rapido import clave, codigo, desde_estado, texto_jugada


def memoria_simple():
    """Memoria nueva sin TT ni orden: alfa-beta y Star1 puros."""
    m = crear_memoria(bits_tt=16)
    m.opciones[:] = 0
    return m


def memoria_con(usar_tt, ordenar, *otras):
    """Memoria con las opciones dadas (las que no se dan, apagadas)."""
    m = crear_memoria(bits_tt=16)
    m.opciones[:] = 0
    m.opciones[: 2 + len(otras)] = (usar_tt, ordenar, *otras)
    return m


def negamax_en_ply_1(m, tab, pts, color, prof, alfa, beta):
    """Llama a negamax con la posición en el ply 1 (en la raíz la TT nunca corta)."""
    m.tabs[1] = tab
    m.ptss[1] = pts
    m.claves[1] = clave(tab, color, pts)
    m.contadores[:2] = 0
    m.limites[:] = (np.inf, float(MAX_NODOS))
    return negamax(m, 1, prof, float(alfa), float(beta), color)


class PruebasNegamax(unittest.TestCase):
    def test_igual_a_expectimax_prof_1_y_2(self):
        for e in muestrear(150, 3):
            tab, pts, color = desde_estado(e)
            for d in (1, 2):
                v, _ = buscar(memoria_simple(), tab, pts, color, d)
                self.assertAlmostEqual(v, expectimax(e, d), delta=1e-9, msg=f"d={d} {e.a_dict()}")

    def test_igual_a_expectimax_prof_3(self):
        for e in muestrear(12, 4):
            tab, pts, color = desde_estado(e)
            v, _ = buscar(memoria_simple(), tab, pts, color, 3)
            self.assertAlmostEqual(v, expectimax(e, 3), delta=1e-9, msg=e.a_dict())

    def test_ventanas_fail_soft(self):
        rng = random.Random(5)
        for e in muestrear(60, 5):
            tab, pts, color = desde_estado(e)
            v, _ = buscar(memoria_simple(), tab, pts, color, 2)
            ventanas = [(v - 0.01, v + 0.01), (v + 0.001, v + 0.2), (v - 0.2, v - 0.001)]
            ventanas += [tuple(sorted((rng.uniform(-1, 1), rng.uniform(-1, 1)))) for _ in range(4)]
            for alfa, beta in ventanas:
                alfa, beta = max(alfa, -1.0), min(beta, 1.0)
                r, _ = buscar(memoria_simple(), tab, pts, color, 2, alfa, beta)
                contexto = f"v={v} ventana=({alfa}, {beta}) r={r} {e.a_dict()}"
                if v <= alfa:  # falla abajo: r es una cota superior de v
                    self.assertTrue(v - 1e-9 <= r <= alfa, contexto)
                elif v >= beta:  # falla arriba: r es una cota inferior de v
                    self.assertTrue(beta <= r <= v + 1e-9, contexto)
                else:
                    self.assertAlmostEqual(r, v, delta=1e-9, msg=contexto)

    def test_jugada_optima(self):
        for e in muestrear(40, 6):
            tab, pts, color = desde_estado(e)
            v, cod = buscar(memoria_simple(), tab, pts, color, 2)
            legales = {str(j): j for j in e.jugadas()}
            self.assertIn(texto_jugada(cod), legales)
            j = legales[texto_jugada(cod)]
            valor_j = sum(p * -expectimax(hijo, 1, ply=1) for p, hijo in e.siguiente(j))
            self.assertAlmostEqual(valor_j, v, delta=1e-9, msg=e.a_dict())


class PruebasTablaYOrden(unittest.TestCase):
    # Claves que caen en la misma cubeta de una TT de 2**16 lugares (mismos 16 bits de abajo).
    H1 = np.uint64(0x1234_0000_0000_0010)
    H2 = np.uint64(0x5678_0000_0001_0010)
    H3 = np.uint64(0x9ABC_0000_0002_0010)
    H4 = np.uint64(0xDEF0_0000_0003_0010)

    def test_valores_iguales_con_tt_y_orden(self):
        for e in muestrear(30, 6):
            tab, pts, color = desde_estado(e)
            v_simple, _ = buscar(memoria_simple(), tab, pts, color, 3)
            v_completa, _ = buscar(memoria_con(1, 1), tab, pts, color, 3)
            self.assertAlmostEqual(v_completa, v_simple, delta=1e-9, msg=e.a_dict())

    def test_valores_iguales_con_cortes_de_tt(self):
        # A profundidad 3 no hay transposiciones dentro de una búsqueda (la TT nunca corta);
        # a profundidad 4 sí: W a, N x, W b llega a lo mismo que W b, N x, W a.
        for e in muestrear(20, 9, sinteticas=0, ply_min=20, ply_max=60):
            tab, pts, color = desde_estado(e)
            v_simple, _ = buscar(memoria_simple(), tab, pts, color, 4)
            v_completa, _ = buscar(memoria_con(1, 1), tab, pts, color, 4)
            self.assertAlmostEqual(v_completa, v_simple, delta=1e-9, msg=e.a_dict())

    def test_tt_respeta_el_tipo_de_cota(self):
        probadas = 0
        for e in muestrear(15, 10, sinteticas=0, ply_min=20, ply_max=60):
            tab, pts, color = desde_estado(e)
            v = negamax_en_ply_1(crear_memoria(bits_tt=16), tab, pts, color, 2, -1.0, 1.0)
            if abs(v) >= 0.5:
                continue
            probadas += 1
            h = np.uint64(clave(tab, color, pts))
            # (tipo y valor guardados, ventana, lo que debe regresar)
            casos = [
                (SUPERIOR, v + 0.3, (v + 0.4, 1.0), v + 0.3),  # cota ≤ α: corta
                (SUPERIOR, v + 0.3, (-1.0, v + 0.1), v),  # cota ≥ β no dice nada: busca
                (INFERIOR, v - 0.3, (-1.0, v - 0.4), v - 0.3),  # cota ≥ β: corta
                (INFERIOR, v - 0.3, (v - 0.1, 1.0), v),  # cota ≤ α no dice nada: busca
                (EXACTO, v, (-1.0, 1.0), v),
            ]
            for tipo, guardado, (alfa, beta), esperado in casos:
                m = crear_memoria(bits_tt=16)
                tt_guardar(m, h, 10, guardado, tipo, -1)
                r = negamax_en_ply_1(m, tab, pts, color, 2, alfa, beta)
                self.assertAlmostEqual(r, esperado, delta=1e-9, msg=(tipo, guardado, alfa, beta))
            # Guardada con menos profundidad de la que se pide: no sirve para cortar.
            m = crear_memoria(bits_tt=16)
            tt_guardar(m, h, 1, 0.123, EXACTO, -1)
            self.assertAlmostEqual(negamax_en_ply_1(m, tab, pts, color, 2, -1.0, 1.0), v, delta=1e-9)
        self.assertGreater(probadas, 5)

    def test_ataques_malos_al_final_dan_el_mismo_valor(self):
        # Mandar al final los ataques que se espera perder solo cambia el orden: el valor es el mismo.
        for e in muestrear(20, 11, sinteticas=0, ply_min=20, ply_max=60):
            tab, pts, color = desde_estado(e)
            for prof in (3, 4):
                v_simple, _ = buscar(memoria_simple(), tab, pts, color, prof)
                v, _ = buscar(memoria_con(1, 1, 1), tab, pts, color, prof)
                self.assertAlmostEqual(v, v_simple, delta=1e-9, msg=(prof, e.a_dict()))

    def test_ultimo_punto_a_mano(self):
        casos = [
            # Blancas con 2 puntos: e8 anota recto.
            (posicion(["e8", "a2"], ["b9", "h9"], "blancas", (2, 0)), 1),
            # Negras con 2 puntos: e2 anota y Blancas no tienen con qué atacarlo.
            (posicion(["b4", "h4"], ["e2", "a8"], "blancas", (0, 2)), -1),
            # d1 puede capturar a e2: no se sabe sin buscar.
            (posicion(["d1", "b4", "h4"], ["e2", "a8", "i8"], "blancas", (0, 2)), 0),
            # Dos amenazas (e2 y h2): d1 solo captura una.
            (posicion(["d1", "b4"], ["e2", "h2", "a8"], "blancas", (0, 2)), -1),
            # e2 solo anotaría atacando (e1, d1 y f1 son blancas): no es amenaza segura.
            (posicion(["d1", "e1", "f1", "a4"], ["e2", "a8"], "blancas", (0, 2)), 0),
            # Blancas pueden anotar el último punto atacando antes que Negras: no se sabe.
            (posicion(["e8", "a4"], ["d9", "e9", "f9", "h2"], "blancas", (2, 2)), 0),
            # Blancas sin peón cerca de su meta; h2 anota sin defensa.
            (posicion(["a4"], ["h2"], "blancas", (2, 2)), -1),
            # Nadie tiene 2 puntos.
            (posicion(["e8"], ["e2"], "blancas", (1, 1)), 0),
        ]
        for estado, esperado in casos:
            tab, pts, color = desde_estado(estado)
            self.assertEqual(ultimo_punto(tab, pts, color), esperado, estado.a_dict())

    def test_ultimo_punto_es_exacto(self):
        # Lo que dice la regla en una hoja es lo que encuentra la búsqueda sin la regla 1 o 2
        # plies más abajo, con el mismo valor (misma pérdida por ply).
        probadas = {1: 0, -1: 0}
        for e in muestrear(3000, 21, sinteticas=1.0):
            tab, pts, color = desde_estado(e)
            veredicto = ultimo_punto(tab, pts, color)
            if veredicto == 0:
                continue
            probadas[veredicto] += 1
            v_regla, _ = buscar(memoria_con(0, 0, 0, 1), tab, pts, color, 0)
            v_busqueda, _ = buscar(memoria_simple(), tab, pts, color, 1 if veredicto > 0 else 2)
            self.assertAlmostEqual(v_regla, v_busqueda, delta=1e-12, msg=e.a_dict())
        self.assertGreater(min(probadas.values()), 30, probadas)

    def test_cache_de_evaluaciones_no_cambia_la_busqueda(self):
        # La misma memoria toda la serie, como en una partida: la caché guarda de una búsqueda
        # a la siguiente. Con y sin caché: misma jugada, mismo valor, mismos nodos.
        con, sin = crear_memoria(bits_tt=16), crear_memoria(bits_tt=16)
        sin.opciones[4] = 0
        for e in muestrear(40, 23, sinteticas=0.25, ply_min=10):
            tab, pts, color = desde_estado(e)
            a = profundizar(con, tab, pts, color, 1e6, 1e6, max_nodos=20_000)
            b = profundizar(sin, tab, pts, color, 1e6, 1e6, max_nodos=20_000)
            self.assertEqual((a.jugada, a.valor, a.prof, a.nodos), (b.jugada, b.valor, b.prof, b.nodos))
        self.assertGreater(np.count_nonzero(con.ev_clave), 1000)
        self.assertEqual(np.count_nonzero(sin.ev_clave), 0)

    def test_menos_nodos(self):
        nodos_simple = nodos_completa = 0
        for e in muestrear(20, 7, sinteticas=0, ply_min=20, ply_max=60):
            tab, pts, color = desde_estado(e)
            m = memoria_simple()
            buscar(m, tab, pts, color, 4)
            nodos_simple += int(m.contadores[0])
            m = memoria_con(1, 1)
            buscar(m, tab, pts, color, 4)
            nodos_completa += int(m.contadores[0])
        self.assertLessEqual(nodos_completa, 0.6 * nodos_simple, (nodos_completa, nodos_simple))

    def test_tt_guardar_y_leer(self):
        m = crear_memoria(bits_tt=16)
        m.contadores[2] = 1
        tt_guardar(m, self.H1, 5, 0.25, EXACTO, 123)
        lugar = tt_buscar(m, self.H1)
        self.assertGreaterEqual(lugar, 0)
        self.assertEqual(
            (m.tt_valor[lugar], m.tt_prof[lugar], m.tt_tipo[lugar], m.tt_jugada[lugar]),
            (0.25, 5, EXACTO, 123),
        )
        self.assertEqual(tt_buscar(m, self.H2), -1)  # misma cubeta, otra clave

    def test_tt_reemplazo(self):
        m = crear_memoria(bits_tt=16)
        m.contadores[2] = 1
        tt_guardar(m, self.H1, 6, 0.1, EXACTO, 1)
        lugar_0 = tt_buscar(m, self.H1)
        tt_guardar(m, self.H2, 3, 0.2, EXACTO, 2)  # más corta: no saca a la profunda
        self.assertEqual(tt_buscar(m, self.H1), lugar_0)
        self.assertEqual(tt_buscar(m, self.H2), lugar_0 + 1)
        tt_guardar(m, self.H3, 2, 0.3, EXACTO, 3)  # el lugar 1 siempre se reemplaza
        self.assertEqual(tt_buscar(m, self.H2), -1)
        self.assertEqual(tt_buscar(m, self.H3), lugar_0 + 1)
        self.assertEqual(tt_buscar(m, self.H1), lugar_0)
        m.contadores[2] = 2  # nueva búsqueda: lo viejo del lugar 0 ya se puede reemplazar
        tt_guardar(m, self.H4, 1, 0.4, EXACTO, 4)
        self.assertEqual(tt_buscar(m, self.H4), lugar_0)
        self.assertEqual(tt_buscar(m, self.H1), -1)

    def test_jugada_tt_ilegal_no_se_juega(self):
        for e in muestrear(30, 8):
            tab, pts, color = desde_estado(e)
            m = crear_memoria(bits_tt=16)
            # Basura en la TT para la raíz: una jugada imposible y una profundidad enorme.
            h = np.uint64(clave(tab, color, pts))
            tt_guardar(m, h, 60, 0.5, EXACTO, codigo(0, 80))
            _, cod = buscar(m, tab, pts, color, 2)
            self.assertIn(texto_jugada(cod), {str(j) for j in e.jugadas()}, e.a_dict())


class PruebasProfundizar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compila todo antes de medir tiempos.
        tab, pts, color = desde_estado(muestrear(1, 0, sinteticas=0, ply_min=20)[0])
        profundizar(crear_memoria(bits_tt=16), tab, pts, color, 0.05, 0.1)
        cls.medio_juego = muestrear(10, 13, sinteticas=0, ply_min=20, ply_max=60)

    def jugar(self, estado, blando, duro, **kwargs):
        tab, pts, color = desde_estado(estado)
        return profundizar(crear_memoria(bits_tt=18), tab, pts, color, blando, duro, **kwargs)

    def test_resultado_legal(self):
        for e in muestrear(50, 8):
            r = self.jugar(e, 0.05, 0.15)
            legales = [str(j) for j in e.jugadas()]
            self.assertIn(texto_jugada(r.jugada), legales, e.a_dict())
            self.assertTrue(r.prof >= 1 or len(legales) == 1, (r, e.a_dict()))

    def test_profundidad_fija_igual_a_buscar(self):
        for e in self.medio_juego:
            r = self.jugar(e, 100.0, 100.0, prof_max=3)
            tab, pts, color = desde_estado(e)
            v, _ = buscar(crear_memoria(bits_tt=18), tab, pts, color, 3)
            self.assertEqual(r.prof, 3)
            self.assertAlmostEqual(r.valor, v, delta=1e-9, msg=e.a_dict())

    def test_corte_por_tiempo(self):
        for e in self.medio_juego[:3]:
            inicio = time.perf_counter()
            r = self.jugar(e, 0.2, 0.3)
            self.assertLessEqual(time.perf_counter() - inicio, 0.35, r)

    def test_corte_por_nodos_determinista(self):
        for e in self.medio_juego[:5]:
            a = self.jugar(e, 100.0, 100.0, max_nodos=20_000)
            b = self.jugar(e, 100.0, 100.0, max_nodos=20_000)
            self.assertEqual(a[:4], b[:4])
            self.assertLess(a.nodos, 20_100)

    def test_una_jugada(self):
        # Blancas solo tienen a2: enfrente está a3 (negra) y la única salida es b3.
        r = self.jugar(posicion(["a2"], ["a3", "h8"], "blancas"), 1.0, 1.0)
        self.assertEqual((texto_jugada(r.jugada), r.prof, r.nodos), ("a2-b3", 0, 0))

    def test_tope_de_profundidad(self):
        r = self.jugar(posicion(["a2"], ["i8"], "blancas"), 1.0, 3.0)
        self.assertLessEqual(r.prof, PROF_MAX)
        self.assertIn(texto_jugada(r.jugada), {"a2-a3", "a2-b3"})

    def test_gana_en_1(self):
        casos = [
            (posicion(["e8", "a2"], ["b9", "h9"], "blancas", (2, 0)), {"e8-d9", "e8-e9", "e8-f9"}),
            (posicion(["b1", "h1"], ["e2", "a8"], "negras", (0, 2)), {"e2-d1", "e2-e1", "e2-f1"}),
        ]
        for estado, ganadoras in casos:
            r = self.jugar(estado, 0.2, 0.6)
            self.assertIn(texto_jugada(r.jugada), ganadoras)
            self.assertAlmostEqual(r.valor, 1 - 0.0005, delta=1e-12)

    def test_evita_perder_en_1(self):
        estado = posicion(["d1", "b4", "h4"], ["e2", "a8", "i8"], "blancas", (0, 2))
        r = self.jugar(estado, 0.2, 0.6)
        self.assertEqual(texto_jugada(r.jugada), "d1-e2")


if __name__ == "__main__":
    unittest.main()
