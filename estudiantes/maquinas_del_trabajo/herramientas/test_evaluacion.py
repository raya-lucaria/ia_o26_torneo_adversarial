"""Pruebas de la evaluación F2 (a mano): rango, simetrías, material y un caso hecho a mano.

Correr desde estudiantes/maquinas_del_trabajo/:
    ../../.venv/bin/python -m unittest herramientas.test_evaluacion -v
"""

import math
import random
import unittest

import numpy as np
from motor import Estado

from evaluacion import (
    CORREDOR_MAX,
    DECIDIDA,
    I_ATAQUES_RIVAL,
    I_CARRERA,
    I_CORREDOR,
    I_MEJOR_ATAQUE,
    I_PATRONES,
    MAX_PEONES,
    N_BASE,
    N_PESOS,
    PESOS_F2,
    patrones,
    rasgos,
)
from herramientas.desde_python import apoyo_eval, apoyo_motor, cono, evaluar
from herramientas.posiciones import muestrear, posicion
from motor_rapido import desde_estado


def arreglos(blancas, negras, turno, puntos=(0, 0)):
    return desde_estado(posicion(blancas, negras, turno, puntos))


def decidida(v):
    """¿Es el valor de una carrera decidida (más allá del tope de la evaluación)?"""
    return abs(v) > 0.95 + 1e-9


# Pesos de F2 con una tabla de patrones al azar: las simetrías deben valer con cualquier tabla.
PESOS_CON_PATRONES = np.concatenate(
    [PESOS_F2, np.random.default_rng(3).normal(0, 0.05, N_PESOS - I_PATRONES)]
)


def lista_de_patrones(tab, color):
    """(índice, signo) de cada peón, ordenados: para comparar sin importar el orden."""
    indices = np.empty(MAX_PEONES, np.int64)
    signos = np.empty(MAX_PEONES, np.int64)
    n = patrones(tab, color, indices, signos)
    return sorted(zip(indices[:n].tolist(), signos[:n].tolist(), strict=True))


def espejo_de_color(tab, pts, color):
    """Misma posición vista desde el otro lado: filas invertidas y colores cambiados."""
    nuevo = np.zeros_like(tab)
    for c in range(81):
        origen = (8 - c // 9) * 9 + c % 9
        nuevo[c] = {0: 0, 1: 2, 2: 1}[int(tab[origen])]
    return nuevo, pts[::-1].copy(), 1 - color


def espejo_lateral(tab):
    nuevo = np.zeros_like(tab)
    for c in range(81):
        nuevo[c] = tab[(c // 9) * 9 + 8 - c % 9]
    return nuevo


class PruebasEvaluacion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.posiciones = [desde_estado(e) for e in muestrear(2000, 2)]

    def test_rango(self):
        for tab, pts, color in self.posiciones:
            v = abs(evaluar(tab, pts, color, PESOS_F2))
            # Una evaluación (hasta 0.95) o una carrera decidida (0.965 − 0.0001·n, n ≤ 12).
            self.assertTrue(v <= 0.95 or DECIDIDA - 0.0012 - 1e-12 <= v <= DECIDIDA, v)
        # Muchísima ventaja, sin corredores cerca de la meta: el valor se queda en el tope.
        tab, pts, color = arreglos(
            [f"{col}{fila}" for col in "abcdefghi" for fila in (3, 4)], ["a9"], "blancas", (2, 0)
        )
        self.assertEqual(evaluar(tab, pts, color, PESOS_F2), 0.95)
        self.assertEqual(evaluar(tab, pts, 1 - color, PESOS_F2), -0.95)

    def test_inicial(self):
        tab, pts, color = desde_estado(Estado.inicial())
        self.assertAlmostEqual(evaluar(tab, pts, color, PESOS_F2), math.tanh(0.05 / 2), delta=1e-12)

    def test_valor_a_mano(self):
        # Blancas d4 y e5 (e5 apoyado por d4), Negras e7; puntos 1–0.
        # x = turno 0.05 + material 0.30 + puntos 1.20 + apoyo 0.06 + avance (0.05+0.10−0.02)
        #     + carrera 0.15·12: Blancas necesitan 4+5 = 9 jugadas; Negras, con un solo peón,
        #       6+20+20 = 46; la diferencia (37) se topa en 12.
        tab, pts, color = arreglos(["d4", "e5"], ["e7"], "blancas", (1, 0))
        self.assertAlmostEqual(evaluar(tab, pts, 0, PESOS_F2), math.tanh(3.54 / 2), delta=1e-12)
        # Mismo tablero, mueven Negras: todo cambia de signo menos el turno.
        self.assertAlmostEqual(evaluar(tab, pts, 1, PESOS_F2), math.tanh(-3.44 / 2), delta=1e-12)

    def test_carrera(self):
        # Solo el peso de carrera: x = CARRERA · (m_rival − m_propio), donde m es la suma de
        # las distancias a la meta de los (3 − puntos) peones más adelantados.
        solo_carrera = np.zeros(N_PESOS)
        solo_carrera[I_CARRERA] = 0.1
        # Blancas (0 puntos): e8 a 1, a2 y b2 a 7 → m = 15. Negras (1 punto): e7 a 6 y
        # a8/h8 a 7 → los 2 mejores: m = 13.
        tab, pts, _ = arreglos(["e8", "a2", "b2"], ["e7", "a8", "h8"], "blancas", (0, 1))
        self.assertAlmostEqual(evaluar(tab, pts, 0, solo_carrera), math.tanh(-0.2 / 2), delta=1e-12)
        self.assertAlmostEqual(evaluar(tab, pts, 1, solo_carrera), math.tanh(0.2 / 2), delta=1e-12)
        # Sin peones suficientes para anotar lo que falta: la diferencia se topa en 12 jugadas.
        tab, pts, _ = arreglos(["e6"], ["a8", "b8", "c8"], "blancas")
        self.assertAlmostEqual(evaluar(tab, pts, 0, solo_carrera), math.tanh(-1.2 / 2), delta=1e-12)

    def test_sin_tabla_es_como_tabla_en_cero(self):
        # Pesos sin tabla (más rápidos) dan exactamente lo mismo que con la tabla toda en cero.
        en_cero = np.concatenate([PESOS_F2, np.zeros(N_PESOS - I_PATRONES)])
        for tab, pts, color in self.posiciones:
            self.assertEqual(evaluar(tab, pts, color, PESOS_F2), evaluar(tab, pts, color, en_cero))

    def test_rasgos_reproducen_a_evaluar(self):
        # Para entrenar, la evaluación debe ser exactamente tanh(x / 2), recortada, con
        # x = pesos_base · rasgos + Σ signo · tabla[patrón] sobre los peones.
        salida = np.empty(N_BASE)
        pesos = PESOS_CON_PATRONES
        for tab, pts, color in self.posiciones:
            if decidida(evaluar(tab, pts, color, pesos)):
                continue  # la carrera decidida no depende de los pesos
            rasgos(tab, pts, color, salida)
            x = float(pesos[:N_BASE] @ salida)
            for indice, signo in lista_de_patrones(tab, color):
                x += signo * pesos[I_PATRONES + indice]
            esperado = min(max(math.tanh(x / 2), -0.95), 0.95)
            self.assertAlmostEqual(evaluar(tab, pts, color, pesos), esperado, delta=1e-12)

    def test_patron_a_mano(self):
        # Peón blanco en c4 (fila de avance 3, columna c): vecinos desde Blancas, en orden
        # adelante-izq b5 (propio=1), adelante c5 (rival=2), adelante-der d5, izq b4, der d4
        # (vacías), atrás-izq b3 (propio=1), atrás c3 (vacía), atrás-der d3 (rival=2).
        # patrón = 1·1 + 2·3 + 1·243 + 2·2187 = 4624; región = 3·5 + 2 = 17.
        tab, _, _ = arreglos(["c4", "b5", "b3"], ["c5", "d3"], "blancas")
        self.assertIn((17 * 3**8 + 4624, 1), lista_de_patrones(tab, 0))
        # Desde Negras, el mismo peón cuenta en contra.
        self.assertIn((17 * 3**8 + 4624, -1), lista_de_patrones(tab, 1))

    def test_patron_con_tablas_igual_a_la_formula(self):
        # `_patron` usa tablas precalculadas: debe dar lo mismo que la fórmula directa
        # (código en base 3 de las 8 vecinas vistas desde el dueño, reflejo en la columna e).
        from evaluacion import (
            COLUMNAS_TABLA, ESPEJO, PATRONES_POR_REGION, POTENCIAS, VALOR_PIEZA, VECINAS,
        )

        for tab, _, color in self.posiciones[:1000]:
            esperados = []
            for c in range(81):
                if tab[c] == 0:
                    continue
                d = int(tab[c]) - 1
                codigo = 0
                for k in range(8):
                    codigo += int(VALOR_PIEZA[d, tab[VECINAS[d, c, k]]]) * int(POTENCIAS[d, c, k])
                fila, col = divmod(c, 9)
                if col == COLUMNAS_TABLA - 1:
                    codigo = min(codigo, int(ESPEJO[codigo]))
                avance = fila if d == 0 else 8 - fila
                region = avance * COLUMNAS_TABLA + min(col, 8 - col)
                esperados.append((region * PATRONES_POR_REGION + codigo, 1 if d == color else -1))
            self.assertEqual(lista_de_patrones(tab, color), sorted(esperados))

    def test_apoyo_con_tablas_igual_al_motor(self):
        # La evaluación cuenta el apoyo con casillas precalculadas: igual que el motor en todas
        # las casillas (con peón o sin él) y para los dos dueños.
        for tab, _, _ in self.posiciones[:1000]:
            for c in range(81):
                for duenio in (0, 1):
                    self.assertEqual(apoyo_eval(tab, c, duenio), apoyo_motor(tab, c, duenio))

    def test_patrones_simetricos(self):
        for tab, pts, color in self.posiciones[:500]:
            lista = lista_de_patrones(tab, color)
            tab2, _, color2 = espejo_de_color(tab, pts, color)
            self.assertEqual(lista_de_patrones(tab2, color2), lista)
            self.assertEqual(lista_de_patrones(espejo_lateral(tab), color), lista)

    def test_rasgos_a_mano(self):
        # Blancas d4, e5 (1 punto) contra Negras e7, desde Blancas.
        tab, pts, _ = arreglos(["d4", "e5"], ["e7"], "blancas", (1, 0))
        salida = np.empty(N_BASE)
        rasgos(tab, pts, 0, salida)
        esperado = np.zeros(N_BASE)
        esperado[[0, 1, 3, 4]] = [1, 1, 1, 1]  # material, punto 1, apoyo, turno
        esperado[5 + 3] += 1  # d4: avance 3
        esperado[5 + 4] += 1  # e5: avance 4
        esperado[5 + 2] -= 1  # e7 negro: avance 2
        esperado[I_CARRERA] = 12  # 46 − 9, topado en 12
        esperado[I_CORREDOR + CORREDOR_MAX + 3] = 1  # e5 a 4 de anotar, con e7 en su cono
        self.assertEqual(salida.tolist(), esperado.tolist())

    def test_simetria_de_color(self):
        for pesos in (PESOS_F2, PESOS_CON_PATRONES):
            for tab, pts, color in self.posiciones:
                v = evaluar(tab, pts, color, pesos)
                tab2, pts2, color2 = espejo_de_color(tab, pts, color)
                self.assertAlmostEqual(evaluar(tab2, pts2, color2, pesos), v, delta=1e-12)

    def test_simetria_lateral(self):
        for pesos in (PESOS_F2, PESOS_CON_PATRONES):
            for tab, pts, color in self.posiciones:
                v = evaluar(tab, pts, color, pesos)
                self.assertAlmostEqual(evaluar(espejo_lateral(tab), pts, color, pesos), v, delta=1e-12)

    def test_material(self):
        for tab, pts, color in self.posiciones:
            v = evaluar(tab, pts, color, PESOS_F2)
            if decidida(v):
                continue
            propios = np.flatnonzero(tab == color + 1)
            rivales = np.flatnonzero(tab == 2 - color)
            if len(rivales) > 0:
                sin_rival = tab.copy()
                sin_rival[rivales[0]] = 0
                v_rival = evaluar(sin_rival, pts, color, PESOS_F2)
                if decidida(v_rival):
                    v_rival = min(v_rival, 0.95)  # quitar un rival puede decidir la carrera
                self.assertGreaterEqual(v_rival, v - 1e-12)
                if -0.95 < v < 0.95:  # en los topes el recorte puede dejarlo igual
                    self.assertGreater(v_rival, v)
            if len(propios) > 1:  # con uno solo, quitarlo deja la posición terminal
                sin_propio = tab.copy()
                sin_propio[propios[0]] = 0
                v_propio = evaluar(sin_propio, pts, color, PESOS_F2)
                # Quitar un peón propio del cono de un corredor puede volverlo seguro.
                if not decidida(v_propio):
                    self.assertLessEqual(v_propio, v + 1e-12)

    def test_cono_a_mano(self):
        # Blanco en e5 (fila 4, a 4 de anotar): su cono llega a la fila 9.
        casos = [
            (["e7"], (1, True)),  # e7 está en el cono: un interceptor
            (["a9"], (0, False)),  # a 4 columnas: fuera hasta del cono ampliado
            (["b8"], (0, True)),  # a 3 columnas en la fila 8: solo en el cono ampliado
            (["d6", "f7"], (2, True)),  # dos interceptores
        ]
        for negras, esperado in casos:
            tab, _, _ = arreglos(["e5", "a2"], negras, "blancas")
            self.assertEqual(cono(tab, 4 * 9 + 4, 0, 4), esperado, negras)
        # Un peón propio en el cono no intercepta, pero el corredor ya no es seguro.
        tab, _, _ = arreglos(["e5", "f7"], ["a9"], "blancas")
        self.assertEqual(cono(tab, 4 * 9 + 4, 0, 4), (0, True))

    def test_ataques_a_mano(self):
        # e5 (apoyado por d4) puede atacar d6 y f6, sin apoyo: p = 0.65 cada uno (2p − 1 = 0.3).
        # d6 y f6 pueden atacar e5 con p = 0.35: para Negras no son ataques que convengan.
        tab, pts, _ = arreglos(["e5", "d4"], ["d6", "f6", "a9"], "blancas")
        salida = np.empty(N_BASE)
        rasgos(tab, pts, 0, salida)
        self.assertAlmostEqual(salida[I_MEJOR_ATAQUE], 0.3)
        self.assertAlmostEqual(salida[I_ATAQUES_RIVAL], 0.0)
        rasgos(tab, pts, 1, salida)
        self.assertAlmostEqual(salida[I_MEJOR_ATAQUE], 0.0)
        self.assertAlmostEqual(salida[I_ATAQUES_RIVAL], 0.6)

    def test_carrera_decidida_a_mano(self):
        # Blancas (2 puntos) con e7 libre a 2 de anotar; Negras solo con peones en su orilla.
        tab, pts, _ = arreglos(["e7"], ["a9", "b9"], "blancas", (2, 0))
        self.assertAlmostEqual(evaluar(tab, pts, 0, PESOS_F2), DECIDIDA - 0.0002, delta=1e-12)
        self.assertAlmostEqual(evaluar(tab, pts, 1, PESOS_F2), -(DECIDIDA - 0.0002), delta=1e-12)
        # Con un negro en el cono de e7 ya no está decidida.
        tab, pts, _ = arreglos(["e7"], ["a9", "d9"], "blancas", (2, 0))
        self.assertLessEqual(abs(evaluar(tab, pts, 0, PESOS_F2)), 0.95)
        # Las dos con 2 puntos: Blancas tardan 2, Negras anotan con a2 en 1 y mueven después;
        # el corredor negro llega primero: perdido para Blancas aunque muevan ellas.
        tab, pts, _ = arreglos(["e7"], ["a2", "i9"], "blancas", (2, 2))
        self.assertAlmostEqual(evaluar(tab, pts, 0, PESOS_F2), -(DECIDIDA - 0.0001), delta=1e-12)

    def test_carrera_decidida_es_segura(self):
        # En posiciones al azar donde la evaluación dice «carrera decidida», el ganador anunciado
        # gana moviendo solo sus corredores seguros, juegue lo que juegue el rival (al azar).
        rng = random.Random(7)
        decididas = 0
        intentos = 0
        while decididas < 40 and intentos < 20000:
            intentos += 1
            blancas = sorted({f"{rng.choice('abcdefghi')}{rng.randint(1, 8)}" for _ in range(rng.randint(1, 5))})
            negras = sorted({f"{rng.choice('abcdefghi')}{rng.randint(2, 9)}" for _ in range(rng.randint(1, 5))} - set(blancas))
            if not negras:
                continue
            turno = rng.choice(["blancas", "negras"])
            puntos = (rng.randint(0, 2), rng.randint(0, 2))
            try:
                estado = posicion(blancas, negras, turno, puntos)
            except ValueError:
                continue
            if estado.terminado():
                continue
            tab, pts, color = desde_estado(estado)
            v = evaluar(tab, pts, color, PESOS_F2)
            if not decidida(v):
                continue
            decididas += 1
            ganador = turno if v > 0 else ("negras" if turno == "blancas" else "blancas")
            for partida in range(5):
                e = estado
                while not e.terminado():
                    if e.turno == ganador:
                        jugada = self._jugada_de_corredor(e)
                    else:
                        jugada = rng.choice(e.jugadas())
                    gana = rng.random() < e.prob_ataque(jugada) if e.es_ataque(jugada) else None
                    e = e.aplicar(jugada, gana)
                self.assertEqual(e.ganador(), ganador, (blancas, negras, turno, puntos, partida))
        self.assertGreaterEqual(decididas, 40)

    def _jugada_de_corredor(self, estado):
        """Avanza recto el corredor seguro más cercano a anotar."""
        tab, _, color = desde_estado(estado)
        mejor = None
        for c in range(81):
            if tab[c] != color + 1:
                continue
            avance = c // 9 if color == 0 else 8 - c // 9
            if 8 - avance <= CORREDOR_MAX and not cono(tab, c, color, avance)[1]:
                if mejor is None or avance > mejor[0]:
                    mejor = (avance, c)
        self.assertIsNotNone(mejor, estado.a_dict())
        c = mejor[1]
        destino = c + (9 if color == 0 else -9)
        texto = f"{'abcdefghi'[c % 9]}{c // 9 + 1}-{'abcdefghi'[destino % 9]}{destino // 9 + 1}"
        legales = {str(j): j for j in estado.jugadas()}
        self.assertIn(texto, legales, estado.a_dict())
        return legales[texto]


if __name__ == "__main__":
    unittest.main()
