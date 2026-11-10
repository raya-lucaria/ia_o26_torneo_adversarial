"""Bot «torneo_breakthrough»: expectimax alfa-beta con nodos de azar sobre bitboards.

- busqueda.py: *-minimax con poda Star1, PVS, quiescencia, tabla de transposición, killers,
  history y caché de evaluación.
- rasgos.py: evaluación lineal aprendida (marcador ordenado + tabla de patrones 3×3), con los pesos
  de data/pesos.npy que genera herramientas/ajustar_marcador.py.
- tablero_bits.py: reglas sobre bitboards por fila, verificadas contra el motor oficial.
Aquí solo viven el reloj y la profundización iterativa. Todo el estado se crea en `preparar`.
Diseño y mediciones: DISENO.md (raíz del repo privado).
"""

import sys
import time

import numpy as np

import busqueda as bq
import rasgos as rg
import tablero_bits as tb
from motor import Estado, reportar

RESERVA_S = 4.0  # nunca se gasta: comunicación con el árbitro y una máquina más lenta
JUGADAS_ESPERADAS = 42  # jugadas propias de una partida típica (medimos 37–40 de media)
MIN_RESTANTES = 12  # aunque la partida se alargue, cada jugada usa a lo más 1/12 de lo disponible
FRACCION_MAX = 0.15  # tope por jugada: 15 % de lo disponible
NPS_INICIAL = 2.0e6  # nodos/s supuestos al empezar; luego se mide en la partida
BITS_TT = 22  # 4 M entradas ≈ 80 MB
BITS_EV = 20  # caché de evaluación: 1 M entradas ≈ 16 MB
PROF_MAX = 60
# Parámetros de búsqueda (el laboratorio los cambia por instancia para comparar variantes).
PARAMETROS = {
    "qprof": 4,  # profundidad máxima de la quiescencia
    "qpmin": 50,  # p mínima (en %) de los ataques que entran a la quiescencia
    "pvs": 1,  # 1 = ventana nula después de la primera jugada (exacto; ahorra nodos)
}


class Motor:
    """Arreglos de la búsqueda y la profundización iterativa con control de tiempo."""

    def __init__(self, pesos, **parametros):
        n = 1 << BITS_TT
        self.tt_k = np.zeros(n, np.uint64)
        self.tt_v = np.zeros(n, np.float64)
        self.tt_p = np.zeros(n, np.int8)
        self.tt_f = np.zeros(n, np.int8)
        self.tt_m = np.zeros(n, np.int16)
        self.ev_k = np.zeros(1 << BITS_EV, np.uint64)
        self.ev_v = np.zeros(1 << BITS_EV, np.float64)
        self.movs = np.zeros((bq.MAX_PLY, bq.MAX_JUG), np.int64)
        self.punt = np.zeros((bq.MAX_PLY, bq.MAX_JUG), np.int64)
        self.killers = np.full((bq.MAX_PLY, 2), -1, np.int64)
        self.historia = np.zeros((2, tb.CASILLAS * tb.CASILLAS), np.int64)
        self.ctx = np.zeros(bq.N_CTX, np.int64)
        self.nps = NPS_INICIAL
        self.pesos = rg.cuantizar(pesos)  # int16: la evaluación suma enteros
        self.par = {**PARAMETROS, **parametros}

    def preparar_ctx(self, max_nodos):
        """Contadores en 0, tope de nodos y parámetros de búsqueda en ctx."""
        ctx = self.ctx
        ctx[:] = 0
        ctx[bq.C_MAX] = max_nodos
        ctx[bq.C_QPMIN] = self.par["qpmin"]
        ctx[bq.C_PVS] = self.par["pvs"]

    def iteracion(self, bb, pts, c, prof, h):
        """Búsqueda a profundidad `prof` desde la raíz, con la ventana completa."""
        return bq.alfabeta(bb, pts, c, prof, -2.0, 2.0, 0, h, self.movs, self.punt, self.killers,
                           self.historia, self.tt_k, self.tt_v, self.tt_p, self.tt_f, self.tt_m,
                           self.ctx, self.par["qprof"], self.pesos, self.ev_k, self.ev_v)

    def pensar(self, bb, pts, c, presupuesto, max_nodos=0):
        """Regresa (jugada codificada, valor, profundidad completa, nodos).

        Cada iteración tiene un tope duro de nodos (2× el presupuesto según los nodos/s medidos):
        numba aborta al llegar y se usa la última iteración completa. No se empieza otra
        iteración si ya se gastó la mitad del presupuesto. Con `max_nodos` > 0 se ignora el reloj
        y el tope son esos nodos (para pruebas y para generar datos).
        """
        inicio = time.perf_counter()
        h = np.uint64(tb.clave(bb, c, pts))
        ctx = self.ctx
        self.preparar_ctx(max_nodos)
        self.killers[:] = -1
        self.historia //= 8
        mejor, valor, hecha = -1, 0.0, 0
        for prof in range(1, PROF_MAX + 1):
            if not max_nodos:
                restante = 2.0 * presupuesto - (time.perf_counter() - inicio)
                usados = ctx[bq.C_NODOS] + ctx[bq.C_QNODOS]
                ctx[bq.C_MAX] = usados + max(2_000, int(restante * self.nps))
            v = self.iteracion(bb, pts, c, prof, h)
            if ctx[bq.C_ABORTA]:
                break
            mejor, valor, hecha = int(ctx[bq.C_MEJOR]), float(v), prof
            if not max_nodos and time.perf_counter() - inicio > 0.5 * presupuesto:
                break
        dt = time.perf_counter() - inicio
        nodos = int(ctx[bq.C_NODOS] + ctx[bq.C_QNODOS])
        if dt > 0.02 and not max_nodos:
            self.nps = 0.7 * self.nps + 0.3 * nodos / dt
        return mejor, valor, hecha, nodos


def asignar(tiempo_restante, jugadas_hechas):
    """Segundos para esta jugada: lo disponible entre las jugadas que esperamos que falten."""
    disponible = tiempo_restante - RESERVA_S
    if disponible <= 0.05:
        return 0.005  # solo la reserva: profundidad mínima
    restantes = max(MIN_RESTANTES, JUGADAS_ESPERADAS - jugadas_hechas)
    return min(disponible / restantes, FRACCION_MAX * disponible)


def preparar(config):
    global motor, jugadas_hechas
    inicio = time.perf_counter()
    motor = Motor(np.load(config.carpeta_datos / "pesos.npy"))
    jugadas_hechas = 0
    bb, pts, c = tb.desde_estado(Estado.inicial())
    motor.pensar(bb, pts, c, 0.05)  # compila los kernels de numba aquí, fuera del reloj
    motor.nps = NPS_INICIAL
    print(f"[preparar] listo en {time.perf_counter() - inicio:.2f} s", file=sys.stderr)


def jugar(estado, tiempo_restante):
    global jugadas_hechas
    jugadas_hechas += 1
    legales = estado.jugadas()
    if len(legales) == 1:
        return legales[0]
    bb, pts, c = tb.desde_estado(estado)
    jugada, valor, prof, nodos = motor.pensar(bb, pts, c, asignar(tiempo_restante, jugadas_hechas))
    reportar(prof=prof, nodos=nodos, eval=round(valor, 3))
    return tb.texto(jugada) if jugada >= 0 else legales[0]
