"""Funciones internas del bot, para llamarlas desde Python (pruebas y herramientas).

En la liga estas funciones numba solo se llaman desde otras funciones numba, así que se compilan
sin envoltura de Python (`njit_interna` en `motor_rapido.py`): `preparar` tarda menos. Aquí cada
una tiene una envoltura `@njit` trivial que solo se compila cuando una prueba o herramienta la usa.
"""

from numba import njit

from busqueda import _ultimo_punto
from busqueda import tt_buscar as _tt_buscar
from busqueda import tt_guardar as _tt_guardar
from evaluacion import _apoyo as _apoyo_eval
from evaluacion import _cono as _cono_interna
from evaluacion import evaluar as _evaluar
from motor_rapido import apoyo as _apoyo_motor
from motor_rapido import hacer as _hacer
from motor_rapido import prob_jugada as _prob_jugada
from motor_rapido import tiene_jugada as _tiene_jugada


@njit
def tiene_jugada(tab, color):
    return _tiene_jugada(tab, color)


@njit
def prob_jugada(tab, color, origen, destino):
    return _prob_jugada(tab, color, origen, destino)


@njit
def hacer(tab, pts, color, origen, destino, gana, h, tab_hijo, pts_hijo):
    return _hacer(tab, pts, color, origen, destino, gana, h, tab_hijo, pts_hijo)


@njit
def evaluar(tab, pts, color, pesos):
    return _evaluar(tab, pts, color, pesos)


@njit
def apoyo_motor(tab, c, duenio):
    return _apoyo_motor(tab, c, duenio)


@njit
def apoyo_eval(tab, c, duenio):
    return _apoyo_eval(tab, c, duenio)


@njit
def cono(tab, c, duenio, avance):
    return _cono_interna(tab, c, duenio, avance)


@njit
def ultimo_punto(tab, pts, color):
    return _ultimo_punto(tab, pts, color)


@njit
def tt_buscar(m, h):
    return _tt_buscar(m, h)


@njit
def tt_guardar(m, h, prof, valor, tipo, jugada):
    _tt_guardar(m, h, prof, valor, tipo, jugada)
