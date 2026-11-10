"""Bot «maquinas_del_trabajo»: expectiminimax con nuestro propio motor en numba.

- `motor_rapido.py`: las reglas en numba, copiadas del motor oficial y validadas contra él.
- `busqueda.py`: negamax con alfa-beta, Star1 para los combates, tabla de transposición y
  profundización iterativa con corte por tiempo.
- `evaluacion.py`: probabilidad de ganar a partir de rasgos de la posición y una tabla de
  patrones; sus pesos entrenados se leen de `data/pesos.npz` (los genera `herramientas/`).
- `reloj.py`: cuánto pensar en cada jugada sin tocar nunca la reserva.

Robustez: la jugada se verifica contra `estado.jugadas()` antes de regresarla y, si algo
falla, se usa una jugada de respaldo. Nunca se lanza una excepción hacia el árbitro.
"""

import time

# El árbitro cuenta en los 20 s de `preparar` también la importación de este archivo: medimos
# desde aquí para reportar el tiempo completo (`prep_total`) en la primera jugada.
INICIO_IMPORTACION = time.perf_counter()

import os
import traceback
from pathlib import Path

import numba.core.config
import numpy as np
from motor import Estado, reportar

# Menos optimización de LLVM al compilar nuestras funciones numba (antes de importarlas): baja el
# tiempo de `preparar` ~5–10 % sin bajar los nodos/s (medido con herramientas/medir.py). Solo
# cambia la configuración de numba de este proceso; el motor oficial ya está compilado.
os.environ["NUMBA_OPT"] = "1"
os.environ["NUMBA_LOOP_VECTORIZE"] = "0"
os.environ["NUMBA_SLP_VECTORIZE"] = "0"
numba.core.config.reload_config()

from busqueda import crear_memoria, profundizar  # noqa: E402 (después de configurar numba)
from evaluacion import N_BASE, N_PESOS  # noqa: E402
from motor_rapido import desde_estado, texto_jugada  # noqa: E402
from reloj import ESCALA_TIEMPO, presupuesto  # noqa: E402


class Bot:
    """Un jugador. Su memoria (tabla de transposición y demás) dura toda la partida."""

    def __init__(self, pesos=None):
        inicio = time.perf_counter()
        self.memoria = crear_memoria(pesos=pesos)
        # Una búsqueda corta compila toda la búsqueda ahora (fuera del reloj de la partida).
        tab, pts, color = desde_estado(Estado.inicial())
        profundizar(self.memoria, tab, pts, color, blando=10.0, duro=10.0, prof_max=3)
        self.segundos_preparar = time.perf_counter() - inicio
        self.segundos_preparar_total = None  # con la importación; lo pone `preparar`
        self.escala_tiempo = ESCALA_TIEMPO  # la arena la cambia para comparar repartos
        self.primera_jugada = True
        self.ya_avisamos = False

    def jugar(self, estado, tiempo_restante):
        legales = [str(j) for j in estado.jugadas()]
        if len(legales) == 1:
            return legales[0]
        try:
            tab, pts, color = desde_estado(estado)
            peones_propios = int(np.count_nonzero(tab == color + 1))
            blando, duro = presupuesto(
                tiempo_restante, peones_propios, int(pts.sum()), self.escala_tiempo
            )
            r = profundizar(self.memoria, tab, pts, color, blando, duro)
            jugada = texto_jugada(r.jugada)
            if jugada not in legales:
                raise RuntimeError(f"la búsqueda propuso {jugada}, que no es legal")
            # Lo que reportamos se ve con -v y en el visor (eval = probabilidad de ganar).
            datos = {"prof": r.prof, "nodos": r.nodos, "eval": round((r.valor + 1) / 2, 3)}
            if self.primera_jugada:
                datos["prep"] = round(self.segundos_preparar, 2)
                if self.segundos_preparar_total is not None:
                    datos["prep_total"] = round(self.segundos_preparar_total, 2)
                self.primera_jugada = False
            reportar(**datos)
            return jugada
        except Exception:
            if not self.ya_avisamos:  # una vez basta: la salida de cada partida tiene límite
                traceback.print_exc()
                self.ya_avisamos = True
        return respaldo(estado, legales)


bot = None  # el jugador de esta partida (cada partida es un proceso nuevo)
ARCHIVO_PESOS = "pesos.npz"  # en data/: lo escribe herramientas/entrenar.py


def cargar_pesos(carpeta_datos):
    """Los pesos entrenados de `data/`, o None si no hay archivo (se usan los pesos a mano)."""
    ruta = Path(carpeta_datos) / ARCHIVO_PESOS
    if not ruta.exists():
        return None
    with np.load(ruta) as archivo:
        pesos = np.asarray(archivo["pesos"], dtype=np.float64)
    if pesos.shape not in ((N_BASE,), (N_PESOS,)) or not np.all(np.isfinite(pesos)):
        raise ValueError(f"{ruta}: se esperaban {N_PESOS} pesos finitos y hay {pesos.shape}")
    return pesos


def preparar(config):
    global bot
    try:
        pesos = cargar_pesos(config.carpeta_datos)
    except Exception:  # con los pesos a mano se juega peor, pero se juega
        traceback.print_exc()
        pesos = None
    bot = Bot(pesos)
    bot.segundos_preparar_total = time.perf_counter() - INICIO_IMPORTACION


def jugar(estado, tiempo_restante):
    if bot is None:  # no debería pasar: el árbitro siempre llama antes a preparar
        return respaldo(estado, [str(j) for j in estado.jugadas()])
    return bot.jugar(estado, tiempo_restante)


def respaldo(estado, legales):
    """Jugada sin pensar: una que anota sin combate si la hay; si no, la primera legal."""
    meta = "9" if estado.turno == "blancas" else "1"
    for j in estado.jugadas():
        if str(j).endswith(meta) and not estado.es_ataque(j):
            return str(j)
    return legales[0]
