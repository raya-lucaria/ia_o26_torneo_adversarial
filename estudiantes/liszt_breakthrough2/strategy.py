"""Bot «liszt_breakthrough»: expectiminimax con poda alfa-beta (Star1), en numba.

La búsqueda vive en alfabeta.py. Aquí se reparte el reloj y se cuida que nunca regresemos una
jugada ilegal ni lancemos una excepción.
"""

import json
import os
import random
import sys
import time

from motor import reportar

from alfabeta import Buscador, calentar

# --- reloj ---
RESERVA_S = 2.5          # segundos que nunca gastamos pensando
JUGADAS_ESPERADAS = 40   # jugadas propias que damos por posibles (partidas medidas: 35 a 63)
MIN_RESTANTES = 8        # nunca suponemos que quedan menos de estas jugadas
FACTOR_CONTACTO = 1.4    # hay ataques posibles: pensar más
FACTOR_QUIETO = 0.7      # nadie puede atacar: pensar menos
TOPE_FRACCION = 0.25     # nunca más de esta fracción de lo disponible en una sola jugada
TOPE_SEG = 2.5           # ni más de estos segundos en una sola jugada
SOBRECOSTO_INICIAL = 0.05  # s por jugada de comunicación con el árbitro (se mide en juego)

# Compilar numba al importar: cuenta dentro de `preparar`, no en el reloj.
calentar()

color = "blancas"
rng = random.Random(0)
buscador = None
origen_pesos = "mano"
n_jug = 0
sobrecosto = SOBRECOSTO_INICIAL
_ult_restante = None
_ult_dt = 0.0


def _cargar_pesos(carpeta):
    """Pesos ajustados offline (data/pesos.json); si no hay, los de alfabeta.PESOS."""
    global origen_pesos
    try:
        ruta = os.path.join(str(carpeta), "pesos.json")
        with open(ruta, encoding="utf-8") as f:
            pesos = json.load(f)["pesos"]
        origen_pesos = "data"
        return pesos
    except Exception as e:
        origen_pesos = "mano"
        print(f"[pesos] no se pudo leer data/pesos.json ({e!r}); uso los de alfabeta.PESOS",
              file=sys.stderr, flush=True)
        return None


def preparar(config):
    global color, rng, buscador, n_jug, sobrecosto, _ult_restante, _ult_dt
    color = config.color
    rng = random.Random(config.semilla)
    buscador = Buscador(pesos=_cargar_pesos(getattr(config, "carpeta_datos", "")))
    n_jug = 0
    sobrecosto = SOBRECOSTO_INICIAL
    _ult_restante = None
    _ult_dt = 0.0


def _hay_contacto(estado):
    """True si alguno de los dos bandos puede atacar en este momento."""
    if any(estado.es_ataque(j) for j in estado.jugadas()):
        return True
    tab = estado._tablero
    rival = 2 if estado._color == 0 else 1
    paso = -9 if rival == 2 else 9  # «adelante» del rival
    for s in range(81):
        if tab[s] != rival:
            continue
        f = s + paso
        if not 0 <= f < 81:
            continue
        col = s % 9
        if col > 0 and tab[f - 1] not in (0, rival):
            return True
        if col < 8 and tab[f + 1] not in (0, rival):
            return True
    return False


def _medir_sobrecosto(tiempo_restante):
    """Lo que el reloj cobró de más entre dos jugadas respecto a lo que medimos pensando."""
    global sobrecosto
    if _ult_restante is None:
        return
    extra = (_ult_restante - tiempo_restante) - _ult_dt
    extra = min(0.2, max(0.0, extra))
    sobrecosto = 0.7 * sobrecosto + 0.3 * extra


def _presupuesto(estado, tiempo_restante):
    restantes = max(MIN_RESTANTES, JUGADAS_ESPERADAS - n_jug)
    disponible = tiempo_restante - RESERVA_S - sobrecosto * restantes
    if disponible <= 0.05:
        return 0.0
    factor = FACTOR_CONTACTO if _hay_contacto(estado) else FACTOR_QUIETO
    return min(TOPE_SEG, TOPE_FRACCION * disponible, factor * disponible / restantes)


def jugar(estado, tiempo_restante):
    global n_jug, _ult_restante, _ult_dt, buscador
    t0 = time.perf_counter()
    _medir_sobrecosto(tiempo_restante)
    n_jug += 1
    jugadas = estado.jugadas()
    elegida = None
    if len(jugadas) == 1:
        elegida = jugadas[0]
    else:
        try:
            if buscador is None:
                buscador = Buscador()
            seg = _presupuesto(estado, tiempo_restante)
            if seg <= 0.0:
                seg = 0.01  # sin tiempo: al menos profundidad 1, que es instantánea
            mov, info = buscador.buscar(estado._tablero, estado._color, estado._puntos, seg)
            if mov is not None:
                o, d = mov
                for j in jugadas:
                    if j.origen == o and j.destino == d:
                        elegida = j
                        break
            reportar(prof=info["prof"], nodos=info["nodos"], eval=info["eval"],
                     seg=round(seg, 2), sobrecosto=round(sobrecosto, 3), pesos=origen_pesos)
        except Exception as e:  # nunca perder por excepción
            print(f"[fallo búsqueda] {e!r}", file=sys.stderr, flush=True)
    if elegida is None:
        elegida = rng.choice(jugadas)
    _ult_restante, _ult_dt = tiempo_restante, time.perf_counter() - t0
    return elegida
