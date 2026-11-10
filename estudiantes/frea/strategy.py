"""Bot de expectiminimax con profundización iterativa. La búsqueda vive en mi_motor.py."""

import time

import numpy as np

import mi_motor
from motor import reportar

NODOS_CALENTAR = 300
PROF_MAX = 40
JUGADAS_ESPERADAS = 42  # jugadas propias en una partida típica
JUGADAS_MINIMAS = 10  # nunca planear con menos jugadas futuras que estas
FRACCION_RESERVA = 0.12  # del reloj inicial, jamás se gasta pensando
TOPE_SOBRE_PRESUPUESTO = 1.6  # una iteración puede pasarse del presupuesto hasta este factor
EMPEZAR_SI_QUEDA = 0.5  # no empezar otra iteración si ya se gastó esta fracción


def _cargar_tablero(estado, buf):
    tab, pts = buf[0], buf[1]
    tab[:] = 0
    for valor, nombre in ((1, "blancas"), (2, "negras")):
        for casilla in estado.piezas(nombre):
            tab[(int(casilla[1]) - 1) * 9 + ord(casilla[0]) - 97] = valor
    pts[0] = estado.puntos["blancas"]
    pts[1] = estado.puntos["negras"]


def _iteracion(turno, h, prof, limite):
    ctx = buf[9]
    ctx[:] = 0
    ctx[1] = limite
    valor = mi_motor.buscar_raiz(buf, turno, h, prof)
    return valor, int(ctx[0]), bool(ctx[2]), int(ctx[3]), int(ctx[4])


def preparar(config):
    global buf, reserva, nps, jugadas_hechas, turno_propio
    archivo = config.carpeta_datos / "pesos.npy"
    pesos = np.load(archivo) if archivo.exists() else mi_motor.PESOS_INICIALES
    buf = mi_motor.crear_buffers(pesos)
    reserva = FRACCION_RESERVA * config.reloj_s
    jugadas_hechas = 0
    turno_propio = 0 if config.color == "blancas" else 1
    # Compila todo aquí (cuenta en preparar, no en el reloj) y mide nodos por segundo.
    from motor import Estado

    _cargar_tablero(Estado.inicial(), buf)
    h = int(mi_motor.clave_de(buf[0], buf[1], 0, buf[13]))
    _iteracion(0, h, 2, NODOS_CALENTAR)
    inicio = time.perf_counter()
    _, nodos, _, _, _ = _iteracion(0, h, 3, 20000)
    nps = max(nodos / max(time.perf_counter() - inicio, 1e-4), 20000.0)


def _pensar(estado, tiempo_restante, legales):
    global nps, jugadas_hechas
    inicio = time.perf_counter()
    jugadas_hechas += 1
    disponible = tiempo_restante - reserva
    faltan = max(JUGADAS_MINIMAS, JUGADAS_ESPERADAS - jugadas_hechas)
    presupuesto = max(disponible / faltan, 0.0)
    if presupuesto <= 0.0:  # solo queda la reserva: búsqueda mínima
        presupuesto = 0.002
    tope = min(presupuesto * TOPE_SOBRE_PRESUPUESTO, max(disponible * 0.4, 0.002))

    _cargar_tablero(estado, buf)
    turno = 0 if estado.turno == "blancas" else 1
    h = int(mi_motor.clave_de(buf[0], buf[1], turno, buf[13]))
    mejor, valor, prof_hecha, total = None, 0.0, 0, 0
    for prof in range(1, PROF_MAX):
        gastado = time.perf_counter() - inicio
        if prof > 1 and gastado > presupuesto * EMPEZAR_SI_QUEDA:
            break
        limite = max(int((tope - gastado) * nps), 200)
        v, nodos, aborto, jugada, terminadas = _iteracion(turno, h, prof, limite)
        total += nodos
        if jugada >= 0 and (not aborto or terminadas > 0):
            candidata = f"{_nombre(jugada // 81)}-{_nombre(jugada % 81)}"
            if candidata in legales and terminadas > 0:
                mejor = candidata
                if not aborto:
                    valor, prof_hecha = v, prof
        if aborto or abs(v) >= 0.999:
            break
    gastado = time.perf_counter() - inicio
    if gastado > 0.01:
        nps = 0.7 * nps + 0.3 * max(total / gastado, 20000.0)
    reportar(prof=prof_hecha, nodos=total, eval=round(float(valor), 3))
    return mejor


def _nombre(c):
    return f"{'abcdefghi'[c % 9]}{c // 9 + 1}"


def jugar(estado, tiempo_restante):
    legales = [str(j) for j in estado.jugadas()]
    if len(legales) == 1:
        return legales[0]
    try:
        mejor = _pensar(estado, tiempo_restante, legales)
    except Exception as error:  # una falla nuestra no debe costar la partida
        print(f"[respaldo] {type(error).__name__}: {error}")
        mejor = None
    return mejor if mejor is not None else legales[0]
