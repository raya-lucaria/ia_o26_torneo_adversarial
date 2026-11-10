"""Simulación al azar con jugadas decisivas y antidecisivas (opción Q).

Es una copia de `nucleo.simular` del motor (con otro nombre, como permite docs/avanzado.md):
cada bando juega al azar uniforme, eligiendo (peón, dirección) al azar y reintentando si no es
legal. La única diferencia es cuando algún bando ya tiene 2 puntos:

- decisiva: si quien mueve tiene 2 puntos y puede llegar a la última fila, lo hace (sin combate
  si puede, porque así gana seguro; si solo puede atacando, con el ataque de mayor p);
- antidecisiva: si el rival tiene 2 puntos y un peón que puede anotar en su siguiente turno,
  quien mueve ataca ese peón si puede (con el ataque de mayor p).

Así una simulación no «regala» el tercer punto que cualquier jugador tomaría o detendría
(Teytaud y Teytaud, 2010; Lanctot et al., 2014). Fuera de esos casos es exactamente el azar.
"""

import numpy as np
from numba import njit

import modelo_oponente as mo

LADO = mo.LADO
CASILLAS = mo.CASILLAS
PUNTOS_PARA_GANAR = mo.PUNTOS_PARA_GANAR
INTENTOS = 16
EXTRA_QUIETUD = 10  # jugadas extra, como máximo, para llegar a una posición quieta
SIN_CORTE = 10**6


@njit
def _puede_anotar(tab, c, t):
    """True si el peón de `c` en `t` tiene una jugada legal a la última fila."""
    avance = mo._paso(c) * LADO
    meta = LADO - 1 if c == 0 else 0
    if (t + avance) // LADO != meta:
        return False
    col = t % LADO
    for g in range(-1, 2):
        if col + g < 0 or col + g >= LADO:
            continue
        v = tab[t + avance + g]
        if v == c + 1 or (g == 0 and v != 0):
            continue
        return True
    return False


@njit
def jugada_tactica(tab, c, pts, lista, cuantos):
    """Jugada decisiva o antidecisiva de `c` como (origen, destino); (-1, -1) si no hay."""
    pieza = c + 1
    avance = mo._paso(c) * LADO
    rival = 1 - c
    if pts[c] == PUNTOS_PARA_GANAR - 1:
        meta = LADO - 1 if c == 0 else 0
        mejor_p = -1.0
        mejor_o = -1
        mejor_d = -1
        for k in range(cuantos[c]):
            o = lista[c, k]
            if (o + avance) // LADO != meta:
                continue
            col = o % LADO
            for g in range(-1, 2):
                if col + g < 0 or col + g >= LADO:
                    continue
                d = o + avance + g
                v = tab[d]
                if v == pieza or (g == 0 and v != 0):
                    continue
                if v == 0:
                    return o, d  # anota sin combate: gana seguro
                p = mo._prob_combate(mo._apoyo(tab, o, c, -1), mo._apoyo(tab, d, rival, -1))
                if p > mejor_p:
                    mejor_p = p
                    mejor_o = o
                    mejor_d = d
        if mejor_o >= 0:
            return mejor_o, mejor_d
    if pts[rival] == PUNTOS_PARA_GANAR - 1:
        mejor_p = -1.0
        mejor_o = -1
        mejor_d = -1
        for k in range(cuantos[rival]):
            t = lista[rival, k]
            if not _puede_anotar(tab, rival, t):
                continue
            # Nuestros peones que atacan a t: desde s, s + avance + g == t.
            col_t = t % LADO
            for g in (-1, 1):
                if col_t - g < 0 or col_t - g >= LADO:
                    continue
                s = t - avance - g
                if s < 0 or s >= CASILLAS or tab[s] != pieza:
                    continue
                p = mo._prob_combate(mo._apoyo(tab, s, c, -1), mo._apoyo(tab, t, rival, -1))
                if p > mejor_p:
                    mejor_p = p
                    mejor_o = s
                    mejor_d = t
        if mejor_o >= 0:
            return mejor_o, mejor_d
    return -1, -1



@njit
def hay_captura(tab, c, lista, cuantos):
    """True si `c` (quien mueve) tiene alguna captura: la posición no es quieta (curso, 7.3.2)."""
    avance = mo._paso(c) * LADO
    for k in range(cuantos[c]):
        o = lista[c, k]
        col = o % LADO
        frente = o + avance
        if col > 0 and tab[frente - 1] == 2 - c:
            return True
        if col < LADO - 1 and tab[frente + 1] == 2 - c:
            return True
    return False


@njit
def simular_tactico(tablero, puntos, color, n, semilla, para, w, corte):
    """Valor medio para `para` (0 Blancas, 1 Negras) de `n` partidas desde (tablero, puntos).

    Cada partida vale 1 si gana `para` y 0 si pierde. Si `corte` es menor que la longitud de la
    partida, a partir de la jugada `corte` se detiene en la primera posición quieta (quien mueve
    no tiene capturas; a lo más EXTRA_QUIETUD jugadas después) y vale lo que dice la evaluación
    `evaluar` con pesos `w`: es la búsqueda de quietud del curso aplicada a una simulación.
    """
    np.random.seed(semilla)
    f = np.empty(D_EVAL)
    tab = np.empty(CASILLAS, np.int8)
    pts = np.empty(2, np.int64)
    origenes = np.empty(mo.MAX_JUGADAS, np.int64)
    destinos = np.empty(mo.MAX_JUGADAS, np.int64)
    lista = np.empty((2, CASILLAS), np.int64)  # casillas de los peones de cada color
    cuantos = np.empty(2, np.int64)
    indice = np.empty(CASILLAS, np.int64)  # posición de cada peón dentro de su lista
    victorias = 0.0
    for _ in range(n):
        tab[:] = tablero
        pts[:] = puntos
        cuantos[:] = 0
        for s in range(CASILLAS):
            if tab[s] != 0:
                k = tab[s] - 1
                lista[k, cuantos[k]] = s
                indice[s] = cuantos[k]
                cuantos[k] += 1
        c = color
        jugadas = 0
        ganador = -1
        while True:
            if jugadas >= corte and (jugadas >= corte + EXTRA_QUIETUD
                                     or not hay_captura(tab, c, lista, cuantos)):
                p_blancas = evaluar(tab, pts, c, w, f)
                victorias += p_blancas if para == 0 else 1.0 - p_blancas
                break
            jugadas += 1
            pieza = c + 1
            avance = mo._paso(c) * LADO
            o = -1
            d = -1
            if pts[0] == PUNTOS_PARA_GANAR - 1 or pts[1] == PUNTOS_PARA_GANAR - 1:
                o, d = jugada_tactica(tab, c, pts, lista, cuantos)
            if o < 0:
                for _intento in range(INTENTOS):
                    if cuantos[c] == 0:
                        break
                    prueba = lista[c, np.random.randint(0, cuantos[c])]
                    giro = np.random.randint(-1, 2)
                    col = prueba % LADO + giro
                    if col < 0 or col >= LADO:
                        continue
                    destino = prueba + avance + giro
                    v = tab[destino]
                    if v == pieza or (giro == 0 and v != 0):
                        continue
                    o = prueba
                    d = destino
                    break
            if o < 0:
                # Muestreo por rechazo fallido: todas las jugadas, una al azar (sigue uniforme).
                m = 0
                for k in range(cuantos[c]):
                    q = lista[c, k]
                    cq = q % LADO
                    for g in range(-1, 2):
                        if cq + g < 0 or cq + g >= LADO:
                            continue
                        dd = q + avance + g
                        v = tab[dd]
                        if v == pieza or (g == 0 and v != 0):
                            continue
                        origenes[m] = q
                        destinos[m] = dd
                        m += 1
                if m == 0:
                    ganador = 1 - c  # sin jugadas: gana el otro
                    break
                k = np.random.randint(0, m)
                o = origenes[k]
                d = destinos[k]
            gana = True
            if tab[d] != 0:
                p = mo._prob_combate(mo._apoyo(tab, o, c, -1), mo._apoyo(tab, d, 1 - c, -1))
                gana = np.random.random() < p
                if gana:
                    mo._quitar(lista, cuantos, indice, 1 - c, d)
                else:
                    mo._quitar(lista, cuantos, indice, c, o)
            if gana:
                meta = LADO - 1 if c == 0 else 0
                if d // LADO == meta:
                    mo._quitar(lista, cuantos, indice, c, o)
                else:
                    i = indice[o]
                    lista[c, i] = d
                    indice[d] = i
            mo._aplicar(tab, pts, c, o, d, gana)
            if pts[c] >= PUNTOS_PARA_GANAR:
                ganador = c
                break
            c = 1 - c
        if ganador == para:
            victorias += 1.0
    return victorias / n


def hijo_tablero(tablero, puntos, turno, jugada, gana):
    """Tablero y puntos tras `jugada` de `turno` (0/1); gana = resultado del ataque."""
    tab = tablero.copy()
    pts = puntos.copy()
    mo._aplicar(tab, pts, turno, jugada.origen, jugada.destino, gana)
    return tab, pts


# --- Evaluación de una posición (regresión logística) -------------------------------------------
# Rasgos desde Blancas (positivo = bueno para Blancas). Los pesos se ajustan offline con
# herramientas/ajustar_eval.py y se guardan en data/eval_posicion.npz.
NOMBRES_EVAL = ("constante", "puntos", "peones", "avance", "punta", "cerca", "contactos_turno", "dos_puntos")
D_EVAL = len(NOMBRES_EVAL)


@njit
def rasgos_posicion(tab, pts, turno, f):
    """Llena f con los rasgos de la posición (tablero, puntos, quién mueve: 0 Blancas, 1 Negras)."""
    n0 = n1 = 0.0
    suma0 = suma1 = 0.0
    max0 = max1 = 0.0
    cerca0 = cerca1 = 0.0
    contactos = 0.0
    for c in range(CASILLAS):
        v = tab[c]
        if v == 0:
            continue
        fila = c // LADO
        if v == 1:
            prog = fila
            n0 += 1
            suma0 += prog
            max0 = max(max0, prog)
            if prog >= LADO - 3:
                cerca0 += 1
            # Contacto: un peón negro en una diagonal de enfrente (se pueden atacar mutuamente).
            frente = c + LADO
            col = c % LADO
            if col > 0 and tab[frente - 1] == 2:
                contactos += 1
            if col < LADO - 1 and tab[frente + 1] == 2:
                contactos += 1
        else:
            prog = LADO - 1 - fila
            n1 += 1
            suma1 += prog
            max1 = max(max1, prog)
            if prog >= LADO - 3:
                cerca1 += 1
    f[0] = 1.0
    f[1] = pts[0] - pts[1]
    f[2] = n0 - n1
    f[3] = (suma0 - suma1) / (LADO - 1)
    f[4] = (max0 - max1) / (LADO - 1)
    f[5] = cerca0 - cerca1
    # Quien mueve ataca primero en un contacto.
    f[6] = contactos if turno == 0 else -contactos
    f[7] = (1.0 if pts[0] == PUNTOS_PARA_GANAR - 1 else 0.0) - (1.0 if pts[1] == PUNTOS_PARA_GANAR - 1 else 0.0)


@njit
def evaluar(tab, pts, turno, w, f):
    """Probabilidad estimada de que ganen Blancas: logística de los rasgos con pesos w."""
    rasgos_posicion(tab, pts, turno, f)
    z = 0.0
    for i in range(f.shape[0]):
        z += w[i] * f[i]
    return 1.0 / (1.0 + np.exp(-z))


# --- Táctica dentro del árbol: riesgo de que el rival anote su tercer punto ---------------------
@njit
def _mejor_anotacion(tab, c):
    """Probabilidad de que `c` anote en su turno con su mejor jugada (1 si puede sin combate)."""
    avance = mo._paso(c) * LADO
    meta = LADO - 1 if c == 0 else 0
    mejor = 0.0
    for o in range(CASILLAS):
        if tab[o] != c + 1 or (o + avance) // LADO != meta:
            continue
        col = o % LADO
        for g in range(-1, 2):
            if col + g < 0 or col + g >= LADO:
                continue
            d = o + avance + g
            v = tab[d]
            if v == c + 1 or (g == 0 and v != 0):
                continue
            if v == 0:
                return 1.0
            p = mo._prob_combate(mo._apoyo(tab, o, c, -1), mo._apoyo(tab, d, 1 - c, -1))
            mejor = max(mejor, p)
    return mejor


@njit
def riesgo_jugada(tab, pts, c, o, d):
    """Probabilidad de que el rival de `c` gane en su siguiente turno si `c` juega o -> d.

    Supone que el rival tiene 2 puntos. Combina los dos resultados del dado si la jugada ataca.
    """
    rival = 1 - c
    t = tab.copy()
    q = pts.copy()
    if tab[d] == 0:
        mo._aplicar(t, q, c, o, d, True)
        return 0.0 if q[c] >= PUNTOS_PARA_GANAR else _mejor_anotacion(t, rival)
    p = mo._prob_combate(mo._apoyo(tab, o, c, -1), mo._apoyo(tab, d, rival, -1))
    mo._aplicar(t, q, c, o, d, True)
    riesgo_gana = 0.0 if q[c] >= PUNTOS_PARA_GANAR else _mejor_anotacion(t, rival)
    t[:] = tab
    q[:] = pts
    mo._aplicar(t, q, c, o, d, False)
    return p * riesgo_gana + (1.0 - p) * _mejor_anotacion(t, rival)
