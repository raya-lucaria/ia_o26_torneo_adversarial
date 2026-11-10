"""Evaluación de una posición: rasgos con pesos y una tabla de patrones, entrenados.

Regresa `v = 2·P(ganar) − 1` desde el bando que mueve, con `P = σ(x)` y `x` una suma lineal
de rasgos (modelo logístico: nada de redes neuronales). Así la búsqueda bajo azar maximiza la
probabilidad de ganar, que es lo que cuenta en la liga.

Rasgos de F2 (s = bando que mueve, o = rival):
- material: peones de s menos peones de o;
- puntos: P(0) = 0, P(1) = PUNTO_1, P(2) = PUNTO_2 (anotar vale más que el peón que sale);
- avance: tabla por filas avanzadas desde la propia orilla (0–7);
- apoyo: suma de los apoyos (0, 1 o 2) de cada peón;
- turno: una constante a favor del que mueve;
- carrera: cuántas jugadas le faltan al rival menos cuántas nos faltan para anotar los puntos
  que quedan, contando solo los (3 − puntos) peones más cercanos a su meta. La última fila no
  defiende: gana quien junte primero sus peones cerca de la meta;
- corredores: el cono de amenaza de un peón son las casillas desde donde un rival podría
  atacarlo o cerrarle el paso antes de anotar: `|Δcol| ≤ ⌊(Δfilas − 1)/2⌋ + 1`
  por delante de él. Un peón a 4 jugadas o menos de anotar vale distinto si en su cono no hay
  rivales (libre), hay uno (interceptado) o hay más (contenido, sin peso propio): una tabla por
  clase y por distancia (1–4);
- carrera libre: como la carrera, pero solo con peones libres (tope de 12 jugadas por bando);
- amenazas: el mejor ataque que tiene el que mueve (2p − 1, si es positivo) y la suma de los
  ataques positivos del rival (el que mueve solo puede responder a uno);
- patrones: una tabla con un valor por cada vecindario 3×3 posible de un peón (cada vecina
  vacía, propia o rival: 3⁸ = 6561), según su fila de avance y su columna (40 regiones). Se ve
  desde el dueño del peón y se refleja para que la columna quede a la izquierda, así que los
  dos colores y los dos lados comparten la tabla: 262 440 valores. Es solo buscar y sumar
  (una «tabla de patrones», permitida; el profesor fijó un máximo de ~40 MB).

El valor se recorta a ±0.95: de ahí para arriba solo caben victorias y derrotas seguras.

**Carrera decidida (exacta, no depende de pesos):** un corredor es *seguro* si en
su cono, ampliado una columna, no hay ningún peón (ni rival ni propio): nadie puede tocarlo y
anota en tantas jugadas como su distancia. Si el que mueve anota lo que le falta con corredores
seguros en `n` jugadas y el rival no puede hacerlo en menos de `n` ni con todos sus peones (cota
inferior: sus distancias, sin bloqueos), ya ganó; si es el rival el que anota en `n` jugadas y al
que mueve le faltan más de `n`, ya perdió. Se regresa ±(0.965 − 0.0001·n): por encima de toda
evaluación (0.95) y por debajo de una victoria vista en el árbol (≥ 0.968).

`PESOS_F2` son los pesos a mano, sin tabla. Los que usa el bot (rasgos y tabla) salen de una
regresión logística sobre partidas (`herramientas/entrenar.py`) y se leen de `data/pesos.npz`.
"""

import math

import numpy as np
from numba import njit

from motor_rapido import CASILLAS, LADO, PUNTOS_PARA_GANAR, VACIA, njit_interna, prob_combate

I_MATERIAL = 0
I_PUNTO_1 = 1
I_PUNTO_2 = 2
I_APOYO = 3
I_TURNO = 4
I_AVANCE = 5  # 8 valores: filas avanzadas 0–7
I_CARRERA = 13
CORREDOR_MAX = 4  # distancia a la meta hasta la que se mira el cono de un peón
I_CORREDOR = 14  # libres (distancia 1–4) y luego interceptados (distancia 1–4)
I_CARRERA_LIBRE = I_CORREDOR + 2 * CORREDOR_MAX
I_MEJOR_ATAQUE = I_CARRERA_LIBRE + 1
I_ATAQUES_RIVAL = I_MEJOR_ATAQUE + 1
N_BASE = I_ATAQUES_RIVAL + 1  # pesos de los rasgos; después vienen los de la tabla de patrones
NOMBRES = ["MATERIAL", "PUNTO_1", "PUNTO_2", "APOYO", "TURNO"] + [f"AVANCE_{k}" for k in range(8)]
NOMBRES += ["CARRERA"] + [f"LIBRE_{d}" for d in range(1, CORREDOR_MAX + 1)]
NOMBRES += [f"INTERCEPTADO_{d}" for d in range(1, CORREDOR_MAX + 1)]
NOMBRES += ["CARRERA_LIBRE", "MEJOR_ATAQUE", "ATAQUES_RIVAL"]
COLUMNAS_TABLA = 5  # columnas a–e (las de la derecha se reflejan)
PATRONES_POR_REGION = 3**8
N_PATRONES = 8 * COLUMNAS_TABLA * PATRONES_POR_REGION
I_PATRONES = N_BASE
N_PESOS = N_BASE + N_PATRONES
MAX_PEONES = 36
# Vecinas de un peón desde su dueño: adelante (izq, centro, der), a los lados, atrás.
DF = (1, 1, 1, 0, 0, -1, -1, -1)
DC = (-1, 0, 1, -1, 1, -1, 0, 1)


def _tablas_de_vecinas() -> tuple[np.ndarray, np.ndarray]:
    """vecinas[dueño, casilla, k]: casilla de la vecina k vista desde el dueño, y potencias:
    3^k, o 0 si esa vecina cae fuera del tablero (entonces vecinas apunta a la misma casilla y
    no suma nada). Así el código del vecindario se calcula sin preguntar por los bordes.

    En las columnas de la derecha ya viene reflejada (izquierda ↔ derecha). Se calculan una vez
    al importar: la evaluación no repite cuentas de filas, columnas y bordes en cada nodo.
    """
    vecinas = np.zeros((2, CASILLAS, 8), np.int64)
    potencias = np.zeros((2, CASILLAS, 8), np.int64)
    for duenio in (0, 1):
        adelante = 1 if duenio == 0 else -1
        for c in range(CASILLAS):
            fila, col = divmod(c, LADO)
            reflejo = 1 if col <= COLUMNAS_TABLA - 1 else -1
            for k in range(8):
                f2, c2 = fila + DF[k] * adelante, col + DC[k] * reflejo
                if 0 <= f2 < LADO and 0 <= c2 < LADO:
                    vecinas[duenio, c, k] = f2 * LADO + c2
                    potencias[duenio, c, k] = 3**k
                else:
                    vecinas[duenio, c, k] = c
    return vecinas, potencias


def _espejo_de_codigos() -> np.ndarray:
    """espejo[código]: el mismo vecindario reflejado (izquierda ↔ derecha)."""
    reflejada = (2, 1, 0, 4, 3, 7, 6, 5)  # la vecina k pasa a ser la vecina reflejada[k]
    espejo = np.zeros(PATRONES_POR_REGION, np.int64)
    for codigo in range(PATRONES_POR_REGION):
        digitos = [codigo // 3**k % 3 for k in range(8)]
        espejo[codigo] = sum(digitos[k] * 3 ** reflejada[k] for k in range(8))
    return espejo


VECINAS, POTENCIAS = _tablas_de_vecinas()
ESPEJO = _espejo_de_codigos()
# Lo que vale cada pieza en el código, según el dueño del peón: vacía 0, propia 1, rival 2.
VALOR_PIEZA = np.array([[0, 1, 2], [0, 2, 1]], np.int64)


def _tablas_rapidas() -> tuple[np.ndarray, ...]:
    """Tablas para que `evaluar` no repita cuentas en cada peón (mismos valores, menos trabajo;
    medido: −25 % de tiempo por evaluación y +12.6 % de nodos/s en la búsqueda).

    - contrib[dueño, casilla, k, pieza]: lo que suma la vecina k al código del vecindario
      (VALOR_PIEZA · POTENCIAS); region[dueño, casilla]: dónde empieza su región en la tabla.
    - atras / adelante[dueño, casilla, i]: las dos casillas diagonales de atrás (apoyo) y de
      adelante (ataques), izquierda y derecha. Si caen fuera del tablero, *_ok vale 0 (la casilla
      guardada es 0 y no cuenta): así no hay que preguntar por los bordes.
    """
    contrib = VALOR_PIEZA[:, None, None, :] * POTENCIAS[:, :, :, None]
    region = np.zeros((2, CASILLAS), np.int64)
    diagonales = np.zeros((2, 2, CASILLAS, 2), np.int64)  # [atrás/adelante, dueño, casilla, i]
    ok = np.zeros((2, 2, CASILLAS, 2), np.int64)
    for duenio in (0, 1):
        paso = 1 if duenio == 0 else -1
        for c in range(CASILLAS):
            fila, col = divmod(c, LADO)
            avance = fila if duenio == 0 else LADO - 1 - fila
            columna = min(col, LADO - 1 - col)
            region[duenio, c] = (avance * COLUMNAS_TABLA + columna) * PATRONES_POR_REGION
            for sentido, f in enumerate((fila - paso, fila + paso)):
                for i, cc in enumerate((col - 1, col + 1)):
                    if 0 <= f < LADO and 0 <= cc < LADO:
                        diagonales[sentido, duenio, c, i] = f * LADO + cc
                        ok[sentido, duenio, c, i] = 1
    return contrib, region, diagonales[0], ok[0], diagonales[1], ok[1]


CONTRIB, REGION, ATRAS, ATRAS_OK, ADELANTE, ADELANTE_OK = _tablas_rapidas()

TOPE = 0.95
SIN_PEON = 20  # distancia de un peón que no existe: no alcanzan los peones para anotar
TOPE_CARRERA = 12  # jugadas de ventaja en la carrera que todavía cuentan
DECIDIDA = 0.965  # valor de una carrera decidida (menos 0.0001 por jugada que falta)
PASO_DECIDIDA = 0.0001

PESOS_F2 = np.array(
    [
        0.30,  # material, por peón
        1.20,  # tener 1 punto
        3.00,  # tener 2 puntos
        0.06,  # por cada apoyo
        0.05,  # turno
        0.00, 0.00, 0.02, 0.05, 0.10, 0.18, 0.30, 0.50,  # avance 0–7
        0.15,  # carrera, por jugada de ventaja (medido en la arena: 0.15 > 0.3 > 0.6 > 0)
        *np.zeros(N_BASE - I_CORREDOR),  # corredores, carrera libre y amenazas: los da la regresión
    ]
)  # sin tabla de patrones: esa sale del entrenamiento (N_PESOS valores en total)


@njit_interna
def _valor_puntos(puntos, pesos):
    if puntos == 0:
        return 0.0
    if puntos == 1:
        return pesos[I_PUNTO_1]
    return pesos[I_PUNTO_2]


@njit_interna
def _insertar(m0, m1, m2, d):
    """Las tres distancias más chicas (m0 ≤ m1 ≤ m2) después de ver una nueva, `d`."""
    if d < m0:
        return d, m0, m1
    if d < m1:
        return m0, d, m1
    if d < m2:
        return m0, m1, d
    return m0, m1, m2


@njit_interna
def _patron(tab, c, duenio):
    """Lugar en la tabla de patrones del peón de `duenio` en la casilla `c`."""
    codigo = 0
    for k in range(8):  # número del vecindario 3×3 en base 3
        codigo += CONTRIB[duenio, c, k, tab[VECINAS[duenio, c, k]]]
    if c % LADO == COLUMNAS_TABLA - 1:  # columna central: el menor de los dos reflejos
        codigo = min(codigo, ESPEJO[codigo])
    return REGION[duenio, c] + codigo


@njit_interna
def _apoyo(tab, c, duenio):
    """Peones de `duenio` en las dos diagonales de atrás de `c` (como `motor_rapido.apoyo`)."""
    pieza = duenio + 1
    return (
        np.int64(tab[ATRAS[duenio, c, 0]] == pieza) * ATRAS_OK[duenio, c, 0]
        + np.int64(tab[ATRAS[duenio, c, 1]] == pieza) * ATRAS_OK[duenio, c, 1]
    )


@njit_interna
def _cono(tab, c, duenio, avance):
    """(interceptores, ocupado) del peón de `duenio` en `c`, mirando hasta su meta.

    `interceptores`: rivales en su cono de amenaza (tope 2). `ocupado`: si hay cualquier peón,
    propio o rival, en el cono ampliado una columna (si no, el peón es un corredor seguro).
    """
    fila = c // LADO
    col = c % LADO
    adelante = 1 if duenio == 0 else -1
    rival = 2 - duenio  # pieza del rival: 1 blancas, 2 negras
    interceptores = 0
    ocupado = False
    for delta in range(1, LADO - avance):
        f = fila + adelante * delta
        ancho = (delta - 1) // 2 + 1
        for dc in range(-ancho - 1, ancho + 2):
            cc = col + dc
            if cc < 0 or cc >= LADO:
                continue
            pieza = tab[f * LADO + cc]
            if pieza != VACIA:
                ocupado = True
                if pieza == rival and abs(dc) <= ancho:
                    interceptores += 1
        if interceptores >= 2:
            return 2, True
    return interceptores, ocupado


@njit_interna
def _ataques(tab, c, duenio, apoyo_propio):
    """(mejor, suma) de los ataques que tiene el peón en `c`: el mayor 2p − 1 (−1 si no puede
    atacar) y la suma de los 2p − 1 positivos."""
    mejor = -1.0
    suma = 0.0
    rival = 2 - duenio
    for i in range(2):  # diagonal izquierda y luego derecha
        if ADELANTE_OK[duenio, c, i] == 1:
            t = ADELANTE[duenio, c, i]
            if tab[t] == rival:
                q = 2.0 * prob_combate(apoyo_propio, _apoyo(tab, t, 1 - duenio)) - 1.0
                mejor = max(mejor, q)
                if q > 0.0:
                    suma += q
    return mejor, suma


@njit_interna
def _jugadas_para_ganar(m0, m1, m2, puntos):
    """Mínimo de jugadas para anotar lo que falta con los peones más cercanos a la meta."""
    faltan = PUNTOS_PARA_GANAR - puntos
    total = m0
    if faltan >= 2:
        total += m1
    if faltan >= 3:
        total += m2
    return total


@njit_interna
def _decidida(sa0, sa1, sa2, sb0, sb1, sb2, a0, a1, a2, b0, b1, b2, pts, color):
    """Jugadas `n` de una carrera decidida: n > 0 si gana el que mueve, −n si gana el rival,
    0 si no está decidida. sa/sb: distancias de los corredores seguros; a/b: de todos los peones
    (a = del que mueve)."""
    n_mueve = _jugadas_para_ganar(sa0, sa1, sa2, pts[color])
    if n_mueve < SIN_PEON and n_mueve <= _jugadas_para_ganar(b0, b1, b2, pts[1 - color]):
        return n_mueve
    n_otro = _jugadas_para_ganar(sb0, sb1, sb2, pts[1 - color])
    if n_otro < SIN_PEON and n_otro < _jugadas_para_ganar(a0, a1, a2, pts[color]):
        return -n_otro
    return 0


@njit_interna
def evaluar(tab, pts, color, pesos):
    """Valor en [−0.95, 0.95] para el bando `color`, que es el que mueve (o ±0.965 − 0.0001·n
    si la carrera ya está decidida; ver el principio del módulo).

    `pesos` trae N_BASE valores (sin tabla de patrones) o N_PESOS (con tabla). Sin tabla no se
    calculan los patrones: la evaluación va casi al doble de rápido.
    """
    con_tabla = len(pesos) > I_PATRONES
    x = pesos[I_TURNO]
    # Distancias a la meta de los 3 peones más adelantados, propios (a) y del rival (b); de los
    # libres (la, lb) y de los corredores seguros (sa, sb).
    # np.int64 y no la constante sola: así numba compila los ayudantes una sola vez.
    a0 = a1 = a2 = b0 = b1 = b2 = np.int64(SIN_PEON)
    la0 = la1 = la2 = lb0 = lb1 = lb2 = np.int64(SIN_PEON)
    sa0 = sa1 = sa2 = sb0 = sb1 = sb2 = np.int64(SIN_PEON)
    mejor_ataque = 0.0
    ataques_rival = 0.0
    for c in range(CASILLAS):
        pieza = tab[c]
        if pieza == VACIA:
            continue
        duenio = pieza - 1
        fila = c // LADO
        avance = fila if duenio == 0 else LADO - 1 - fila
        apoyo_peon = _apoyo(tab, c, duenio)
        valor = pesos[I_MATERIAL] + pesos[I_AVANCE + avance] + pesos[I_APOYO] * apoyo_peon
        if con_tabla:
            valor += pesos[I_PATRONES + _patron(tab, c, duenio)]
        distancia = LADO - 1 - avance  # jugadas para anotar con este peón
        libre = False
        seguro = False
        if distancia <= CORREDOR_MAX:
            interceptores, ocupado = _cono(tab, c, duenio, avance)
            if interceptores < 2:
                valor += pesos[I_CORREDOR + interceptores * CORREDOR_MAX + distancia - 1]
            libre = interceptores == 0
            seguro = not ocupado
        mejor, suma = _ataques(tab, c, duenio, apoyo_peon)
        if duenio == color:
            x += valor
            a0, a1, a2 = _insertar(a0, a1, a2, distancia)
            if libre:
                la0, la1, la2 = _insertar(la0, la1, la2, distancia)
            if seguro:
                sa0, sa1, sa2 = _insertar(sa0, sa1, sa2, distancia)
            mejor_ataque = max(mejor_ataque, mejor)
        else:
            x -= valor
            b0, b1, b2 = _insertar(b0, b1, b2, distancia)
            if libre:
                lb0, lb1, lb2 = _insertar(lb0, lb1, lb2, distancia)
            if seguro:
                sb0, sb1, sb2 = _insertar(sb0, sb1, sb2, distancia)
            ataques_rival += suma
    n = _decidida(sa0, sa1, sa2, sb0, sb1, sb2, a0, a1, a2, b0, b1, b2, pts, color)
    if n > 0:
        return DECIDIDA - PASO_DECIDIDA * n
    if n < 0:
        return -(DECIDIDA + PASO_DECIDIDA * n)
    x += _valor_puntos(pts[color], pesos) - _valor_puntos(pts[1 - color], pesos)
    carrera = _jugadas_para_ganar(b0, b1, b2, pts[1 - color]) - _jugadas_para_ganar(
        a0, a1, a2, pts[color]
    )
    x += pesos[I_CARRERA] * min(max(carrera, -TOPE_CARRERA), TOPE_CARRERA)
    libre_propia = min(_jugadas_para_ganar(la0, la1, la2, pts[color]), TOPE_CARRERA)
    libre_rival = min(_jugadas_para_ganar(lb0, lb1, lb2, pts[1 - color]), TOPE_CARRERA)
    x += pesos[I_CARRERA_LIBRE] * (libre_rival - libre_propia)
    x += pesos[I_MEJOR_ATAQUE] * mejor_ataque + pesos[I_ATAQUES_RIVAL] * ataques_rival
    v = math.tanh(x / 2.0)  # = 2·σ(x) − 1
    return min(max(v, -TOPE), TOPE)


@njit
def rasgos(tab, pts, color, salida):
    """Llena `salida` (N_BASE valores) con los rasgos de la posición desde el bando `color`.

    Es la otra cara de `evaluar`, junto con `patrones`: `evaluar = tanh(x / 2)`, recortado a
    ±0.95, con x = pesos[:N_BASE] · rasgos + Σ signo · pesos[I_PATRONES + índice] (salvo en
    una carrera decidida, que no depende de pesos). Sirve para entrenar los pesos con regresión
    logística (herramientas/entrenar.py); en la partida se usa `evaluar`, que suma directo y es
    más rápida.
    """
    salida[:] = 0.0
    salida[I_TURNO] = 1.0
    a0 = a1 = a2 = b0 = b1 = b2 = np.int64(SIN_PEON)
    la0 = la1 = la2 = lb0 = lb1 = lb2 = np.int64(SIN_PEON)
    mejor_ataque = 0.0
    ataques_rival = 0.0
    for c in range(CASILLAS):
        pieza = tab[c]
        if pieza == VACIA:
            continue
        duenio = pieza - 1
        fila = c // LADO
        avance = fila if duenio == 0 else LADO - 1 - fila
        signo = 1.0 if duenio == color else -1.0
        apoyo_peon = _apoyo(tab, c, duenio)
        salida[I_MATERIAL] += signo
        salida[I_AVANCE + avance] += signo
        salida[I_APOYO] += signo * apoyo_peon
        distancia = LADO - 1 - avance
        libre = False
        if distancia <= CORREDOR_MAX:
            interceptores, _ = _cono(tab, c, duenio, avance)
            if interceptores < 2:
                salida[I_CORREDOR + interceptores * CORREDOR_MAX + distancia - 1] += signo
            libre = interceptores == 0
        mejor, suma = _ataques(tab, c, duenio, apoyo_peon)
        if duenio == color:
            a0, a1, a2 = _insertar(a0, a1, a2, distancia)
            if libre:
                la0, la1, la2 = _insertar(la0, la1, la2, distancia)
            mejor_ataque = max(mejor_ataque, mejor)
        else:
            b0, b1, b2 = _insertar(b0, b1, b2, distancia)
            if libre:
                lb0, lb1, lb2 = _insertar(lb0, lb1, lb2, distancia)
            ataques_rival += suma
    for puntos, signo in ((pts[color], 1.0), (pts[1 - color], -1.0)):
        if puntos == 1:
            salida[I_PUNTO_1] += signo
        elif puntos >= 2:
            salida[I_PUNTO_2] += signo
    carrera = _jugadas_para_ganar(b0, b1, b2, pts[1 - color]) - _jugadas_para_ganar(
        a0, a1, a2, pts[color]
    )
    salida[I_CARRERA] = min(max(carrera, -TOPE_CARRERA), TOPE_CARRERA)
    libre_propia = min(_jugadas_para_ganar(la0, la1, la2, pts[color]), TOPE_CARRERA)
    libre_rival = min(_jugadas_para_ganar(lb0, lb1, lb2, pts[1 - color]), TOPE_CARRERA)
    salida[I_CARRERA_LIBRE] = libre_rival - libre_propia
    salida[I_MEJOR_ATAQUE] = mejor_ataque
    salida[I_ATAQUES_RIVAL] = ataques_rival


@njit
def patrones(tab, color, indices, signos):
    """Lugar en la tabla de cada peón (+1 propio, −1 rival desde `color`); regresa cuántos."""
    n = 0
    for c in range(CASILLAS):
        pieza = tab[c]
        if pieza == VACIA:
            continue
        duenio = pieza - 1
        indices[n] = _patron(tab, c, duenio)
        signos[n] = 1 if duenio == color else -1
        n += 1
    return n
