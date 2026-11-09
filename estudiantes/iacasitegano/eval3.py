"""Evaluación eval3: las mismas seis ideas de `eval.py` más cuatro familias nuevas, todas
calculables en numba sobre el tablero crudo.

**El truco de diseño.** Las tablas de patrones y de filas son *pesos aprendidos*, no features
precalculadas. Entonces hay dos formas de usarlas y tienen que coincidir exactamente:

- Para ENTRENAR hace falta el vector ancho (un histograma de patrones, un conteo por fila…),
  porque la regresión necesita una columna por peso. Eso se calcula una vez por posición,
  offline.
- Para JUGAR no se arma el vector: se acumula `W[patrón]` peón por peón. Es O(peones)
  lookups en vez de construir y multiplicar 259 números en cada hoja de la búsqueda.

`evaluar_nb(...)` (jugar) debe dar exactamente `dot(W, features_anchas_nb(...))` (entrenar).
Eso lo verifica `herramientas/lab/test_eval3_features.py`, junto con la igualdad entre la
versión numba y la de Python de cada familia.

Convenio: todo es "propio menos rival", igual que `eval.py`, para que la evaluación sea
antisimétrica (si se intercambian los colores, el valor cambia de signo).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from motor.nucleo import CASILLAS, LADO, VACIA, apoyo, generar, prob_combate

# --- Distribución del vector ancho -----------------------------------------------------
# Cada familia ocupa un rango contiguo; los nombres se usan en la ablación del entrenador.
CELDAS_PATRON = 5           # 3 casillas de adelante + 2 diagonales de atrás (apoyo)
N_PATRONES = 3 ** CELDAS_PATRON   # 243: cada celda es vacía / propia / rival
N_FILAS = LADO              # 9 filas de avance

I_PUNTOS = 0
I_PATRONES = 1
I_FILAS = I_PATRONES + N_PATRONES
I_AMENAZA = I_FILAS + N_FILAS       # 3: capturables nuestros, capturables rivales, apoyo
I_LIBRES = I_AMENAZA + 3            # 2: corredores libres, distancia ajustada
I_MOVILIDAD = I_LIBRES + 2          # 1
# Las tres features que la evaluación entregada tiene y a eval3 le faltaban. Importan por
# una razón estructural, no por ser tres más: **dos de ellas no son aditivas sobre peones**,
# y todo el resto de eval3 sí lo es.
#   pasado   suma del avance de los peones sin rival en su camino  (aditiva)
#   vacias   columnas rivales desocupadas menos propias desocupadas (cuenta de columnas
#            DISTINTAS: dos peones en la misma columna no cuentan doble)
#   carrera  distancia a la meta del peón MÁS AVANZADO, propia menos rival (un MÁXIMO)
# Una suma de términos por peón no puede representar un máximo ni un conteo de columnas
# distintas, así que con 259 features aditivas eval3 no podía expresar lo que la evaluación
# a mano expresa con dos. Y la carrera es justo lo que decide los finales.
I_EXTRA = I_MOVILIDAD + 1           # 3: pasado, vacias, carrera
# --- Versión 2: tres familias nuevas (material, marcador, exposición) ------------------
# `data/pesos_eval3.npz` YA tiene un candidato jugable (267 pesos, versión 1 = hasta aquí)
# que Codex puede estar midiendo en este momento con partidas reales. Si cada partida
# relanza el proceso del bot, un cambio aquí que no sea compatible rompería la carga de esos
# pesos a mitad de medición. `N_FEATURES_V1`/`N_PESOS_V1` quedan fijos como los valores
# HISTÓRICOS de la versión 1 (262/267); `busqueda_eval3.cargar_pesos` detecta un archivo de
# ese largo exacto y lo relincha con las familias nuevas apagadas, reproduciendo el
# comportamiento de antes bit a bit (verificado en test_compat_v1.py).
N_FEATURES_V1 = I_EXTRA + 3          # 262 — NUNCA cambiar este número
I_MATERIAL = I_EXTRA + 3             # 1: peones vivos, propios menos rivales
I_MARCADOR = I_MATERIAL + 1          # 3: marcador no lineal, ver más abajo
I_EXPOSICION = I_MARCADOR + 3        # 1: exposición táctica esperada
N_FEATURES = I_EXPOSICION + 1        # 267

FAMILIAS = {
    "puntos": (I_PUNTOS, I_PUNTOS + 1),
    "patrones": (I_PATRONES, I_FILAS),
    "filas": (I_FILAS, I_AMENAZA),
    "amenaza": (I_AMENAZA, I_LIBRES),
    "libres": (I_LIBRES, I_MOVILIDAD),
    "movilidad": (I_MOVILIDAD, I_EXTRA),
    "extra": (I_EXTRA, N_FEATURES_V1),
    "material": (I_MATERIAL, I_MARCADOR),
    "marcador": (I_MARCADOR, I_EXPOSICION),
    "exposicion": (I_EXPOSICION, N_FEATURES),
}

# Las 3 celdas del marcador no lineal: una por par (i,j) con i<j, i,j en {0,1,2} (no puede
# haber más puntos sin que la partida termine). Construcción ANTISIMÉTRICA (ver
# `_marcador_signo`): solo 3 parámetros libres, no 9, y así no hace falta forzar
# w[i,j]=-w[j,i] a mano — la propia feature ya sale con ese signo.
PARES_MARCADOR = ((0, 1), (0, 2), (1, 2))

# --- Banderas de familia (apagar una familia AHORRA su cómputo) -------------------------
# Poner los pesos de una familia en cero NO abarata nada: `evaluar_nb` calcularía igual el
# patrón, las amenazas y los corredores, y solo multiplicaría por cero. Como el criterio de
# aceptación incluye un tope de costo por nodo, una versión "barata" tiene que saltarse el
# cómputo, no solo anular su contribución.
#
# Las banderas viajan en las ÚLTIMAS posiciones del propio vector de pesos. Así la firma de
# `evaluar_nb` no cambia (`float64[::1]` no fija el largo) y no hay que tocar la cadena de
# llamadas de la búsqueda, que pasa `pesos` por seis niveles. `puntos` y `filas` no tienen
# bandera: son una multiplicación y un lookup dentro de un recorrido que ya ocurre.
#
# Cuidado con los dos largos: el entrenador trabaja con N_FEATURES (una columna por peso) y
# el que juega usa N_PESOS. `empaquetar`/`desempaquetar` son el único punto de conversión.
BANDERAS_V1 = ("patrones", "amenaza", "libres", "movilidad", "extra")
N_PESOS_V1 = N_FEATURES_V1 + len(BANDERAS_V1)   # 267 — el largo que YA tiene el archivo
                                                # jugable de la versión 1; no cambia nunca.
BANDERAS = BANDERAS_V1 + ("material", "marcador", "exposicion")
N_BANDERAS = len(BANDERAS)
B_PATRONES = N_FEATURES
B_AMENAZA = N_FEATURES + 1
B_LIBRES = N_FEATURES + 2
B_MOVILIDAD = N_FEATURES + 3
B_EXTRA = N_FEATURES + 4
B_MATERIAL = N_FEATURES + 5
B_MARCADOR = N_FEATURES + 6
B_EXPOSICION = N_FEATURES + 7
N_PESOS = N_FEATURES + N_BANDERAS    # 275


def empaquetar(W, activas=BANDERAS):
    """Vector de `N_PESOS` listo para jugar: los pesos más las banderas de familia.

    Apaga además los pesos de las familias inactivas, para que el vector no pueda mentir
    sobre lo que la evaluación realmente usa."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (N_FEATURES,):
        raise ValueError(f"empaquetar espera {N_FEATURES} pesos, llegaron {W.shape}")
    desconocidas = set(activas) - set(FAMILIAS)
    if desconocidas:
        raise ValueError(f"familias desconocidas: {sorted(desconocidas)}")
    salida = np.zeros(N_PESOS, np.float64)
    salida[:N_FEATURES] = W
    for nombre in BANDERAS:
        if nombre in activas:
            salida[N_FEATURES + BANDERAS.index(nombre)] = 1.0
        else:
            i0, i1 = FAMILIAS[nombre]
            salida[i0:i1] = 0.0
    return np.ascontiguousarray(salida)


def activas_de(W):
    """Las familias con algún peso distinto de cero, para no tener que recordarlas aparte.

    Es lo que hace automática la versión barata: si la ablación dejó una familia fuera, sus
    pesos quedan en cero y aquí se apaga su cómputo, que es de donde sale el ahorro."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape not in {(N_FEATURES,), (N_PESOS,)}:
        raise ValueError(f"se esperaban {N_FEATURES} o {N_PESOS} pesos, llegaron {W.shape}")
    return tuple(n for n in BANDERAS if np.any(W[slice(*FAMILIAS[n])] != 0.0))


def empaquetar_auto(W):
    """`empaquetar` con las familias deducidas de los pesos. Es lo que se quiere casi siempre."""
    return empaquetar(W[:N_FEATURES], activas_de(W))


def desempaquetar(W):
    """Los `N_FEATURES` pesos y el conjunto de familias activas."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (N_PESOS,):
        raise ValueError(f"desempaquetar espera {N_PESOS} pesos, llegaron {W.shape}")
    activas = tuple(n for i, n in enumerate(BANDERAS) if W[N_FEATURES + i] != 0.0)
    return np.ascontiguousarray(W[:N_FEATURES]), activas


def desde_v1(W_v1):
    """Convierte un vector empaquetado de la VERSIÓN 1 (`N_PESOS_V1`=267, las familias
    hasta `extra`) al formato actual, con `material`/`marcador`/`exposicion` en cero y
    apagadas. Reproduce EXACTAMENTE el comportamiento de la v1 (verificado en
    `test_compat_v1.py`): es la función que mantiene cargable el candidato que ya está en
    `eval3_listo.flag` aunque este archivo siga creciendo."""
    W_v1 = np.asarray(W_v1, dtype=np.float64)
    if W_v1.shape != (N_PESOS_V1,):
        raise ValueError(f"desde_v1 espera {N_PESOS_V1} pesos, llegaron {W_v1.shape}")
    salida = np.zeros(N_PESOS, np.float64)
    salida[:N_FEATURES_V1] = W_v1[:N_FEATURES_V1]        # los pesos, sin tocar
    for i, nombre in enumerate(BANDERAS_V1):             # las 5 banderas viejas, en orden
        salida[N_FEATURES + BANDERAS.index(nombre)] = W_v1[N_FEATURES_V1 + i]
    return salida


# --- Utilidades compartidas ------------------------------------------------------------
@njit(cache=True)
def _avance_nb(fila, color):
    return fila if color == 0 else (LADO - 1 - fila)


@njit(cache=True)
def _estado_celda(tablero, fila, col, color):
    """0 = vacía o fuera del tablero, 1 = propia, 2 = rival."""
    if fila < 0 or fila >= LADO or col < 0 or col >= LADO:
        return 0
    v = tablero[fila * LADO + col]
    if v == VACIA:
        return 0
    return 1 if v == color + 1 else 2


@njit(cache=True)
def _patron(tablero, casilla, color):
    """Código base 3 de la vecindad de un peón: las 3 casillas de adelante y las 2
    diagonales de atrás (las que lo apoyan). Devuelve 0..242."""
    fila = casilla // LADO
    col = casilla % LADO
    d = 1 if color == 0 else -1
    p = 0
    p += _estado_celda(tablero, fila + d, col - 1, color)
    p += 3 * _estado_celda(tablero, fila + d, col, color)
    p += 9 * _estado_celda(tablero, fila + d, col + 1, color)
    p += 27 * _estado_celda(tablero, fila - d, col - 1, color)
    p += 81 * _estado_celda(tablero, fila - d, col + 1, color)
    return p


@njit(cache=True)
def _es_libre(tablero, casilla, color):
    """Ningún rival en las tres columnas que le faltan por recorrer (mismo criterio que la
    feature `pasado` de `eval.py`)."""
    fila = casilla // LADO
    col = casilla % LADO
    d = 1 if color == 0 else -1
    f = fila + d
    while 0 <= f < LADO:
        for c in range(col - 1, col + 2):
            if 0 <= c < LADO and _estado_celda(tablero, f, c, color) == 2:
                return False
        f += d
    return True


@njit(cache=True)
def _filas_rivales_por_columna(tablero):
    """Extremos de fila de las piezas rivales, por columna y color de pieza."""
    maxrow_color2 = np.full(LADO, -1, np.int64)
    minrow_color1 = np.full(LADO, LADO, np.int64)
    for c in range(CASILLAS):
        pieza = tablero[c]
        if pieza == VACIA:
            continue
        fila = c // LADO
        col = c % LADO
        if pieza == 2:
            if fila > maxrow_color2[col]:
                maxrow_color2[col] = fila
        elif fila < minrow_color1[col]:
            minrow_color1[col] = fila
    return maxrow_color2, minrow_color1


@njit(cache=True)
def _es_libre_rapido(casilla, color, maxrow_color2, minrow_color1):
    """Misma condición que `_es_libre`, consultando extremos por columna en O(1)."""
    fila = casilla // LADO
    col = casilla % LADO
    for dc in (-1, 0, 1):
        cc = col + dc
        if cc < 0 or cc >= LADO:
            continue
        if color == 0:
            if maxrow_color2[cc] > fila:
                return False
        elif minrow_color1[cc] < fila:
            return False
    return True


@njit(cache=True)
def _amenazas_de(tablero, casilla, color):
    """(prob de que nos capturen este peón, prob de que capturemos a un rival desde él).

    Un ataque va en diagonal hacia adelante sobre casilla ocupada, así que un rival en
    `fila + d_rival` diagonal puede capturarnos. La probabilidad es la real del motor,
    `prob_combate(apoyo del atacante, apoyo del defensor)`, no una inventada."""
    fila = casilla // LADO
    col = casilla % LADO
    d = 1 if color == 0 else -1
    rival = 1 - color
    peor = 0.0
    mejor = 0.0
    # Nos capturan: un rival está una fila por delante nuestro en diagonal (para él, esa
    # es su dirección de avance).
    for dc in (-1, 1):
        fr = fila + d
        cr = col + dc
        if 0 <= fr < LADO and 0 <= cr < LADO:
            if _estado_celda(tablero, fr, cr, color) == 2:
                p = prob_combate(apoyo(tablero, casilla, rival), apoyo(tablero, casilla, color))
                if p > peor:
                    peor = p
    # Capturamos: hay un rival en nuestra diagonal de avance.
    for dc in (-1, 1):
        fr = fila + d
        cr = col + dc
        if 0 <= fr < LADO and 0 <= cr < LADO:
            if _estado_celda(tablero, fr, cr, color) == 2:
                destino = fr * LADO + cr
                p = prob_combate(apoyo(tablero, destino, color), apoyo(tablero, destino, rival))
                if p > mejor:
                    mejor = p
    return peor, mejor


@njit(cache=True)
def _exposicion_de(tablero, casilla, color):
    """Exposición táctica neta de ESTE peón: p(nos lo capturan) · (1 − p(lo recapturamos)).

    Es una pérdida de material ESPERADA, no solo "probabilidad de ataque" (que ya existe en
    `amenaza`/`peor`): si tengo con qué recapturar, la pérdida neta baja aunque me ataquen.

    **Aproximación ESTÁTICA**, documentada porque es la pieza menos exacta de esta
    evaluación: la probabilidad de recaptura usa el apoyo que el atacante tiene HOY en su
    propia casilla, no el que tendría en la casilla de destino después de capturar (para
    eso haría falta ver el tablero tras el cambio, es decir, un ply más de búsqueda). Es la
    misma clase de aproximación de "no mirar más allá" que usa el resto de la evaluación.

    Entre varios atacantes posibles se usa el de mayor probabilidad (igual que `peor` en
    `_amenazas_de`), y la recaptura se evalúa contra ESE atacante en particular."""
    fila = casilla // LADO
    col = casilla % LADO
    d = 1 if color == 0 else -1
    rival = 1 - color
    peor = 0.0
    recaptura = 0.0
    for dc in (-1, 1):
        fr = fila + d
        cr = col + dc
        if 0 <= fr < LADO and 0 <= cr < LADO:
            if _estado_celda(tablero, fr, cr, color) == 2:
                p = prob_combate(apoyo(tablero, casilla, rival), apoyo(tablero, casilla, color))
                if p > peor:
                    peor = p
                    recaptura = prob_combate(apoyo(tablero, casilla, color),
                                             apoyo(tablero, fr * LADO + cr, rival))
    return peor * (1.0 - recaptura)


@njit(cache=True)
def _marcador_indice_signo(i, j):
    """Para el marcador (i,j) = (puntos propios, puntos rivales), con i,j en {0,1,2}:
    el índice de celda entre los 3 pares posibles {0,1}/{0,2}/{1,2} y el signo (+1 si
    vamos ganando, −1 si vamos perdiendo, 0 empatado). Construcción ANTISIMÉTRICA: solo
    se activa UNA celda, con signo, nunca las dos simétricas (i,j) y (j,i) a la vez — así
    un único peso por par alcanza, sin tener que forzar w[i,j]=−w[j,i] a mano."""
    if i == j:
        return 0, 0.0
    if i < j:
        lo, hi, signo = i, j, -1.0
    else:
        lo, hi, signo = j, i, 1.0
    if lo == 0 and hi == 1:
        idx = 0
    elif lo == 0 and hi == 2:
        idx = 1
    else:
        idx = 2
    return idx, signo


@njit(cache=True)
def _distancia_ajustada(tablero, casilla, color):
    """Distancia a la meta menos la del rival más cercano que puede interceptar.

    Un rival solo puede interceptar si le alcanzan las columnas: necesita `|Δcol|`
    movimientos laterales y tiene `dg` turnos. Es la familia más caro de calcular
    (O(peones) por peón), así que la ablación decide si se queda."""
    fila = casilla // LADO
    col = casilla % LADO
    d = 1 if color == 0 else -1
    dg = LADO - 1 - _avance_nb(fila, color)
    mejor_rival = LADO * 2
    for c in range(CASILLAS):
        if _estado_celda(tablero, c // LADO, c % LADO, color) != 2:
            continue
        fr = c // LADO
        cr = c % LADO
        dcol = cr - col
        if dcol < 0:
            dcol = -dcol
        if dcol > dg:
            continue  # no le alcanzan las columnas para cruzarse
        # Filas que le faltan al rival para ponerse delante de nuestro peón.
        filas = (fila + d * dg) - fr if d > 0 else fr - (fila + d * dg)
        if filas < 0:
            filas = -filas
        costo = filas if filas > dcol else dcol
        if costo < mejor_rival:
            mejor_rival = costo
    return float(mejor_rival - dg)


# --- Vector ancho (para entrenar) ------------------------------------------------------
@njit(cache=True)
def features_anchas_nb(tablero, puntos, color):
    """Vector de `N_FEATURES`, "propio menos rival". Offline: lo usa el entrenador."""
    f = np.zeros(N_FEATURES, np.float64)
    rival = 1 - color
    f[I_PUNTOS] = puntos[color] - puntos[rival]

    cap_propios = 0.0
    cap_rivales = 0.0
    apoyo_neto = 0.0
    libres = 0.0
    dist = 0.0
    pasado_p = 0.0
    pasado_r = 0.0
    col_p = np.zeros(LADO, np.int8)
    col_r = np.zeros(LADO, np.int8)
    avance_max_p = -1
    avance_max_r = -1
    material = 0.0
    exposicion = 0.0
    maxrow_color2, minrow_color1 = _filas_rivales_por_columna(tablero)
    for c in range(CASILLAS):
        v = tablero[c]
        if v == VACIA:
            continue
        mio = (v == color + 1)
        quien = color if mio else rival
        signo = 1.0 if mio else -1.0
        material += signo
        f[I_PATRONES + _patron(tablero, c, quien)] += signo
        a = _avance_nb(c // LADO, quien)
        f[I_FILAS + a] += signo
        peor, mejor = _amenazas_de(tablero, c, quien)
        cap_propios += signo * peor
        cap_rivales += signo * mejor
        apoyo_neto += signo * apoyo(tablero, c, quien)
        exposicion += signo * _exposicion_de(tablero, c, quien)
        es_libre = _es_libre_rapido(c, quien, maxrow_color2, minrow_color1)
        if es_libre:
            libres += signo
            dist += signo * _distancia_ajustada(tablero, c, quien)
        # Las tres de la familia `extra`, con la MISMA definición que `eval.py` de la
        # evaluación entregada (ver su `_features_numba`): `pasado` pesa por avance, las
        # columnas se cuentan una sola vez, y la carrera usa el peón más avanzado.
        if mio:
            col_p[c % LADO] = 1
            if a > avance_max_p:
                avance_max_p = a
            if es_libre:
                pasado_p += a
        else:
            col_r[c % LADO] = 1
            if a > avance_max_r:
                avance_max_r = a
            if es_libre:
                pasado_r += a
    f[I_AMENAZA] = cap_propios
    f[I_AMENAZA + 1] = cap_rivales
    f[I_AMENAZA + 2] = apoyo_neto
    f[I_LIBRES] = libres
    f[I_LIBRES + 1] = dist

    vacias_p = LADO - col_p.sum()
    vacias_r = LADO - col_r.sum()
    distancia_p = LADO if avance_max_p < 0 else LADO - 1 - avance_max_p
    distancia_r = LADO if avance_max_r < 0 else LADO - 1 - avance_max_r
    f[I_EXTRA] = pasado_p - pasado_r
    f[I_EXTRA + 1] = float(vacias_r - vacias_p)
    f[I_EXTRA + 2] = float(distancia_p - distancia_r)

    o = np.empty(256, np.int64)
    d = np.empty(256, np.int64)
    n_propio = generar(tablero, color, o, d)
    n_rival = generar(tablero, rival, o, d)
    f[I_MOVILIDAD] = float(n_propio - n_rival)

    f[I_MATERIAL] = material
    idx, signo_m = _marcador_indice_signo(int(puntos[color]), int(puntos[rival]))
    f[I_MARCADOR + idx] = signo_m
    f[I_EXPOSICION] = exposicion
    return f


# --- Evaluación acumulada (para jugar) -------------------------------------------------
@njit(cache=True)
def evaluar_nb(tablero, puntos, color, W):
    """Igual que `dot(W, features_anchas_nb(...))` pero sin armar el vector: acumula los
    pesos de las tablas peón por peón. Eso es lo que corre en cada hoja de la búsqueda."""
    rival = 1 - color
    # Banderas: una familia apagada se SALTA (no se calcula). Ver `empaquetar` en este mismo
    # archivo; `W` tiene que venir de ahí, con largo N_PESOS.
    usa_patrones = W[B_PATRONES] != 0.0
    usa_amenaza = W[B_AMENAZA] != 0.0
    usa_libres = W[B_LIBRES] != 0.0
    usa_extra = W[B_EXTRA] != 0.0
    usa_material = W[B_MATERIAL] != 0.0
    usa_marcador = W[B_MARCADOR] != 0.0
    usa_exposicion = W[B_EXPOSICION] != 0.0
    v = W[I_PUNTOS] * (puntos[color] - puntos[rival])
    pasado_p = 0.0
    pasado_r = 0.0
    col_p = np.zeros(LADO, np.int8)
    col_r = np.zeros(LADO, np.int8)
    avance_max_p = -1
    avance_max_r = -1
    material = 0.0
    exposicion = 0.0
    maxrow_color2 = np.empty(LADO, np.int64)
    minrow_color1 = np.empty(LADO, np.int64)
    if usa_libres or usa_extra:
        maxrow_color2, minrow_color1 = _filas_rivales_por_columna(tablero)
    for c in range(CASILLAS):
        t = tablero[c]
        if t == VACIA:
            continue
        mio = (t == color + 1)
        quien = color if mio else rival
        signo = 1.0 if mio else -1.0
        if usa_material:
            material += signo
        if usa_patrones:
            v += signo * W[I_PATRONES + _patron(tablero, c, quien)]
        a = _avance_nb(c // LADO, quien)
        v += signo * W[I_FILAS + a]
        if usa_amenaza:
            peor, mejor = _amenazas_de(tablero, c, quien)
            v += signo * (W[I_AMENAZA] * peor + W[I_AMENAZA + 1] * mejor
                           + W[I_AMENAZA + 2] * apoyo(tablero, c, quien))
        if usa_exposicion:
            exposicion += signo * _exposicion_de(tablero, c, quien)
        # `_es_libre` lo necesitan `libres` y `pasado`, así que se calcula una sola vez
        # cuando alguna de las dos está activa.
        libre = False
        if usa_libres or usa_extra:
            libre = _es_libre_rapido(c, quien, maxrow_color2, minrow_color1)
        if usa_libres and libre:
            v += signo * (W[I_LIBRES] + W[I_LIBRES + 1] * _distancia_ajustada(tablero, c, quien))
        if usa_extra:
            if mio:
                col_p[c % LADO] = 1
                if a > avance_max_p:
                    avance_max_p = a
                if libre:
                    pasado_p += a
            else:
                col_r[c % LADO] = 1
                if a > avance_max_r:
                    avance_max_r = a
                if libre:
                    pasado_r += a
    if usa_extra:
        vacias_p = LADO - col_p.sum()
        vacias_r = LADO - col_r.sum()
        distancia_p = LADO if avance_max_p < 0 else LADO - 1 - avance_max_p
        distancia_r = LADO if avance_max_r < 0 else LADO - 1 - avance_max_r
        v += W[I_EXTRA] * (pasado_p - pasado_r)
        v += W[I_EXTRA + 1] * float(vacias_r - vacias_p)
        v += W[I_EXTRA + 2] * float(distancia_p - distancia_r)
    if W[B_MOVILIDAD] != 0.0:
        o = np.empty(256, np.int64)
        d = np.empty(256, np.int64)
        v += W[I_MOVILIDAD] * float(generar(tablero, color, o, d)
                                     - generar(tablero, rival, o, d))
    if usa_material:
        v += W[I_MATERIAL] * material
    if usa_marcador:
        idx, signo_m = _marcador_indice_signo(int(puntos[color]), int(puntos[rival]))
        if signo_m != 0.0:
            v += W[I_MARCADOR + idx] * signo_m
    if usa_exposicion:
        v += W[I_EXPOSICION] * exposicion
    return v


# --- Referencia en Python (para verificar la de numba) ---------------------------------
def _estado_celda_py(tablero, fila, col, color):
    if not (0 <= fila < LADO and 0 <= col < LADO):
        return 0
    v = int(tablero[fila * LADO + col])
    if v == VACIA:
        return 0
    return 1 if v == color + 1 else 2


def _patron_py(tablero, casilla, color):
    fila, col = divmod(int(casilla), LADO)
    d = 1 if color == 0 else -1
    pesos = [1, 3, 9, 27, 81]
    celdas = [(fila + d, col - 1), (fila + d, col), (fila + d, col + 1),
              (fila - d, col - 1), (fila - d, col + 1)]
    return sum(w * _estado_celda_py(tablero, f, c, color)
               for w, (f, c) in zip(pesos, celdas, strict=True))


def _es_libre_py(tablero, casilla, color):
    fila, col = divmod(int(casilla), LADO)
    d = 1 if color == 0 else -1
    f = fila + d
    while 0 <= f < LADO:
        for c in range(col - 1, col + 2):
            if 0 <= c < LADO and _estado_celda_py(tablero, f, c, color) == 2:
                return False
        f += d
    return True


def _amenazas_de_py(tablero, casilla, color):
    fila, col = divmod(int(casilla), LADO)
    d = 1 if color == 0 else -1
    rival = 1 - color
    peor = mejor = 0.0
    for dc in (-1, 1):
        fr, cr = fila + d, col + dc
        if 0 <= fr < LADO and 0 <= cr < LADO and _estado_celda_py(tablero, fr, cr, color) == 2:
            peor = max(peor, float(prob_combate(int(apoyo(tablero, casilla, rival)),
                                                 int(apoyo(tablero, casilla, color)))))
            destino = fr * LADO + cr
            mejor = max(mejor, float(prob_combate(int(apoyo(tablero, destino, color)),
                                                   int(apoyo(tablero, destino, rival)))))
    return peor, mejor


def _exposicion_de_py(tablero, casilla, color):
    fila, col = divmod(int(casilla), LADO)
    d = 1 if color == 0 else -1
    rival = 1 - color
    peor = recaptura = 0.0
    for dc in (-1, 1):
        fr, cr = fila + d, col + dc
        if 0 <= fr < LADO and 0 <= cr < LADO and _estado_celda_py(tablero, fr, cr, color) == 2:
            p = float(prob_combate(int(apoyo(tablero, casilla, rival)),
                                   int(apoyo(tablero, casilla, color))))
            if p > peor:
                peor = p
                origen = fr * LADO + cr
                recaptura = float(prob_combate(int(apoyo(tablero, casilla, color)),
                                               int(apoyo(tablero, origen, rival))))
    return peor * (1.0 - recaptura)


def _marcador_indice_signo_py(i, j):
    if i == j:
        return 0, 0.0
    lo, hi = (i, j) if i < j else (j, i)
    signo = -1.0 if i < j else 1.0
    idx = {(0, 1): 0, (0, 2): 1, (1, 2): 2}[(lo, hi)]
    return idx, signo


def _distancia_ajustada_py(tablero, casilla, color):
    fila, col = divmod(int(casilla), LADO)
    d = 1 if color == 0 else -1
    dg = LADO - 1 - (fila if color == 0 else LADO - 1 - fila)
    mejor_rival = LADO * 2
    for c in range(CASILLAS):
        if _estado_celda_py(tablero, c // LADO, c % LADO, color) != 2:
            continue
        fr, cr = divmod(c, LADO)
        dcol = abs(cr - col)
        if dcol > dg:
            continue
        objetivo = fila + d * dg
        filas = abs(objetivo - fr)
        mejor_rival = min(mejor_rival, max(filas, dcol))
    return float(mejor_rival - dg)


def features_anchas_py(tablero, puntos, color):
    """Misma definición que `features_anchas_nb`, escrita aparte en Python: si las dos
    coinciden en posiciones al azar, la versión de numba hace lo que dice."""
    f = np.zeros(N_FEATURES)
    rival = 1 - color
    f[I_PUNTOS] = puntos[color] - puntos[rival]
    # Para la familia `extra`: se acumulan aparte porque dos de las tres NO son sumas por
    # peón (un conteo de columnas distintas y un máximo).
    pasado = {0: 0.0, 1: 0.0}
    columnas = {0: set(), 1: set()}
    avance_max = {0: -1, 1: -1}
    material = 0.0
    exposicion = 0.0
    for c in range(CASILLAS):
        v = int(tablero[c])
        if v == VACIA:
            continue
        mio = (v == color + 1)
        quien = color if mio else rival
        signo = 1.0 if mio else -1.0
        material += signo
        f[I_PATRONES + _patron_py(tablero, c, quien)] += signo
        a = c // LADO if quien == 0 else LADO - 1 - c // LADO
        f[I_FILAS + a] += signo
        peor, mejor = _amenazas_de_py(tablero, c, quien)
        f[I_AMENAZA] += signo * peor
        f[I_AMENAZA + 1] += signo * mejor
        f[I_AMENAZA + 2] += signo * int(apoyo(tablero, c, quien))
        exposicion += signo * _exposicion_de_py(tablero, c, quien)
        libre = _es_libre_py(tablero, c, quien)
        if libre:
            f[I_LIBRES] += signo
            f[I_LIBRES + 1] += signo * _distancia_ajustada_py(tablero, c, quien)
        lado = 0 if mio else 1          # 0 = propio, 1 = rival
        columnas[lado].add(c % LADO)
        avance_max[lado] = max(avance_max[lado], a)
        if libre:
            pasado[lado] += a
    vacias_p = LADO - len(columnas[0])
    vacias_r = LADO - len(columnas[1])
    dist_p = LADO if avance_max[0] < 0 else LADO - 1 - avance_max[0]
    dist_r = LADO if avance_max[1] < 0 else LADO - 1 - avance_max[1]
    f[I_EXTRA] = pasado[0] - pasado[1]
    f[I_EXTRA + 1] = float(vacias_r - vacias_p)
    f[I_EXTRA + 2] = float(dist_p - dist_r)
    o = np.empty(256, np.int64)
    d = np.empty(256, np.int64)
    f[I_MOVILIDAD] = float(generar(tablero, color, o, d) - generar(tablero, rival, o, d))
    f[I_MATERIAL] = material
    idx, signo_m = _marcador_indice_signo_py(int(puntos[color]), int(puntos[rival]))
    f[I_MARCADOR + idx] = signo_m
    f[I_EXPOSICION] = exposicion
    return f
