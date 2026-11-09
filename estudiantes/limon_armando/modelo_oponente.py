"""Modelo probabilístico de las jugadas del rival (logit condicional) y simulación guiada.

El modelo da a cada jugada legal a una probabilidad

    pi(a | s) = exp(theta · phi(s, a)) / suma_a' exp(theta · phi(s, a'))

con phi(s, a) un vector de características de la jugada vista desde quien mueve. Es una
regresión logística multinomial (sin capas ocultas: no es una red neuronal).

- `caracteristicas` calcula phi para una lista de jugadas (numba).
- `probabilidades` aplica el softmax y lo mezcla con una uniforme.
- `map_laplace` actualiza los pesos con las jugadas observadas del rival (Bayes con prior
  gaussiano y aproximación de Laplace).
- `reconstruir_jugada_rival` averigua qué jugó el rival comparando estados.
- `SimuladorRapido` juega partidas en las que cada bando elige jugadas con el modelo,
  con los pesos precalculados en una tabla (`llenar_tabla`).

Las reglas (generar jugadas, apoyo, probabilidad de combate, aplicar) están copiadas de
`sistema/motor/nucleo.py` con otro nombre, como permite docs/avanzado.md («Modificar el
motor»). La jugada que entrega el bot siempre sale de `estado.jugadas()` del motor oficial.
"""

import numpy as np
from numba import njit

LADO = 9
CASILLAS = 81
MAX_JUGADAS = 18 * 3
PUNTOS_PARA_GANAR = 3

NOMBRES = (
    "ataque",           # 1 si la jugada ataca
    "ventaja",          # (p - 0.5) si ataca, 0 si no
    "anota",            # 1 si llega a la última fila
    "avance",           # qué tan adelante queda el destino (0 a 1)
    "expuesto",         # 1 si el destino queda atacable por un peón rival
    "apoyo",            # apoyo del peón en el destino (0, 0.5 o 1)
    "defiende",         # 1 si ataca a un peón rival a 2 filas o menos de anotar
    "punta",            # 1 si mueve su peón más adelantado
    "recto",            # 1 si avanza recto
    "ataque_urgencia",  # ataque cuando el otro bando ya tiene 2 puntos
)
D = len(NOMBRES)


# --- Reglas (copiadas de sistema/motor/nucleo.py) ------------------------------------------
@njit
def _paso(color):
    return 1 if color == 0 else -1


@njit
def _apoyo(tablero, casilla, color, ignorar):
    """Peones propios en las dos diagonales de atrás de `casilla`, sin contar `ignorar`."""
    fila = casilla // LADO - _paso(color)
    if fila < 0 or fila >= LADO:
        return 0
    col = casilla % LADO
    pieza = color + 1
    total = 0
    if col > 0:
        c = fila * LADO + col - 1
        if tablero[c] == pieza and c != ignorar:
            total += 1
    if col < LADO - 1:
        c = fila * LADO + col + 1
        if tablero[c] == pieza and c != ignorar:
            total += 1
    return total


@njit
def _prob_combate(apoyo_atacante, apoyo_defensor):
    p = 0.5 + 0.15 * (apoyo_atacante - apoyo_defensor)
    return min(max(p, 0.2), 0.9)


@njit
def _aplicar(tablero, puntos, color, origen, destino, gana_ataque):
    tablero[origen] = 0
    if tablero[destino] != 0:
        if not gana_ataque:
            return
        tablero[destino] = 0
    meta = LADO - 1 if color == 0 else 0
    if destino // LADO == meta:
        puntos[color] += 1
    else:
        tablero[destino] = color + 1


@njit
def _progreso(fila, color):
    """Filas avanzadas desde la salida (0 a 8) para `color`."""
    return fila if color == 0 else LADO - 1 - fila


# --- Características --------------------------------------------------------------------------
# Todas las características menos «punta» dependen solo de un contexto local pequeño de la
# jugada (fila de destino, dirección, apoyos, si queda expuesta, urgencia). Por eso se pueden
# precalcular en una tabla de pesos para simular rápido (ver `llenar_tabla`).
N_CONTEXTOS = 9 * 3 * 2 * 3 * 3 * 2 * 3 * 2


@njit
def _indice_contexto(prog, giro, ataque, a_atq, a_def, expuesto, apoyo_d, urgencia):
    """Número único (0 a N_CONTEXTOS - 1) de un contexto local; giro es -1, 0 o 1."""
    i = prog * 3 + giro + 1
    i = i * 2 + ataque
    i = i * 3 + a_atq
    i = i * 3 + a_def
    i = i * 2 + expuesto
    i = i * 3 + apoyo_d
    return i * 2 + urgencia


@njit
def _contexto(tablero, color, o, d, urgencia):
    """Contexto local de la jugada o -> d de `color` (ver `_indice_contexto`).

    Escrita a mano con índices de casilla (sin llamar a `_apoyo`): se llama para cada jugada
    en cada turno de cada simulación, y así es ~3 veces más rápida.
    """
    fila_d = d // LADO
    col_d = d - fila_d * LADO
    adelante = LADO if color == 0 else -LADO
    prog = fila_d if color == 0 else LADO - 1 - fila_d
    giro = d - o - adelante
    mia = color + 1
    suya = 2 - color
    ataque = 0
    a_atq = 0
    a_def = 0
    if tablero[d] != 0:
        ataque = 1
        col_o = col_d - giro
        atras_o = o - adelante  # fila de atrás del atacante: sus apoyos
        if 0 <= atras_o < CASILLAS:
            if col_o > 0 and tablero[atras_o - 1] == mia:
                a_atq += 1
            if col_o < LADO - 1 and tablero[atras_o + 1] == mia:
                a_atq += 1
        atras_def = d + adelante  # fila de atrás del defensor (visto desde su lado)
        if 0 <= atras_def < CASILLAS:
            if col_d > 0 and tablero[atras_def - 1] == suya:
                a_def += 1
            if col_d < LADO - 1 and tablero[atras_def + 1] == suya:
                a_def += 1
    expuesto = 0
    apoyo_d = 0
    # Si anota, el peón sale del tablero: no queda expuesto ni con apoyo.
    if prog != LADO - 1:
        frente = d + adelante
        if col_d > 0 and tablero[frente - 1] == suya:
            expuesto = 1
        if col_d < LADO - 1 and tablero[frente + 1] == suya:
            expuesto = 1
        atras = d - adelante  # apoyos en el destino, sin contar el peón que se mueve
        if col_d > 0 and atras - 1 != o and tablero[atras - 1] == mia:
            apoyo_d += 1
        if col_d < LADO - 1 and atras + 1 != o and tablero[atras + 1] == mia:
            apoyo_d += 1
    return _indice_contexto(prog, giro, ataque, a_atq, a_def, expuesto, apoyo_d, urgencia)


@njit
def _phi_local(prog, giro, ataque, a_atq, a_def, expuesto, apoyo_d, urgencia, fila):
    """Llena `fila` con phi de un contexto local («punta» queda en 0: no es local)."""
    anota = prog == LADO - 1
    fila[0] = ataque
    fila[1] = _prob_combate(a_atq, a_def) - 0.5 if ataque else 0.0
    fila[2] = 1.0 if anota else 0.0
    fila[3] = prog / (LADO - 1)
    fila[4] = 0.0 if anota else expuesto
    fila[5] = 0.0 if anota else apoyo_d / 2.0
    # «defiende»: el peón atacado está a 2 filas o menos de anotar.
    fila[6] = 1.0 if ataque and LADO - 1 - prog >= LADO - 3 else 0.0
    fila[7] = 0.0
    fila[8] = 1.0 if giro == 0 else 0.0
    fila[9] = ataque * urgencia


@njit
def caracteristicas(tablero, color, puntos, origenes, destinos, m, F):
    """Llena F[:m] con las características de las jugadas (origenes[k], destinos[k])."""
    pieza = color + 1
    mas_adelante = -1
    for c in range(CASILLAS):
        if tablero[c] == pieza:
            p = _progreso(c // LADO, color)
            if p > mas_adelante:
                mas_adelante = p
    urgencia = 1 if puntos[1 - color] >= PUNTOS_PARA_GANAR - 1 else 0
    for k in range(m):
        o = origenes[k]
        d = destinos[k]
        i = _contexto(tablero, color, o, d, urgencia)
        _phi_desde_indice(i, F[k])
        F[k, 7] = 1.0 if _progreso(o // LADO, color) == mas_adelante else 0.0


@njit
def _phi_desde_indice(i, fila):
    """Inverso de `_indice_contexto`: phi local del contexto número i."""
    urgencia = i % 2
    i //= 2
    apoyo_d = i % 3
    i //= 3
    expuesto = i % 2
    i //= 2
    a_def = i % 3
    i //= 3
    a_atq = i % 3
    i //= 3
    ataque = i % 2
    i //= 2
    giro = i % 3 - 1
    prog = i // 3
    _phi_local(prog, giro, ataque, a_atq, a_def, expuesto, apoyo_d, urgencia, fila)


@njit
def llenar_tabla(theta, escala, W):
    """W[i] = exp(escala·theta·phi(contexto i)), dividido entre el máximo (no se desborda).

    Es el mismo logit, pero precalculado: al simular, el peso de una jugada se busca en la
    tabla en vez de calcular sus 10 características y un exponencial.
    shortcut: omite «punta» (uno de los pesos más chicos); si se le diera peso, habría que
    llevar en la simulación cuál es el peón más adelantado.
    """
    fila = np.zeros(D)
    maximo = -1e300
    for i in range(N_CONTEXTOS):
        _phi_desde_indice(i, fila)
        z = 0.0
        for j in range(D):
            z += theta[j] * fila[j]
        W[i] = escala * z
        maximo = max(maximo, W[i])
    for i in range(N_CONTEXTOS):
        W[i] = np.exp(W[i] - maximo)


# --- Simulación rápida con tabla de pesos -----------------------------------------------------
# Copia de `nucleo.simular` (lista de peones incremental y elección uniforme por rechazo), pero
# el bando guiado elige con probabilidad proporcional a W[contexto] (buscado en la tabla).
INTENTOS = 16  # intentos de (peón, dirección) al azar antes de enumerar todo


@njit
def _quitar(lista, cuantos, indice, color, casilla):
    i = indice[casilla]
    ultimo = lista[color, cuantos[color] - 1]
    lista[color, i] = ultimo
    indice[ultimo] = i
    cuantos[color] -= 1


@njit
def _proponer(tab, c, lista, cuantos):
    """Jugada legal uniforme por rechazo (como el motor); (-1, -1) si no la encontró."""
    pieza = c + 1
    avance = _paso(c) * LADO
    for _ in range(INTENTOS):
        if cuantos[c] == 0:
            break
        o = lista[c, np.random.randint(0, cuantos[c])]
        giro = np.random.randint(-1, 2)
        col = o % LADO + giro
        if col < 0 or col >= LADO:
            continue
        d = o + avance + giro
        v = tab[d]
        if v == pieza or (giro == 0 and v != 0):
            continue
        return o, d
    return -1, -1


@njit
def _enumerar(tab, c, lista, cuantos, Wc, urgencia, con_pesos, origenes, destinos, acumulado):
    """Todas las jugadas legales desde la lista de peones; con pesos, su suma acumulada."""
    pieza = c + 1
    avance = _paso(c) * LADO
    m = 0
    total = 0.0
    for k in range(cuantos[c]):
        o = lista[c, k]
        col = o % LADO
        for giro in range(-1, 2):
            if col + giro < 0 or col + giro >= LADO:
                continue
            d = o + avance + giro
            v = tab[d]
            if v == pieza or (giro == 0 and v != 0):
                continue
            origenes[m] = o
            destinos[m] = d
            total += Wc[_contexto(tab, c, o, d, urgencia)] if con_pesos else 1.0
            acumulado[m] = total
            m += 1
    return m, total


@njit
def _elegir(tab, c, lista, cuantos, Wc, guiado, eps, urgencia, origenes, destinos, acumulado):
    """Jugada de `c`: uniforme si no es guiado (o con prob. eps); si no, proporcional a W.

    (Muestrear por rechazo con propuestas uniformes no sirve aquí: los pesos varían tanto que
    se aceptaba menos del 1 % de las propuestas. Por eso se enumeran las jugadas.)
    """
    con_pesos = guiado and np.random.random() >= eps
    if not con_pesos:
        o, d = _proponer(tab, c, lista, cuantos)
        if o >= 0:
            return o, d
    m, total = _enumerar(tab, c, lista, cuantos, Wc, urgencia, con_pesos,
                         origenes, destinos, acumulado)
    if m == 0:
        return -1, -1
    u = np.random.random() * total
    for k in range(m):
        if u < acumulado[k]:
            return origenes[k], destinos[k]
    return origenes[m - 1], destinos[m - 1]


@njit
def _simular_tabla(tablero, puntos, color, n, semilla, para, W, guiado, eps, pasos_guiados,
                   tab, pts, lista, cuantos, indice, origenes, destinos, acumulado):
    """Fracción de n partidas que gana `para`; el bando c elige con la tabla W[c] si guiado[c].

    Solo las primeras `pasos_guiados` jugadas de cada simulación usan el modelo; las demás
    son al azar (como el motor).
    """
    np.random.seed(semilla)
    victorias = 0
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
        while True:
            urgencia = 1 if pts[1 - c] >= PUNTOS_PARA_GANAR - 1 else 0
            guia = guiado[c] and jugadas < pasos_guiados
            jugadas += 1
            o, d = _elegir(tab, c, lista, cuantos, W[c], guia, eps, urgencia,
                           origenes, destinos, acumulado)
            if o < 0:
                ganador = 1 - c  # sin jugadas: gana el otro
                break
            gana = True
            if tab[d] != 0:
                p = _prob_combate(_apoyo(tab, o, c, -1), _apoyo(tab, d, 1 - c, -1))
                gana = np.random.random() < p
                if gana:
                    _quitar(lista, cuantos, indice, 1 - c, d)
                else:
                    _quitar(lista, cuantos, indice, c, o)
            if gana:
                meta = LADO - 1 if c == 0 else 0
                if d // LADO == meta:
                    _quitar(lista, cuantos, indice, c, o)
                else:
                    i = indice[o]
                    lista[c, i] = d
                    indice[d] = i
            _aplicar(tab, pts, c, o, d, gana)
            if pts[c] >= PUNTOS_PARA_GANAR:
                ganador = c
                break
            c = 1 - c
        if ganador == para:
            victorias += 1
    return victorias / n


# --- Puente con el motor oficial (solo API pública) -------------------------------------------
COLORES = ("blancas", "negras")


def _casilla(nombre):
    """'e4' -> índice fila*9 + col, como en el motor."""
    return (int(nombre[1:]) - 1) * LADO + (ord(nombre[0]) - ord("a"))


def tablero_de(estado):
    """(tablero int8[81], puntos int64[2], color) a partir de la API pública de Estado."""
    tablero = np.zeros(CASILLAS, np.int8)
    for k, nombre_color in enumerate(COLORES):
        for nombre in estado.piezas(nombre_color):
            tablero[_casilla(nombre)] = k + 1
    puntos = np.array([estado.puntos["blancas"], estado.puntos["negras"]], np.int64)
    return tablero, puntos, COLORES.index(estado.turno)


def matriz_caracteristicas(estado, jugadas):
    """Matriz |jugadas| × D con phi(estado, jugada) para quien tiene el turno."""
    tablero, puntos, color = tablero_de(estado)
    m = len(jugadas)
    origenes = np.array([j.origen for j in jugadas], np.int64)
    destinos = np.array([j.destino for j in jugadas], np.int64)
    F = np.zeros((m, D))
    caracteristicas(tablero, color, puntos, origenes, destinos, m, F)
    return F


def probabilidades(theta, F, eps=0.0):
    """Softmax estable de F @ theta, mezclado con uniforme: (1 - eps) * pi + eps / m."""
    z = F @ theta
    e = np.exp(z - z.max())
    return (1.0 - eps) * e / e.sum() + eps / len(z)


class SimuladorRapido:
    """Simulación guiada por tablas: W[color] = pesos del logit de ese bando, ya precalculados.

    `poner_pesos` se llama cuando cambia theta (una vez por turno para el rival); simular no
    calcula características ni exponenciales.
    """

    def __init__(self, eps=0.1, pasos_guiados=10**6):
        self.eps = eps
        self.pasos_guiados = pasos_guiados  # jugadas guiadas al inicio de cada simulación
        self.W = np.ones((2, N_CONTEXTOS))
        self.guiado = np.zeros(2, np.bool_)
        self.tab = np.zeros(CASILLAS, np.int8)
        self.pts = np.zeros(2, np.int64)
        self.lista = np.zeros((2, CASILLAS), np.int64)
        self.cuantos = np.zeros(2, np.int64)
        self.indice = np.zeros(CASILLAS, np.int64)
        self.origenes = np.zeros(MAX_JUGADAS, np.int64)
        self.destinos = np.zeros(MAX_JUGADAS, np.int64)
        self.acumulado = np.zeros(MAX_JUGADAS)

    def poner_pesos(self, color, theta, escala=1.0):
        """Guía al bando `color` ("blancas"/"negras") con theta; theta None = juega al azar."""
        k = COLORES.index(color)
        self.guiado[k] = theta is not None
        if theta is not None:
            llenar_tabla(np.asarray(theta, np.float64), escala, self.W[k])

    def fraccion_tablero(self, tablero, puntos, turno, n, semilla, para):
        return _simular_tabla(
            tablero, puntos, turno, n, semilla % 2**32, COLORES.index(para),
            self.W, self.guiado, self.eps, self.pasos_guiados,
            self.tab, self.pts, self.lista, self.cuantos, self.indice,
            self.origenes, self.destinos, self.acumulado,
        )

    def fraccion(self, estado, n, semilla, para):
        """Fracción de las n partidas simuladas desde `estado` que gana `para`."""
        tablero, puntos, turno = tablero_de(estado)
        return self.fraccion_tablero(tablero, puntos, turno, n, semilla, para)


# --- Bayes: prior gaussiano + aproximación de Laplace -----------------------------------------
def _gradiente_y_precision(theta, mu0, prec0, observaciones):
    g = -prec0 @ (theta - mu0)
    H = prec0.copy()
    for F, b in observaciones:
        pi = probabilidades(theta, F)
        media = pi @ F
        g += F[b] - media
        H += (F * pi[:, None]).T @ F - np.outer(media, media)  # Cov_pi[phi]
    return g, H


PASO_MAXIMO = 1.0  # tope por coordenada de cada paso de Newton


def log_posterior(theta, mu0, prec0, observaciones):
    """Log-prior gaussiano + log-verosimilitud del logit (salvo una constante)."""
    total = -0.5 * (theta - mu0) @ prec0 @ (theta - mu0)
    for F, b in observaciones:
        z = F @ theta
        maximo = z.max()
        total += z[b] - maximo - np.log(np.exp(z - maximo).sum())
    return total


def map_laplace(mu0, prec0, observaciones, theta, pasos=6, tol=1e-6):
    """Media posterior (MAP) por Newton amortiguado y precisión evaluada en ella.

    observaciones: lista de (F_k, índice de la jugada que hizo el rival).
    Newton puro se pasa de largo con la logística (oscila); por eso cada paso se acota y se
    reduce a la mitad hasta que la log-posterior no empeore (búsqueda de línea).
    """
    actual = log_posterior(theta, mu0, prec0, observaciones)
    for _ in range(pasos):
        g, H = _gradiente_y_precision(theta, mu0, prec0, observaciones)
        paso = np.linalg.solve(H, g)
        mayor = np.abs(paso).max()
        if mayor > PASO_MAXIMO:
            paso = paso * (PASO_MAXIMO / mayor)
        t = 1.0
        while t > 1e-4:
            candidato = theta + t * paso
            valor = log_posterior(candidato, mu0, prec0, observaciones)
            if valor >= actual:
                break
            t /= 2
        else:
            break  # ningún paso mejora: ya estamos en el máximo (a la precisión numérica)
        theta, actual = candidato, valor
        if t * np.abs(paso).max() < tol:
            break
    _, H = _gradiente_y_precision(theta, mu0, prec0, observaciones)
    return theta, H


def reconstruir_jugada_rival(estado_previo, jugada_previa, estado):
    """Pares (estado del rival, jugada del rival) compatibles con llegar a `estado`.

    `estado_previo` es donde jugamos `jugada_previa` (None si el rival abrió la partida y
    `estado_previo` es la posición inicial). Puede haber más de un candidato (p. ej. anotar
    recto o en diagonal deja el mismo estado); quien llama decide qué hacer.
    """
    if jugada_previa is None:
        intermedios = [estado_previo]
    else:
        intermedios = [s for _, s in estado_previo.siguiente(jugada_previa)]
    clave = estado.clave
    candidatos = []
    for s in intermedios:
        if s.ganador() is not None or s.turno == estado.turno:
            continue
        for b in s.jugadas():
            if any(s2.clave == clave for _, s2 in s.siguiente(b)):
                candidatos.append((s, b))
    return candidatos
