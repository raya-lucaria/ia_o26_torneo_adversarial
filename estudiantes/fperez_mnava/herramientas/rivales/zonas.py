"""Bot «expectimax»: búsqueda con valor esperado en los ataques y evaluación = probabilidad de ganar.

Idea (ver INFORME): el juego es de suma cero, con información perfecta y azar solo en los ataques.
El valor de un estado es P(ganar) con juego óptimo; aquí se aproxima en las hojas con una regresión
logística (suma lineal de rasgos + sigmoide) y se propaga con max / min / valor esperado.

- Búsqueda: negamax con poda alfa-beta y nodos de azar (poda tipo Star1), profundización iterativa.
- Representación propia (lista de 81 casillas, hacer/deshacer en sitio): el `Estado` oficial es
  demasiado lento para buscar. La jugada final se revisa contra `estado.jugadas()`.
- PODA_ZONAS = True activa el experimento de mitades: tus jugadas se restringen a las columnas a-e
  o e-i (la e es de ambas), el rival responde con todas las suyas y la evaluación ve todo el tablero.
- Los pesos de PESOS son una primera estimación a mano: hay que ajustarlos con partidas (arena.py).
"""

import math
import time

from motor import reportar

PODA_ZONAS = True  # experimento de las dos mitades (ver arriba)
PROF_MAX = 12
RESERVA = 0.12  # fracción del reloj que nunca se gasta pensando
MOVS_ESTIMADOS = 45  # jugadas propias que esperamos en una partida (para repartir el reloj)

# z = suma de peso * rasgo (desde el punto de vista de quien mueve); P(ganar) = sigmoide(z)
PESOS = {
    "puntos": 3.0,  # puntos anotados (cada uno vale casi un tercio de la partida)
    "carrera": 0.30,  # jugadas que faltan a los peones más avanzados para anotar lo que falta
    "mueve": 0.20,  # ventaja de tener el turno
    "corredor": 0.12,  # peones sin defensores por delante, más cerca de la meta = más
    "carriles": 0.25,  # corredores en zonas distintas del tablero (izq./centro/der.)
    "apoyo": 0.04,  # apoyos totales (diagonales de atrás ocupadas por peones propios)
    "amenaza_mia": 1.2,  # ataques disponibles para quien mueve, ponderados por p - 0.5
    "amenaza_rival": 0.6,  # lo mismo para el rival (le tocará tras nuestra jugada)
    "material": 0.06,
}


# --- Tablero: lista de 81 enteros (0 vacía, 1 blanca, 2 negra), casilla = fila*9 + col ---------------


def _desde_estado(estado):
    tab = [0] * 81
    pos = (set(), set())
    for k, nombre in enumerate(("blancas", "negras")):
        for c in estado.piezas(nombre):
            i = (int(c[1]) - 1) * 9 + ord(c[0]) - 97
            tab[i] = k + 1
            pos[k].add(i)
    pts = estado.puntos
    return tab, pos, [pts["blancas"], pts["negras"]]


def _nombre(i):
    return "abcdefghi"[i % 9] + str(i // 9 + 1)


def _apoyo(tab, c, lado):
    """Peones propios en las dos diagonales de atrás de la casilla c."""
    fila = c // 9 - (1 if lado == 0 else -1)
    if fila < 0 or fila > 8:
        return 0
    col = c % 9
    pieza = lado + 1
    base = fila * 9
    n = 0
    if col > 0 and tab[base + col - 1] == pieza:
        n += 1
    if col < 8 and tab[base + col + 1] == pieza:
        n += 1
    return n


def _generar(tab, pos, lado):
    """Jugadas legales como (origen, destino, p); p = 1.0 si no es ataque."""
    pieza = lado + 1
    av = 9 if lado == 0 else -9
    res = []
    for c in pos[lado]:
        col = c % 9
        f = c + av
        for dc in (-1, 0, 1):
            if (dc == -1 and col == 0) or (dc == 1 and col == 8):
                continue
            d = f + dc
            v = tab[d]
            if dc == 0:
                if v == 0:
                    res.append((c, d, 1.0))
            elif v == 0:
                res.append((c, d, 1.0))
            elif v != pieza:
                p = 0.5 + 0.15 * (_apoyo(tab, c, lado) - _apoyo(tab, d, 1 - lado))
                res.append((c, d, min(max(p, 0.2), 0.9)))
    return res


def _ordenar(jugadas, lado):
    """Primero anotar, luego los ataques con mejor p, luego avanzar."""
    meta = 8 if lado == 0 else 0

    def clave(j):
        o, d, p = j
        k = 10.0 if d // 9 == meta else 0.0
        if p < 1.0:
            k += 2.0 + 4.0 * (p - 0.5)
        return -(k + 0.1 * (d // 9 if lado == 0 else 8 - d // 9))

    return sorted(jugadas, key=clave)


def _hacer(tab, pos, pts, lado, o, d, gana):
    """Aplica la jugada en sitio y regresa lo necesario para deshacerla."""
    pieza = lado + 1
    capturada = tab[d]
    tab[o] = 0
    pos[lado].discard(o)
    if capturada:
        if not gana:  # sale el atacante; el defensor se queda
            return (o, d, capturada, False, False)
        pos[1 - lado].discard(d)
        tab[d] = 0
    if d // 9 == (8 if lado == 0 else 0):
        pts[lado] += 1  # anota y sale del tablero
        return (o, d, capturada, gana, True)
    tab[d] = pieza
    pos[lado].add(d)
    return (o, d, capturada, gana, False)


def _deshacer(tab, pos, pts, lado, u):
    o, d, capturada, gana, anota = u
    pieza = lado + 1
    if anota:
        pts[lado] -= 1
    elif capturada == 0 or gana:
        tab[d] = 0
        pos[lado].discard(d)
    if capturada and gana:
        tab[d] = capturada
        pos[1 - lado].add(d)
    tab[o] = pieza
    pos[lado].add(o)


# --- Evaluación --------------------------------------------------------------------------------------


def _rasgos(tab, pos, pts, c):
    av = 9 if c == 0 else -9
    rival = 2 - c  # valor de un peón del rival en el tablero
    necesito = 3 - pts[c]
    dists, cand = [], []
    apoyo_total = 0
    amenaza = 0.0
    for s in pos[c]:
        fila, col = s // 9, s % 9
        dist = 8 - fila if c == 0 else fila
        dists.append(dist)
        a = _apoyo(tab, s, c)
        apoyo_total += a
        f = s + av
        if col > 0 and tab[f - 1] == rival:
            p = 0.5 + 0.15 * (a - _apoyo(tab, f - 1, 1 - c))
            if p > 0.5:
                amenaza += min(p, 0.9) - 0.5
        if col < 8 and tab[f + 1] == rival:
            p = 0.5 + 0.15 * (a - _apoyo(tab, f + 1, 1 - c))
            if p > 0.5:
                amenaza += min(p, 0.9) - 0.5
        if dist <= 5:
            cand.append((dist, col, fila))
    dists.sort()
    carrera = sum(dists[:necesito]) + 12 * max(0, necesito - len(dists))
    libres = []
    for dist, col, fila in cand:  # libre = ningún peón rival adelante dentro del cono
        libre = True
        for e in pos[1 - c]:
            er = e // 9
            if (er > fila if c == 0 else er < fila) and abs(e % 9 - col) <= abs(er - fila):
                libre = False
                break
        if libre:
            libres.append((dist, col))
    libres.sort()
    mejores = libres[: max(necesito, 0)]
    corredor = sum(6 - d for d, _ in mejores)
    carriles = len({col // 3 for _, col in mejores})
    return carrera, corredor, carriles, apoyo_total, amenaza, len(pos[c])


def _evaluar(tab, pos, pts, lado):
    """P(ganar) para quien tiene el turno (`lado`), en (0, 1)."""
    otro = 1 - lado
    cl, rl, ll, sl, al, ml = _rasgos(tab, pos, pts, lado)
    co, ro, lo, so, ao, mo = _rasgos(tab, pos, pts, otro)
    w = PESOS
    z = (
        w["puntos"] * (pts[lado] - pts[otro])
        + w["carrera"] * (co - cl)
        + w["mueve"]
        + w["corredor"] * (rl - ro)
        + w["carriles"] * (ll - lo)
        + w["apoyo"] * (sl - so)
        + w["amenaza_mia"] * al
        - w["amenaza_rival"] * ao
        + w["material"] * (ml - mo)
    )
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# --- Búsqueda ----------------------------------------------------------------------------------------


class _Tiempo(Exception):
    """Se acabó el tiempo de esta jugada: se usa el resultado de la profundidad anterior."""


class _Buscador:
    def __init__(self, tab, pos, pts, yo):
        self.tab, self.pos, self.pts, self.yo = tab, pos, pts, yo
        self.nodos = 0
        self.limite = float("inf")
        self.cols = None  # columnas permitidas para NUESTRAS jugadas (experimento de mitades)

    def _jugadas(self, lado):
        js = _ordenar(_generar(self.tab, self.pos, lado), lado)
        if self.cols is not None and lado == self.yo:
            filtradas = [j for j in js if j[0] % 9 in self.cols or j[2] < 1.0]
            if filtradas:
                js = filtradas
        return js

    def valor(self, lado, prof, alfa, beta):
        """P(ganar) de `lado` (que mueve), fail-soft dentro de la ventana (alfa, beta)."""
        self.nodos += 1
        if not self.nodos & 127 and time.perf_counter() > self.limite:
            raise _Tiempo
        if prof == 0:
            return _evaluar(self.tab, self.pos, self.pts, lado)
        js = self._jugadas(lado)
        if not js:
            return 0.0  # sin jugadas: pierde
        mejor = -1.0
        for o, d, p in js:
            mv = self.valor_jugada(lado, prof, o, d, p, max(alfa, mejor), beta)
            if mv > mejor:
                mejor = mv
                if mejor >= beta:
                    break
        return mejor

    def valor_jugada(self, lado, prof, o, d, p, alfa, beta):
        """Valor esperado de una jugada para `lado`; si es ataque, promedia los dos resultados."""
        tab, pos, pts = self.tab, self.pos, self.pts
        if p >= 1.0:
            u = _hacer(tab, pos, pts, lado, o, d, True)
            mv = 1.0 if pts[lado] >= 3 else 1.0 - self.valor(1 - lado, prof - 1, 1.0 - beta, 1.0 - alfa)
            _deshacer(tab, pos, pts, lado, u)
            return mv
        ventana_a, ventana_b = (alfa - (1.0 - p)) / p, beta / p  # ventana del resultado «gana»
        u = _hacer(tab, pos, pts, lado, o, d, True)
        a = 1.0
        if pts[lado] < 3:
            a = 1.0 - self.valor(1 - lado, prof - 1, 1.0 - ventana_b, 1.0 - ventana_a)
        _deshacer(tab, pos, pts, lado, u)
        if a >= ventana_b:
            return p * a  # cota inferior ya >= beta
        if a <= ventana_a:
            return p * a + (1.0 - p)  # cota superior ya <= alfa
        a2, b2 = (alfa - p * a) / (1.0 - p), (beta - p * a) / (1.0 - p)
        u = _hacer(tab, pos, pts, lado, o, d, False)
        b = 1.0 - self.valor(1 - lado, prof - 1, 1.0 - b2, 1.0 - a2)
        _deshacer(tab, pos, pts, lado, u)
        return p * a + (1.0 - p) * b

    def raiz(self, prof, previo):
        """Mejor jugada (o, d), su valor y los valores/cotas de cada jugada."""
        js = self._jugadas(self.yo)
        if previo:
            js.sort(key=lambda j: -previo.get((j[0], j[1]), -2.0))
        mejor, mov, vals = -1.0, None, {}
        for o, d, p in js:
            mv = self.valor_jugada(self.yo, prof, o, d, p, mejor, 1.0 + 1e-9)
            vals[(o, d)] = mv
            if mv > mejor:
                mejor, mov = mv, (o, d)
            if mejor >= 1.0:
                break
        return mov, mejor, vals


# --- Interfaz del torneo -----------------------------------------------------------------------------


def preparar(config):
    global reserva, n_mov
    reserva = RESERVA * config.reloj_s
    n_mov = 0


def _decidir(estado, tiempo_restante):
    t0 = time.perf_counter()
    tab, pos, pts = _desde_estado(estado)
    yo = 0 if estado.turno == "blancas" else 1
    disponible = tiempo_restante - reserva
    asignado = max(0.02, disponible / max(8, MOVS_ESTIMADOS - n_mov))
    b = _Buscador(tab, pos, pts, yo)
    zonas = [set(range(0, 5)), set(range(4, 9))] if PODA_ZONAS else [None]
    mejor_mov, mejor_val, prof, previo = None, -1.0, 0, {}
    for p in range(1, PROF_MAX + 1):
        b.limite = float("inf") if p == 1 else t0 + asignado  # la capa 1 siempre se completa
        try:
            resultado = (None, -2.0, {})
            vals_total = {}
            for cols in zonas:
                b.cols = cols
                mov, val, vals = b.raiz(p, previo)
                vals_total.update(vals)
                if mov is not None and val > resultado[1]:
                    resultado = (mov, val, vals)
            if resultado[0] is None:
                break
        except _Tiempo:
            break
        mejor_mov, mejor_val, prof, previo = resultado[0], resultado[1], p, vals_total
        if mejor_val >= 1.0 or time.perf_counter() - t0 > asignado * 0.5:
            break  # ganamos seguro, o la siguiente capa ya no cabe
    reportar(prof=prof, nodos=b.nodos, eval=round(mejor_val, 3))
    return mejor_mov


def jugar(estado, tiempo_restante):
    global n_mov
    legales = [str(j) for j in estado.jugadas()]
    try:
        o, d = _decidir(estado, tiempo_restante)
        texto = _nombre(o) + "-" + _nombre(d)
    except Exception:  # noqa: BLE001  (si algo falla, jugamos legal en vez de perder)
        texto = legales[0]
    if texto not in legales:
        texto = legales[0]
    n_mov += 1
    return texto
