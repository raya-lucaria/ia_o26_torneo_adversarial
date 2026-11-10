"""Expectiminimax con corte, escrito en numba sobre las reglas del motor oficial.

Las reglas (jugadas legales, combate, puntos) vienen de `motor.nucleo`; aquí
solo vive la búsqueda y la evaluación. Forma negamax: todo valor es desde el
lado que mueve y vive en [-1, 1] (1 = gano seguro, -1 = pierdo seguro).

- Nodos MAX/MIN con poda alfa-beta.
- Cada ataque es un nodo de AZAR: p·V(gana) + (1 − p)·V(pierde), con la poda
  Star1 de Ballard (1983): la cota |V| ≤ 1 permite cortar sin buscar el
  segundo resultado del dado.
- Búsqueda de quietud en las hojas (anotar y ataques con p ≥ 0.5).
- Tabla de transposición con la clave Zobrist del motor, jugadas asesinas
  (killers) e historial para ordenar.
- La búsqueda se detiene sola al agotar un presupuesto de nodos; el lado
  Python lo calcula a partir del reloj (numba no puede leer el reloj).
"""

import numpy as np
from numba import njit
from motor import nucleo

LADO = nucleo.LADO
CASILLAS = nucleo.CASILLAS
MAX_JUGADAS = nucleo.MAX_JUGADAS
MAX_PLY = 64
QUIETUD = 4  # plies máximos de quietud
TAM_TABLA = 1 << 20
TAM_EVAL = 1 << 18   # caché de evaluaciones (clave → valor)
GANAR = 1.0
EXACTO, PISO, TECHO = 0, 1, 2

# Rasgos de la evaluación, siempre propio − rival desde el lado que mueve.
TABLA = 0      # 8 avances × 5 columnas (doblando el tablero por el centro)
APOYO1 = 40    # peones con al menos un apoyo
APOYO2 = 41    # peones con dos apoyos
AMENAZA = 42   # peones que el rival puede atacar con p ≥ 0.5
AMENAZA_GRAVE = 43  # ... con p ≥ 0.65
CORREDOR = 44  # 5 rasgos: peones sin rival que los frene, avance 3..7
PUNTOS = 49    # diferencia de puntos
DOS_MIOS = 50  # tengo 2 puntos
DOS_SUYOS = 51  # el rival tiene 2 puntos
ATAQUES = 52   # Σ (p − 0.5) de mis ataques posibles menos los del rival
TURNO = 53     # siempre 1: ventaja de mover
SEGURO = 54    # 3 rasgos: peones en avance 5..7 que ningún rival puede atacar
COLGADO = 57   # peones atacables con p ≥ 0.5 y sin apoyo
FALANGE = 58   # pares de peones propios lado a lado
BLOQUEADO = 59  # peones con un rival justo enfrente
MOVILIDAD = 60  # jugadas posibles / 10
CARRERA_GANO = 61    # mi corredor más cercano llega antes (o igual, porque muevo yo)
CARRERA_PIERDO = 62  # el corredor rival llega antes
CASI_CORREDOR = 63   # 3 rasgos: peones con un solo rival que los puede frenar, avance 4..6
CARRERA_MARGEN = 66  # (llegada rival − la mía) / 3, recortado a ±1, si los dos tienen corredor
MATERIAL_FINAL = 67  # (peones propios − rivales) × (1 − piezas / 36): el material pesa más al final
DOS_Y_CERCA = 68     # tengo 2 puntos y un peón en avance ≥ 5, menos lo mismo del rival
RASGOS = 69
SIN_CORREDOR = 99


def pesos_iniciales():
    """Punto de partida a mano (en logits); el entrenamiento los reemplaza."""
    w = np.zeros(RASGOS)
    for avance in range(8):
        valor = 0.25 + 0.02 * avance * avance
        for col in range(5):
            w[TABLA + avance * 5 + col] = valor
    w[APOYO1], w[APOYO2] = 0.05, 0.05
    w[AMENAZA], w[AMENAZA_GRAVE] = -0.1, -0.3
    for i, v in enumerate((0.1, 0.2, 0.4, 0.7, 1.0)):
        w[CORREDOR + i] = v
    w[PUNTOS], w[DOS_MIOS], w[DOS_SUYOS] = 1.5, 0.8, -0.8
    w[ATAQUES], w[TURNO] = 0.5, 0.1
    return w


@njit(cache=True)
def avance_de(casilla, color):
    fila = casilla // LADO
    return fila if color == 0 else LADO - 1 - fila


@njit(cache=True)
def rasgos(tab, pts, c, f):
    """Llena `f` con los rasgos del tablero desde el lado `c` que mueve."""
    f[:] = 0.0
    carrera = np.full(2, SIN_CORREDOR)
    propios = 0  # peones del lado que mueve
    rivales = 0
    lejos_propio = 0  # avance máximo de cada lado
    lejos_rival = 0
    for s in range(CASILLAS):
        v = tab[s]
        if v == 0:
            continue
        dueno = v - 1
        signo = 1.0 if dueno == c else -1.0
        fila = s // LADO
        col = s % LADO
        avance = avance_de(s, dueno)
        if dueno == c:
            propios += 1
            lejos_propio = max(lejos_propio, avance)
        else:
            rivales += 1
            lejos_rival = max(lejos_rival, avance)
        f[TABLA + avance * 5 + min(col, LADO - 1 - col)] += signo
        apoyos = nucleo.apoyo(tab, s, dueno)
        if apoyos >= 1:
            f[APOYO1] += signo
        if apoyos == 2:
            f[APOYO2] += signo
        # Rivales que pueden atacar este peón: en diagonal, una fila más adelante.
        frente = fila + nucleo.paso(dueno)
        if col < LADO - 1 and tab[s + 1] == v:
            f[FALANGE] += signo
        if 0 <= frente < LADO:
            delante = tab[frente * LADO + col]
            if delante == 0:
                f[MOVILIDAD] += 0.1 * signo
            elif delante == 2 - dueno:
                f[BLOQUEADO] += signo
            for dc in (-1, 1):
                cc = col + dc
                if 0 <= cc < LADO and tab[frente * LADO + cc] != v:
                    f[MOVILIDAD] += 0.1 * signo
        peor = 0.0
        if 0 <= frente < LADO:
            for dc in (-1, 1):
                cc = col + dc
                if 0 <= cc < LADO and tab[frente * LADO + cc] == 2 - dueno:
                    a = frente * LADO + cc
                    p = nucleo.prob_combate(nucleo.apoyo(tab, a, 1 - dueno), apoyos)
                    # Ese mismo par también es un ataque posible de este peón.
                    f[ATAQUES] -= signo * (p - 0.5)
                    if p > peor:
                        peor = p
        if peor >= 0.65:
            f[AMENAZA_GRAVE] += signo
        elif peor >= 0.5:
            f[AMENAZA] += signo
        if peor >= 0.5 and apoyos == 0:
            f[COLGADO] += signo
        if peor == 0.0 and avance >= 5:
            f[SEGURO + avance - 5] += signo
        if avance >= 3:
            # Rivales en el cono de delante (hasta 2: solo importa 0, 1 o más).
            frenos = 0
            paso = nucleo.paso(dueno)
            r = fila + paso
            while 0 <= r < LADO and frenos < 2:
                for cc in range(max(0, col - 1), min(LADO, col + 2)):
                    if tab[r * LADO + cc] == 2 - dueno:
                        frenos += 1
                r += paso
            if frenos == 1 and 4 <= avance <= 6:
                f[CASI_CORREDOR + avance - 4] += signo
            if frenos == 0:
                f[CORREDOR + avance - 3] += signo
                carrera[dueno] = min(carrera[dueno], LADO - 1 - avance)
    f[PUNTOS] = pts[c] - pts[1 - c]
    f[DOS_MIOS] = 1.0 if pts[c] == 2 else 0.0
    f[DOS_SUYOS] = 1.0 if pts[1 - c] == 2 else 0.0
    f[TURNO] = 1.0
    if carrera[c] < SIN_CORREDOR and carrera[1 - c] < SIN_CORREDOR:
        f[CARRERA_MARGEN] = min(1.0, max(-1.0, (carrera[1 - c] - carrera[c]) / 3.0))
    f[MATERIAL_FINAL] = (propios - rivales) * (1.0 - (propios + rivales) / 36.0)
    if pts[c] == 2 and lejos_propio >= 5:
        f[DOS_Y_CERCA] += 1.0
    if pts[1 - c] == 2 and lejos_rival >= 5:
        f[DOS_Y_CERCA] -= 1.0
    if carrera[c] < SIN_CORREDOR and carrera[c] <= carrera[1 - c]:
        f[CARRERA_GANO] = 1.0
    elif carrera[1 - c] < carrera[c]:
        f[CARRERA_PIERDO] = 1.0


@njit(cache=True)
def clave_hijo(h, tab, hijo, pt, pts_hijo, c, o, d, zp, zt, zpu):
    """Clave Zobrist del hijo a partir de la del padre: solo cambian el origen, el destino,
    los puntos de quien movió y el turno (igual que nucleo.clave, sin recorrer el tablero)."""
    for s in (o, d):
        if tab[s] != 0:
            h ^= zp[tab[s] - 1, s]
        if hijo[s] != 0:
            h ^= zp[hijo[s] - 1, s]
    if pts_hijo[c] != pt[c]:
        h ^= zpu[c, pt[c]] ^ zpu[c, pts_hijo[c]]
    return h ^ zt


@njit(cache=True)
def evaluar(tab, pts, c, w, f):
    rasgos(tab, pts, c, f)
    z = 0.0
    for i in range(RASGOS):
        z += w[i] * f[i]
    return np.tanh(0.5 * z)  # = 2·σ(z) − 1


class Contexto:
    """Arreglos de trabajo que se reutilizan entre búsquedas."""

    def __init__(self, pesos):
        w = np.zeros(RASGOS)
        w[:len(pesos)] = pesos  # pesos de versiones con menos rasgos: el resto vale 0
        self.w = w
        self.f = np.zeros(RASGOS)
        self.tabs = np.zeros((MAX_PLY + QUIETUD + 2, CASILLAS), np.int8)
        self.pts = np.zeros((MAX_PLY + QUIETUD + 2, 2), np.int64)
        self.origenes = np.zeros((MAX_PLY + QUIETUD + 2, MAX_JUGADAS), np.int64)
        self.destinos = np.zeros((MAX_PLY + QUIETUD + 2, MAX_JUGADAS), np.int64)
        self.orden = np.zeros((MAX_PLY + QUIETUD + 2, MAX_JUGADAS), np.float64)
        self.t_clave = np.zeros(TAM_TABLA, np.uint64)
        self.t_valor = np.zeros(TAM_TABLA, np.float64)
        self.t_prof = np.full(TAM_TABLA, -1, np.int8)
        self.t_tipo = np.zeros(TAM_TABLA, np.int8)
        self.t_jugada = np.full(TAM_TABLA, -1, np.int32)
        # La evaluación solo depende de la posición, así que el caché no se limpia nunca.
        self.e_clave = np.zeros(TAM_EVAL, np.uint64)
        self.e_valor = np.zeros(TAM_EVAL, np.float64)
        self.asesinas = np.full((MAX_PLY + QUIETUD + 2, 2), -1, np.int32)
        self.historial = np.zeros((2, CASILLAS * CASILLAS), np.float64)
        # [nodos, límite de nodos, abortado, mejor jugada de la raíz, plies de quietud]
        self.control = np.zeros(6, np.int64)
        self.control[4] = QUIETUD

    def limpiar(self):
        self.t_clave[:] = 0
        self.t_prof[:] = -1
        self.asesinas[:] = -1
        self.historial[:] = 0.0

    def args(self):
        return (self.w, self.f, self.tabs, self.pts, self.origenes, self.destinos,
                self.orden, self.t_clave, self.t_valor, self.t_prof, self.t_tipo,
                self.t_jugada, self.asesinas, self.historial, self.control,
                nucleo.Z_PIEZAS, nucleo.Z_TURNO, nucleo.Z_PUNTOS, self.e_clave, self.e_valor)


@njit(cache=True)
def negamax(ply, c, prof, alfa, beta, w, f, tabs, pts, origenes, destinos, orden,
            t_clave, t_valor, t_prof, t_tipo, t_jugada, asesinas, historial,
            control, zp, zt, zpu, e_clave, e_valor, h):
    """Valor de tabs[ply]/pts[ply] para el lado `c` que mueve, ventana (alfa, beta)."""
    control[0] += 1
    if control[0] >= control[1]:
        control[2] = 1
    if control[2]:
        return 0.0
    tab = tabs[ply]
    pt = pts[ply]
    if pt[1 - c] >= nucleo.PUNTOS_PARA_GANAR:
        return -(GANAR - 1e-4 * ply)

    quietud = prof <= 0
    if quietud:
        ie = np.int64(h & np.uint64(TAM_EVAL - 1))
        if e_clave[ie] == h:
            propio = e_valor[ie]
        else:
            propio = evaluar(tab, pt, c, w, f)
            e_clave[ie] = h
            e_valor[ie] = propio
        if propio >= beta or prof <= -control[4]:
            return propio
        if propio > alfa:
            alfa = propio
        mejor = propio
    else:
        mejor = -2.0

    m = nucleo.generar(tab, c, origenes[ply], destinos[ply])
    if m == 0:
        return -(GANAR - 1e-4 * ply)

    # Tabla de transposición (solo fuera de la quietud).
    jugada_tabla = -1
    indice = np.int64(0)
    clave = np.uint64(0)
    alfa0 = alfa
    if not quietud:
        clave = h
        indice = np.int64(clave & np.uint64(TAM_TABLA - 1))
        if t_clave[indice] == clave:
            jugada_tabla = t_jugada[indice]
            # En la raíz la tabla solo ordena: hay que regresar una jugada.
            if t_prof[indice] >= prof and ply > 0:
                v = t_valor[indice]
                tipo = t_tipo[indice]
                if tipo == EXACTO:
                    return v
                if tipo == PISO and v >= beta:
                    return v
                if tipo == TECHO and v <= alfa:
                    return v

    meta = LADO - 1 if c == 0 else 0
    os_ = origenes[ply]
    ds = destinos[ply]
    sc = orden[ply]
    for k in range(m):
        o = os_[k]
        d = ds[k]
        cod = o * CASILLAS + d
        ataque = tab[d] != 0
        anota = d // LADO == meta
        if quietud and not anota and not (ataque and nucleo.prob_jugada(tab, c, o, d) >= 0.5):
            sc[k] = -1e18  # en quietud solo jugadas ruidosas
            continue
        if cod == jugada_tabla:
            s = 1e12
        elif anota:
            s = 1e10 + (0.0 if ataque else 1.0)
        elif ataque:
            s = 1e8 * nucleo.prob_jugada(tab, c, o, d)
        elif cod == asesinas[ply, 0] or cod == asesinas[ply, 1]:
            s = 1e7
        else:
            s = historial[c, cod]
        sc[k] = s

    hijo = ply + 1
    mejor_jugada = -1
    hechas = 0
    for _ in range(m):
        # Selección: la mejor que falte por mirar.
        k = 0
        for i in range(1, m):
            if sc[i] > sc[k]:
                k = i
        if sc[k] <= -1e17:
            break
        orden_k = sc[k]
        sc[k] = -1e18
        o = os_[k]
        d = ds[k]
        hechas += 1
        if tab[d] != 0:
            p = nucleo.prob_jugada(tab, c, o, d)
            # Primero el resultado más probable; luego el otro con Star1.
            gana1 = p >= 0.5
            p1 = p if gana1 else 1.0 - p
            p2 = 1.0 - p1
            a1 = max(-GANAR, (alfa - p2 * GANAR) / p1)
            b1 = min(GANAR, (beta + p2 * GANAR) / p1)
            tabs[hijo, :] = tab
            pts[hijo, :] = pt
            nucleo.aplicar(tabs[hijo], pts[hijo], c, o, d, gana1)
            hh = clave_hijo(h, tab, tabs[hijo], pt, pts[hijo], c, o, d, zp, zt, zpu)
            x1 = -negamax(hijo, 1 - c, prof - 1, -b1, -a1, w, f, tabs, pts, origenes,
                          destinos, orden, t_clave, t_valor, t_prof, t_tipo, t_jugada,
                          asesinas, historial, control, zp, zt, zpu, e_clave, e_valor, hh)
            if control[2]:
                return 0.0
            if x1 <= a1:
                v = p1 * x1 + p2 * GANAR
            elif x1 >= b1:
                v = p1 * x1 - p2 * GANAR
            else:
                a2 = max(-GANAR, (alfa - p1 * x1) / p2)
                b2 = min(GANAR, (beta - p1 * x1) / p2)
                tabs[hijo, :] = tab
                pts[hijo, :] = pt
                nucleo.aplicar(tabs[hijo], pts[hijo], c, o, d, not gana1)
                hh = clave_hijo(h, tab, tabs[hijo], pt, pts[hijo], c, o, d, zp, zt, zpu)
                x2 = -negamax(hijo, 1 - c, prof - 1, -b2, -a2, w, f, tabs, pts, origenes,
                              destinos, orden, t_clave, t_valor, t_prof, t_tipo, t_jugada,
                              asesinas, historial, control, zp, zt, zpu, e_clave, e_valor, hh)
                if control[2]:
                    return 0.0
                v = p1 * x1 + p2 * x2
        else:
            tabs[hijo, :] = tab
            pts[hijo, :] = pt
            nucleo.aplicar(tabs[hijo], pts[hijo], c, o, d, True)
            hh = clave_hijo(h, tab, tabs[hijo], pt, pts[hijo], c, o, d, zp, zt, zpu)
            fila = d // LADO
            avance = fila if c == 0 else LADO - 1 - fila
            # Reducción de jugadas tardías: una jugada tranquila, lejos de la meta y
            # mal ordenada se mira primero con un nivel menos y ventana nula.
            reducir = (control[5] != 0 and not quietud and prof >= 3 and hechas > 3
                       and orden_k < 1e7 and avance < LADO - 3)
            v = alfa + 1.0
            if reducir:
                v = -negamax(hijo, 1 - c, prof - 2, -alfa - 1e-6, -alfa, w, f, tabs, pts,
                             origenes, destinos, orden, t_clave, t_valor, t_prof, t_tipo,
                             t_jugada, asesinas, historial, control, zp, zt, zpu, e_clave, e_valor, hh)
                if control[2]:
                    return 0.0
            if v > alfa:
                v = -negamax(hijo, 1 - c, prof - 1, -beta, -alfa, w, f, tabs, pts, origenes,
                             destinos, orden, t_clave, t_valor, t_prof, t_tipo, t_jugada,
                             asesinas, historial, control, zp, zt, zpu, e_clave, e_valor, hh)
                if control[2]:
                    return 0.0
        if v > mejor:
            mejor = v
            mejor_jugada = o * CASILLAS + d
            if ply == 0:
                control[3] = mejor_jugada
        if v > alfa:
            alfa = v
        if alfa >= beta:
            if not quietud and tab[d] == 0:
                cod = o * CASILLAS + d
                if asesinas[ply, 0] != cod:
                    asesinas[ply, 1] = asesinas[ply, 0]
                    asesinas[ply, 0] = cod
                historial[c, cod] += prof * prof
            break

    if not quietud:
        if mejor <= alfa0:
            tipo = TECHO
        elif mejor >= beta:
            tipo = PISO
        else:
            tipo = EXACTO
        if t_prof[indice] <= prof or t_clave[indice] != clave:
            t_clave[indice] = clave
            t_valor[indice] = mejor
            t_prof[indice] = prof
            t_tipo[indice] = tipo
            t_jugada[indice] = mejor_jugada
    return mejor


def buscar(ctx, tablero, puntos, color, prof, limite_nodos):
    """Una búsqueda a profundidad fija. Regresa (valor, jugada o None si se abortó)."""
    ctx.tabs[0, :] = tablero
    ctx.pts[0, :] = puntos
    ctx.control[0] = 0
    ctx.control[1] = limite_nodos
    ctx.control[2] = 0
    ctx.control[3] = -1
    h = nucleo.clave(ctx.tabs[0], color, ctx.pts[0], nucleo.Z_PIEZAS, nucleo.Z_TURNO, nucleo.Z_PUNTOS)
    v = negamax(0, color, prof, -2.0, 2.0, *ctx.args(), np.uint64(h))
    if ctx.control[2] or ctx.control[3] < 0:
        return v, None
    return v, int(ctx.control[3])


def texto(cod):
    o, d = divmod(cod, CASILLAS)
    nombre = lambda s: chr(ord("a") + s % LADO) + str(s // LADO + 1)
    return f"{nombre(o)}-{nombre(d)}"
