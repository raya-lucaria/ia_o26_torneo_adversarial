"""Evaluación lineal aprendida (método de Texel) y sus rasgos, en numba.

Cada rasgo por bando se cuenta en coordenadas relativas a su color (fila 0 = su fila de casa) y
entra como (Blancas − Negras). Así el modelo es antisimétrico: cambiar los colores cambia el signo.
Con pesos w, la evaluación es z = w·φ(s) y P(ganan Blancas) = σ(z); desde quien mueve,
v = ±tanh(z/2) en (-1, 1), que es lo que usa la búsqueda.

Dos partes, las dos lineales en los pesos (no hay red neuronal):
- 46 rasgos por bando (`_lado`) con pesos según el marcador ordenado: cada bando usa
  W[sus puntos][los del rival] (9 juegos de pesos), así un corredor vale distinto con 0 o con 2
  puntos y distinto para quien va adelante que para quien va atrás;
- una tabla de patrones 3×3 (`puntaje_patrones`): un peso por ventana de 3×3 casillas según su
  contenido, su fila y si está en la orilla. Hay 4 tablas, una por grupo de marcador del bando dueño
  de la ventana (nadie en 2 puntos, él en 2, su rival en 2, 2-2), y cada bando lee la de su grupo.
Los pesos van cuantizados a int16 (`cuantizar`): la suma es entera y solo el final es flotante.

Una sola implementación: `puntaje` suma w·φ(s) directamente, sin armar φ (es lo que corre en la
búsqueda). Como z es lineal en w, el rasgo k es el puntaje con el peso k = 1 y los demás en 0; así
calcula `rasgos` los datos de entrenamiento (herramientas/). Lo que se entrena es exactamente lo
que se juega.
"""

import numpy as np
from numba import njit

from tablero_bits import LADO, MASCARA

BANDAS = ("atras", "medio", "adelante")  # filas relativas 0-2, 3-5 y 6-7
NOMBRES = (
    [f"fila_{r}" for r in range(8)]  # peones propios por fila relativa
    + [f"apoyo_1_{b}" for b in BANDAS]  # con exactamente 1 peón propio en sus diagonales de atrás
    + [f"apoyo_2_{b}" for b in BANDAS]  # con 2
    + [f"atacable_{b}" for b in BANDAS]  # un rival lo puede atacar
    + [f"atacable_sin_apoyo_{b}" for b in BANDAS]
    + [f"pasado_{d}" for d in range(1, 9)]  # sin rivales en su cono, a distancia d de la meta
    # Sin rivales en su cono salvo la fila de casa rival: solo lo pueden atacar al llegar a la
    # penúltima fila, y desde la fila de casa (sin apoyo), con p = 0.5 − 0.15·(su apoyo).
    + [f"pasado_sin_casa_{d}" for d in range(2, 6)]
    # En la penúltima fila: nadie lo puede atacar (punto seguro) o atacable con apoyo 1 o 2.
    + ["libre_7", "atacable_7_apoyo_1", "atacable_7_apoyo_2"]
    + [f"orilla_{b}" for b in BANDAS]  # columnas a e i
    + ["hueco_fila_1", "hueco_fila_2"]  # columnas donde un corredor ya no puede ser atacado
    # Con 2 puntos, el siguiente gana: peones en la penúltima fila y corredores (pasados salvo la
    # fila de casa) a distancia 2 o 3.
    + ["fila_7_con_2", "corredor_con_2"]
    + ["marcador_1_0", "marcador_2_0", "marcador_2_1"]  # (puntos propios, rivales)
    + ["turno"]  # +1 si mueven Blancas
)
K = len(NOMBRES)
I_FILA = NOMBRES.index("fila_0")
I_APOYO_1 = NOMBRES.index("apoyo_1_atras")
I_APOYO_2 = NOMBRES.index("apoyo_2_atras")
I_ATACABLE = NOMBRES.index("atacable_atras")
I_ATACABLE_SIN = NOMBRES.index("atacable_sin_apoyo_atras")
I_PASADO = NOMBRES.index("pasado_1")
I_PASADO_SC = NOMBRES.index("pasado_sin_casa_2")
I_LIBRE_7 = NOMBRES.index("libre_7")
I_AT7_APOYO_1 = NOMBRES.index("atacable_7_apoyo_1")
I_AT7_APOYO_2 = NOMBRES.index("atacable_7_apoyo_2")
I_ORILLA = NOMBRES.index("orilla_atras")
I_HUECO = NOMBRES.index("hueco_fila_1")
I_FILA7_CON_2 = NOMBRES.index("fila_7_con_2")
I_CORREDOR_CON_2 = NOMBRES.index("corredor_con_2")
I_MARCADOR = NOMBRES.index("marcador_1_0")
I_TURNO = NOMBRES.index("turno")
POP = np.array([bin(i).count("1") for i in range(1 << LADO)], np.int64)
ORILLA = 1 | (1 << (LADO - 1))
BANDA = np.array([0, 0, 0, 1, 1, 1, 2, 2], np.int64)  # fila relativa → banda
PENULTIMA = LADO - 2  # fila relativa 7


@njit(cache=False)
def _dilatar(m):
    """Columnas a distancia ≤ 1 de alguna columna de `m`."""
    return (m | (m << 1) | (m >> 1)) & MASCARA


@njit(cache=False)
def _lado(bb, yo, pts_yo, w):
    """w·(rasgos del bando `yo`). Fila absoluta de la fila relativa r: r o 8 - r."""
    otro = 1 - yo
    z = w[0] * 0  # del tipo de los pesos: entera si están cuantizados
    cono = 0  # columnas que algún rival de más adelante alcanza
    cono_sc = 0  # lo mismo sin contar la fila de casa rival
    for r in range(PENULTIMA, -1, -1):
        f = r if yo == 0 else LADO - 1 - r
        rival = bb[otro, f + 1 if yo == 0 else f - 1]  # la fila de adelante
        cono = _dilatar(rival | cono)
        if r < PENULTIMA:
            cono_sc = _dilatar(rival | cono_sc)
        propios = bb[yo, f]
        if propios == 0:
            continue
        b = BANDA[r]
        if r > 0:
            atras = bb[yo, f - 1 if yo == 0 else f + 1]
            a1 = (atras << 1) & MASCARA
            a2 = atras >> 1
            uno = propios & (a1 | a2)  # al menos un apoyo
            dos = propios & a1 & a2
        else:
            uno = 0
            dos = 0
        atacables = propios & (((rival << 1) & MASCARA) | (rival >> 1))
        z += (w[I_FILA + r] * POP[propios]
              + w[I_APOYO_1 + b] * (POP[uno] - POP[dos]) + w[I_APOYO_2 + b] * POP[dos]
              + w[I_ATACABLE + b] * POP[atacables]
              + w[I_ATACABLE_SIN + b] * POP[atacables & ~uno]
              + w[I_PASADO + PENULTIMA - r] * POP[propios & ~cono]  # distancia 8 - r
              + w[I_ORILLA + b] * POP[propios & ORILLA])
        if r == PENULTIMA:  # `rival` es la fila de casa rival
            z += (w[I_LIBRE_7] * POP[propios & ~atacables]
                  + w[I_AT7_APOYO_1] * POP[atacables & uno & ~dos]
                  + w[I_AT7_APOYO_2] * POP[atacables & dos])
            if pts_yo == 2:
                z += w[I_FILA7_CON_2] * POP[propios]
        elif r >= 3:
            corredores = POP[propios & ~cono_sc]
            z += w[I_PASADO_SC + PENULTIMA - 1 - r] * corredores  # distancia 8 - r, de 2 a 5
            if pts_yo == 2 and r >= 5:
                z += w[I_CORREDOR_CON_2] * corredores
    # Huecos: columnas c sin peón propio en (fila relativa 0, c±1) o (fila relativa 1, c±1).
    casa = bb[yo, 0 if yo == 0 else LADO - 1]
    segunda = bb[yo, 1 if yo == 0 else LADO - 2]
    cubre1 = ((casa << 1) | (casa >> 1)) & MASCARA
    cubre2 = ((segunda << 1) | (segunda >> 1)) & MASCARA
    return z + w[I_HUECO] * (LADO - POP[cubre1]) + w[I_HUECO + 1] * (LADO - POP[cubre2])


@njit(cache=False)
def puntaje(bb, pts, c, w):
    """z = w·φ(s): log-momio de que ganen Blancas."""
    return _lado(bb, 0, pts[0], w) - _lado(bb, 1, pts[1], w) + _marcador_turno(pts, c, w)


@njit(cache=False)
def _marcador_turno(pts, c, w):
    """Lo que no depende del tablero: ventaja en el marcador y quién mueve."""
    z = w[0] * 0  # del tipo de los pesos: entera si están cuantizados
    pb, pn = pts[0], pts[1]
    if pb != pn:
        alto, bajo = max(pb, pn), min(pb, pn)
        signo = 1 if pb > pn else -1
        if alto == 1:
            z += signo * w[I_MARCADOR]  # 1-0
        elif bajo == 0:
            z += signo * w[I_MARCADOR + 1]  # 2-0
        else:
            z += signo * w[I_MARCADOR + 2]  # 2-1
    return z + (w[I_TURNO] if c == 0 else -w[I_TURNO])


@njit(cache=False)
def rasgos(bb, pts, c, phi):
    """Llena phi (float64[K]) desde la perspectiva de Blancas: φ_k = puntaje con w = e_k."""
    e = np.zeros(K, np.float64)
    for k in range(K):
        e[k] = 1.0
        phi[k] = puntaje(bb, pts, c, e)
        e[k] = 0.0


@njit(cache=False)
def rasgos_lado(bb, yo, pts_yo, phi):
    """Para entrenar el marcador ordenado: phi (float64[K]) = los rasgos del bando `yo` solo, en su
    orientación (φ_k = `_lado` con w = e_k). Marcador y turno no salen de `_lado`: quedan en 0."""
    e = np.zeros(K, np.float64)
    for k in range(K):
        e[k] = 1.0
        phi[k] = _lado(bb, yo, pts_yo, e)
        e[k] = 0.0


# Tabla de patrones 3×3. Sus pesos van en `pesos` después de los K rasgos, por (fila del
# ancla, orilla o centro, índice base 3 de las 9 casillas: vacía 0, propia 1, rival 2).
ANCLAS = LADO - 2  # anclas por fila y por columna
NCONF = 3**9
TAM_TABLA = ANCLAS * 2 * NCONF
# Código base 3 de un tramo de 3 casillas, por (propios3 | rivales3 << 3).
TRAMO = np.zeros(64, np.int64)
for _p in range(8):
    for _r in range(8):
        if _p & _r == 0:
            TRAMO[_p | (_r << 3)] = sum((((_p >> k) & 1) + 2 * ((_r >> k) & 1)) * 3**k for k in range(3))


GRUPOS_TABLA = 4  # tablas por grupo de marcador del dueño de la ventana


@njit(cache=False)
def grupo_tabla(mios, suyos):
    """0 = nadie con 2 puntos, 1 = el bando con 2, 2 = su rival con 2, 3 = 2-2."""
    return (1 if mios == 2 else 0) + (2 if suyos == 2 else 0)


@njit(cache=False)
def puntaje_patrones(bb, w, base_b, base_n):
    """Suma de la tabla de patrones: Blancas − Negras, cada bando en su orientación y con su tabla
    (la de Blancas empieza en base_b y la de Negras en base_n)."""
    z = w[0] * 0  # del tipo de los pesos: entera si están cuantizados
    for col in range(ANCLAS):
        orilla = 0 if (col == 0 or col == ANCLAS - 1) else 1
        # Blancas: anclas en las filas 0..6, ventana hacia arriba.
        t0 = TRAMO[((bb[0, 0] >> col) & 7) | (((bb[1, 0] >> col) & 7) << 3)]
        t1 = TRAMO[((bb[0, 1] >> col) & 7) | (((bb[1, 1] >> col) & 7) << 3)]
        for rr in range(ANCLAS):
            f = rr + 2
            t2 = TRAMO[((bb[0, f] >> col) & 7) | (((bb[1, f] >> col) & 7) << 3)]
            z += w[base_b + (rr * 2 + orilla) * NCONF + t0 + 27 * t1 + 729 * t2]
            t0, t1 = t1, t2
        # Negras: fila relativa rr = fila absoluta 8 - rr, ventana hacia abajo.
        t0 = TRAMO[((bb[1, LADO - 1] >> col) & 7) | (((bb[0, LADO - 1] >> col) & 7) << 3)]
        t1 = TRAMO[((bb[1, LADO - 2] >> col) & 7) | (((bb[0, LADO - 2] >> col) & 7) << 3)]
        for rr in range(ANCLAS):
            f = LADO - 3 - rr
            t2 = TRAMO[((bb[1, f] >> col) & 7) | (((bb[0, f] >> col) & 7) << 3)]
            z -= w[base_n + (rr * 2 + orilla) * NCONF + t0 + 27 * t1 + 729 * t2]
            t0, t1 = t1, t2
    return z


@njit(cache=False)
def indices_patrones(bb, idx):
    """Para entrenar la tabla: idx[0] / idx[1] (int64[ANCLAS²]) = índice en la tabla de cada ventana de
    Blancas / Negras, de modo que `puntaje_patrones(bb, w, base_b, base_n)` = Σ w[base_b + idx[0]] −
    Σ w[base_n + idx[1]] (herramientas/ajustar_marcador.py lo verifica)."""
    for col in range(ANCLAS):
        orilla = 0 if (col == 0 or col == ANCLAS - 1) else 1
        for yo in range(2):
            otro = 1 - yo
            f0 = 0 if yo == 0 else LADO - 1
            paso = 1 if yo == 0 else -1
            t0 = TRAMO[((bb[yo, f0] >> col) & 7) | (((bb[otro, f0] >> col) & 7) << 3)]
            t1 = TRAMO[((bb[yo, f0 + paso] >> col) & 7) | (((bb[otro, f0 + paso] >> col) & 7) << 3)]
            for rr in range(ANCLAS):
                f = f0 + paso * (rr + 2)
                t2 = TRAMO[((bb[yo, f] >> col) & 7) | (((bb[otro, f] >> col) & 7) << 3)]
                idx[yo, col * ANCLAS + rr] = (rr * 2 + orilla) * NCONF + t0 + 27 * t1 + 729 * t2
                t0, t1 = t1, t2


TAM_PESOS = 9 * K + GRUPOS_TABLA * TAM_TABLA  # data/pesos.npy: 9 clases × K pesos y las 4 tablas
ESCALA = 4096  # pesos cuantizados: round(w · 2^12) en int16


def cuantizar(pesos):
    """Los pesos en float64 → int16 con ESCALA (la tabla cabe en un cuarto de la memoria)."""
    pesos = np.asarray(pesos, np.float64)
    if pesos.shape != (TAM_PESOS,) and pesos.shape != (K,):
        raise ValueError(f"se esperaban {TAM_PESOS} pesos (o {K}); llegaron {pesos.shape}")
    q = np.round(pesos * ESCALA)
    if np.abs(q).max() > np.iinfo(np.int16).max:
        raise ValueError("un peso no cabe en int16 con esta ESCALA")
    return q.astype(np.int16)


@njit(cache=False)
def evaluar_lineal(bb, pts, c, pesos):
    """Valor en (-1, 1) desde quien mueve, con pesos cuantizados: la suma es entera y solo el final
    pasa a punto flotante, ±tanh(z/2) con z = suma / ESCALA. Con TAM_PESOS pesos (los de
    data/pesos.npy), el modelo completo; con K, los 46 rasgos solos (los de versiones anteriores,
    que herramientas/ todavía usa como maestro de las etiquetas)."""
    if pesos.shape[0] == TAM_PESOS:
        pb, pn = pts[0], pts[1]
        wb = pesos[(pb * 3 + pn) * K:(pb * 3 + pn + 1) * K]  # Blancas: W[sus puntos][los de Negras]
        wn = pesos[(pn * 3 + pb) * K:(pn * 3 + pb + 1) * K]  # Negras: W[sus puntos][los de Blancas]
        z = (_lado(bb, 0, pb, wb) - _lado(bb, 1, pn, wn) + _marcador_turno(pts, c, wb)
             + puntaje_patrones(bb, pesos, 9 * K + grupo_tabla(pb, pn) * TAM_TABLA,
                                9 * K + grupo_tabla(pn, pb) * TAM_TABLA))
    else:
        z = puntaje(bb, pts, c, pesos[0:K])
    v = np.tanh((0.5 / ESCALA) * z)
    return v if c == 0 else -v
