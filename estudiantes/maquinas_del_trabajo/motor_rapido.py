"""Motor rápido del equipo: las reglas del juego en numba, copiadas del motor oficial.

Es una copia de `sistema/motor/nucleo.py` (se vale: docs/avanzado.md, «Modificar el motor»)
con lo que necesita nuestra búsqueda:
- `hacer`: aplica una jugada sobre una COPIA del tablero y actualiza la clave Zobrist sin
  recalcularla completa (la búsqueda escribe cada hijo en un arreglo preasignado por ply);
- `tiene_jugada`: dice si un bando puede mover, sin generar todas sus jugadas;
- conversiones entre el `Estado` oficial, nuestros arreglos y el texto de las jugadas.

La jugada que elegimos la revisa siempre el árbitro con el motor oficial. Por eso
`herramientas/validar_motor.py` compara este archivo contra el oficial en muchas posiciones.

Representación (la misma del motor oficial):
- `tab`: `int8[81]`; casilla = fila*9 + col (fila 0 = fila «1»). 0 vacía, 1 blanca, 2 negra.
- `color`: 0 = Blancas, 1 = Negras. La pieza de un color vale `color + 1`.
- `pts`: `int64[2]` (Blancas, Negras).

Las funciones numba no usan `cache=True`: en la liga la carpeta es de solo lectura y cada
partida es un proceso nuevo, así que se compilan en `preparar`.
"""

import numpy as np
from numba import njit

# Para las funciones que en la liga solo se llaman desde otras funciones numba: sin la envoltura
# que permite llamarlas desde Python, `preparar` compila menos (~14 % menos tiempo, medido). Las
# pruebas y herramientas que las necesitan desde Python usan `herramientas/desde_python.py`.
njit_interna = njit(no_cpython_wrapper=True)

VACIA = 0
LADO = 9
CASILLAS = LADO * LADO
PEONES = 18
MAX_JUGADAS = PEONES * 3
PUNTOS_PARA_GANAR = 3
COLUMNAS = "abcdefghi"


def _splitmix64(semilla: int, cuantos: int) -> np.ndarray:
    """Números de 64 bits fijos (copia del motor oficial, para tener sus mismas claves)."""
    mascara = (1 << 64) - 1
    x = semilla
    salida = []
    for _ in range(cuantos):
        x = (x + 0x9E3779B97F4A7C15) & mascara
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mascara
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mascara
        salida.append(z ^ (z >> 31))
    return np.array(salida, dtype=np.uint64)


# Tablas Zobrist idénticas a las oficiales: así nuestra clave es igual a `estado.clave`.
_Z = _splitmix64(20261018, 2 * CASILLAS + 1 + 2 * (PUNTOS_PARA_GANAR + 1))
Z_PIEZAS = _Z[: 2 * CASILLAS].reshape(2, CASILLAS)
Z_TURNO = _Z[2 * CASILLAS]
Z_PUNTOS = _Z[2 * CASILLAS + 1 :].reshape(2, PUNTOS_PARA_GANAR + 1)


@njit_interna
def paso(color):
    """Dirección «adelante»: +1 fila para Blancas, −1 para Negras."""
    return 1 if color == 0 else -1


@njit
def generar(tab, color, origenes, destinos):
    """Llena `origenes`/`destinos` con las jugadas legales y regresa cuántas son.

    Mismo orden que el oficial: casilla de origen creciente; por origen, diagonal izquierda,
    recto, diagonal derecha.
    """
    pieza = color + 1
    avance = paso(color) * LADO
    n = 0
    for c in range(CASILLAS):
        if tab[c] != pieza:
            continue
        col = c % LADO
        frente = c + avance  # nunca sale del tablero: al llegar a la meta la pieza sale
        if col > 0 and tab[frente - 1] != pieza:
            origenes[n] = c
            destinos[n] = frente - 1
            n += 1
        if tab[frente] == VACIA:
            origenes[n] = c
            destinos[n] = frente
            n += 1
        if col < LADO - 1 and tab[frente + 1] != pieza:
            origenes[n] = c
            destinos[n] = frente + 1
            n += 1
    return n


@njit_interna
def tiene_jugada(tab, color):
    """True si `color` tiene al menos una jugada legal (para en la primera que encuentra)."""
    pieza = color + 1
    avance = paso(color) * LADO
    for c in range(CASILLAS):
        if tab[c] != pieza:
            continue
        col = c % LADO
        frente = c + avance
        if tab[frente] == VACIA:
            return True
        if col > 0 and tab[frente - 1] != pieza:
            return True
        if col < LADO - 1 and tab[frente + 1] != pieza:
            return True
    return False


@njit_interna
def apoyo(tab, casilla, color):
    """Peones propios en las dos casillas diagonales de atrás de `casilla` (0, 1 o 2)."""
    fila = casilla // LADO - paso(color)
    if fila < 0 or fila >= LADO:
        return 0
    col = casilla % LADO
    pieza = color + 1
    total = 0
    if col > 0 and tab[fila * LADO + col - 1] == pieza:
        total += 1
    if col < LADO - 1 and tab[fila * LADO + col + 1] == pieza:
        total += 1
    return total


@njit_interna
def prob_combate(apoyo_atacante, apoyo_defensor):
    """p = clip(0.5 + 0.15·(apoyo_atacante − apoyo_defensor), 0.2, 0.9)."""
    p = 0.5 + 0.15 * (apoyo_atacante - apoyo_defensor)
    return min(max(p, 0.2), 0.9)


@njit_interna
def prob_jugada(tab, color, origen, destino):
    """Probabilidad de que la jugada salga bien: p si es ataque, 1.0 si no."""
    if tab[destino] == VACIA:
        return 1.0
    return prob_combate(apoyo(tab, origen, color), apoyo(tab, destino, 1 - color))


def clave(tab, color, pts) -> np.uint64:
    """Hash Zobrist de 64 bits (piezas, turno y puntos), igual al oficial.

    Es Python con numpy y no numba: solo se calcula una vez por jugada (en la raíz; adentro de la
    búsqueda la actualiza `hacer`), así que compilarla solo alargaría `preparar`.
    """
    ocupadas = np.flatnonzero(tab != VACIA)
    h = np.bitwise_xor.reduce(Z_PIEZAS[tab[ocupadas] - 1, ocupadas], initial=np.uint64(0))
    if color == 1:
        h ^= Z_TURNO
    h ^= Z_PUNTOS[0, pts[0]]
    h ^= Z_PUNTOS[1, pts[1]]
    return np.uint64(h)


@njit_interna
def hacer(tab, pts, color, origen, destino, gana, h, tab_hijo, pts_hijo):
    """Escribe en `tab_hijo`/`pts_hijo` la posición después de la jugada y regresa su clave.

    `h` es la clave de la posición actual; `tab` y `pts` no cambian. `gana` solo importa si la
    jugada es un ataque. La clave regresada ya tiene el turno del otro bando.
    """
    # Copia con bucles y no con `tab_hijo[:] = tab`: la asignación por rebanada obliga a numba a
    # compilar la revisión de formas con su mensaje de error, y eso alarga `preparar` ~0.8 s.
    for c in range(CASILLAS):
        tab_hijo[c] = tab[c]
    pts_hijo[0] = pts[0]
    pts_hijo[1] = pts[1]
    h ^= Z_TURNO
    h ^= Z_PIEZAS[color, origen]
    tab_hijo[origen] = VACIA
    if tab[destino] != VACIA:  # ataque
        if not gana:
            return h  # sale el atacante; el defensor se queda
        h ^= Z_PIEZAS[1 - color, destino]
        tab_hijo[destino] = VACIA
    meta = LADO - 1 if color == 0 else 0
    if destino // LADO == meta:  # anota y la pieza sale del tablero
        h ^= Z_PUNTOS[color, pts[color]] ^ Z_PUNTOS[color, pts[color] + 1]
        pts_hijo[color] += 1
    else:
        tab_hijo[destino] = color + 1
        h ^= Z_PIEZAS[color, destino]
    return h


# --- Conversiones (Python) ------------------------------------------------------------------


def codigo(origen: int, destino: int) -> int:
    """Una jugada como un solo entero: origen*81 + destino."""
    return origen * CASILLAS + destino


def _nombre(casilla: int) -> str:
    return f"{COLUMNAS[casilla % LADO]}{casilla // LADO + 1}"


def _casilla(texto: str) -> int:
    return (int(texto[1]) - 1) * LADO + COLUMNAS.index(texto[0])


def texto_jugada(cod: int) -> str:
    """Código de jugada → texto como lo espera el árbitro, p. ej. "e2-e3"."""
    origen, destino = divmod(int(cod), CASILLAS)
    return f"{_nombre(origen)}-{_nombre(destino)}"


def desde_estado(estado) -> tuple[np.ndarray, np.ndarray, int]:
    """(tablero, puntos, color) del `Estado` que manda el árbitro, como copias que podemos cambiar.

    Lee los arreglos internos del estado (lo mismo que hace `simular_al_azar` oficial). Si una
    versión nueva del motor los cambiara, reconstruye todo con la API pública.
    """
    try:
        tab = np.array(estado._tablero, dtype=np.int8)
        pts = np.array(estado._puntos, dtype=np.int64)
        color = int(estado._color)
    except AttributeError:
        tab = np.zeros(CASILLAS, np.int8)
        for color_pieza, nombre in enumerate(("blancas", "negras")):
            for texto in estado.piezas(nombre):
                tab[_casilla(texto)] = color_pieza + 1
        puntos = estado.puntos
        pts = np.array([puntos["blancas"], puntos["negras"]], dtype=np.int64)
        color = 0 if estado.turno == "blancas" else 1
    return tab, pts, color
