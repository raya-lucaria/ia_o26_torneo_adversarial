"""Todo lo que el CLI imprime: tablas de partidas, avisos, jugadas (`-v`) y tableros.

Las funciones regresan texto (no imprimen) para poder probarlas; `comandos` imprime.
Los registros son del formato v1 (`arbitro.registro`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import config
from motor import Estado, Jugada

COLORES = ("blancas", "negras")
LETRA = {"blancas": "B", "negras": "N"}
AVISO_RELOJ = 0.9  # avisar si un bot usó más de este porcentaje de su reloj
LINEAS_SALIDA = 12  # líneas finales de lo que imprimió un bot que perdió por error

# Cómo terminó la partida, visto desde el ganador (al final de la fila si no fue «puntos»).
MOTIVOS = {
    "puntos": "3 puntos",
    "sin_jugadas": "rival sin jugadas",
    "reloj": "rival sin reloj",
    "preparar": "rival tardó en preparar",
    "memoria": "rival sin memoria",
    "ilegal": "rival hizo jugada ilegal",
    "error": "rival con error",
}

# Cómo terminó la partida, en una frase (para `final`).
FINALES = {
    "puntos": "llegó a 3 puntos",
    "sin_jugadas": "el rival se quedó sin jugadas",
    "reloj": "el rival se quedó sin reloj",
    "preparar": "el rival tardó demasiado en preparar",
    "memoria": "el rival se pasó de memoria",
    "ilegal": "el rival hizo una jugada ilegal",
    "error": "el rival terminó con error (o se rindió)",
}

AVISO_SIN_DOCKER = (
    "⚠ sin Docker: el reloj y el tiempo de preparar sí cuentan, pero no hay límite de "
    "memoria ni de CPU y tu código corre sin aislamiento. Lo oficial es con Docker o la "
    "prueba de GitHub (Actions)."
)


# --- Utilidades ----------------------------------------------------------------------------
# Secuencias ANSI (colores, mover el cursor...) y caracteres de control salvo \n y \t.
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b.")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def limpio(texto: str) -> str:
    """Texto de un bot listo para la terminal: sin ANSI ni caracteres de control."""
    return _CONTROL.sub("?", _ANSI.sub("", texto))


def ruta_visible(ruta: Path) -> str:
    """Ruta como la ve el alumno en su máquina (fuera del contenedor del CLI)."""
    ruta = Path(ruta).resolve()
    host = os.environ.get("TORNEO_RAIZ_HOST")
    if host:
        try:
            return str(Path(host) / ruta.relative_to(config.RAIZ))
        except ValueError:
            pass
    return str(ruta)


def columnas(filas: list[list[str]], derecha: frozenset[int] = frozenset()) -> list[str]:
    """Alinea una tabla de texto (la primera fila es el encabezado)."""
    anchos = [max(len(f[i]) for f in filas) for i in range(len(filas[0]))]
    return [
        "  ".join(
            c.rjust(anchos[i]) if i in derecha else c.ljust(anchos[i]) for i, c in enumerate(f)
        ).rstrip()
        for f in filas
    ]


def _segundos(valor: float) -> str:
    return f"{valor:.1f} s"


def _equipo(registro: dict, color: str) -> str:
    return registro["meta"][color]


# --- Tabla de partidas ---------------------------------------------------------------------
class TablaPartidas:
    """Tabla compacta, una fila por partida, que se imprime conforme terminan.

    Los anchos se fijan desde el principio con los nombres de los equipos.
    """

    ENCABEZADO = ("#", "Blancas", "Negras", "Ganador", "Puntos", "Jugadas", "Reloj máx")
    DERECHA = frozenset({0, 4, 5, 6})

    def __init__(self, nombres: list[str], partidas: int) -> None:
        nombre = max([len(n) for n in nombres] + [7])
        self.anchos = [max(1, len(str(partidas))), nombre, nombre, nombre, 6, 7, 10]

    def _linea(self, celdas: list[str]) -> str:
        return "  ".join(
            c.rjust(a) if i in self.DERECHA else c.ljust(a)
            for i, (c, a) in enumerate(zip(celdas, self.anchos, strict=True))
        ).rstrip()

    def encabezado(self) -> str:
        return self._linea(list(self.ENCABEZADO))

    def fila(self, n: int, registro: dict) -> str:
        """Fila de la partida `n`. Si no terminó por puntos, el motivo va al final."""
        ganador = registro["ganador"]
        nota = ""
        if ganador in COLORES:
            texto = _equipo(registro, ganador)
            if registro["motivo"] != "puntos":
                nota = f"  ({MOTIVOS.get(registro['motivo'], registro['motivo'])})"
        else:
            texto, nota = "—", "  (no contó: falla de infraestructura)"
        puntos = registro["puntos"]
        reloj = registro.get("reloj_usado") or {}
        color_max = max(COLORES, key=lambda c: reloj.get(c, 0.0))
        maximo = f"{_segundos(reloj.get(color_max, 0.0))} ({LETRA[color_max]})"
        linea = self._linea(
            [
                str(n),
                _equipo(registro, "blancas"),
                _equipo(registro, "negras"),
                texto,
                f"{puntos.get('blancas', 0)}–{puntos.get('negras', 0)}",
                str(len(registro["jugadas"])),
                maximo,
            ]
        )
        return linea + nota


def resultado(registros: list[dict]) -> str:
    """«Resultado: montecarlo 3 – 1 random» (partidas ganadas por cada uno)."""
    validos = [r for r in registros if r["ganador"] in COLORES]
    nombres = sorted({_equipo(r, c) for r in registros for c in COLORES})
    if len(nombres) == 1:  # el mismo bot en ambos colores: por color
        ganadas = {c: sum(r["ganador"] == c for r in validos) for c in COLORES}
        texto = f"blancas {ganadas['blancas']} – {ganadas['negras']} negras"
    else:
        ganadas = dict.fromkeys(nombres, 0)
        for r in validos:
            ganadas[_equipo(r, r["ganador"])] += 1
        orden = sorted(nombres, key=lambda n: -ganadas[n])
        if len(orden) == 2:
            a, b = orden
            texto = f"{a} {ganadas[a]} – {ganadas[b]} {b}"
        else:
            texto = ", ".join(f"{n} {ganadas[n]}" for n in orden)
    fallas = len(registros) - len(validos)
    if fallas:
        texto += f"  ({fallas} no contaron por falla de infraestructura)"
    return f"Resultado: {texto}"


def avisos(n: int, registro: dict, limite_s: float) -> list[str]:
    """⚠ si un bot usó más del 90 % de su reloj (y no perdió por reloj)."""
    lineas = []
    for color in COLORES:
        usado = (registro.get("reloj_usado") or {}).get(color, 0.0)
        perdio_por_reloj = registro["motivo"] == "reloj" and registro["ganador"] != color
        if usado > AVISO_RELOJ * limite_s and not perdio_por_reloj:
            lineas.append(
                f"⚠ partida {n}: {_equipo(registro, color)} usó {usado:.1f} de {limite_s:g} s "
                f"de reloj ({round(100 * usado / limite_s)} %). Está cerca de perder por reloj."
            )
    return lineas


def _cola(texto: str, n: int = LINEAS_SALIDA) -> list[str]:
    lineas = limpio(texto).rstrip().splitlines()
    return lineas[-n:]


def problemas(
    n: int, registro: dict, publicos: config.Limites, con_salida: bool = True
) -> list[str]:
    """✗ por cada derrota que no fue «normal» (reloj, error, ilegal...), con qué hacer.

    `con_salida`: agrega las últimas líneas que imprimió el bot que perdió.
    """
    if registro["ganador"] not in COLORES:
        nota = next(
            (
                linea
                for color in COLORES
                for linea in registro["salidas"].get(color, "").splitlines()
                if "Falla de infraestructura" in linea
            ),
            "",
        )
        return [
            f"✗ partida {n}: falla de infraestructura (no es culpa de ningún bot); no cuenta. "
            "Vuelve a intentarlo.",
            *([f"    {limpio(nota).strip()}"] if nota else []),
        ]
    motivo = registro["motivo"]
    if motivo in ("puntos", "sin_jugadas"):
        return []
    perdedor = COLORES[1 - COLORES.index(registro["ganador"])]
    quien = f"{_equipo(registro, perdedor)} ({perdedor})"
    jugada = len(registro["jugadas"]) + 1
    usado = (registro.get("reloj_usado") or {}).get(perdedor, 0.0)
    explicacion = {
        "reloj": (
            f"se quedó sin reloj en la jugada {jugada}: usó {usado:.1f} s y el límite es "
            f"{publicos.reloj_s:g} s por partida. Usa tiempo_restante para repartir tu tiempo."
        ),
        "preparar": (
            f"preparar tardó más de {publicos.preparar_s:g} s (cuenta también importar "
            "strategy.py). Carga menos cosas o hazlas más rápido."
        ),
        "memoria": f"se pasó de memoria (límite {publicos.memoria_mb} MB).",
        "ilegal": (
            f"regresó una jugada ilegal en la jugada {jugada}. jugar debe regresar una "
            "jugada de estado.jugadas()."
        ),
        "error": "terminó con error (una excepción o se cerró).",
    }.get(motivo, motivo)
    lineas = [f"✗ partida {n}: {quien} {explicacion}"]
    cola = _cola(registro["salidas"].get(perdedor, ""))
    if con_salida and cola and motivo != "reloj":
        lineas.append(f"    Lo que imprimió {_equipo(registro, perdedor)} (últimas líneas):")
        lineas += [f"    | {linea}" for linea in cola]
    return lineas


# --- -v: una línea por jugada ----------------------------------------------------------------
def _dato(info: dict | None, clave: str) -> str:
    valor = (info or {}).get(clave)
    if valor is None:
        return "-"
    if isinstance(valor, bool):
        return str(valor)
    if isinstance(valor, float):
        return f"{valor:.3g}"
    return limpio(str(valor))[:10]


COLUMNAS_INFO = ("prof", "nodos", "eval")
EXTRAS_MAX = 4  # datos extra de `reportar` por línea (el resto, en el visor)


def _extras(info: dict | None) -> str:
    """Lo demás que reportó el bot (p. ej. `muestras=120`), compacto: «muestras=120 ...»."""
    if not isinstance(info, dict):
        return ""
    claves = [k for k in info if k not in COLUMNAS_INFO]
    partes = [f"{limpio(str(k))[:12]}={_dato(info, k)}" for k in claves[:EXTRAS_MAX]]
    if len(claves) > EXTRAS_MAX:
        partes.append("…")
    return " ".join(partes)


def _combate(combate: dict | None) -> str:
    if not combate:
        return ""
    resultado = "ganó" if combate["gano"] else "perdió (sale el atacante)"
    return f"⚔ ataque p={combate['p']:.2f} → {resultado}"


def _anota(estado: Estado | None, jugada: dict) -> tuple[Estado | None, str]:
    """Aplica la jugada; regresa el estado nuevo y «★ anota (1–0)» si subió el marcador.

    Nunca falla: si la jugada no se puede aplicar (ilegal, registro mal formado) regresa
    `None` y ya no se marca nada más.
    """
    if estado is None:
        return None, ""
    try:
        combate = jugada["combate"]
        gana = combate["gano"] if combate else None
        nuevo = estado.aplicar(Jugada.desde_texto(jugada["jugada"]), gana_ataque=gana)
    except Exception:
        return None, ""
    antes, ahora = estado.puntos, nuevo.puntos
    if ahora == antes:
        return nuevo, ""
    return nuevo, f"★ anota ({ahora['blancas']}–{ahora['negras']})"


def lineas_jugadas(registro: dict) -> list[str]:
    """Una línea por jugada: jug, quién, jugada, prof, nodos, eval, t, reloj.

    Al final: combate, «★ anota» y los demás datos que el bot pasó a `reportar`.
    """
    filas = [["jug", "quién", "jugada", "prof", "nodos", "eval", "t", "reloj", ""]]
    estado = Estado.inicial()
    for j in registro["jugadas"]:
        estado, anota = _anota(estado, j)
        filas.append(
            [
                str(j["n"]),
                _equipo(registro, j["color"]),
                j["jugada"],
                _dato(j["info"], "prof"),
                _dato(j["info"], "nodos"),
                _dato(j["info"], "eval"),
                f"{j['tiempo']:.2f}",
                f"{j['reloj']:.1f}",
                " ".join(x for x in (_combate(j["combate"]), anota, _extras(j["info"])) if x),
            ]
        )
    return columnas(filas, derecha=frozenset({0, 3, 4, 5, 6, 7}))


def salidas_bots(registro: dict) -> list[str]:
    """Lo que imprimió cada bot (con las notas del árbitro), aparte de las jugadas."""
    lineas = []
    for color in COLORES:
        texto = limpio(registro["salidas"].get(color, "")).rstrip()
        if not texto:
            continue
        lineas.append(f"── Lo que imprimió {_equipo(registro, color)} ({color}) ──")
        lineas += texto.splitlines()
    if lineas:
        lineas.append("── fin de lo que imprimieron los bots ──")
    return lineas


def titulo_partida(n: int, registro: dict) -> str:
    return (
        f"── Partida {n}: {_equipo(registro, 'blancas')} (blancas) vs "
        f"{_equipo(registro, 'negras')} (negras) ──"
    )


# --- Tableros --------------------------------------------------------------------------------
def tablero(estado: Estado, titulo: str) -> str:
    return f"{titulo}\n{estado.dibujar()}"


def repeticion(registro: dict) -> list[str]:
    """Tableros de una partida guardada: el inicial y uno después de cada jugada."""
    estado = Estado.inicial()
    textos = [tablero(estado, "Inicio")]
    for j in registro["jugadas"]:
        jugada = Jugada.desde_texto(j["jugada"])
        combate = j["combate"]
        if combate:
            estado = estado.aplicar(jugada, gana_ataque=combate["gano"])
        else:
            estado = estado.aplicar(jugada)
        titulo = f"Jugada {j['n']} · {_equipo(registro, j['color'])} ({j['color']}): {j['jugada']}"
        if combate:
            titulo += f"  {_combate(combate)}"
        textos.append(tablero(estado, titulo))
    return textos


def final(registro: dict) -> str:
    if registro["ganador"] not in COLORES:
        return "La partida no contó: falla de infraestructura."
    ganador = registro["ganador"]
    puntos = registro["puntos"]
    return (
        f"Ganó {_equipo(registro, ganador)} ({ganador}): "
        f"{FINALES.get(registro['motivo'], registro['motivo'])} · "
        f"puntos {puntos.get('blancas', 0)}–{puntos.get('negras', 0)}"
    )


# --- Liga -----------------------------------------------------------------------------------
def tabla_liga(tabla: list) -> list[str]:
    """Tabla de la liga (FilaTabla), ya ordenada."""
    filas = [["Lugar", "Equipo", "Puntos", "Partidas", "SB", "Anotados"]]
    for f in tabla:
        filas.append(
            [
                str(f.posicion),
                f.equipo,
                str(f.puntos),
                str(f.partidas),
                f"{f.sb:g}",
                str(f.anotados),
            ]
        )
    return columnas(filas, derecha=frozenset({0, 2, 3, 4, 5}))
