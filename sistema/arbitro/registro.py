"""Registro (log) de una partida en formato v1 y escape de la salida de los bots.

Formato v1 (`a_json`):
    {"version": 1,
     "meta": {"blancas", "negras", "pareja", "juego", "semilla_maestra"},
     "jugadas": [{"n", "color", "jugada", "combate": {"p", "u", "gano"} | null,
                  "tiempo", "reloj", "info"}],
     "ganador", "motivo", "puntos", "reloj_usado", "preparar_usado", "salidas"}

`preparar_usado` (segundos de `preparar` por color) puede faltar en registros viejos.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arbitro.partida import ResultadoPartida

VERSION = 1
CAMPOS_META = ("blancas", "negras", "pareja", "juego", "semilla_maestra")

# Caracteres de control (salvo salto de línea y tabulador), incluidos \r, ESC, los de
# control C1 y los separadores Unicode de línea/párrafo.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029]")
# Caracteres invisibles de formato: ancho cero, marcas de dirección, BOM, etc.
_INVISIBLES = re.compile(r"[\u00ad\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
# Comandos de GitHub Actions: `:` al inicio de línea, con cualquier espacio Unicode antes.
_COMANDO = re.compile(r"^([^\S\n]*):", re.MULTILINE)
# Comandos de la sintaxis vieja (`##[error]...`), que Actions reconoce en cualquier parte.
_COMANDO_V1 = "##["


def escapar(texto: str) -> str:
    """Neutraliza la salida de un bot para que no se interprete como comando.

    - Cambia los caracteres de control e invisibles por `?` (así `\\r::error::` no sirve).
    - Si una línea empieza con `:` (p. ej. `::error::`, aun con espacios antes), le
      antepone `> `.
    - Rompe `##[` (comandos viejos de Actions) como `# #[`.
    """
    texto = _CONTROL.sub("?", texto)
    texto = _INVISIBLES.sub("?", texto)
    texto = _COMANDO.sub(r"\1> :", texto)
    while _COMANDO_V1 in texto:
        texto = texto.replace(_COMANDO_V1, "# #[")
    return texto


def a_json(r: ResultadoPartida, meta: dict) -> dict:
    """Resultado de una partida → diccionario v1 listo para `json.dump`."""
    faltan = [c for c in CAMPOS_META if c not in meta]
    if faltan:
        raise ValueError(f"Faltan datos en meta: {', '.join(faltan)}.")
    return {
        "version": VERSION,
        "meta": {c: meta[c] for c in CAMPOS_META},
        "jugadas": [
            {
                "n": j["n"],
                "color": j["color"],
                "jugada": j["jugada"],
                "combate": j["combate"],
                "tiempo": j["tiempo"],
                "reloj": j["reloj"],
                "info": j["info"],
            }
            for j in r.jugadas
        ],
        "ganador": r.ganador,
        "motivo": r.motivo,
        "puntos": dict(r.puntos),
        "reloj_usado": dict(r.reloj_usado),
        "preparar_usado": dict(r.preparar_usado),
        "salidas": {color: escapar(texto) for color, texto in r.salidas.items()},
    }
