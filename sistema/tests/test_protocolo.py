"""Tests del protocolo y del registro v1."""

import os
import threading

import pytest

from arbitro import protocolo
from arbitro.partida import ResultadoPartida
from arbitro.protocolo import LectorLineas, MensajeInvalido, codificar, decodificar
from arbitro.registro import a_json, escapar


def _lector_con(datos: bytes) -> LectorLineas:
    r, w = os.pipe()

    def escribir():
        with os.fdopen(w, "wb") as f:
            f.write(datos)

    threading.Thread(target=escribir, daemon=True).start()
    return LectorLineas(r)


def test_ida_y_vuelta():
    msg = protocolo.jugar({"turno": "blancas"}, 12.5)
    linea = codificar(msg)
    assert linea.endswith(b"\n") and linea.count(b"\n") == 1
    assert decodificar(linea[:-1]) == msg


def test_mensaje_grande_se_rechaza():
    with pytest.raises(MensajeInvalido):
        codificar({"tipo": "jugada", "jugada": "x" * (70 * 1024)})
    grande = b'{"tipo":"jugada","jugada":"' + b"x" * (70 * 1024) + b'"}\n'
    lector = _lector_con(grande + codificar(protocolo.listo()))
    with pytest.raises(MensajeInvalido):
        lector.leer()
    # Después de la línea larga, la siguiente se lee normal.
    assert decodificar(lector.leer()) == {"tipo": "listo"}
    assert lector.leer() is None


def test_linea_larga_sin_salto_no_acumula_memoria():
    lector = _lector_con(b"y" * (1024 * 1024))
    with pytest.raises(MensajeInvalido):
        lector.leer()
    assert lector.leer() is None
    assert len(lector._buf) <= protocolo.MAX_BYTES + 64 * 1024


@pytest.mark.parametrize(
    "linea",
    [
        b"no es json",
        b"[1, 2]",
        b'{"jugada": "e2-e3"}',
        b'{"tipo": 3}',
        b"[" * 60000,
        b'{"tipo":"x","n":' + b"9" * 5000 + b"}",
    ],
)
def test_basura_es_invalida(linea):
    with pytest.raises(MensajeInvalido):
        decodificar(linea)


def test_escapar_neutraliza_comandos_y_control():
    texto = "::error::x\nhola\n  ::set-output name=a::b\nfin\r::warning::y\x1b[31m ::z"
    limpio = escapar(texto)
    assert not limpio.startswith("::")
    assert all(not linea.lstrip().startswith(":") for linea in limpio.split("\n"))
    assert "\r" not in limpio and "\x1b" not in limpio and " " not in limpio
    assert escapar(limpio) == limpio  # escapar dos veces no cambia nada
    assert escapar("hola: mundo\nlinea normal") == "hola: mundo\nlinea normal"


@pytest.mark.parametrize(
    "prefijo", ["\u00a0", "\u3000", "\u2003", "\u200b", "\ufeff", "\u200e", " \u200b\u00a0"]
)
def test_escapar_espacios_unicode_y_ancho_cero(prefijo):
    limpio = escapar(f"hola\n{prefijo}::error::x\n{prefijo}::warning::y")
    for linea in limpio.split("\n"):
        assert not linea.lstrip().startswith(":"), repr(linea)
        assert not any(c in linea for c in "\u200b\u200c\u200d\u200e\u200f\ufeff")
    assert escapar(limpio) == limpio


def test_escapar_comando_v1():
    limpio = escapar("hola ##[error]x\n##[set-output name=a;]b")
    assert "##[" not in limpio
    assert escapar(limpio) == limpio


def test_registro_v1():
    r = ResultadoPartida(
        ganador="blancas",
        motivo="puntos",
        puntos={"blancas": 3, "negras": 1},
        jugadas=[
            {
                "n": 1,
                "color": "blancas",
                "jugada": "e2-e3",
                "combate": None,
                "tiempo": 0.01,
                "reloj": 0.01,
                "info": {"prof": 3},
            }
        ],
        reloj_usado={"blancas": 0.01, "negras": 0.0},
        salidas={"blancas": "::error::x", "negras": ""},
    )
    meta = {
        "blancas": "a",
        "negras": "b",
        "pareja": 0,
        "juego": 1,
        "semilla_maestra": 9,
        "otro": "se ignora",
    }
    d = a_json(r, meta)
    assert d["version"] == 1
    assert d["meta"] == {
        "blancas": "a",
        "negras": "b",
        "pareja": 0,
        "juego": 1,
        "semilla_maestra": 9,
    }
    assert set(d) == {
        "version",
        "meta",
        "jugadas",
        "ganador",
        "motivo",
        "puntos",
        "reloj_usado",
        "preparar_usado",
        "salidas",
    }
    assert set(d["jugadas"][0]) == {"n", "color", "jugada", "combate", "tiempo", "reloj", "info"}
    assert not d["salidas"]["blancas"].startswith("::")
    with pytest.raises(ValueError, match="pareja"):
        a_json(r, {"blancas": "a", "negras": "b", "juego": 1, "semilla_maestra": 9})
