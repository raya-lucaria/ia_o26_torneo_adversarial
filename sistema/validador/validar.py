"""Revisión de la carpeta de un equipo: `validar_equipo` y `validar_nombre_nuevo`.

Revisa, en este orden: nombre, archivos (enlaces, cachés, binarios, tamaños,
permisos), `equipo.toml`, `INFORME.md`, código (`strategy.py` e imports),
`requirements.txt` y, si todo lo anterior pasa, 2 partidas de humo contra `random`.
Todos los mensajes son para alumnos: en español y con qué hacer para arreglarlo.
Nunca lanza excepciones por el contenido de la carpeta: todo termina en el reporte.
"""

from __future__ import annotations

import difflib
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from arbitro.azar import Dados, semilla_bots
from arbitro.partida import COLORES, ResultadoPartida, jugar_partida
from arbitro.procesos import BotDocker, BotLocal
from config import RAIZ, Limites, es_reservado, limites_prueba
from validador import reglas
from validador.archivos import revisar_archivos
from validador.codigo import nombres_definidos, revisar_python
from validador.requirements import auditar_archivo

INFORME_EJEMPLO = RAIZ / "estudiantes" / "montecarlo" / "INFORME.md"
RIVAL_HUMO = RAIZ / "estudiantes" / "random"
IMAGEN_RIVAL = "torneo-bot-base:local"  # random no tiene requirements.txt
SEMILLA_HUMO = 2026
# Avisar si `preparar` usó más de esta fracción de su límite: en la liga puede tardar más.
AVISO_PREPARAR = 0.5

# Motivos de derrota en el humo que indican un bot roto (no solo un bot débil).
_MOTIVOS_ROTOS = {
    "preparar": "preparar (incluye importar strategy.py) tardó demasiado",
    "error": "tu bot lanzó una excepción o se cerró",
    "ilegal": "tu bot regresó una jugada ilegal o mal formada",
    "memoria": "tu bot se pasó de memoria",
}


@dataclass
class Reporte:
    """Resultado de la validación. `ok` es False si hay al menos un error.

    `humo_jugado` es True solo si las partidas de humo se jugaron de verdad (sin
    falla de infraestructura). Quien exija el humo (p. ej. el PR) debe revisarlo.
    """

    ok: bool
    errores: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    humo_jugado: bool = False


def validar_nombre_nuevo(nombre: str) -> list[str]:
    """Errores del nombre de un equipo de alumnos (formato y nombres reservados)."""
    errores = _errores_formato_nombre(nombre)
    if not errores and es_reservado(nombre):
        errores.append(
            f"El nombre «{nombre}» está reservado para los participantes del curso "
            "(random, montecarlo, humano, tier_*, o que empiece con _). Elige otro nombre "
            "y renombra tu carpeta."
        )
    return errores


def _errores_formato_nombre(nombre: str) -> list[str]:
    if reglas.PATRON_NOMBRE.fullmatch(nombre):
        return []
    return [
        f"El nombre {nombre!r} no es válido. Usa de 3 a 30 caracteres: solo minúsculas sin "
        "acentos, dígitos y guion bajo (por ejemplo: los_alfa_beta)."
    ]


def validar_equipo(
    carpeta: Path,
    limites: Limites,
    jugar_humo: bool = True,
    docker: bool = False,
    oficial: bool = False,
    entrega: bool = True,
) -> Reporte:
    """Revisa la carpeta de un equipo.

    `oficial=True` es para nuestros participantes (random, montecarlo, tier_*):
    se permiten nombres reservados, claves de ejemplo y el informe de ejemplo.
    `docker=True` juega el humo en contenedores (`torneo-bot:<equipo>`).
    `entrega=False` es el modo prueba (`./torneo validar`, prueba.yml): lo que falta
    por escribir en INFORME.md (secciones vacías, texto de la plantilla, copia del de
    montecarlo) es aviso; con `entrega=True` (el PR) es error.
    """
    carpeta = Path(carpeta)
    if carpeta.is_symlink() or not carpeta.is_dir():
        return Reporte(False, [f"No existe la carpeta {carpeta}. Revisa el nombre."], [])
    nombre = carpeta.resolve().name
    errores = _errores_formato_nombre(nombre) if oficial else validar_nombre_nuevo(nombre)
    inventario = revisar_archivos(carpeta, limites)
    errores += inventario.errores
    avisos = list(inventario.avisos)
    _revisar_equipo_toml(carpeta, nombre, oficial, errores, avisos)
    _revisar_informe(carpeta, nombre, oficial, entrega, errores, avisos)
    _revisar_codigo(carpeta, inventario.archivos, errores, avisos)
    requisitos = _leer(carpeta / "requirements.txt", errores)
    if requisitos is not None:
        errores += auditar_archivo(requisitos)[0]
    humo_jugado = False
    if jugar_humo and not errores:
        humo_jugado = _jugar_humo(carpeta, nombre, limites, docker, errores, avisos)
    return Reporte(not errores, errores, avisos, humo_jugado)


def _leer(ruta: Path, errores: list[str]) -> bytes | None:
    """Bytes de un archivo normal; None si no existe o es un enlace (eso ya es error)."""
    if ruta.is_symlink() or not ruta.is_file():
        return None
    try:
        return ruta.read_bytes()
    except OSError:
        errores.append(f"{ruta.name}: no se pudo leer. Revisa sus permisos.")
        return None


def _texto(datos: bytes, archivo: str, errores: list[str]) -> str | None:
    try:
        return datos.decode("utf-8")
    except UnicodeDecodeError as e:
        errores.append(f"{archivo} no es UTF-8 válido (byte {e.start}). Guárdalo como UTF-8.")
        return None


# --- equipo.toml ------------------------------------------------------------------------


def _revisar_equipo_toml(
    carpeta: Path, nombre: str, oficial: bool, errores: list[str], avisos: list[str]
) -> None:
    ruta = carpeta / "equipo.toml"
    toml = f"estudiantes/{nombre}/equipo.toml"  # el archivo a editar, tal cual
    crudo = _leer(ruta, errores)
    if crudo is None:
        if not ruta.is_symlink():
            errores.append(f"Falta {toml}. Copia el de estudiantes/montecarlo/ y cámbialo.")
        return
    texto = _texto(crudo, toml, errores)
    if texto is None:
        return
    try:
        datos = tomllib.loads(texto)
    except tomllib.TOMLDecodeError as e:
        linea = re.search(r"line (\d+)", str(e))
        donde = f", línea {linea[1]}" if linea else ""
        errores.append(
            f"{toml}{donde}: está mal escrito (detalle técnico: {e}). Revisa comillas, "
            "signos = y corchetes; copia el formato de estudiantes/montecarlo/equipo.toml."
        )
        return
    if datos.get("nombre") != nombre:
        errores.append(
            f"En {toml}, nombre = {datos.get('nombre')!r}, pero la carpeta se llama "
            f'«{nombre}». Deben ser iguales: escribe nombre = "{nombre}".'
        )
    integrantes = datos.get("integrantes")
    if not isinstance(integrantes, list) or not all(isinstance(i, dict) for i in integrantes):
        errores.append(
            f'En {toml} falta la lista integrantes = [{{ nombre = "...", '
            'clave_unica = "..." }, ...].'
        )
        return
    if len(integrantes) != 2:
        if oficial and len(integrantes) == 1:
            avisos.append(f"{toml} tiene 1 integrante (participante del curso).")
        else:
            errores.append(
                f"{toml} tiene {len(integrantes)} integrantes; los equipos son de "
                "exactamente 2. Si el profesor les dio permiso para otro número, él lo "
                "acepta a mano: díganle."
            )
    claves = []
    patron = r"[0-9]+" if oficial else r"[0-9]{6,12}"
    for n, persona in enumerate(integrantes, start=1):
        if not isinstance(persona.get("nombre"), str) or not persona["nombre"].strip():
            errores.append(f"{toml}, integrante {n}: falta su nombre.")
        clave = persona.get("clave_unica")
        if not isinstance(clave, str) or not re.fullmatch(patron, clave):
            errores.append(
                f"{toml}, integrante {n}: clave_unica debe ser texto de 6 a 12 dígitos, "
                f'entre comillas, por ejemplo clave_unica = "000123456" (tiene {clave!r}).'
            )
            continue
        claves.append(clave)
    if oficial:
        return
    if any(set(c) == {"0"} for c in claves):
        errores.append(
            f"{toml}: hay una clave_unica de ejemplo (puros ceros). Pon la clave única "
            "real de cada integrante."
        )
    if len(set(claves)) < len(claves):
        errores.append(
            f"{toml}: dos integrantes tienen la misma clave_unica. Cada integrante "
            "pone su propia clave."
        )


# --- INFORME.md -------------------------------------------------------------------------


def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().split())


def _sin_bloques_de_codigo(texto: str) -> str:
    """El texto con las líneas de bloques ``` o ~~~ en blanco (mismas posiciones)."""
    lineas, cerca = [], None
    for linea in texto.split("\n"):
        marca = re.match(r"[ \t]{0,3}(`{3,}|~{3,})", linea)
        if cerca is None and marca:
            cerca = marca[1][0]
            lineas.append("")
        elif cerca is not None:
            if marca and marca[1][0] == cerca:
                cerca = None
            lineas.append("")
        else:
            lineas.append(linea)
    return "\n".join(lineas)


# Título de sección numerada: «## 3. ...» o «# 3) ...»; «### 3.1» es subsección.
_TITULO = re.compile(r"^#{1,6}[ \t]*(\d+)[ \t]*[.)](?!\d)", re.MULTILINE)


def _revisar_informe(
    carpeta: Path,
    nombre: str,
    oficial: bool,
    entrega: bool,
    errores: list[str],
    avisos: list[str],
) -> None:
    pendientes: list[str] = []  # sin llenar: error al entregar, aviso en modo prueba
    informe = f"estudiantes/{nombre}/INFORME.md"  # el archivo a editar, tal cual
    _revisar_contenido_informe(carpeta, informe, oficial, errores, avisos, pendientes)
    if entrega:
        errores += pendientes
    else:
        avisos += [f"{p} Antes de entregar completa tu INFORME.md." for p in pendientes]


def _revisar_contenido_informe(
    carpeta: Path,
    informe: str,
    oficial: bool,
    errores: list[str],
    avisos: list[str],
    pendientes: list[str],
) -> None:
    ruta = carpeta / "INFORME.md"
    datos = _leer(ruta, errores)
    if datos is None:
        if not ruta.is_symlink():
            errores.append(
                f"Falta {informe}. Es obligatorio: usa las 9 secciones del de "
                "estudiantes/montecarlo/INFORME.md."
            )
        return
    texto = _texto(datos, informe, errores)
    if texto is None:
        return
    limpio = _sin_bloques_de_codigo(texto)
    titulos = list(_TITULO.finditer(limpio))
    encontrados: dict[int, tuple[int, int]] = {}  # sección → (fin del título, inicio siguiente)
    for k, t in enumerate(titulos):
        siguiente = titulos[k + 1].start() if k + 1 < len(titulos) else len(limpio)
        encontrados.setdefault(int(t[1]), (t.end(), siguiente))
    faltan = [i for i in range(1, 10) if i not in encontrados]
    if faltan:
        lista = "; ".join(f"{i}. {reglas.SECCIONES_INFORME[i - 1]}" for i in faltan)
        errores.append(
            f"A {informe} le faltan secciones: {lista}. Escribe cada una como título, "
            "por ejemplo «## 3. Búsqueda (algoritmo y por qué)»."
        )
    vacias = []
    for i, (fin, siguiente) in sorted(encontrados.items()):
        if not 1 <= i <= 9:
            continue
        resto = limpio[fin:siguiente].split("\n", 1)  # sin el resto de la línea del título
        if len(resto) < 2 or not resto[1].strip():
            vacias.append(str(i))
    if vacias:
        pendientes.append(
            f"{informe}: la sección {', '.join(vacias)} está vacía. Explica qué hicieron."
            if len(vacias) == 1
            else f"{informe}: las secciones {', '.join(vacias)} están vacías. "
            "Explica qué hicieron en cada una."
        )
    if oficial:
        return
    plantilla = [t for t in reglas.TEXTOS_PLANTILLA if t.lower() in texto.lower()]
    if plantilla:
        pendientes.append(
            f"{informe} todavía tiene texto de la plantilla de montecarlo («{plantilla[0]}»). "
            "Bórralo y escribe sobre su propio bot."
        )
    if not INFORME_EJEMPLO.is_file():
        avisos.append(
            "No se encontró estudiantes/montecarlo/INFORME.md: no se pudo revisar que tu "
            "informe no sea una copia del de ejemplo."
        )
        return
    ejemplo = INFORME_EJEMPLO.read_text(encoding="utf-8")
    tope = 4 * len(ejemplo)  # un informe enorme no es copia y compararlo sería lento
    similitud = difflib.SequenceMatcher(
        None, _normalizar(texto[:tope]), _normalizar(ejemplo)
    ).ratio()
    if similitud >= reglas.SIMILITUD_MAXIMA_INFORME:
        pendientes.append(
            f"{informe} es casi una copia del de montecarlo ({similitud:.0%} igual). "
            "Escribe su propio informe: qué hace su bot, cómo y por qué."
        )


# --- Código -----------------------------------------------------------------------------


def _revisar_codigo(
    carpeta: Path, archivos: list[Path], errores: list[str], avisos: list[str]
) -> None:
    estrategia = carpeta / "strategy.py"
    if estrategia.is_symlink() or not estrategia.is_file():
        errores.append(
            "Falta strategy.py. Es lo único que se ejecuta: copia el de estudiantes/montecarlo/."
        )
    for ruta in archivos:
        if ruta.suffix != ".py":
            continue
        rel = ruta.relative_to(carpeta).as_posix()
        fuente = _leer(ruta, errores)
        if fuente is None:
            continue
        en_herramientas = rel.split("/")[0] == reglas.CARPETA_HERRAMIENTAS
        h = revisar_python(rel, fuente, en_herramientas)
        errores += h.errores
        avisos += h.avisos
        if rel != "strategy.py":
            continue
        definidos = nombres_definidos(fuente)
        if definidos is None:
            continue  # el error de sintaxis ya está en el reporte
        for funcion, firma in (("preparar", "config"), ("jugar", "estado, tiempo_restante")):
            if funcion not in definidos:
                errores.append(
                    f"strategy.py no define la función {funcion}. Agrega "
                    f"«def {funcion}({firma}):» (mira estudiantes/montecarlo/strategy.py)."
                )


# --- Humo -------------------------------------------------------------------------------


def _bot(carpeta: Path, imagen: str, publicos: Limites, docker: bool):
    if docker:
        return BotDocker(carpeta, imagen, None, publicos)
    return BotLocal(carpeta, publicos.salida_kb)


def _cola(texto: str, n: int = 1500) -> str:
    return texto if len(texto) <= n else "..." + texto[-n:]


def _jugar_humo(
    carpeta: Path,
    nombre: str,
    limites: Limites,
    docker: bool,
    errores: list[str],
    avisos: list[str],
) -> bool:
    """2 partidas contra random (una de cada color). Un bot roto es error.

    Si `preparar` pasa de `AVISO_PREPARAR` de su límite, es aviso (deja margen).
    Regresa True si se jugaron sin fallas de infraestructura.
    """
    # Reloj corto para que sea rápido; `preparar` con el límite real (es justo).
    publicos = replace(limites, reloj_s=min(limites.reloj_s, limites_prueba().reloj_s))
    jugado = True
    preparar_max = 0.0
    for juego, color in enumerate(COLORES):
        propio = _bot(carpeta, f"torneo-bot:{nombre}", publicos, docker)
        rival = _bot(RIVAL_HUMO, IMAGEN_RIVAL, publicos, docker)
        blancas, negras = (propio, rival) if color == "blancas" else (rival, propio)
        r: ResultadoPartida = jugar_partida(
            blancas,
            negras,
            Dados(SEMILLA_HUMO, 0, juego),
            publicos,
            publicos.efectivos(),
            semilla_bots(SEMILLA_HUMO, 0, juego),
        )
        partida = f"Partida de prueba {juego + 1} contra random (tu bot con {color})"
        if r.falla_infraestructura:
            avisos.append(
                f"{partida}: no se pudo jugar por una falla de infraestructura (no es culpa "
                "de tu bot). Vuelve a intentarlo."
            )
            jugado = False
            continue
        preparar_max = max(preparar_max, r.preparar_usado.get(color, 0.0))
        if r.ganador == color:
            continue
        if r.motivo in _MOTIVOS_ROTOS:
            salida = _cola(r.salidas.get(color, "")).strip()
            errores.append(
                f"{partida}: perdió por «{r.motivo}»: {_MOTIVOS_ROTOS[r.motivo]}. "
                "Arréglalo antes de entregar."
                + (f"\nLo que imprimió tu bot:\n{salida}" if salida else "")
            )
            return True
        if r.motivo == "reloj":
            avisos.append(
                f"{partida}: se quedó sin reloj (con {publicos.reloj_s:g} s de prueba). "
                "Revisa que tu bot use tiempo_restante para repartir el tiempo."
            )
    if preparar_max > AVISO_PREPARAR * publicos.preparar_s:
        avisos.append(aviso_preparar(preparar_max, publicos.preparar_s))
    return jugado


def aviso_preparar(usado: float, limite: float) -> str:
    """Aviso de `preparar` lento (sin ⚠: lo pone quien lo imprime)."""
    return (
        f"Tu preparar tardó {usado:.1f} s de {limite:g}. En la liga los runners de GitHub "
        "pueden ser hasta ~2,5× más lentos al compilar numba: deja margen. Ver "
        "docs/avanzado.md#numba-que-compila-rápido"
    )
