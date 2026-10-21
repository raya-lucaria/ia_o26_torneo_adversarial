"""Comandos de `./torneo`: validar, jugar, liga, mini, repetir, visor, calentar, construir e
imagenes.

Todos aceptan `--sin-docker` (bots como subprocesos, sin límites ni aislamiento: solo
para depurar rápido). Con Docker, cada bot corre en su contenedor con los límites
oficiales. `main(argv)` regresa el código de salida (0 bien, 1 la carpeta del equipo
no pasa, 2 error de uso o de infraestructura). La liga oficial vive en `cli.oficial`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import config
from arbitro import registro
from arbitro.azar import Dados, semilla_bots
from arbitro.partida import jugar_partida
from arbitro.procesos import ProcesoBot
from cli import ErrorCLI, ErrorEquipo, salida
from cli.comun import (
    ESTUDIANTES,
    REFERENCIAS,
    Entorno,
    avisos_y_problemas,
    carpeta_resultados,
    cierre,
    disponibles,
    es_del_curso,
    imprimir,
    imprimir_tabla,
    jugar_liga,
    limites,
    linea_limites,
    nombre_de_participante,
    nueva_corrida,
    resolver_equipo,
    semilla_nueva,
    tabla_partidas,
)
from cli.humano import RELOJ_HUMANO_S, BotHumano, Mirador, RelojNormal, Vista
from cli.oficial import cmd_liga_oficial
from config import cargar_calificacion, es_reservado
from liga.calificacion import calificar, nota_por_referencias, referencias_vencidas
from liga.tabla import construir_tabla
from visor.generar import visor_partida

# argparse en español (sus mensajes salen de gettext; aquí van los que ve un alumno).
_ESPANOL = {
    "usage: ": "uso: ",
    "positional arguments": "argumentos",
    "options": "opciones",
    "show this help message and exit": "muestra esta ayuda y termina",
    "show program's version number and exit": "muestra la versión y termina",
    "the following arguments are required: %s": "faltan argumentos: %s",
    "unrecognized arguments: %s": "no conozco estos argumentos: %s",
    "invalid choice: %(value)r (choose from %(choices)s)": (
        "comando desconocido: %(value)r (usa uno de: %(choices)s)"
    ),
    "argument %(argument_name)s: %(message)s": "%(argument_name)s: %(message)s",
    "invalid %(type)s value: %(value)r": "valor inválido: %(value)r",
    "expected one argument": "falta su valor",
}
argparse._ = lambda texto: _ESPANOL.get(texto, texto)


def _version() -> str:
    with (config.SISTEMA / "pyproject.toml").open("rb") as archivo:
        return tomllib.load(archivo)["project"]["version"]


def _guardar(reg: dict, destino: Path, numero: int) -> None:
    """Visor HTML y registro JSON (mismo nombre) de la partida `numero` de la corrida."""
    html = visor_partida(reg, destino, numero)
    html.with_suffix(".json").write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")


def _semilla(args) -> int:
    return args.semilla if args.semilla is not None else semilla_nueva()


# --- validar -------------------------------------------------------------------------------
def cmd_validar(args, entorno: Entorno) -> int:
    from validador import validar_equipo

    nombre, carpeta = resolver_equipo(args.equipo)
    oficial = es_del_curso(nombre)
    publicos = limites()
    imprimir(f"Validando {salida.ruta_visible(carpeta)} ...")
    if entorno.docker:
        reporte = validar_equipo(
            carpeta, publicos, jugar_humo=False, oficial=oficial, entrega=args.entrega
        )
        if reporte.ok:
            try:
                entorno.preparar({nombre: carpeta})
            except ErrorEquipo as e:  # su requirements.txt: problema de la carpeta
                imprimir(f"✗ {e}")
                imprimir(f"✗ {nombre} NO pasa la validación.")
                return 1
            imprimir("Jugando 2 partidas de prueba contra random (una con cada color)...")
            reporte = validar_equipo(
                carpeta, publicos, docker=True, oficial=oficial, entrega=args.entrega
            )
    else:
        imprimir(salida.AVISO_SIN_DOCKER)
        imprimir("Jugando 2 partidas de prueba contra random (si lo demás está bien)...")
        reporte = validar_equipo(
            carpeta, publicos, docker=False, oficial=oficial, entrega=args.entrega
        )

    for aviso in reporte.avisos:
        imprimir(f"⚠ {salida.limpio(aviso)}")
    for error in reporte.errores:
        imprimir(f"✗ {salida.limpio(error)}")
    if not reporte.ok:
        entrega = " --entrega" if args.entrega else ""
        imprimir(
            f"✗ {nombre} NO pasa la validación ({len(reporte.errores)} errores). Arregla lo de "
            f"arriba y vuelve a correr ./torneo validar {args.equipo}{entrega}"
        )
        return 1
    if not reporte.humo_jugado:
        raise ErrorCLI(
            "No se pudieron jugar las partidas de prueba por una falla de infraestructura "
            "(no es culpa de tu bot). Vuelve a intentarlo."
        )
    imprimir(f"✓ {nombre} pasa la validación{' de entrega' if args.entrega else ''}.")
    return 0


# --- jugar ---------------------------------------------------------------------------------
def cmd_jugar(args, entorno: Entorno) -> int:
    nombres = [args.a, args.b]
    if nombres.count("humano") == 2:
        raise ErrorCLI("Al menos uno de los dos debe ser un bot (p. ej. humano montecarlo).")
    if args.partidas < 1:
        raise ErrorCLI("--partidas debe ser 1 o más.")
    carpetas = {n: resolver_equipo(n) for n in nombres if n != "humano"}
    equipos = {nombre: carpeta for nombre, carpeta in carpetas.values()}
    hay_humano = "humano" in nombres
    nombres = [n if n == "humano" else carpetas[n][0] for n in nombres]
    entorno.preparar(equipos)

    publicos = limites()
    reglas = replace(publicos, reloj_s=RELOJ_HUMANO_S) if hay_humano else publicos
    semilla = _semilla(args)
    destino = nueva_corrida()
    a, b = nombres
    cuantas = "1 partida" if args.partidas == 1 else f"{args.partidas} partidas"
    imprimir(f"Partidas: {a} vs {b} · {cuantas} · semilla {semilla}")
    imprimir(linea_limites(publicos, entorno.docker))
    if hay_humano:
        imprimir("Contra humano nadie pierde por reloj (el bot ve su reloj normal).")

    tabla = salida.TablaPartidas(nombres, args.partidas)
    en_vivo = not (args.verbose or args.ver or hay_humano)
    if en_vivo:
        imprimir("")
        imprimir(tabla.encabezado())
    registros: list[dict] = []
    for k in range(args.partidas):
        juego = k // 2
        blancas, negras = (a, b) if k % 2 == 0 else (b, a)
        vista = Vista()

        def crear(nombre: str, vista: Vista = vista) -> ProcesoBot:
            if nombre == "humano":
                return BotHumano(vista)
            bot = entorno.bot(nombre, equipos[nombre], publicos)
            if hay_humano:
                return RelojNormal(bot, publicos.reloj_s, nombre, vista)
            return bot

        mirador = None
        if args.ver and not hay_humano:
            mirador = Mirador(
                {"blancas": blancas, "negras": negras},
                publicos.reloj_s,
                imprimir,
                limpiar=sys.stdout.isatty(),
            )
        r = jugar_partida(
            crear(blancas),
            crear(negras),
            Dados(semilla, 0, juego),
            reglas,
            reglas.efectivos(),
            semilla_bots(semilla, 0, juego),
            observador=mirador,
        )
        meta = {
            "blancas": blancas,
            "negras": negras,
            "pareja": 0,
            "juego": juego,
            "semilla_maestra": semilla,
        }
        reg = registro.a_json(r, meta)
        _guardar(reg, destino, k + 1)
        registros.append(reg)
        if args.verbose:
            imprimir("")
            imprimir(salida.titulo_partida(k + 1, reg))
            for linea in [*salida.lineas_jugadas(reg), *salida.salidas_bots(reg)]:
                imprimir(linea)
        if args.ver and not hay_humano:
            imprimir(salida.repeticion(reg)[-1])  # el tablero final (con la última jugada)
        if en_vivo:
            imprimir(tabla.fila(k + 1, reg))
        else:
            imprimir(salida.final(reg))

    if not en_vivo:
        imprimir("")
        imprimir(tabla.encabezado())
        for k, reg in enumerate(registros):
            imprimir(tabla.fila(k + 1, reg))
    imprimir("")
    avisos_y_problemas(registros, publicos, con_salida=not args.verbose, sin=("humano",))
    imprimir(salida.resultado(registros))
    cierre(registros, destino)
    imprimir(f"Para repetir exactamente estas partidas: --semilla {semilla}")
    return 0


# --- liga y mini ---------------------------------------------------------------------------
def cmd_liga(args, entorno: Entorno) -> int:
    if args.oficial:
        return cmd_liga_oficial(args, entorno)
    if len(args.equipos) < 2:
        raise ErrorCLI("La liga necesita al menos 2 equipos: ./torneo liga random montecarlo")
    if args.shard or args.publicar or args.estado:
        raise ErrorCLI("--shard, --publicar y --estado son solo para la liga --oficial.")
    participantes = dict(resolver_equipo(e) for e in args.equipos)
    if len(participantes) != len(args.equipos):
        raise ErrorCLI("Hay equipos repetidos en la lista.")
    entorno.preparar(participantes)
    publicos = limites()
    semilla = _semilla(args)
    destino = nueva_corrida()
    registros = jugar_liga(participantes, args.juegos_dados, semilla, destino, entorno, publicos)
    tabla_partidas(registros, publicos, con_salida=True)
    tabla = construir_tabla(registros, semilla, list(participantes))
    imprimir_tabla("Tabla", tabla)
    cierre(registros, destino, tabla)
    return 0


def _primeras_lineas(errores: list[str], n: int = 5) -> list[str]:
    return [f"  ✗ {salida.limpio(e).splitlines()[0]}" for e in errores[:n]]


def cmd_mini(args, entorno: Entorno) -> int:
    from validador import validar_equipo

    participantes = {n: (ESTUDIANTES / n).resolve() for n in REFERENCIAS}
    alumno = None
    if args.equipo:
        alumno, carpeta = resolver_equipo(args.equipo)
        if alumno in participantes or es_reservado(alumno):
            raise ErrorCLI(
                f"«{alumno}» es un participante del curso. Usa el nombre de tu equipo: "
                "./torneo mini <tu_equipo>"
            )
        participantes[alumno] = carpeta
    entorno.preparar(participantes)
    publicos = limites()
    no_calificables: set[str] = set()
    errores: list[str] = []
    faltan_entrega: list[str] = []
    if alumno:
        reporte = validar_equipo(participantes[alumno], publicos, jugar_humo=False, entrega=False)
        if not reporte.ok:
            no_calificables, errores = {alumno}, reporte.errores
        else:
            entrega = validar_equipo(
                participantes[alumno], publicos, jugar_humo=False, entrega=True
            )
            faltan_entrega = entrega.errores

    semilla = _semilla(args)
    destino = nueva_corrida()
    imprimir(f"Mini torneo: {', '.join(sorted(participantes))} (como la liga, pero corto)")
    registros = jugar_liga(participantes, 1, semilla, destino, entorno, publicos)
    tabla_partidas(registros, publicos, con_salida=True)
    tabla = construir_tabla(registros, semilla, list(participantes))
    referencias = cargar_calificacion()
    notas = calificar(tabla, referencias, {alumno} if alumno else set(), no_calificables)
    imprimir_tabla("Tabla (si la liga fuera solo esto)", tabla)

    if alumno is None:
        imprimir(
            "Calificación: no aplica (random y montecarlo son participantes del curso). "
            "Prueba con ./torneo mini <tu_equipo>."
        )
    elif alumno not in notas:
        imprimir(
            f"Calificación: {alumno} no recibiría calificación hasta que su carpeta pase "
            "la validación (INFORME.md, equipo.toml...):"
        )
        for linea in _primeras_lineas(errores):
            imprimir(linea)
        imprimir(f"  Detalle completo: ./torneo validar {args.equipo}")
    else:
        _imprimir_calificacion(tabla, referencias, alumno)
        if faltan_entrega:
            imprimir("⚠ Para recibirla en la liga real falta:")
            for linea in _primeras_lineas(faltan_entrega):
                imprimir(linea)
            imprimir(f"  Detalle completo: ./torneo validar {args.equipo} --entrega")
    cierre(registros, destino, tabla)
    imprimir(f"Semilla: {semilla} (repite este mini torneo con --semilla {semilla})")
    return 0


def _imprimir_calificacion(tabla, referencias: dict[str, float], alumno: str) -> None:
    """Calificación por referencias vencidas (sin el top 3: aquí serías el único alumno)."""
    nota = nota_por_referencias(tabla, referencias, alumno)
    vencidas = referencias_vencidas(tabla, referencias, alumno)
    if not vencidas:
        detalle = "no le ganas a nadie: necesitas estrictamente más puntos que random para un 1"
    elif len(vencidas) == 1:
        detalle = f"le ganas a {vencidas[0]}"
    else:
        detalle = f"le ganas a {' y '.join(vencidas)}; cuenta la más alta"
    filas = {f.equipo: f for f in tabla}
    empates = [
        r for r in REFERENCIAS if r not in vencidas and filas[r].puntos == filas[alumno].puntos
    ]
    if empates:
        detalle += f"; empatas con {' y '.join(empates)} y empatar no es ganarle"
    imprimir(f"Calificación si la liga fuera solo esto: {nota:g} ({detalle}).")
    imprimir(
        "  En la liga real hay más rivales (tier_1 … tier_5) y los lugares 1–3 entre "
        "alumnos sacan 10."
    )


# --- imagenes ------------------------------------------------------------------------------
def cmd_imagenes(args, entorno: Entorno) -> int:
    """Construye las imágenes de equipo (la liga oficial las prepara antes de empezar)."""
    if not entorno.docker:
        raise ErrorCLI("imagenes necesita Docker (quita --sin-docker).")
    from cli import imagenes

    nombres = args.equipos or [n for n in disponibles() if nombre_de_participante(n)]
    equipos = dict(resolver_equipo(n) for n in nombres)
    c = entorno.cliente()
    imagenes.imagen_base(c)
    fallas = 0
    for nombre, carpeta in equipos.items():
        try:
            imagenes.asegurar_imagen_equipo(c, nombre, carpeta, avisar=imprimir)
            imprimir(f"✓ {imagenes.imagen_equipo(nombre)}")
        except ErrorEquipo as e:
            imprimir(f"✗ {e}")
            fallas += 1
    return 1 if fallas else 0


# --- repetir, visor y calentar -----------------------------------------------------------------
def _leer_json(ruta: Path) -> dict:
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ErrorCLI(f"No pude leer {ruta}: {e}") from None


def _registros_de(datos: object) -> list[dict]:
    if not isinstance(datos, dict):
        return []
    if isinstance(datos.get("registros"), list):
        return datos["registros"]
    if datos.get("version") == registro.VERSION and "jugadas" in datos:
        return [datos]
    return []


def cmd_repetir(args, entorno: Entorno) -> int:
    registros = _registros_de(_leer_json(args.archivo))
    if not registros:
        raise ErrorCLI(f"{args.archivo} no es el registro de una partida.")
    pausa = args.pausa if args.pausa is not None else (0.4 if sys.stdout.isatty() else 0.0)
    for n, reg in enumerate(registros, 1):
        imprimir(salida.titulo_partida(n, reg))
        for texto in salida.repeticion(reg):
            imprimir(texto)
            imprimir("")
            if pausa:
                time.sleep(pausa)
        imprimir(salida.final(reg))
    return 0


def cmd_visor(args, entorno: Entorno) -> int:
    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        raise ErrorCLI(f"No existe la carpeta {carpeta}.")
    registros = []
    for ruta in sorted(carpeta.glob("*.json")) + sorted(
        (carpeta / "parejas").glob("*.json"), key=lambda r: (len(r.stem), r.stem)
    ):
        registros += _registros_de(_leer_json(ruta))
    if not registros:
        raise ErrorCLI(f"No encontré partidas (.json) en {carpeta}.")
    cierre(registros, carpeta)
    return 0


def cmd_calentar(args, entorno: Entorno) -> int:
    if not entorno.docker:
        imprimir("Sin Docker no hace falta: numba guarda su caché junto al motor.")
        return 0
    from cli import imagenes

    c = entorno.cliente()
    imagenes.limpiar_contenedores(c)
    imagenes.calentar(c, carpeta_resultados() / ".cache", avisar=imprimir)
    return 0


def cmd_del_script(args, entorno: Entorno) -> int:
    """`instalar`, `diagnostico` y `actualizar` los hace el script ./torneo (funcionan aunque
    falte Python o .venv/); aquí solo aparecen en la ayuda."""
    raise ErrorCLI(f"{args.comando} lo hace el script: corre ./torneo {args.comando} en tu repo.")


def cmd_construir(args, entorno: Entorno) -> int:
    """Las imágenes ya las reconstruyó el script ./torneo (TORNEO_CONSTRUIR=1); aquí falta
    la caché de numba, igual que en `calentar`."""
    if not entorno.docker:
        imprimir("Sin Docker no hay imágenes que construir.")
        return 0
    imprimir("Imágenes del torneo listas.")
    return cmd_calentar(args, entorno)


# --- Argumentos ----------------------------------------------------------------------------
class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(2, f"✗ {message}\nAyuda: ./torneo {self.prog.split()[-1]} --help\n")


def construir_parser() -> argparse.ArgumentParser:
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument(
        "--sin-docker",
        action="store_true",
        default=argparse.SUPPRESS,
        help="bots como subprocesos con el Python de .venv/ (se prepara solo la primera vez): "
        "cuentan el reloj y preparar, pero sin límite de memoria ni CPU. "
        "En Codespaces se pone solo",
    )
    parser = _Parser(
        prog="torneo",
        description="Torneo «Breakthrough de combate». Ej.: ./torneo jugar mi_equipo montecarlo",
        parents=[comun],
    )
    parser.add_argument("--version", action="version", version=f"torneo {_version()}")
    sub = parser.add_subparsers(dest="comando", metavar="comando", parser_class=_Parser)

    p = sub.add_parser("validar", parents=[comun], help="revisa tu carpeta antes de entregar")
    p.add_argument("equipo", help="nombre de tu carpeta en estudiantes/")
    p.add_argument(
        "--entrega",
        action="store_true",
        help="revisión de entrega (la del PR): el INFORME.md debe estar completo",
    )
    p.set_defaults(funcion=cmd_validar)

    p = sub.add_parser("jugar", parents=[comun], help="juega partidas entre dos bots")
    p.add_argument("a", help="equipo (o «humano» para jugar tú)")
    p.add_argument("b", help="equipo rival")
    p.add_argument("--partidas", type=int, default=4, help="cuántas partidas (default 4)")
    p.add_argument("--semilla", type=int, help="semilla de los dados (para repetir)")
    p.add_argument("-v", "--verbose", action="store_true", help="una línea por jugada")
    p.add_argument("--ver", action="store_true", help="tablero en vivo")
    p.set_defaults(funcion=cmd_jugar)

    p = sub.add_parser("liga", parents=[comun], help="todos contra todos")
    p.add_argument("equipos", nargs="*", help="equipos que juegan")
    p.add_argument("--juegos-dados", type=int, default=2, help="juegos de dados por pareja")
    p.add_argument("--semilla", type=int, help="semilla maestra")
    p.add_argument("--oficial", action="store_true", help="liga oficial (profesor y CI)")
    p.add_argument("--shard", help="con --oficial: juega solo el pedazo i/n (i de 1 a n)")
    p.add_argument("--estado", help="con --oficial: carpeta de la liga")
    p.add_argument("--publicar", action="store_true", help="con --oficial: solo publica")
    p.add_argument(
        "--aceptar-incidentes",
        action="store_true",
        help="con --publicar: publica aunque haya parejas excluidas",
    )
    p.set_defaults(funcion=cmd_liga)

    p = sub.add_parser(
        "mini", parents=[comun], help="mini torneo contra random y montecarlo, con calificación"
    )
    p.add_argument("equipo", nargs="?", help="tu equipo (sin él: random vs montecarlo)")
    p.add_argument("--semilla", type=int, help="semilla maestra")
    p.set_defaults(funcion=cmd_mini)

    p = sub.add_parser("repetir", parents=[comun], help="repite una partida guardada")
    p.add_argument("archivo", help="registro .json de resultados/")
    p.add_argument("--pausa", type=float, help="segundos entre jugadas")
    p.set_defaults(funcion=cmd_repetir)

    p = sub.add_parser("visor", parents=[comun], help="genera el index.html de una carpeta")
    p.add_argument("carpeta", help="carpeta de resultados/")
    p.set_defaults(funcion=cmd_visor)

    p = sub.add_parser("calentar", parents=[comun], help="prepara la caché de numba")
    p.set_defaults(funcion=cmd_calentar)

    p = sub.add_parser(
        "construir",
        parents=[comun],
        help="reconstruye las imágenes del torneo y la caché de numba (unos minutos)",
        description="Reconstruye desde cero, en tu máquina, las imágenes de Docker del torneo "
        "(CLI y bots) y luego corre «calentar». Úsalo si algo raro pasa con Docker o después "
        "de actualizar el sistema. Tarda unos minutos.",
    )
    p.set_defaults(funcion=cmd_construir)

    for nombre, ayuda in (
        ("instalar", "crea o repara .venv/: el Python del torneo para --sin-docker y tu editor"),
        ("diagnostico", "revisa Docker, .venv/, git y tu carpeta, y dice qué hacer"),
        ("actualizar", "trae la última versión del curso (nunca toca tu carpeta)"),
    ):
        p = sub.add_parser(nombre, parents=[comun], help=ayuda)
        p.set_defaults(funcion=cmd_del_script)

    p = sub.add_parser("imagenes", parents=[comun], help="construye las imágenes de equipos")
    p.add_argument("equipos", nargs="*", help="equipos (sin nombres: todos los de estudiantes/)")
    p.set_defaults(funcion=cmd_imagenes)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "comando", None):
        parser.print_help()
        return 2
    entorno = Entorno(sin_docker=getattr(args, "sin_docker", False))
    try:
        return args.funcion(args, entorno)
    except ErrorCLI as e:
        print(f"✗ {e}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("\nInterrumpido.", file=sys.stderr, flush=True)
        return 130
