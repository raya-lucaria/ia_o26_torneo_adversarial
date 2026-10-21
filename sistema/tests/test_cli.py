"""CLI `./torneo`: comandos sin Docker (rápidos) y, marcados `docker`, con Docker de verdad."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from cli import salida
from cli.comandos import main
from cli.oficial import leer_shard
from config import RAIZ

ROTO = RAIZ / "sistema" / "tests" / "equipos" / "roto"
LIMITES_RAPIDOS = """
reloj_s = 3.0
preparar_s = 60.0
memoria_mb = 1024
carpeta_mb = 100
archivo_mb = 50
salida_kb = 64
requirements_mb = 300
escala_limites = 1.0
"""
IGNORAR_COPIA = shutil.ignore_patterns(
    ".git",
    ".venv",
    "resultados",
    "privado",
    ".superpowers",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "*.egg-info",
)


@pytest.fixture(autouse=True)
def entorno(tmp_path, monkeypatch):
    """Reloj de 3 s (partidas rápidas) y resultados en una carpeta temporal."""
    limites = tmp_path / "limites.toml"
    limites.write_text(LIMITES_RAPIDOS, encoding="utf-8")
    monkeypatch.setenv("TORNEO_LIMITES", str(limites))
    monkeypatch.setenv("TORNEO_RESULTADOS", str(tmp_path / "resultados"))
    monkeypatch.setenv("TORNEO_SIMULTANEAS", "1")
    return tmp_path


def una_corrida(tmp_path: Path) -> Path:
    corridas = [c for c in (tmp_path / "resultados").iterdir() if c.is_dir()]
    assert len(corridas) == 1, corridas
    return corridas[0]


def registros_de(carpeta: Path) -> list[dict]:
    return [json.loads(r.read_text(encoding="utf-8")) for r in sorted(carpeta.glob("*.json"))]


# --- jugar ---------------------------------------------------------------------------------
def test_jugar_sin_docker_imprime_tabla(capsys, tmp_path):
    codigo = main(
        ["jugar", "random", "montecarlo", "--partidas", "2", "--sin-docker", "--semilla", "1"]
    )
    out = capsys.readouterr().out
    assert codigo == 0
    assert "sin Docker" in out and "sin aislamiento" in out
    assert re.search(r"#\s+Blancas\s+Negras\s+Ganador\s+Puntos\s+Jugadas\s+Reloj máx", out)
    assert re.search(r"^\s*1\s+random\s+montecarlo\s", out, re.MULTILINE)
    assert re.search(r"^\s*2\s+montecarlo\s+random\s", out, re.MULTILINE)
    assert "Resultado:" in out
    corrida = una_corrida(tmp_path)
    assert (corrida / "index.html").is_file()
    assert str(corrida / "index.html") in out
    assert len(registros_de(corrida)) == 2


def test_jugar_misma_semilla_mismas_partidas(capsys, tmp_path):
    def jugadas() -> list:
        main(["jugar", "random", "random", "--partidas", "1", "--sin-docker", "--semilla", "7"])
        corrida = una_corrida(tmp_path)
        regs = registros_de(corrida)
        shutil.rmtree(corrida)
        return [(j["jugada"], j["combate"]) for j in regs[0]["jugadas"]]

    assert jugadas() == jugadas()


def test_verbose_una_linea_por_jugada(capsys, tmp_path):
    main(
        ["jugar", "random", "montecarlo", "--partidas", "1", "--sin-docker", "--semilla", "3", "-v"]
    )
    out = capsys.readouterr().out
    registro = registros_de(una_corrida(tmp_path))[0]
    lineas = re.findall(r"^\s*\d+\s+(?:random|montecarlo)\s+[a-i]\d-[a-i]\d", out, re.MULTILINE)
    assert len(lineas) == len(registro["jugadas"]) > 0
    assert re.search(r"jug\s+quién\s+jugada\s+prof\s+nodos\s+eval\s+t\s+reloj", out)
    if any(j["combate"] for j in registro["jugadas"]):
        assert "ataque" in out


def test_ver_dibuja_el_tablero(capsys):
    main(
        ["jugar", "random", "random", "--partidas", "1", "--sin-docker", "--semilla", "2", "--ver"]
    )
    out = capsys.readouterr().out
    assert out.count("  a b c d e f g h i") > 10


def test_reloj_cerca_del_limite_avisa():
    registro = {
        "meta": {"blancas": "lento", "negras": "random", "pareja": 0, "juego": 0},
        "jugadas": [],
        "ganador": "blancas",
        "motivo": "puntos",
        "puntos": {"blancas": 3, "negras": 0},
        "reloj_usado": {"blancas": 28.5, "negras": 0.1},
        "salidas": {"blancas": "", "negras": ""},
    }
    avisos = salida.avisos(3, registro, 30.0)
    assert avisos == [
        "⚠ partida 3: lento usó 28.5 de 30 s de reloj (95 %). Está cerca de perder por reloj."
    ]


def test_jugador_inexistente_es_error_claro(capsys):
    assert main(["jugar", "nadie_existe", "random", "--sin-docker"]) == 2
    err = capsys.readouterr().err
    assert "estudiantes/nadie_existe" in err and "random" in err


# --- humano --------------------------------------------------------------------------------
def test_humano_puede_rendirse(capsys, monkeypatch):
    respuestas = iter(["zz", "a1-a9", "salir"])
    monkeypatch.setattr("builtins.input", lambda _texto="": next(respuestas))
    assert (
        main(["jugar", "humano", "random", "--partidas", "1", "--sin-docker", "--semilla", "1"])
        == 0
    )
    out = capsys.readouterr().out
    assert "mal escrita" in out  # «zz»
    assert "no es legal" in out  # «a1-a9»
    assert "Te rendiste" in out


# --- validar -------------------------------------------------------------------------------
def test_validar_falla_con_codigo_1(capsys):
    assert main(["validar", str(ROTO), "--sin-docker"]) == 1
    out = capsys.readouterr().out
    assert "✗" in out


def test_validar_participante_oficial_pasa(capsys):
    assert main(["validar", "random", "--sin-docker"]) == 0
    assert "✓" in capsys.readouterr().out


# --- repetir y visor -----------------------------------------------------------------------
def test_repetir_reproduce_log(capsys, tmp_path):
    main(["jugar", "random", "random", "--partidas", "1", "--sin-docker", "--semilla", "4"])
    capsys.readouterr()
    corrida = una_corrida(tmp_path)
    archivo = next(corrida.glob("*.json"))
    registro = json.loads(archivo.read_text(encoding="utf-8"))
    assert main(["repetir", str(archivo), "--pausa", "0"]) == 0
    out = capsys.readouterr().out
    assert out.count("  a b c d e f g h i") == 2 * (len(registro["jugadas"]) + 1)
    assert registro["jugadas"][-1]["jugada"] in out
    assert "Ganó" in out


def test_visor_regenera_index(capsys, tmp_path):
    main(["jugar", "random", "random", "--partidas", "1", "--sin-docker", "--semilla", "5"])
    corrida = una_corrida(tmp_path)
    (corrida / "index.html").unlink()
    assert main(["visor", str(corrida)]) == 0
    assert (corrida / "index.html").is_file()


# --- mini y liga ---------------------------------------------------------------------------
def test_mini_sin_equipo_e2e(capsys, tmp_path):
    assert main(["mini", "--sin-docker", "--semilla", "1"]) == 0
    out = capsys.readouterr().out
    corrida = una_corrida(tmp_path)
    assert (corrida / "index.html").is_file()
    assert str(corrida / "index.html") in out
    assert re.search(r"^Calificación", out, re.MULTILINE)
    assert re.search(r"^\s*1\s+(random|montecarlo)\s", out, re.MULTILINE)
    assert len(list((corrida / "parejas").glob("*.json"))) == 1
    assert "SB = calidad de victorias (desempate) · Anotados = peones que llegaron" in out


def test_mini_con_equipo_imprime_su_calificacion(capsys, tmp_path):
    equipo = tmp_path / "mi_equipo"
    shutil.copytree(RAIZ / "estudiantes" / "random", equipo)
    assert main(["mini", str(equipo), "--sin-docker", "--semilla", "1"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^\s*\d\s+mi_equipo\s", out, re.MULTILINE)
    # La copia de random no tiene equipo.toml ni INFORME válidos de alumno.
    assert "mi_equipo no recibiría calificación" in out


def equipo_valido(tmp_path: Path) -> Path:
    """Copia de random con equipo.toml e INFORME.md propios (pasa la validación)."""
    equipo = tmp_path / "mi_equipo"
    shutil.copytree(RAIZ / "estudiantes" / "random", equipo)
    (equipo / "equipo.toml").write_text(
        'nombre = "mi_equipo"\nintegrantes = [\n'
        '  { nombre = "Ana", clave_unica = "123456" },\n'
        '  { nombre = "Luis", clave_unica = "654321" },\n]\n',
        encoding="utf-8",
    )
    secciones = (RAIZ / "estudiantes" / "montecarlo" / "INFORME.md").read_text(encoding="utf-8")
    titulos = [linea for linea in secciones.splitlines() if linea.startswith("## ")]
    (equipo / "INFORME.md").write_text(
        "# Informe: mi_equipo\n\n"
        + "".join(
            f"{t}\n\nNuestro bot de prueba número {i} juega al azar; aquí lo explicamos.\n\n"
            for i, t in enumerate(titulos)
        ),
        encoding="utf-8",
    )
    return equipo


def test_mini_con_equipo_valido_imprime_calificacion(capsys, tmp_path):
    equipo = equipo_valido(tmp_path)
    assert main(["mini", str(equipo), "--sin-docker", "--semilla", "1"]) == 0
    out = capsys.readouterr().out
    linea = re.search(r"^Calificación si la liga fuera solo esto: (\d+) \((.*)\)\.$", out, re.M)
    assert linea, out
    assert "los lugares 1–3 entre alumnos sacan 10" in out


def test_liga_local(capsys, tmp_path):
    assert (
        main(
            [
                "liga",
                "random",
                "montecarlo",
                "--juegos-dados",
                "1",
                "--sin-docker",
                "--semilla",
                "2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Tabla" in out and "Calificación" not in out
    assert (una_corrida(tmp_path) / "index.html").is_file()


def test_leer_shard_es_base_1():
    assert leer_shard("1/4") == (0, 4)
    assert leer_shard("4/4") == (3, 4)
    for malo in ("0/4", "5/4", "a/b", "3"):
        with pytest.raises(ValueError):
            leer_shard(malo)


def test_liga_oficial_shard_y_publicar_sin_llave(capsys, tmp_path):
    estado = tmp_path / "estado"
    args = ["liga", "--oficial", "--sin-docker", "--estado", str(estado), "--juegos-dados", "1"]
    assert main([*args, "--semilla", "9", "--shard", "1/1"]) == 0
    assert (estado / "parejas" / "0.json").is_file()
    assert (estado / "manifest.json").is_file()
    capsys.readouterr()
    # Publicar sin la llave del profesor: se niega con un mensaje claro.
    llave = RAIZ / "sistema" / "llave_profesor.age.pub"
    if llave.exists():
        pytest.skip("este repo ya tiene la llave del profesor")
    publicar = ["liga", "--oficial", "--publicar", "--sin-docker", "--estado", str(estado)]
    assert main([*publicar, "--juegos-dados", "1"]) == 2
    assert "llave" in capsys.readouterr().err


def test_liga_oficial_no_publica_resultados_ajenos_al_calendario(capsys, tmp_path):
    estado = tmp_path / "estado"
    args = ["liga", "--oficial", "--sin-docker", "--estado", str(estado), "--juegos-dados", "1"]
    assert main([*args, "--semilla", "9", "--shard", "1/1"]) == 0
    capsys.readouterr()
    ruta = estado / "parejas" / "0.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["registros"][1]["meta"]["negras"] = "intruso"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    publicar = ["liga", "--oficial", "--publicar", "--sin-docker", "--estado", str(estado)]
    assert main([*publicar, "--juegos-dados", "1"]) == 2
    err = capsys.readouterr().err
    assert "no coinciden con el calendario" in err and "ajena" in err
    # Con otro número de juegos de dados el calendario guardado no coincide.
    assert main([*publicar, "--juegos-dados", "2"]) == 2
    assert "otro calendario" in capsys.readouterr().err


def test_liga_oficial_shard_sin_semilla_es_error(capsys, tmp_path):
    estado = tmp_path / "estado"
    assert (
        main(["liga", "--oficial", "--sin-docker", "--estado", str(estado), "--shard", "1/2"]) == 2
    )
    assert "--semilla" in capsys.readouterr().err


# --- Revisiones sueltas --------------------------------------------------------------------
def test_limpio_quita_ansi_y_control():
    sucio = "hola\x1b[31mrojo\x1b[0m\x1b]0;titulo\x07 \x07fin\r\nsigue\tok"
    assert salida.limpio(sucio) == "holarojo ?fin?\nsigue\tok"


def test_salida_de_bots_impresa_sin_ansi():
    registro = {
        "meta": {"blancas": "a", "negras": "b"},
        "salidas": {"blancas": "\x1b[2Jborra\x1b[H", "negras": ""},
    }
    assert "\x1b" not in "\n".join(salida.salidas_bots(registro))


def test_ver_dibuja_tras_cada_jugada_sin_costo_de_reloj():
    from arbitro.azar import Dados
    from arbitro.partida import jugar_partida
    from arbitro.procesos import BotLocal
    from cli.humano import Mirador
    from config import limites_prueba

    lim = limites_prueba()
    lineas: list[str] = []

    def lento(*partes, **kw):  # una terminal lentísima
        time.sleep(0.05)
        lineas.append(" ".join(map(str, partes)))

    mirador = Mirador({"blancas": "uno", "negras": "dos"}, lim.reloj_s, lento)
    llamadas: list = []

    def observador(estado, jugada):
        llamadas.append(jugada)
        mirador(estado, jugada)

    bots = RAIZ / "sistema" / "tests" / "bots"
    r = jugar_partida(
        BotLocal(bots / "fiel"),
        BotLocal(bots / "fiel"),
        Dados(1, 0, 0),
        lim,
        lim.efectivos(),
        1,
        observador=observador,
    )
    assert not r.falla_infraestructura, r.salidas
    assert llamadas[0] is None and llamadas[1:] == r.jugadas  # una vez por jugada
    # Dibujar tarda ≥ 0.15 s por jugada y no se cobra a nadie.
    assert all(j["tiempo"] < 0.15 for j in r.jugadas), [j["tiempo"] for j in r.jugadas]
    assert lineas[0].startswith("Le toca a uno (blancas)")
    assert any(linea.startswith(f"uno jugó {r.jugadas[0]['jugada']}") for linea in lineas)


def test_validar_falla_de_infraestructura_sale_con_2(capsys, monkeypatch):
    import validador
    from validador import Reporte

    monkeypatch.setattr(
        validador, "validar_equipo", lambda *a, **k: Reporte(True, [], [], humo_jugado=False)
    )
    assert main(["validar", "random", "--sin-docker"]) == 2
    assert "infraestructura" in capsys.readouterr().err


def test_mini_avisa_lo_que_falta_para_la_entrega(capsys, tmp_path, monkeypatch):
    import validador
    from validador import Reporte

    original = validador.validar_equipo

    def falso(carpeta, limites, *, entrega=True, **kw):
        if Path(carpeta).name == "mi_equipo":
            if entrega:
                return Reporte(False, ["INFORME.md: falta la sección 7."], [])
            return Reporte(True, [], [])
        return original(carpeta, limites, entrega=entrega, **kw)

    monkeypatch.setattr(validador, "validar_equipo", falso)
    assert main(["mini", str(equipo_valido(tmp_path)), "--sin-docker", "--semilla", "1"]) == 0
    out = capsys.readouterr().out
    assert "Calificación si la liga fuera solo esto" in out
    assert "⚠ Para recibirla en la liga real falta:" in out and "sección 7" in out


def test_participantes_oficiales_usan_las_reglas_del_validador(tmp_path, monkeypatch):
    from cli import comun, oficial

    for nombre in ("random", "tier_1", "mi_equipo", "humano", "_oculto", "Mal Nombre"):
        (tmp_path / nombre).mkdir()
        (tmp_path / nombre / "strategy.py").write_text("")
    monkeypatch.setattr(comun, "ESTUDIANTES", tmp_path)
    monkeypatch.setattr(oficial, "ESTUDIANTES", tmp_path)
    assert sorted(oficial.participantes_oficiales()) == ["mi_equipo", "random", "tier_1"]


class Contenedor:
    def __init__(self, nombre, estado, segundos, dueno=None):
        from datetime import UTC, datetime, timedelta

        creado = datetime.now(UTC) - timedelta(seconds=segundos)
        self.name = nombre
        self.status = estado
        self.attrs = {
            "State": {"Status": estado},
            "Created": creado.strftime("%Y-%m-%dT%H:%M:%S.0Z"),
            "Config": {"Labels": {"torneo.dueno": dueno} if dueno else {}},
        }
        self.borrado = False

    def remove(self, force):
        self.borrado = True


def test_limpiar_contenedores_solo_borra_huerfanos_reales():
    import socket

    import docker

    from cli import imagenes

    yo = socket.gethostname()
    muerto = 2**22 + 12345  # mayor que pid_max por defecto: no existe
    contenedores = {
        "torneo-bot-terminado": Contenedor("torneo-bot-terminado", "exited", 5),
        "torneo-bot-recien-creado": Contenedor("torneo-bot-recien-creado", "created", 5),
        "torneo-bot-creado-viejo": Contenedor("torneo-bot-creado-viejo", "created", 120),
        "torneo-bot-jugando": Contenedor("torneo-bot-jugando", "running", 60),
        "torneo-bot-sin-dueno-viejo": Contenedor("torneo-bot-sin-dueno-viejo", "paused", 3600),
        "torneo-calentar-viejo": Contenedor("torneo-calentar-viejo", "exited", 5),
        "torneo-torneo-run-1": Contenedor("torneo-torneo-run-1", "exited", 5),
        # Partida humana de 2 horas en otra terminal (contenedor del CLI vivo): se respeta.
        "torneo-bot-humano-largo": Contenedor(
            "torneo-bot-humano-largo", "paused", 7200, dueno="cli_vivo:7:c"
        ),
        # Su contenedor del CLI ya no existe (Ctrl+C): huérfano aunque sea reciente.
        "torneo-bot-cli-borrado": Contenedor(
            "torneo-bot-cli-borrado", "paused", 30, dueno="cli_borrado:7:c"
        ),
        # Lanzado desde esta misma máquina/contenedor: se revisa el proceso.
        "torneo-bot-mio-vivo": Contenedor(
            "torneo-bot-mio-vivo", "running", 7200, dueno=f"{yo}:{os.getpid()}:h"
        ),
        "torneo-bot-mio-muerto": Contenedor(
            "torneo-bot-mio-muerto", "running", 30, dueno=f"{yo}:{muerto}:h"
        ),
    }
    clis = {"cli_vivo": Contenedor("torneo-torneo-run-2", "running", 7200)}

    class Cliente:
        class containers:  # noqa: N801
            @staticmethod
            def list(all, filters):
                return list(contenedores.values())

            @staticmethod
            def get(nombre):
                if nombre not in clis:
                    raise docker.errors.NotFound("no existe")
                return clis[nombre]

    assert imagenes.limpiar_contenedores(Cliente) == 6
    borrados = {n for n, c in contenedores.items() if c.borrado}
    assert borrados == {
        "torneo-bot-terminado",
        "torneo-bot-creado-viejo",
        "torneo-bot-sin-dueno-viejo",
        "torneo-calentar-viejo",
        "torneo-bot-cli-borrado",
        "torneo-bot-mio-muerto",
    }


def test_bots_llevan_etiqueta_de_dueno():
    from arbitro.procesos import BotDocker, dueno
    from config import limites_prueba

    a = BotDocker(RAIZ / "estudiantes" / "random", "img", None, limites_prueba()).argumentos()
    pares = list(zip(a, a[1:], strict=False))
    assert ("--label", f"torneo.dueno={dueno()}") in pares
    host, pid, lugar = dueno().rsplit(":", 2)
    assert int(pid) > 0 and lugar in {"c", "h"}


def test_liga_oficial_nunca_construye_imagenes(monkeypatch, tmp_path):
    import docker

    from cli import ErrorCLI, imagenes

    equipo = tmp_path / "con_paquetes"
    equipo.mkdir()
    (equipo / "requirements.txt").write_text("sortedcontainers==2.4.0\n")

    class Imagen:
        id = "sha256:base"
        labels: dict = {}

    class Cliente:
        class images:  # noqa: N801
            @staticmethod
            def get(nombre):
                if nombre == imagenes.IMAGEN_BOTS:
                    return Imagen()
                raise docker.errors.ImageNotFound("no")

        class api:  # noqa: N801
            @staticmethod
            def build(**kw):
                raise AssertionError("no debe construir")

    monkeypatch.setenv("TORNEO_IMAGEN_DIGEST", "sha256:base")
    with pytest.raises(ErrorCLI, match="no se construye nada"):
        imagenes.asegurar_imagen_equipo(Cliente, "con_paquetes", equipo, avisar=lambda t: None)


def test_huella_igual_desde_cualquier_carpeta(tmp_path):
    script = RAIZ / "sistema" / "herramientas" / "huella.sh"
    for etapa in ("bot", "cli"):
        desde_raiz = subprocess.run([script, etapa], cwd=RAIZ, capture_output=True, text=True)
        desde_otra = subprocess.run([script, etapa], cwd=tmp_path, capture_output=True, text=True)
        assert desde_raiz.returncode == 0 and re.fullmatch(r"[0-9a-f]{16}\n", desde_raiz.stdout)
        assert desde_raiz.stdout == desde_otra.stdout


def test_script_usa_la_huella_y_nombres_locales():
    script = (RAIZ / "torneo").read_text(encoding="utf-8")
    assert "sistema/herramientas/huella.sh bot" in script
    assert "sistema/herramientas/huella.sh cli" in script
    assert '--label "torneo.fuente=$esperada"' in script
    assert 'IMAGEN_CLI="torneo-base:local"' in script
    assert 'IMAGEN_BOTS="torneo-bot-base:local"' in script
    compose = yaml.safe_load((RAIZ / "docker-compose.yml").read_text(encoding="utf-8"))
    servicio = compose["services"]["torneo"]
    assert servicio["image"] == "torneo-base:local"
    assert servicio["pull_policy"] == "never"


# --- script ./torneo -----------------------------------------------------------------------
def test_script_nunca_descarga_imagenes():
    script = (RAIZ / "torneo").read_text(encoding="utf-8")
    assert "docker pull" not in script
    assert "ghcr.io" not in script


def test_script_avisa_una_vez_que_construye_las_imagenes():
    script = (RAIZ / "torneo").read_text(encoding="utf-8")
    aviso = (
        "Construyendo las imágenes del torneo en tu máquina (la primera vez o tras una "
        "actualización del curso; tarda unos minutos)..."
    )
    assert script.count(aviso) == 1
    # El aviso va después de revisar si la imagen ya está (no sale si no hay nada que hacer).
    funcion = script[script.index("asegurar_imagen() {") :]
    assert funcion.index('== "$esperada" ]] && return 0') < funcion.index(aviso)


def test_script_construir_fuerza_la_reconstruccion():
    script = (RAIZ / "torneo").read_text(encoding="utf-8")
    bloque = script[script.index('if [[ "$comando" == construir ]]') :]
    bloque = bloque[: bloque.index("\nfi\n")]
    assert "export TORNEO_CONSTRUIR=1" in bloque
    assert "TORNEO_IMAGEN_DIGEST" in bloque  # en la liga no se construye nada


def test_construir_aparece_en_la_ayuda(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "construir" in capsys.readouterr().out


def test_construir_sin_docker_no_hace_nada(capsys):
    assert main(["construir", "--sin-docker"]) == 0
    assert "Sin Docker" in capsys.readouterr().out


def copiar_repo(tmp_path: Path) -> Path:
    destino = tmp_path / "ruta con espacio" / "mi repo"
    shutil.copytree(RAIZ, destino, ignore=IGNORAR_COPIA, symlinks=True)
    return destino


def test_script_desde_subdirectorio_con_espacios(tmp_path):
    repo = copiar_repo(tmp_path)
    ambiente = {**os.environ, "TORNEO_PYTHON": sys.executable}
    ambiente.pop("TORNEO_RESULTADOS")
    r = subprocess.run(
        ["../torneo", "--sin-docker", "--version"],
        cwd=repo / "estudiantes",
        capture_output=True,
        text=True,
        env=ambiente,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("torneo ")
    r = subprocess.run(
        ["../torneo", "--sin-docker", "jugar", "random", "random", "--partidas", "1"],
        cwd=repo / "estudiantes",
        capture_output=True,
        text=True,
        env=ambiente,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert list((repo / "resultados").glob("*/index.html"))


# --- Docker --------------------------------------------------------------------------------
@pytest.mark.docker
def test_compose_desde_ruta_con_espacios(tmp_path):
    repo = copiar_repo(tmp_path)
    limites = repo / "resultados" / "limites_prueba.toml"
    limites.parent.mkdir()
    limites.write_text(LIMITES_RAPIDOS, encoding="utf-8")
    ambiente = {**os.environ, "TORNEO_LIMITES": str(limites)}
    ambiente.pop("TORNEO_RESULTADOS")
    r = subprocess.run(
        ["../torneo", "jugar", "random", "montecarlo", "--partidas", "1", "--semilla", "1"],
        cwd=repo / "estudiantes",
        capture_output=True,
        text=True,
        env=ambiente,
        timeout=1800,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Ganador" in r.stdout
    indices = list((repo / "resultados").glob("*/index.html"))
    assert len(indices) == 1
    # Sin archivos de root en resultados/.
    assert indices[0].stat().st_uid == os.getuid()
    registro = registros_de(indices[0].parent)[0]
    # Los bots vieron su carpeta: cargaron strategy.py y jugaron. Es solo una prueba de humo:
    # con el reloj corto de LIMITES_RAPIDOS, perder por reloj (máquina cargada) también vale.
    assert registro["motivo"] in {"puntos", "sin_jugadas", "reloj"}, registro["salidas"]
    assert len(registro["jugadas"]) > 0


@pytest.mark.docker
def test_validar_con_docker(capsys, monkeypatch):
    monkeypatch.delenv("TORNEO_RAIZ_HOST", raising=False)
    monkeypatch.delenv("TORNEO_IMAGEN_DIGEST", raising=False)
    assert main(["validar", "montecarlo"]) == 0
    assert "✓" in capsys.readouterr().out


def _jugada(n, color, texto, combate=None):
    return {"n": n, "color": color, "jugada": texto, "combate": combate, "tiempo": 0.1,
            "reloj": 0.1, "info": None}  # fmt: skip


def _registro_minimo(jugadas):
    return {"meta": {"blancas": "a", "negras": "b"}, "jugadas": jugadas}


def test_v_jugada_ilegal_al_final_no_truena():
    reg = _registro_minimo(
        [_jugada(1, "blancas", "e2-e3"), _jugada(2, "negras", "a1-a1"), _jugada(3, "blancas", "x")]
    )
    lineas = salida.lineas_jugadas(reg)
    assert len(lineas) == 4  # encabezado + 3 jugadas
    assert not any("★" in linea for linea in lineas)


def test_v_muestra_los_datos_extra_de_reportar():
    j = _jugada(1, "blancas", "e2-e3")
    j["info"] = {"eval": 0.5, "muestras": 120, "tasa": 0.123456, "x\x1b[31m": "a" * 50}
    lineas = salida.lineas_jugadas(_registro_minimo([j]))
    assert lineas[1].rstrip().endswith("muestras=120 tasa=0.123 x=aaaaaaaaaa")
    assert "eval=" not in lineas[1]
    j["info"] = {f"k{i}": i for i in range(6)}
    assert salida.lineas_jugadas(_registro_minimo([j]))[1].rstrip().endswith("k3=3 …")


def test_v_marca_la_jugada_que_anota(monkeypatch):
    from motor import Estado

    # Posición fija: el peón blanco de e8 llega a la meta con e8-e9.
    fijo = Estado.desde_dict({"blancas": ["e8", "a3"], "negras": ["i7"], "turno": "blancas"})
    monkeypatch.setattr(salida.Estado, "inicial", classmethod(lambda cls: fijo))
    jugadas = [_jugada(1, "blancas", "e8-e9"), _jugada(2, "negras", "i7-i6")]
    lineas = salida.lineas_jugadas(_registro_minimo(jugadas))
    assert "★ anota (1–0)" in lineas[1]
    assert "★" not in lineas[2]

def test_actualizar_trae_todo_lo_del_curso():
    # Todo lo rastreado fuera de estudiantes/<equipo>/ es del curso: `./torneo actualizar` debe
    # traerlo (si falta algo, los repos de los alumnos se quedan con una versión vieja).
    script = (RAIZ / "torneo").read_text(encoding="utf-8")
    bloque = script[script.index("RUTAS_CURSO=(") :]
    rutas = set(bloque[len("RUTAS_CURSO=(") : bloque.index(")")].split())
    rastreados = subprocess.run(
        ["git", "-C", str(RAIZ), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    del_curso = {r.split("/")[0] for r in rastreados if not r.startswith("estudiantes/")}
    del_curso |= {"/".join(r.split("/")[:2]) for r in rastreados if r.startswith("estudiantes/")}
    assert del_curso <= rutas, del_curso - rutas
    assert not any(r == "estudiantes" for r in rutas)  # nunca la carpeta de un equipo


def test_diagnostico_no_falla_ni_cambia_nada():
    antes = subprocess.run(["git", "-C", str(RAIZ), "status", "--porcelain"], capture_output=True)
    hecho = subprocess.run([RAIZ / "torneo", "diagnostico"], capture_output=True, text=True)
    assert hecho.returncode == 0, hecho.stderr
    assert "Diagnóstico del torneo" in hecho.stdout
    despues = subprocess.run(["git", "-C", str(RAIZ), "status", "--porcelain"], capture_output=True)
    assert antes.stdout == despues.stdout
