"""Reglas de seguridad y forma de los workflows de GitHub y del Codespace."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import yaml

from config import RAIZ

WORKFLOWS = RAIZ / ".github" / "workflows"
REPO_ORG = "raya-lucaria/ia_o26_torneo_adversarial"
GUARDIA_ORG = f"github.repository == '{REPO_ORG}'"
GUARDIA_ALUMNOS = f"github.repository != '{REPO_ORG}'"


def cargar(nombre: str) -> dict:
    datos = yaml.safe_load((WORKFLOWS / nombre).read_text(encoding="utf-8"))
    datos["on"] = datos.get("on", datos.get(True))  # PyYAML lee «on» como True
    return datos


def todos() -> list[str]:
    return sorted(p.name for p in WORKFLOWS.glob("*.yml"))


def test_existen_los_workflows():
    assert set(todos()) >= {
        "prueba.yml",
        "revisar-pr.yml",
        "liga-oficial.yml",
        "liga-mini.yml",
        "_liga.yml",
    }


def test_ningun_workflow_usa_pull_request_target():
    for nombre in todos():
        w = cargar(nombre)
        eventos = w["on"] if isinstance(w["on"], (dict, list)) else [w["on"]]
        assert "pull_request_target" not in eventos, nombre
        assert "pull_request_target" not in (WORKFLOWS / nombre).read_text(), nombre


def test_workflows_de_org_tienen_guardia_de_repo():
    for nombre in todos():
        if nombre == "prueba.yml":
            continue
        for job, datos in cargar(nombre)["jobs"].items():
            assert GUARDIA_ORG in str(datos.get("if", "")), f"{nombre}:{job}"


def test_prueba_solo_corre_en_repos_de_alumnos():
    w = cargar("prueba.yml")
    for job, datos in w["jobs"].items():
        assert GUARDIA_ALUMNOS in str(datos.get("if", "")), job
    assert w["on"]["push"]["branches"] == ["main"]
    assert w["on"]["push"]["paths"] == ["estudiantes/**"]
    assert w["on"]["workflow_dispatch"]["inputs"]["partidas"]["default"] == 2
    assert w["concurrency"]["cancel-in-progress"] is True
    assert w["permissions"] == {"contents": "read"}
    texto = (WORKFLOWS / "prueba.yml").read_text()
    assert "./torneo validar" in texto and "./torneo calentar" in texto
    assert "random" in texto and "montecarlo" in texto
    assert "repeticiones" in texto


def test_revisar_pr_sin_secretos_y_solo_lectura():
    w = cargar("revisar-pr.yml")
    assert set(w["on"]) == {"pull_request"}
    assert w["permissions"] == {"contents": "read"}
    texto = (WORKFLOWS / "revisar-pr.yml").read_text()
    assert "${{ secrets." not in texto and "secrets: inherit" not in texto
    for datos in w["jobs"].values():
        assert datos.get("permissions", {"contents": "read"}) == {"contents": "read"}
        assert "environment" not in datos


def test_liga_oficial_solo_manual_y_con_environment():
    w = cargar("liga-oficial.yml")
    assert set(w["on"]) == {"workflow_dispatch"}
    entradas = w["on"]["workflow_dispatch"]["inputs"]
    assert entradas["juegos_dados"]["default"] == 2
    assert entradas["shards"]["default"] == 20
    assert entradas["verbose"]["default"] is False
    (llamada,) = w["jobs"].values()
    assert llamada["uses"] == "./.github/workflows/_liga.yml"
    assert llamada["with"]["modo"] == "oficial"

    liga = cargar("_liga.yml")
    assert set(liga["on"]) == {"workflow_call"}
    jobs = liga["jobs"]
    assert list(jobs)[:4] == ["preparar", "imagenes", "jugar", "publicar"]
    # El Environment con aprobación protege la llave de los tiers (solo en modo oficial).
    assert "'liga-oficial'" in jobs["preparar"]["environment"]
    assert "inputs.modo == 'oficial'" in jobs["preparar"]["environment"]
    # imagenes: la única llave es LIGA_LLAVE, solo en el paso que restaura el paquete.
    pasos_con_secretos = [
        paso["name"] for paso in jobs["imagenes"]["steps"] if "${{ secrets." in json.dumps(paso)
    ]
    assert pasos_con_secretos == ["Restaurar participantes (único paso con la llave)"]
    assert "TIERS_DEPLOY_KEY" not in json.dumps(jobs["imagenes"])
    jugar = jobs["jugar"]
    assert jugar["runs-on"] == "${{ vars.MAQUINA_LIGA || 'ubuntu-latest' }}"
    assert jugar["strategy"]["max-parallel"] == 20
    assert jugar["strategy"]["fail-fast"] is False
    assert jobs["publicar"]["permissions"]["contents"] == "write"
    for nombre in ("preparar", "imagenes", "jugar"):
        assert jobs[nombre]["permissions"] == {"contents": "read"}, nombre
    for nombre, job in jobs.items():
        assert "github.ref == 'refs/heads/main'" in job["if"], nombre
    assert "github.ref == 'refs/heads/main'" in llamada["if"]
    # publicar: el token solo en el paso del commit.
    publicar = jobs["publicar"]
    assert publicar["steps"][0]["with"]["persist-credentials"] is False
    con_token = [p.get("name") for p in publicar["steps"] if "github.token" in json.dumps(p)]
    assert con_token == ["Commit a la rama ${{ env.RAMA }}"]


def test_liga_usa_imagenes_construidas_para_esa_corrida():
    jobs = cargar("_liga.yml")["jobs"]
    pasos = json.dumps(jobs["imagenes"]["steps"])
    assert "TORNEO_CONSTRUIR" in pasos and "docker save" in pasos
    jugar = jobs["jugar"]
    assert jugar["env"]["TORNEO_IMAGEN_DIGEST"] == "${{ needs.imagenes.outputs.digest }}"
    assert jugar["needs"] == ["preparar", "imagenes"]
    assert "docker load" in json.dumps(jugar["steps"])
    assert jobs["publicar"]["env"]["TORNEO_IMAGEN_DIGEST"] == "${{ needs.imagenes.outputs.digest }}"


def test_liga_fija_las_llaves_de_github():
    texto = (WORKFLOWS / "_liga.yml").read_text()
    assert "StrictHostKeyChecking=yes" in texto and "accept-new" not in texto
    assert (
        "github.com ssh-ed25519 "
        "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" in texto
    )


def test_liga_mini_llama_al_mismo_workflow_sin_tiers():
    w = cargar("liga-mini.yml")
    assert set(w["on"]) == {"workflow_dispatch"}
    (llamada,) = w["jobs"].values()
    assert llamada["uses"] == "./.github/workflows/_liga.yml"
    assert llamada["with"]["modo"] == "mini"
    assert llamada["with"]["shards"] == 2
    texto = (WORKFLOWS / "_liga.yml").read_text()
    assert "ensayo" in texto and "oficial" in texto


def test_liga_desconfia_de_los_logs_de_los_bots():
    texto = (WORKFLOWS / "_liga.yml").read_text()
    assert "::stop-commands::" in texto
    assert "TIERS_DEPLOY_KEY" in texto and "rm -f" in texto
    assert "--shard" in texto and "--oficial" in texto


def test_ningun_workflow_publica_imagenes():
    # Las imágenes se construyen en cada máquina; nada se sube a un registro.
    assert not (WORKFLOWS / "imagen-base.yml").exists()
    for nombre in todos():
        texto = (WORKFLOWS / nombre).read_text(encoding="utf-8")
        assert "ghcr.io" not in texto, nombre
        assert "docker/login-action" not in texto, nombre
        assert "docker push" not in texto, nombre
        assert "push: true" not in texto, nombre
        assert "packages: write" not in texto, nombre


def test_codeowners():
    assert (RAIZ / ".github" / "CODEOWNERS").read_text().split() == ["*", "@uumami"]


def test_devcontainer_sin_docker_con_venv():
    # Codespaces no levanta Docker (docker-in-docker no funcionó): ./torneo corre ahí con
    # --sin-docker usando .venv/, que crea `./torneo instalar` al crear el Codespace.
    datos = json.loads((RAIZ / ".devcontainer" / "devcontainer.json").read_text())
    assert "docker" not in json.dumps(datos.get("features", {}))
    assert "3.12" in datos["image"]
    assert "preparar.sh" in datos["postCreateCommand"]
    ajustes = datos["customizations"]["vscode"]["settings"]
    assert ajustes["python.defaultInterpreterPath"].endswith("/.venv/bin/python")
    script = (RAIZ / ".devcontainer" / "preparar.sh").read_text()
    assert "./torneo instalar" in script
    torneo = (RAIZ / "torneo").read_text()
    assert '"${CODESPACES:-}" == true' in torneo and 'set -- "$@" --sin-docker' in torneo


# --- Resumen de prueba.yml / revisar-pr.yml --------------------------------------------------
def _modulo_resumen():
    ruta = RAIZ / ".github" / "scripts" / "resumen_prueba.py"
    spec = importlib.util.spec_from_file_location("resumen_prueba", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _partida(carpeta: Path, nombre: str, blancas: str, negras: str, ganador: str, **kw):
    reg = {
        "version": 1,
        "meta": {"blancas": blancas, "negras": negras, "pareja": 0, "juego": 0},
        "jugadas": [],
        "ganador": ganador,
        "motivo": kw.get("motivo", "puntos"),
        "puntos": {"blancas": 3, "negras": 0},
        "reloj_usado": kw.get("reloj", {"blancas": 1.0, "negras": 1.0}),
        "salidas": {},
    }
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / nombre).write_text(json.dumps(reg), encoding="utf-8")


def test_resumen_marca_bien_y_avisos(tmp_path):
    r = _modulo_resumen()
    res = tmp_path / "resultados"
    _partida(res / "a", "1.json", "mi_eq", "random", "blancas", reloj={"blancas": 5.0, "negras": 1})
    _partida(res / "a", "2.json", "random", "mi_eq", "negras")
    _partida(
        res / "b", "1.json", "mi_eq", "montecarlo", "negras", reloj={"blancas": 29.5, "negras": 3}
    )
    _partida(res / "b", "2.json", "montecarlo", "mi_eq", "blancas", motivo="error")
    (res / ".cache").mkdir()
    (res / ".cache" / "x.json").write_text("{}")
    memoria = tmp_path / "memoria.log"
    memoria.write_text("torneo-bot:mi_eq 120.5MiB / 1GiB\ntorneo-bot:mi_eq 1.2GiB / 1GiB\nbasura\n")
    texto = r.resumen(
        "mi_eq", validacion_ok=True, resultados=res, memoria=memoria, reloj_s=30.0, memoria_mb=1024
    )
    assert "✅" in texto and "⚠️" in texto
    assert "2/2" in texto  # contra random
    assert "0/2" in texto  # contra montecarlo
    assert "29.5" in texto  # reloj máximo
    assert "Error del bot" in texto or "error" in texto
    assert "1229" in texto  # 1.2 GiB en MB


def test_resumen_cuenta_solo_partidas_jugadas(tmp_path):
    r = _modulo_resumen()
    _partida(tmp_path, "1.json", "mi_eq", "random", "blancas")
    _partida(tmp_path, "2.json", "random", "mi_eq", "")  # falla de infraestructura
    assert r.contar_partidas(tmp_path, "mi_eq") == 1


def test_resumen_validacion_fallida_y_sin_partidas(tmp_path):
    r = _modulo_resumen()
    texto = r.resumen(
        "mi_eq",
        validacion_ok=False,
        resultados=tmp_path / "nada",
        memoria=tmp_path / "no.log",
        reloj_s=30.0,
        memoria_mb=1024,
    )
    assert "⚠️" in texto and "No se jugaron" in texto


def test_revisar_pr_job_estable_y_diff_de_git():
    w = cargar("revisar-pr.yml")
    assert list(w["jobs"]) == ["revisar-entrega"]
    job = w["jobs"]["revisar-entrega"]
    assert job["name"] == "revisar-entrega"
    assert job["steps"][0]["with"]["fetch-depth"] == 0
    texto = (WORKFLOWS / "revisar-pr.yml").read_text()
    assert "git diff --raw --no-renames --no-abbrev -z" in texto
    assert "::stop-commands::" in texto
    assert 'kill "$muestreo"' in texto
    assert "--contar" in texto and "no se pudieron jugar" not in texto
    assert "repeticiones" in texto
    assert 'validar "$EQUIPO" --entrega' in texto
    assert "--entrega" not in (WORKFLOWS / "prueba.yml").read_text()


def test_gitattributes_lf():
    texto = (RAIZ / ".gitattributes").read_text()
    assert "* text=auto eol=lf" in texto
    for patron in ("*.npy", "*.npz", "*.bin", "*.pkl", "*.joblib", "*.gz", "*.png", "*.jpg"):
        assert f"{patron} binary" in texto, patron


def test_gitattributes_data_de_equipos_sin_conversion():
    # Un .csv o .txt de data/ no se toca (ni finales de línea): su hash es el registrado.
    salida = subprocess.run(
        ["git", "-C", str(RAIZ), "check-attr", "text", "--", "estudiantes/eq/data/t.csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert salida.strip().endswith("text: unset"), salida


def test_proteger_main_cubre_ramas_de_publicacion():
    texto = (RAIZ / "sistema" / "herramientas" / "proteger_main.sh").read_text()
    assert '"context": "revisar-entrega"' in texto
    assert '"refs/heads/oficial", "refs/heads/ensayo"' in texto
    assert texto.count('{"type": "deletion"}') == 2


def _paso(job: dict, nombre: str) -> dict:
    return next(p for p in job["steps"] if p.get("name") == nombre)


def test_liga_cifra_con_destinatario_publico_tras_codigo_de_alumnos():
    liga = cargar("_liga.yml")
    assert liga["defaults"]["run"]["shell"] == "bash"  # bash -eo pipefail
    jobs = liga["jobs"]
    for job, paso in (("imagenes", "Guardar imágenes"), ("jugar", "Empaquetar el shard")):
        datos = json.dumps(_paso(jobs[job], paso))
        assert "LIGA_LLAVE" not in datos and "LIGA_DESTINATARIO" in datos, paso
    script = (RAIZ / ".github" / "scripts" / "liga_restaurar.sh").read_text()
    assert "LIGA_DESTINATARIO=$(age-keygen -y" in script
    # En jugar, la llave solo se usa antes de correr bots.
    nombres = [p.get("name", p.get("uses")) for p in jobs["jugar"]["steps"]]
    con_llave = [
        i for i, p in enumerate(jobs["jugar"]["steps"]) if "secrets.LIGA_LLAVE" in json.dumps(p)
    ]
    assert max(con_llave) < nombres.index("Preparar Docker y la caché de numba")
    pages = jobs["pages"]["steps"][0]
    assert pages["with"]["persist-credentials"] is False


def test_liga_no_imprime_detalles_de_tiers():
    paso = _paso(
        cargar("_liga.yml")["jobs"]["imagenes"],
        "Construir imágenes de participantes con requirements.txt",
    )["run"]
    assert 'if [[ "$eq" == tier_* ]]' in paso
    assert '>>"$RUNNER_TEMP/registros_tiers.txt" 2>&1' in paso


def test_prueba_usa_los_mismos_nombres_que_el_validador():
    from config import RESERVADOS
    from validador import reglas

    texto = (WORKFLOWS / "prueba.yml").read_text()
    m = re.search(r"=~ (\^\S+\$) \]\]", texto)
    assert m and m.group(1) == reglas.PATRON_NOMBRE.pattern
    caso = re.search(r"^\s*(random .*)\) continue ;;$", texto, re.MULTILINE)
    assert caso
    patrones = {p.strip() for p in caso.group(1).split("|")}
    assert patrones == set(RESERVADOS) | {"tier_*", "_*"}


def test_juntar_shards_rechaza_estados_de_otra_liga():
    texto = (WORKFLOWS / "_liga.yml").read_text()
    assert 'cmp -s "$global" "$ESTADO/estado_liga.json"' in texto
