"""Tests de la configuración: límites, calificación y nombres reservados."""

from pathlib import Path

import pytest

from config import (
    RAIZ,
    cargar_calificacion,
    cargar_limites,
    es_reservado,
    limites_prueba,
)

LIMITES_ESCALADOS = """
reloj_s = 30.0
preparar_s = 20.0
memoria_mb = 1024
carpeta_mb = 100
archivo_mb = 50
salida_kb = 64
requirements_mb = 300
escala_limites = 2.0
"""


@pytest.fixture
def toml_escalado(tmp_path: Path) -> Path:
    p = tmp_path / "limites.toml"
    p.write_text(LIMITES_ESCALADOS, encoding="utf-8")
    return p


def test_limites_publicos() -> None:
    lim = cargar_limites()
    assert (lim.reloj_s, lim.preparar_s, lim.memoria_mb, lim.escala_limites) == (30, 20, 1024, 1.0)
    tamanos = (lim.carpeta_mb, lim.archivo_mb, lim.salida_kb, lim.requirements_mb)
    assert tamanos == (100, 50, 64, 300)


def test_escala_multiplica_reloj_y_memoria(toml_escalado: Path) -> None:
    assert cargar_limites(toml_escalado).efectivos().reloj_s == pytest.approx(60.0)
    assert cargar_limites(toml_escalado).efectivos().memoria_mb == 2048


def test_escala_no_toca_preparar(toml_escalado: Path) -> None:
    assert cargar_limites(toml_escalado).efectivos().preparar_s == 20


def test_variable_de_entorno_sobreescribe(
    toml_escalado: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORNEO_LIMITES", str(toml_escalado))
    assert cargar_limites().escala_limites == pytest.approx(2.0)


def test_falta_un_limite(tmp_path: Path) -> None:
    p = tmp_path / "limites.toml"
    p.write_text("reloj_s = 30.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Faltan: preparar_s, memoria_mb"):
        cargar_limites(p)


def test_limites_prueba() -> None:
    lim = limites_prueba()
    assert (lim.reloj_s, lim.preparar_s) == (3, 5)
    assert lim.memoria_mb == cargar_limites().memoria_mb


def test_calificacion() -> None:
    assert cargar_calificacion() == {
        "random": 1,
        "montecarlo": 3,
        "tier_1": 5,
        "tier_2": 6,
        "tier_3": 7,
        "tier_4": 8.5,
        "tier_5": 10,
    }


def test_reservados() -> None:
    assert es_reservado("random") and es_reservado("tier_3") and es_reservado("_plantilla")
    assert es_reservado("montecarlo") and es_reservado("humano")
    assert not es_reservado("los_alfa_beta")


def test_raiz_del_repo() -> None:
    assert (RAIZ / "sistema" / "config.py").is_file()


def _fijadas(ruta: Path) -> dict[str, str]:
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    pares = (ln.split("#")[0].strip().split("==") for ln in lineas)
    return {p[0].lower(): p[1] for p in pares if len(p) == 2}


def test_constraints_coinciden_con_requirements() -> None:
    sistema = RAIZ / "sistema"
    constraints = _fijadas(sistema / "constraints.txt")
    requirements = _fijadas(sistema / "requirements.txt")
    assert {"numpy", "scipy", "numba"} <= constraints.keys()
    for paquete in ("numpy", "scipy", "numba"):
        assert constraints[paquete] == requirements[paquete]


def test_ruta_explicita_gana_a_variable_de_entorno(
    toml_escalado: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORNEO_LIMITES", str(toml_escalado))
    publicos = Path(__file__).resolve().parents[1] / "limites.toml"
    assert cargar_limites(publicos).escala_limites == 1.0


def test_clave_desconocida_se_rechaza(tmp_path: Path) -> None:
    p = tmp_path / "limites.toml"
    p.write_text(LIMITES_ESCALADOS + "nucleos = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Sobran: nucleos"):
        cargar_limites(p)


@pytest.mark.parametrize(
    ("linea", "nueva"),
    [
        ("reloj_s = 30.0", "reloj_s = 0"),
        ("memoria_mb = 1024", "memoria_mb = -5"),
        ("salida_kb = 64", 'salida_kb = "64"'),
        ("carpeta_mb = 100", "carpeta_mb = 100.5"),
        ("escala_limites = 2.0", "escala_limites = true"),
    ],
)
def test_valores_invalidos_se_rechazan(tmp_path: Path, linea: str, nueva: str) -> None:
    p = tmp_path / "limites.toml"
    p.write_text(LIMITES_ESCALADOS.replace(linea, nueva), encoding="utf-8")
    campo = nueva.split(" = ")[0]
    with pytest.raises(ValueError, match=campo):
        cargar_limites(p)
