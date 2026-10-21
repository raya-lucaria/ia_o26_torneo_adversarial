"""La documentación dice lo mismo que el código.

- Las tablas de límites y calificación se generan desde los TOML (`sincronizar_docs`).
- Cada ejemplo resuelto de `docs/reglas.md` y `docs/calificacion.md` está aquí como test.
- Los enlaces relativos existen y los comandos `./torneo ...` de los docs son válidos.
"""

import re
import shlex

import pytest

from config import RAIZ, cargar_calificacion
from herramientas import sincronizar_docs
from liga.calificacion import calificar
from liga.tabla import FilaTabla, construir_tabla
from motor import Estado, Jugada, prob_combate

README = RAIZ / "README.md"
AGENTS = RAIZ / "AGENTS.md"  # lo leen los asistentes de IA de los alumnos
DOCS = sorted((RAIZ / "docs").glob("*.md"))
GUIAS = ["reglas", "empezar", "probar", "entregar", "calificacion", "avanzado", "preguntas"]
# Capturas de pantalla que el profesor agrega después (docs/img/capturas/).
CAPTURAS_PENDIENTES = "img/capturas/"


def J(texto: str) -> Jugada:
    return Jugada.desde_texto(texto)


# --- Estructura -----------------------------------------------------------------------


def test_existen_todas_las_guias():
    for nombre in GUIAS:
        assert (RAIZ / "docs" / f"{nombre}.md").is_file(), f"Falta docs/{nombre}.md"


def test_docs_sincronizados():
    assert sincronizar_docs.verificar() == [], (
        "Tablas desactualizadas: corre `python -m herramientas.sincronizar_docs` desde sistema/."
    )


def test_sincronizar_reemplaza_solo_entre_marcadores():
    texto = "antes\n<!-- limites:inicio -->\nviejo\n<!-- limites:fin -->\ndespués\n"
    nuevo = sincronizar_docs.sincronizar(texto)
    assert nuevo.startswith("antes\n<!-- limites:inicio -->\n| Recurso")
    assert nuevo.endswith("<!-- limites:fin -->\ndespués\n")
    assert "viejo" not in nuevo
    assert sincronizar_docs.sincronizar(nuevo) == nuevo


@pytest.mark.parametrize(
    "malo",
    [
        "<!-- limites:inicio -->\nsin cierre\n",
        "<!-- limites:fin -->\nal revés\n<!-- limites:inicio -->\n",
        "<!-- limites:inicio --> en una línea <!-- limites:fin -->\n",
        "<!-- limites:inicio -->\n<!-- limites:inicio -->\n<!-- limites:fin -->\n"
        "<!-- limites:fin -->\n",
    ],
)
def test_sincronizar_marcadores_mal_puestos_son_error(malo):
    with pytest.raises(ValueError, match="limites"):
        sincronizar_docs.sincronizar(malo)


def test_tablas_tienen_los_valores_del_toml():
    limites = sincronizar_docs.tabla_limites()
    for valor in ("30 s", "10 s", "1 núcleo", "1024 MB", "100 MB", "50 MB", "64 KB", "300 MB"):
        assert valor in limites
    assert "escala" not in limites.lower()
    notas = sincronizar_docs.tabla_calificacion()
    for rival, nota in cargar_calificacion().items():
        assert f"| {nota:g} |" in notas, rival
    assert "Lugares 1, 2 y 3" in notas
    assert "Mejor bot de referencia al que le ganas" in notas


def test_readme_tiene_secciones():
    readme = README.read_text(encoding="utf-8")
    secciones = [
        "Empieza en 3 pasos",
        "Tu carpeta",
        "Entrada y salida",
        "Los participantes",
        "Cómo se califica",
        "Reglas que no puedes romper",
        "Equipos de 2",
        "Herramientas recomendadas",
    ]
    posiciones = []
    for t in secciones:
        assert f"## {t}" in readme, f"Falta la sección «{t}» en el README"
        posiciones.append(readme.index(f"## {t}"))
    assert posiciones == sorted(posiciones), "Las secciones del README están en otro orden"
    for svg in ("moverse.svg", "combate.svg", "anotar.svg"):
        assert f"docs/img/{svg}" in readme
    assert "cp -r estudiantes/montecarlo estudiantes/" in readme
    for guia in GUIAS:
        assert f"docs/{guia}.md" in readme


def test_readme_y_entregar_dicen_que_solo_cambias_tu_carpeta():
    for archivo in (README, RAIZ / "docs" / "entregar.md"):
        texto = archivo.read_text(encoding="utf-8")
        assert "revisar-entrega" in texto, archivo.name
        assert "estudiantes/<tu_equipo>/" in texto, archivo.name


def test_enlaces_relativos_existen():
    enlace = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
    for archivo in [README, AGENTS, *DOCS]:
        texto = archivo.read_text(encoding="utf-8")
        for destino in enlace.findall(texto):
            if re.match(r"[a-z]+:", destino) or destino.startswith("#"):
                continue
            if CAPTURAS_PENDIENTES in destino:
                continue
            ruta = (archivo.parent / destino.split("#")[0]).resolve()
            assert ruta.exists(), f"{archivo.name}: enlace roto a {destino}"


def test_comandos_torneo_de_los_docs_son_validos():
    comandos = pytest.importorskip("cli.comandos")
    parser = comandos.construir_parser()
    vistos = 0
    for archivo in [README, AGENTS, *DOCS]:
        for bloque in re.findall(r"```[a-z]*\n(.*?)```", archivo.read_text(encoding="utf-8"), re.S):
            for linea in bloque.splitlines():
                linea = linea.split(" #")[0].strip()
                if not linea.startswith("./torneo "):
                    continue
                linea = re.sub(r"<[^>]+>", "mi_equipo", linea)
                try:
                    parser.parse_args(shlex.split(linea)[1:])
                except SystemExit:
                    pytest.fail(f"{archivo.name}: comando inválido: {linea}")
                vistos += 1
    assert vistos >= 10


# --- Ejemplos de docs/reglas.md -------------------------------------------------------


def test_ejemplo_movimiento():
    e = Estado.desde_dict({"blancas": ["e4"], "negras": ["f5"], "turno": "blancas"})
    assert {str(j) for j in e.jugadas()} == {"e4-e5", "e4-d5", "e4-f5"}
    assert e.es_ataque(J("e4-f5")) and not e.es_ataque(J("e4-d5"))


def test_ejemplo_recto_bloqueado():
    # Un rival enfrente bloquea el paso recto; no se ataca recto.
    e = Estado.desde_dict({"blancas": ["e4"], "negras": ["e5"], "turno": "blancas"})
    assert {str(j) for j in e.jugadas()} == {"e4-d5", "e4-f5"}
    # Una pieza propia en diagonal tampoco deja pasar.
    e = Estado.desde_dict({"blancas": ["e4", "d5"], "negras": ["e5"], "turno": "blancas"})
    assert {str(j) for j in e.jugadas()} == {"e4-f5", "d5-d6", "d5-c6", "d5-e6"}


def test_tabla_de_probabilidades():
    dif = {-2: (0, 2), -1: (0, 1), 0: (1, 1), 1: (2, 1), 2: (2, 0)}
    esperado = {-2: 0.20, -1: 0.35, 0: 0.50, 1: 0.65, 2: 0.80}
    for d, (a, b) in dif.items():
        assert prob_combate(a, b) == pytest.approx(esperado[d])


def test_ejemplo_combate_blancas():
    # Atacante e4 con apoyo en d3 y f3 (2); defensor f5 con apoyo en g6 (1): p = 0.65.
    e = Estado.desde_dict(
        {"blancas": ["e4", "d3", "f3"], "negras": ["f5", "g6"], "turno": "blancas"}
    )
    j = J("e4-f5")
    assert e.es_ataque(j)
    assert e.prob_ataque(j) == pytest.approx(0.65)
    (p_gana, gana), (p_pierde, pierde) = e.siguiente(j)
    assert p_gana == pytest.approx(0.65) and p_pierde == pytest.approx(0.35)
    assert gana.piezas("blancas") == ["d3", "f3", "f5"] and gana.piezas("negras") == ["g6"]
    assert pierde.piezas("blancas") == ["d3", "f3"]
    assert pierde.piezas("negras") == ["f5", "g6"]


def test_ejemplo_apoyo_desde_el_origen():
    # g4 está detrás del destino f5, no del origen e4: no cuenta. p = 0.5.
    e = Estado.desde_dict({"blancas": ["e4", "g4"], "negras": ["f5"], "turno": "blancas"})
    assert e.prob_ataque(J("e4-f5")) == pytest.approx(0.5)


def test_ejemplo_combate_negras():
    # Negras d6 (apoyo c7: 1) atacan e5 (apoyo d4 y f4: 2): p = 0.35.
    e = Estado.desde_dict(
        {"blancas": ["e5", "d4", "f4"], "negras": ["d6", "c7"], "turno": "negras"}
    )
    assert e.prob_ataque(J("d6-e5")) == pytest.approx(0.35)


def test_ejemplo_anotar_caminando():
    e = Estado.desde_dict({"blancas": ["e8"], "negras": ["a9"], "turno": "blancas"})
    nuevo = e.aplicar(J("e8-e9"))
    assert nuevo.puntos == {"blancas": 1, "negras": 0}
    assert nuevo.piezas("blancas") == []


def test_ejemplo_anotar_ganando_ataque():
    e = Estado.desde_dict({"blancas": ["e8", "a2"], "negras": ["f9", "a9"], "turno": "blancas"})
    (p, gana), (_, pierde) = e.siguiente(J("e8-f9"))
    assert p == pytest.approx(0.5)
    assert gana.puntos["blancas"] == 1
    assert gana.piezas("blancas") == ["a2"] and gana.piezas("negras") == ["a9"]
    assert pierde.puntos["blancas"] == 0
    assert pierde.piezas("blancas") == ["a2"] and pierde.piezas("negras") == ["a9", "f9"]


def test_ejemplo_tres_puntos():
    e = Estado.desde_dict(
        {
            "blancas": ["e8", "a2"],
            "negras": ["a9", "b9"],
            "turno": "blancas",
            "puntos": {"blancas": 2, "negras": 0},
        }
    )
    assert e.aplicar(J("e8-e9")).ganador() == "blancas"


def test_ejemplo_sin_jugadas():
    e = Estado.desde_dict({"blancas": [], "negras": ["e5"], "turno": "blancas"})
    assert e.jugadas() == [] and e.ganador() == "negras"
    # Peón bloqueado: a4 tiene un rival enfrente (a5) y una pieza propia en su única
    # diagonal (b5). No se mueve, pero b5 sí: Blancas todavía tienen jugadas.
    e = Estado.desde_dict({"blancas": ["a4", "b5"], "negras": ["a5", "b6"], "turno": "blancas"})
    assert {str(j) for j in e.jugadas()} == {"b5-a6", "b5-c6"}
    assert not e.terminado()


def test_ejemplo_texto_o_jugada():
    e = Estado.inicial()
    assert e.aplicar("e2-e3") == e.aplicar(J("e2-e3"))
    assert str(J("e2-e3")) == "e2-e3"


# --- Ejemplos de docs/calificacion.md ---------------------------------------------------


def _registro(blancas: str, negras: str, gana: str, pareja: int) -> dict:
    ganador = "blancas" if gana == blancas else "negras"
    puntos = {"blancas": 0, "negras": 0}
    puntos[ganador] = 3
    return {
        "meta": {"blancas": blancas, "negras": negras, "pareja": pareja, "juego": 0},
        "ganador": ganador,
        "puntos": puntos,
    }


def _pareja(a: str, b: str, victorias_a: int, pareja: int) -> list[dict]:
    """4 partidas entre a y b; a gana `victorias_a`."""
    ganadores = [a] * victorias_a + [b] * (4 - victorias_a)
    return [
        _registro(*((a, b) if k % 2 == 0 else (b, a)), g, pareja) for k, g in enumerate(ganadores)
    ]


def test_ejemplo_desempate():
    registros = [
        *_pareja("alfa", "beta", 2, 1),
        *_pareja("alfa", "montecarlo", 1, 2),
        *_pareja("alfa", "random", 4, 3),
        *_pareja("beta", "montecarlo", 2, 4),
        *_pareja("beta", "random", 3, 5),
        *_pareja("montecarlo", "random", 4, 6),
    ]
    tabla = {f.equipo: f for f in construir_tabla(registros, semilla_maestra=1)}
    assert {e: f.puntos for e, f in tabla.items()} == {
        "montecarlo": 9,
        "alfa": 7,
        "beta": 7,
        "random": 1,
    }
    assert tabla["alfa"].sb == 27 and tabla["beta"].sb == 35
    assert [e for e, _ in sorted(tabla.items(), key=lambda x: x[1].posicion)] == [
        "montecarlo",
        "beta",
        "alfa",
        "random",
    ]


def test_ejemplo_calificacion():
    puntos = {
        "tier_2": 70,
        "top_1": 69,
        "top_2": 66,
        "top_3": 64,
        "tier_1": 58,
        "ana_luis": 58,
        "los_alfa_beta": 59,
        "montecarlo": 40,
        "random": 5,
    }
    orden = sorted(puntos, key=lambda e: -puntos[e])
    tabla = [
        FilaTabla(e, puntos[e], puntos[e], 0, 0, posicion=i + 1, partidas=4)
        for i, e in enumerate(orden)
    ]
    alumnos = {"top_1", "top_2", "top_3", "ana_luis", "los_alfa_beta"}
    notas = calificar(tabla, cargar_calificacion(), alumnos)
    assert notas == {
        "top_1": 10,
        "top_2": 10,
        "top_3": 10,
        "los_alfa_beta": 5,  # 59 > 58 de tier_1
        "ana_luis": 3,  # 58 = 58 de tier_1: empatar no es ganarle
    }
