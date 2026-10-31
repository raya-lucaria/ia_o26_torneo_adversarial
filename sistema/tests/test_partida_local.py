"""Partidas con BotLocal (subprocesos): reglas del árbitro y bots que fallan."""

from dataclasses import replace
from pathlib import Path

from arbitro.azar import Dados, semilla_bots
from arbitro.partida import jugar_partida
from arbitro.procesos import BotLocal, BotNoSeCongela
from config import limites_prueba
from motor import Estado, Jugada

BOTS = Path(__file__).parent / "bots"
LIMITES = limites_prueba()


def local(nombre: str) -> BotLocal:
    return BotLocal(BOTS / nombre)


def partida(blancas: str, negras: str, semilla: int = 1, publicos=LIMITES, efectivos=None):
    return jugar_partida(
        local(blancas),
        local(negras),
        Dados(semilla, 0, 0),
        publicos,
        efectivos or publicos.efectivos(),
        semilla_bots(semilla, 0, 0),
    )


def proyeccion(r) -> tuple:
    """Lo que debe repetirse con la misma semilla: jugadas, combates, estado y resultado."""
    estado = Estado.inicial()
    for j in r.jugadas:
        c = j["combate"]
        estado = estado.aplicar(Jugada.desde_texto(j["jugada"]), None if c is None else c["gano"])
    return (
        [(j["color"], j["jugada"], j["combate"]) for j in r.jugadas],
        estado.a_dict(),
        r.ganador,
        r.motivo,
        r.puntos,
    )


def test_fiel_contra_fiel_termina():
    r = partida("fiel", "fiel")
    assert r.motivo in {"puntos", "sin_jugadas"}
    assert not r.falla_infraestructura
    assert r.ganador in {"blancas", "negras"}
    assert r.jugadas and all(j["reloj"] < LIMITES.reloj_s for j in r.jugadas)
    # El estado final reconstruido coincide con los puntos reportados.
    assert proyeccion(r)[1]["puntos"] == r.puntos
    assert any(j["combate"] for j in r.jugadas)
    assert set(r.preparar_usado) == {"blancas", "negras"}
    assert all(0 <= t < LIMITES.preparar_s for t in r.preparar_usado.values())


def test_misma_semilla_misma_partida():
    r1, r2 = partida("azaroso", "azaroso", 3), partida("azaroso", "azaroso", 3)
    assert proyeccion(r1) == proyeccion(r2)
    assert proyeccion(r1) != proyeccion(partida("azaroso", "azaroso", 4))


def test_bot_ve_reloj_publico_aunque_el_efectivo_sea_mayor():
    publicos = replace(LIMITES, reloj_s=3.0)
    efectivos = replace(LIMITES, reloj_s=6.0)
    r = partida("excede_publico", "fiel", publicos=publicos, efectivos=efectivos)
    assert r.motivo in {"puntos", "sin_jugadas"}, r.salidas["blancas"]
    assert r.reloj_usado["blancas"] > 3.0
    vistos = [j["info"]["tiempo_restante"] for j in r.jugadas if j["color"] == "blancas"]
    assert vistos[0] == 3.0 and vistos[-1] == 0


def test_bucle_pierde_por_reloj():
    r = partida("bucle", "fiel")
    assert (r.ganador, r.motivo) == ("negras", "reloj")
    assert "sin reloj" in r.salidas["blancas"]


def test_excepcion_pierde_por_error():
    r = partida("fiel", "excepcion")
    assert (r.ganador, r.motivo) == ("blancas", "error")
    assert "ups, mi bot falló" in r.salidas["negras"] and "Traceback" in r.salidas["negras"]


def test_ilegal_pierde():
    r = partida("ilegal", "fiel")
    assert (r.ganador, r.motivo) == ("negras", "ilegal")
    assert "a1-a9" in r.salidas["blancas"]


def test_jugada_como_texto_es_valida():
    r = partida("texto", "fiel")
    assert r.motivo in {"puntos", "sin_jugadas"}
    assert r.jugadas[0]["jugada"] == "a2-a3"  # primera jugada legal de blancas, como texto


def test_jugada_tupla_es_ilegal():
    r = partida("tupla", "fiel")
    assert (r.ganador, r.motivo) == ("negras", "ilegal")
    assert "tupla" in r.salidas["blancas"]


def test_import_lento_pierde_por_preparar():
    r = partida("import_lento", "fiel")
    assert (r.ganador, r.motivo) == ("negras", "preparar")
    assert "importar strategy.py" in r.salidas["blancas"]


def test_preparar_lento_pierde_por_preparar():
    r = partida("fiel", "preparar_lento")
    assert (r.ganador, r.motivo) == ("blancas", "preparar")
    assert r.preparar_usado["negras"] >= LIMITES.preparar_s


def test_spam_se_recorta_y_escapa():
    r = partida("spam", "fiel")
    assert (r.ganador, r.motivo) == ("negras", "reloj")
    s = r.salidas["blancas"]
    assert len(s.encode()) <= 64 * 1024 and not s.startswith("::") and "\n::" not in s
    assert "se recortó" in s


def test_comandos_actions_se_escapan():
    r = partida("comandos_actions", "fiel")
    assert r.motivo in {"puntos", "sin_jugadas"}
    s = r.salidas["blancas"]
    assert "gané yo" in s
    assert all(not linea.lstrip().startswith("::") for linea in s.split("\n"))
    assert "\r" not in s and "\x1b" not in s


def test_escribe_fd1_no_rompe_protocolo():
    r = partida("escribe_fd1", "fiel")
    assert r.motivo in {"puntos", "sin_jugadas"}
    assert "basura sin salto" in r.salidas["blancas"]
    assert "y un print normal" in r.salidas["blancas"]


def test_basura_en_protocolo_es_ilegal():
    r = partida("basura_protocolo", "fiel")
    assert (r.ganador, r.motivo) == ("negras", "ilegal")


def test_sigue_pensando_tras_responder():
    r = partida("sigue_pensando", "tranquilo")  # el rival tarda 30 ms por jugada
    assert r.motivo in {"puntos", "sin_jugadas"}
    mias = [j for j in r.jugadas if j["color"] == "blancas"][1:]
    # Velocidad del hilo cuando corre solo (dentro de jugar, que solo duerme).
    velocidad = sum(j["info"]["dentro"] for j in mias) / sum(j["info"]["duracion"] for j in mias)
    # Tiempo que el hilo trabajó fuera de `jugar` (al menos fuera/velocidad) menos lo que
    # sí se le cobró fuera de `jugar` (reanudar, mensajes): lo que trabajó gratis.
    trabajo_fuera = sum(j["info"]["fuera"] for j in mias) / velocidad
    cobrado_fuera = sum(j["tiempo"] - j["info"]["duracion"] for j in mias)
    gratis = trabajo_fuera - cobrado_fuera
    assert gratis < 0.005 * len(mias), (gratis, trabajo_fuera, cobrado_fuera, len(mias))
    # El rival no se ve afectado: sus jugadas tardan lo de siempre (~30 ms).
    rival = [j["tiempo"] for j in r.jugadas if j["color"] == "negras"][1:]
    assert sum(rival) / len(rival) < 0.03 + 0.01


def test_sin_strategy_pierde_por_error(tmp_path):
    r = jugar_partida(
        local("fiel"), BotLocal(tmp_path), Dados(1, 0, 0), LIMITES, LIMITES.efectivos(), 1
    )
    assert (r.ganador, r.motivo) == ("blancas", "error")
    assert "strategy.py" in r.salidas["negras"]


# --- Datos maliciosos del bot: pierde el bot, nunca es falla de infraestructura ----------


def test_info_anidada_no_es_falla_de_infraestructura():
    r = partida("info_anidada", "fiel")
    assert not r.falla_infraestructura, r.salidas
    assert r.motivo in {"puntos", "sin_jugadas"}
    info = r.jugadas[0]["info"]
    profundidad = 0
    while isinstance(info, dict) and info.get("arbol", info.get("x")) is not None:
        info = info.get("arbol", info.get("x"))
        profundidad += 1
    assert profundidad <= 33 and "recort" in str(info)


def test_numeros_raros_no_son_falla_de_infraestructura():
    blancas = local("numeros_raros")
    r = jugar_partida(blancas, local("fiel"), Dados(1, 0, 0), LIMITES, LIMITES.efectivos(), 1)
    assert not r.falla_infraestructura, r.salidas
    assert r.motivo in {"puntos", "sin_jugadas"}
    info = r.jugadas[0]["info"]
    assert info["nan"] is None and info["inf"] is None and info["menos_inf"] is None
    assert blancas.memoria_pico_mb() is None


def test_adios_falso_con_numero_enorme_pierde():
    r = partida("adios_falso", "fiel")
    assert not r.falla_infraestructura, r.salidas
    assert (r.ganador, r.motivo) == ("negras", "ilegal")


def test_campos_raros_pierde():
    r = partida("campos_raros", "fiel")
    assert not r.falla_infraestructura, r.salidas
    assert (r.ganador, r.motivo) == ("negras", "ilegal")


def test_mensaje_fuera_de_turno_es_ilegal():
    r = partida("doble", "fiel")
    assert not r.falla_infraestructura, r.salidas
    assert (r.ganador, r.motivo) == ("negras", "ilegal")
    assert len(r.jugadas) >= 2  # su primera jugada sí contó
    # Si la pausa cae entre sus dos escrituras, el mensaje extra llega como respuesta
    # del turno siguiente (jugada vieja → ilegal); si no, se detecta antes de enviarle
    # el turno. En ambos casos pierde por «ilegal».
    assert "fuera de turno" in r.salidas["blancas"] or "Jugada ilegal" in r.salidas["blancas"]


class BotSinHola:
    """Proceso que arrancó pero nunca manda `hola` (p. ej. un .pth que truena o se cuelga)."""

    def __init__(self, causa):
        self._causa = causa

    def iniciar(self): ...
    def enviar(self, msg): ...
    def recibir(self, timeout_s):
        return None

    def pausar(self): ...
    def reanudar(self): ...
    def terminar(self): ...
    def salida(self):
        return "Traceback: algo en sitecustomize"

    def causa_muerte(self):
        return self._causa

    def memoria_pico_mb(self):
        return None


def test_sin_hola_es_culpa_del_bot():
    for causa in ("error", None):  # murió o se colgó antes de `hola`
        r = jugar_partida(
            BotSinHola(causa), local("fiel"), Dados(1, 0, 0), LIMITES, LIMITES.efectivos(), 1
        )
        assert not r.falla_infraestructura
        assert (r.ganador, r.motivo) == ("negras", "error")
        assert "no arrancó" in r.salidas["blancas"]


def test_ignora_el_pycache_de_la_carpeta_del_bot(bot_con_pyc_viejo):
    r = jugar_partida(
        BotLocal(bot_con_pyc_viejo), local("fiel"), Dados(1, 0, 0), LIMITES, LIMITES.efectivos(), 1
    )
    assert r.jugadas[0]["info"] == {"valor": "src"}, r.salidas


class BotLocalQueNoSePausa(BotLocal):
    """Su `pausar` falla como un contenedor vivo que Docker no logra congelar."""

    def __init__(self, carpeta, falla_en: int):
        super().__init__(carpeta)
        self.pausas = 0
        self.falla_en = falla_en

    def pausar(self):
        self.pausas += 1
        if self.pausas >= self.falla_en:
            raise BotNoSeCongela("Docker no pudo pausar el contenedor vivo del bot: prueba")
        super().pausar()


def test_contenedor_vivo_que_no_se_pausa_pierde_el_bot():
    for falla_en in (1, 4):  # al arrancar y a media partida
        r = jugar_partida(
            BotLocalQueNoSePausa(BOTS / "fiel", falla_en),
            local("fiel"),
            Dados(1, 0, 0),
            LIMITES,
            LIMITES.efectivos(),
            1,
        )
        assert not r.falla_infraestructura, r.salidas
        assert (r.ganador, r.motivo) == ("negras", "error")
        assert "No se pudo pausar tu bot" in r.salidas["blancas"]
