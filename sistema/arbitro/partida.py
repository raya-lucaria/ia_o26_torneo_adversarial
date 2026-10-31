"""Una partida entre dos bots. El árbitro es la única fuente de verdad del resultado.

Reloj: corre desde que termina `reanudar` hasta que llega la respuesta; al llegar,
el bot se pausa antes de procesarla. El bot ve el reloj PÚBLICO
(`tiempo_restante = max(0, publicos.reloj_s − usado)`) y pierde al pasar el
EFECTIVO (`efectivos.reloj_s`). `preparar` no se escala.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from arbitro import protocolo
from arbitro.azar import Dados, semilla_de_color
from arbitro.procesos import BotNoSeCongela, ProcesoBot
from arbitro.registro import escapar
from config import Limites
from motor import Estado, Jugada

COLORES = ("blancas", "negras")
# Recibe el estado después de cada jugada y esa jugada (None antes de la primera).
Observador = Callable[[Estado, "dict | None"], None]
ARRANQUE_S = 60.0  # arranque del contenedor + copia de caché (fuera de todo reloj)
DESPEDIDA_S = 2.0  # espera de `adios` tras `fin` (memoria pico informativa)
INFO_MAX = 4096  # bytes de `info` por jugada que se guardan en el registro
INFO_PROFUNDIDAD = 32  # niveles de anidación de `info` que se guardan


@dataclass
class ResultadoPartida:
    """Resultado de una partida.

    Si `falla_infraestructura` es True, la partida no cuenta (se repite) y
    `ganador` queda vacío.
    """

    ganador: str
    motivo: str
    puntos: dict[str, int]
    jugadas: list[dict] = field(default_factory=list)
    reloj_usado: dict[str, float] = field(default_factory=dict)
    preparar_usado: dict[str, float] = field(default_factory=dict)  # segundos de `preparar`
    salidas: dict[str, str] = field(default_factory=dict)
    falla_infraestructura: bool = False


class _Pierde(Exception):
    """El bot de `color` pierde la partida por `motivo`."""

    def __init__(self, color: str, motivo: str, nota: str) -> None:
        super().__init__(nota)
        self.color, self.motivo, self.nota = color, motivo, nota


def _rival(color: str) -> str:
    return COLORES[1 - COLORES.index(color)]


def _limpiar_info(info: object) -> dict | None:
    """`info` del bot: solo diccionarios, textos escapados, sin NaN, máx. 4 KB.

    Profundidad máxima `INFO_PROFUNDIDAD`: lo que esté más abajo se cambia por un
    aviso (así un diccionario anidado mil veces no rompe nada).
    """
    if not isinstance(info, dict):
        return None
    aviso = f"recortado: más de {INFO_PROFUNDIDAD} niveles"

    def limpiar(valor: object, nivel: int) -> object:
        if isinstance(valor, str):
            return escapar(valor)
        if isinstance(valor, bool) or valor is None:
            return valor
        if isinstance(valor, float):
            return valor if math.isfinite(valor) else None
        if isinstance(valor, int):
            return valor if abs(valor) < 2**63 else None
        if isinstance(valor, list | dict):
            if nivel >= INFO_PROFUNDIDAD:
                return aviso
            if isinstance(valor, list):
                return [limpiar(v, nivel + 1) for v in valor]
            return {escapar(str(k)): limpiar(v, nivel + 1) for k, v in valor.items()}
        return None  # JSON no trae otros tipos

    limpio = limpiar(info, 0)
    if len(json.dumps(limpio)) > INFO_MAX:
        return {"aviso": f"info recortada: pasa de {INFO_MAX // 1024} KB"}
    return limpio


class _Arbitro:
    def __init__(self, bots, dados, publicos, efectivos, semilla_bots, observador=None) -> None:
        self.bots: dict[str, ProcesoBot] = bots
        self.dados: Dados = dados
        self.publicos: Limites = publicos
        self.efectivos: Limites = efectivos
        self.semilla_bots: int = semilla_bots
        self.estado = Estado.inicial()
        self.jugadas: list[dict] = []
        self.usado = dict.fromkeys(COLORES, 0.0)
        self.preparar_usado: dict[str, float] = {}
        self.notas: dict[str, list[str]] = {c: [] for c in COLORES}
        self.observador: Observador | None = observador

    # --- Fases ---------------------------------------------------------------------------
    def arrancar(self) -> None:
        for bot in self.bots.values():
            bot.iniciar()
        for color, bot in self.bots.items():
            try:
                mensaje = bot.recibir(ARRANQUE_S)
            except protocolo.MensajeInvalido:
                mensaje = None  # basura antes de `hola`: algo del bot imprimió en stdout
            if mensaje is None or mensaje.get("tipo") != "hola":
                # El proceso sí arrancó (si Docker no pudo crearlo o iniciarlo,
                # `causa_muerte` lanza FallaInfraestructura): es culpa del bot, p. ej.
                # un paquete de su requirements.txt que truena o se cuelga al iniciar.
                if bot.causa_muerte() == "memoria":
                    raise _Pierde(color, "memoria", self._nota_memoria())
                raise _Pierde(
                    color,
                    "error",
                    (
                        "[árbitro] Tu bot no arrancó: Python falló o se colgó antes de "
                        "cargar tu código. Revisa los paquetes de tu requirements.txt."
                    ),
                )
            self._congelar(color, pausar=True)

    def preparar(self, color: str) -> None:
        bot = self.bots[color]
        limite = self.efectivos.preparar_s
        mensaje, transcurrido = self._turno(
            color,
            protocolo.preparar(
                color, semilla_de_color(self.semilla_bots, color), self.publicos.reloj_s
            ),
            limite,
            invalido=("error", "[árbitro] Tu bot mandó un mensaje inválido durante preparar."),
        )
        self.preparar_usado[color] = transcurrido
        if transcurrido > limite or mensaje is None:
            motivo = self._causa(bot, transcurrido, limite)
            if motivo == "reloj":
                raise _Pierde(
                    color,
                    "preparar",
                    (
                        f"[árbitro] preparar tardó más de {limite:g} s (cuenta también "
                        "importar strategy.py). Carga menos cosas o hazlas más rápido."
                    ),
                )
            raise _Pierde(color, motivo, self._nota_muerte(motivo, "preparar"))
        if mensaje.get("tipo") != "listo":
            raise _Pierde(color, "error", "[árbitro] Tu bot no respondió bien a preparar.")

    def jugar(self) -> tuple[str, str]:
        self._observar(None)
        while (ganador := self.estado.ganador()) is None:
            self._una_jugada(self.estado.turno, len(self.jugadas) + 1)
        motivo = "puntos" if max(self.estado.puntos.values()) >= 3 else "sin_jugadas"
        return ganador, motivo

    def despedir(self, ganador: str, motivo: str, perdedor_roto: str | None) -> None:
        """Manda `fin` a los bots que siguen sanos (para la memoria pico)."""
        resultado = protocolo.fin(
            {"ganador": ganador, "motivo": motivo, "puntos": self.estado.puntos}
        )
        for color, bot in self.bots.items():
            if color == perdedor_roto:
                continue
            try:
                bot.reanudar()
                bot.enviar(resultado)
                bot.recibir(DESPEDIDA_S)
            except Exception:  # la partida ya está decidida; esto es informativo
                pass

    # --- Turnos --------------------------------------------------------------------------
    def _turno(self, color, mensaje, limite, invalido) -> tuple[dict | None, float]:
        """Reanuda, manda, espera la respuesta (máx. `limite`) y pausa al recibirla.

        El reloj corre desde que termina `reanudar` hasta que termina `pausar`: lo
        que el bot haga mientras se congela también se le cobra.
        """
        bot = self.bots[color]
        try:
            viejo = bot.recibir(0)  # algo que mandó fuera de su turno
        except protocolo.MensajeInvalido:
            viejo = {}
        if viejo is not None:
            raise _Pierde(
                color,
                invalido[0],
                (
                    "[árbitro] Tu bot mandó mensajes fuera de turno (p. ej. un hilo que "
                    "escribe en el protocolo después de responder)."
                ),
            )
        self._congelar(color, pausar=False)
        inicio = time.monotonic()
        bot.enviar(mensaje)
        try:
            respuesta = bot.recibir(limite)
        except protocolo.MensajeInvalido:
            self._congelar(color, pausar=True)
            raise _Pierde(color, *invalido) from None
        if respuesta is not None:
            self._congelar(color, pausar=True)
        return respuesta, time.monotonic() - inicio

    def _congelar(self, color: str, pausar: bool) -> None:
        """Pausa o reanuda al bot; si su contenedor vivo no se deja, pierde él."""
        bot = self.bots[color]
        try:
            if pausar:
                bot.pausar()
            else:
                bot.reanudar()
        except BotNoSeCongela as e:
            raise _Pierde(
                color,
                "error",
                (
                    f"[árbitro] No se pudo {'pausar' if pausar else 'reanudar'} tu bot "
                    f"({e}). Suele pasar si crea muchos procesos o hilos."
                ),
            ) from None

    def _una_jugada(self, color: str, n: int) -> None:
        bot = self.bots[color]
        limite = self.efectivos.reloj_s - self.usado[color]
        restante = max(0.0, self.publicos.reloj_s - self.usado[color])
        mensaje, transcurrido = self._turno(
            color,
            protocolo.jugar(self.estado.a_dict(), restante),
            limite,
            invalido=("ilegal", f"[árbitro] Jugada {n}: tu bot mandó un mensaje mal formado."),
        )
        self.usado[color] += transcurrido
        if mensaje is None or self.usado[color] > self.efectivos.reloj_s:
            motivo = self._causa(bot, transcurrido, limite)
            if motivo == "reloj":
                raise _Pierde(
                    color,
                    "reloj",
                    (
                        f"[árbitro] Te quedaste sin reloj en la jugada {n}: usaste más de "
                        f"{self.publicos.reloj_s:g} s en la partida."
                    ),
                )
            raise _Pierde(color, motivo, self._nota_muerte(motivo, f"la jugada {n}"))
        try:  # todo lo que sigue procesa datos del bot: si algo truena, pierde él
            jugada = self._jugada_legal(color, n, mensaje)
            info = _limpiar_info(mensaje.get("info"))
        except _Pierde:
            raise
        except Exception:
            raise _Pierde(
                color,
                "ilegal",
                (f"[árbitro] Jugada {n}: tu bot mandó una respuesta que no se pudo leer."),
            ) from None
        combate = None
        if self.estado.es_ataque(jugada):
            p = self.estado.prob_ataque(jugada)
            u = self.dados.tirar(color)
            combate = {"p": p, "u": u, "gano": u < p}
            self.estado = self.estado.aplicar(jugada, gana_ataque=u < p)
        else:
            self.estado = self.estado.aplicar(jugada)
        self.jugadas.append(
            {
                "n": n,
                "color": color,
                "jugada": str(jugada),
                "combate": combate,
                "tiempo": transcurrido,
                "reloj": self.usado[color],
                "info": info,
            }
        )
        self._observar(self.jugadas[-1])

    def _observar(self, jugada: dict | None) -> None:
        """Avisa al observador con los dos bots en pausa: fuera de todo reloj."""
        if self.observador is not None:
            self.observador(self.estado, jugada)

    def _jugada_legal(self, color: str, n: int, mensaje: dict) -> Jugada:
        texto = mensaje.get("jugada")
        if mensaje.get("tipo") == "jugada" and isinstance(texto, str):
            try:
                jugada = Jugada.desde_texto(texto)
            except ValueError:
                jugada = None
            if jugada is not None and jugada in self.estado.jugadas():
                return jugada
        muestra = repr(texto)[:80] if isinstance(texto, str) else "nada"
        raise _Pierde(
            color,
            "ilegal",
            (
                f"[árbitro] Jugada ilegal en la jugada {n}: tu bot regresó {muestra}. "
                "jugar debe regresar una jugada de estado.jugadas() (o su texto, "
                "como 'e2-e3'); una tupla u otra cosa no sirve."
            ),
        )

    # --- Causas y notas -------------------------------------------------------------------
    def _causa(self, bot: ProcesoBot, transcurrido: float, limite: float) -> str:
        causa = bot.causa_muerte()
        if causa is not None:
            return causa
        return "reloj" if transcurrido >= limite else "error"

    def _nota_memoria(self) -> str:
        return (
            f"[árbitro] Tu bot se pasó de memoria (límite {self.publicos.memoria_mb} MB) "
            "y el sistema lo detuvo."
        )

    def _nota_muerte(self, motivo: str, donde: str) -> str:
        if motivo == "memoria":
            return self._nota_memoria()
        return (
            f"[árbitro] Tu bot terminó con error en {donde}. "
            "Arriba está lo que imprimió (busca el Traceback)."
        )

    def salida_final(self, color: str) -> str:
        """Salida del bot escapada, con las notas del árbitro, máx. `salida_kb` (en bytes)."""
        limite = self.publicos.salida_kb * 1024
        cruda = self.bots[color].salida()
        notas = list(self.notas[color])
        if len(cruda.encode()) >= limite:
            notas.insert(0, f"[árbitro] Tu salida se recortó a {self.publicos.salida_kb} KB.")
        pie = escapar("\n" + "\n".join(notas)).encode() if notas else b""
        cuerpo = escapar(cruda).encode()[: max(0, limite - len(pie))]
        return (cuerpo + pie).decode("utf-8", errors="ignore")


def jugar_partida(
    blancas: ProcesoBot,
    negras: ProcesoBot,
    dados: Dados,
    publicos: Limites,
    efectivos: Limites,
    semilla_bots: int,
    observador: Observador | None = None,
) -> ResultadoPartida:
    """Juega una partida completa y regresa el resultado. Siempre termina a los bots.

    `semilla_bots` sale de `azar.semilla_bots(semilla_maestra, pareja, juego)`; cada
    color recibe `azar.semilla_de_color(semilla_bots, color)` en `config.semilla`.
    `observador(estado, jugada)` (opcional, p. ej. `--ver`) se llama una vez antes de la
    primera jugada (`jugada=None`) y tras registrar cada jugada (el dict del registro),
    siempre con los dos bots en pausa: lo que tarde no cuenta en ningún reloj.
    Excepciones del árbitro o de Docker → `falla_infraestructura=True`.
    """
    a = _Arbitro(
        {"blancas": blancas, "negras": negras},
        dados,
        publicos,
        efectivos,
        semilla_bots,
        observador,
    )
    falla = False
    try:
        roto = None
        try:
            a.arrancar()
            for color in COLORES:
                a.preparar(color)
            ganador, motivo = a.jugar()
        except _Pierde as p:
            ganador, motivo = _rival(p.color), p.motivo
            a.notas[p.color].append(p.nota)
            roto = p.color if p.motivo != "ilegal" else None
        a.despedir(ganador, motivo, roto)
    except Exception as e:  # nuestro: Docker, runner o árbitro
        falla = True
        ganador, motivo = "", "error"
        for color in COLORES:
            a.notas[color].append(
                f"[árbitro] Falla de infraestructura (no es culpa de ningún bot): "
                f"{type(e).__name__}: {e}. Esta partida se repite."
            )
    finally:
        for bot in (blancas, negras):
            try:
                bot.terminar()
            except Exception as e:  # el resultado ya está decidido
                print(f"[árbitro] Aviso: no se pudo terminar un bot: {e}", file=sys.stderr)
    return ResultadoPartida(
        ganador=ganador,
        motivo=motivo,
        puntos=a.estado.puntos,
        jugadas=a.jugadas,
        reloj_usado=dict(a.usado),
        preparar_usado=dict(a.preparar_usado),
        salidas={c: a.salida_final(c) for c in COLORES},
        falla_infraestructura=falla,
    )
