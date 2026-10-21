"""Procesos de bot: `BotLocal` (subproceso sin límites) y `BotDocker` (contenedor aislado).

Ambos corren `python -m arbitro.runner <carpeta>` y cumplen `ProcesoBot`.
- stdout del proceso = protocolo (lo lee un hilo, con máximo 64 KB por línea).
- stderr del proceso = lo que imprime el bot (lo drena otro hilo todo el tiempo;
  se guardan los primeros `salida_kb` y el resto se descarta).
"""

from __future__ import annotations

import math
import os
import platform
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Protocol

import config
from arbitro import protocolo
from config import Limites

_FIN = object()  # marca de fin de archivo en la cola de mensajes
_ESPERA_MUERTE_S = 10.0  # cuánto esperar a que termine un proceso que cerró su salida
_ASENTAR_S = 2.0  # cuánto esperar a que Docker reporte bien un contenedor que muere
# Python busca sus .pyc aquí y no en los __pycache__ de la carpeta del bot. En la
# imagen de bots es un árbol ya compilado (stdlib, numpy, numba, sistema/) y de solo
# lectura al jugar: no tiene entradas para /bot, así que el .pyc de un equipo nunca se usa.
PYCACHE_DOCKER = "/opt/pyc"


ETIQUETA_DUENO = "torneo.dueno"


def dueno() -> str:
    """Quién lanza los contenedores: «<hostname>:<pid>:<c|h>» (c = dentro de un contenedor).

    Dentro del contenedor del CLI, el hostname es el id corto de ese contenedor: si ya
    no existe (o el proceso murió), sus bots son huérfanos y se pueden borrar.
    """
    lugar = "c" if Path("/.dockerenv").exists() else "h"
    return f"{socket.gethostname()}:{os.getpid()}:{lugar}"


class FallaInfraestructura(RuntimeError):
    """Algo nuestro falló (Docker, el runner, el árbitro): la partida se repite."""


class BotNoSeCongela(RuntimeError):
    """Docker no pudo pausar/reanudar un contenedor que sigue vivo: se cobra al bot.

    Solo se lanza tras esperar `_ASENTAR_S` y confirmar con Docker que el contenedor
    sigue corriendo y no quedó como se pedía (p. ej. un bot que crea procesos sin
    parar). Si se cobrara como infraestructura, un bot perdiendo podría forzar que su
    partida se repita.
    """


class ProcesoBot(Protocol):
    def iniciar(self) -> None: ...
    def enviar(self, msg: dict) -> None: ...
    def recibir(self, timeout_s: float) -> dict | None: ...
    def pausar(self) -> None: ...
    def reanudar(self) -> None: ...
    def terminar(self) -> None: ...
    def salida(self) -> str: ...
    def causa_muerte(self) -> str | None: ...
    def memoria_pico_mb(self) -> float | None: ...


def numero_seguro(valor: object, maximo: float = 1e9) -> float | None:
    """Número que mandó el bot → float finito entre 0 y `maximo`; si no, None.

    Nunca lanza: el bot controla este dato (enteros enormes, NaN, infinito, texto...).
    """
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        return None
    if isinstance(valor, int) and not 0 <= valor <= maximo:
        return None
    numero = float(valor)
    return numero if math.isfinite(numero) and 0 <= numero <= maximo else None


class _BotProceso:
    """Lo común: subproceso, hilo lector del protocolo e hilo que drena stderr."""

    def __init__(self, salida_kb: int) -> None:
        self._max_salida = salida_kb * 1024
        self._proc: subprocess.Popen | None = None
        self._mensajes: queue.Queue = queue.Queue()
        self._salida = bytearray()
        self._hilos: list[threading.Thread] = []
        self._fin_visto = False
        self._memoria_pico: float | None = None

    # --- Subclases ---------------------------------------------------------------------
    def _comando(self) -> list[str]:
        raise NotImplementedError

    def _popen_kw(self) -> dict:
        return {}

    # --- ProcesoBot ----------------------------------------------------------------------
    def iniciar(self) -> None:
        self._proc = subprocess.Popen(
            self._comando(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **self._popen_kw(),
        )
        for destino in (self._leer_protocolo, self._drenar_salida):
            hilo = threading.Thread(target=destino, daemon=True)
            hilo.start()
            self._hilos.append(hilo)

    def enviar(self, msg: dict) -> None:
        try:
            self._proc.stdin.write(protocolo.codificar(msg))
        except (BrokenPipeError, ValueError, OSError):
            pass  # el bot ya murió: lo verá `recibir` (fin de archivo) y `causa_muerte`

    def recibir(self, timeout_s: float) -> dict | None:
        """Siguiente mensaje; None si se acabó el tiempo o el bot cerró su salida.

        MensajeInvalido si llegó una línea mal formada o de más de 64 KB.
        """
        if self._fin_visto:
            return None
        try:
            item = self._mensajes.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None
        if item is _FIN:
            self._fin_visto = True
            return None
        if isinstance(item, protocolo.MensajeInvalido):
            raise item
        if item.get("tipo") == "adios":
            self._memoria_pico = numero_seguro(item.get("memoria_pico_mb"))
        return item

    def salida(self) -> str:
        return bytes(self._salida).decode("utf-8", errors="replace")

    def memoria_pico_mb(self) -> float | None:
        return self._memoria_pico

    # --- Hilos ---------------------------------------------------------------------------
    def _leer_protocolo(self) -> None:
        lector = protocolo.LectorLineas(self._proc.stdout.fileno())
        while True:
            try:
                linea = lector.leer()
                if linea is None:
                    break
                self._mensajes.put(protocolo.decodificar(linea))
            except protocolo.MensajeInvalido as e:
                self._mensajes.put(e)
            except OSError:
                break
            except Exception as e:  # cualquier otra cosa al leer datos del bot: culpa del bot
                self._mensajes.put(protocolo.MensajeInvalido(f"Mensaje ilegible: {e!r}"[:200]))
        self._mensajes.put(_FIN)

    def _drenar_salida(self) -> None:
        fd = self._proc.stderr.fileno()
        while True:
            try:
                trozo = os.read(fd, 64 * 1024)
            except OSError:
                break
            if not trozo:
                break
            falta = self._max_salida - len(self._salida)
            if falta > 0:
                self._salida += trozo[:falta]

    def _esperar_hilos(self) -> None:
        for hilo in self._hilos:
            hilo.join(timeout=5)

    def _esperar_salida_del_proceso(self) -> int | None:
        """Tras ver fin de archivo, espera un poco a que el proceso termine."""
        if self._fin_visto:
            try:
                return self._proc.wait(timeout=_ESPERA_MUERTE_S)
            except subprocess.TimeoutExpired:
                return None
        return self._proc.poll()


class BotLocal(_BotProceso):
    """Bot como subproceso de este Python, sin límites (modo de depuración).

    Pausa con SIGSTOP y reanuda con SIGCONT a todo su grupo de procesos.
    """

    def __init__(self, carpeta: Path, salida_kb: int | None = None) -> None:
        super().__init__(salida_kb or config.cargar_limites().salida_kb)
        self.carpeta = Path(carpeta).resolve()

    def _comando(self) -> list[str]:
        return [sys.executable, "-m", "arbitro.runner", str(self.carpeta)]

    def _popen_kw(self) -> dict:
        # Ambiente mínimo: el bot no hereda tokens ni secretos del ambiente.
        env = {
            k: v
            for k, v in os.environ.items()
            if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "NUMBA_CACHE_DIR")
        }
        previo = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = str(config.SISTEMA) + (os.pathsep + previo if previo else "")
        # Ignora los __pycache__ de la carpeta del bot. Aquí la caché sí se escribe (fuera
        # de la carpeta): sin ella, importar numpy/numba tardaría ~7 s en cada arranque.
        env["PYTHONPYCACHEPREFIX"] = str(Path(tempfile.gettempdir()) / "torneo_pycache")
        # Un hilo, como en la liga (`BotDocker`): numba/numpy no reparten el trabajo en
        # todos los núcleos de la máquina (el tiempo se parece más al oficial).
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[variable] = "1"
        env["NUMBA_NUM_THREADS"] = "1"
        return {"env": env, "start_new_session": True}

    def _senal(self, senal: int) -> None:
        try:
            os.killpg(self._proc.pid, senal)
        except (ProcessLookupError, PermissionError):
            pass  # ya terminó

    def pausar(self) -> None:
        self._senal(signal.SIGSTOP)

    def reanudar(self) -> None:
        self._senal(signal.SIGCONT)

    def terminar(self) -> None:
        if self._proc is None:
            return
        self._senal(signal.SIGKILL)
        self._proc.wait()
        self._esperar_hilos()

    def causa_muerte(self) -> str | None:
        return None if self._esperar_salida_del_proceso() is None else "error"


def arquitectura() -> str:
    """`amd64` o `arm64` (para el volumen de caché de numba de la máquina)."""
    maquina = platform.machine().lower()
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(maquina, maquina)


class BotDocker(_BotProceso):
    """Bot en un contenedor aislado.

    `limites` deben ser los límites PÚBLICOS (`config.cargar_limites()`), nunca los
    ya efectivos: aquí se aplica `escala_limites` UNA vez (`limites.efectivos()`) a la
    memoria del contenedor. Pasar `limites.efectivos()` la aplicaría dos veces.
    `raiz_host` (o `$TORNEO_RAIZ_HOST`): raíz del repo en el anfitrión, para
    traducir la carpeta cuando este código corre dentro del contenedor del CLI.

    El motor vive horneado en la imagen (/torneo/sistema, de root, sistema de
    archivos de solo lectura): el bot no lo puede modificar.
    """

    IMAGEN_COMANDO = ("python", "-m", "arbitro.runner", "/bot")

    def __init__(
        self,
        carpeta: Path,
        imagen: str,
        nucleo: int | None,
        limites: Limites,
        raiz_host: Path | None = None,
    ) -> None:
        super().__init__(limites.salida_kb)
        self.carpeta = Path(carpeta).resolve()
        self.imagen = imagen
        self.nucleo = nucleo
        self.limites = limites
        host = raiz_host or os.environ.get("TORNEO_RAIZ_HOST")
        self.raiz_host = Path(host) if host else None
        self.nombre = f"torneo-bot-{uuid.uuid4().hex[:12]}"
        self._cliente = None
        self._inspeccion: dict | None = None

    def carpeta_host(self) -> Path:
        """Ruta de la carpeta del bot vista desde el anfitrión (para `-v`)."""
        if self.raiz_host is None:
            return self.carpeta
        try:
            relativa = self.carpeta.relative_to(config.RAIZ)
        except ValueError:
            raise ValueError(
                f"La carpeta {self.carpeta} no está dentro del repo ({config.RAIZ}). "
                "Pon tu bot en estudiantes/<tu_equipo>/."
            ) from None
        return self.raiz_host / relativa

    def argumentos(self) -> list[str]:
        memoria = self.limites.efectivos().memoria_mb
        args = [
            "docker",
            "run",
            "-i",
            "--name",
            self.nombre,
            "--label",
            f"{ETIQUETA_DUENO}={dueno()}",
        ]
        args.append("--cpus=1")
        if self.nucleo is not None and os.environ.get("TORNEO_SIN_CPUSET") != "1":
            args.append(f"--cpuset-cpus={self.nucleo}")
        args += [
            f"--memory={memoria}m",
            f"--memory-swap={memoria}m",
            "--pids-limit=64",
            "--network=none",
            "--read-only",
            "--tmpfs",
            "/tmp/numba:size=64m,noexec",
            "--log-driver=none",
            "--user",
            "65534:65534",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{self.carpeta_host()}:/bot:ro",
            "-v",
            f"torneo_numba_{arquitectura()}:/opt/numba_listo:ro",
        ]
        for variable in (
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMBA_NUM_THREADS=1",
            "NUMBA_CACHE_DIR=/tmp/numba",
            f"PYTHONPYCACHEPREFIX={PYCACHE_DOCKER}",
        ):
            args += ["-e", variable]
        return [*args, self.imagen, *self.IMAGEN_COMANDO]

    def _comando(self) -> list[str]:
        return self.argumentos()

    def _popen_kw(self) -> dict:
        return {"start_new_session": True}  # Ctrl+C en la terminal no mata al bot a medias

    def _api(self):
        if self._cliente is None:
            import docker  # solo en la imagen del CLI, nunca en la de bots

            self._cliente = docker.from_env()
        return self._cliente.api

    def _congelar(self, pausar: bool) -> None:
        """Pausa/reanuda con el SDK (más rápido que el CLI). Si ya murió, no hace nada."""
        import docker

        operacion = self._api().pause if pausar else self._api().unpause
        try:
            operacion(self.nombre)
            return
        except docker.errors.APIError as e:
            error = e
        # Falló: casi siempre porque el bot está muriendo justo ahora (p. ej. un hilo
        # que hace os._exit tras responder) y Docker aún lo reporta como vivo. Se
        # espera a que el estado se asiente; los errores de conexión con el daemon
        # (no APIError) suben tal cual y son falla de infraestructura. Si tras la
        # espera sigue vivo y sin congelarse, es del bot (`BotNoSeCongela`).
        limite = time.monotonic() + _ASENTAR_S
        espera = 0.01
        while True:
            # Primero Docker (fuente de verdad): si solo murió el cliente `docker run`
            # pero el contenedor sigue vivo, no se culpa al bot ni se deja sin pausar.
            estado = self._inspeccionar()
            if estado is None or not estado["State"]["Running"]:
                return  # murió: lo verán `recibir` y `causa_muerte`
            if bool(estado["State"]["Paused"]) == pausar:
                return  # ya estaba como se pedía
            if time.monotonic() > limite:
                raise BotNoSeCongela(
                    f"Docker no pudo {'pausar' if pausar else 'reanudar'} el contenedor "
                    f"vivo del bot: {error}"
                )
            time.sleep(espera)
            espera = min(espera * 2, 0.2)
            try:
                operacion(self.nombre)
                return
            except docker.errors.APIError as e:
                error = e

    def pausar(self) -> None:
        self._congelar(True)

    def reanudar(self) -> None:
        self._congelar(False)

    def _inspeccionar(self) -> dict | None:
        import docker

        try:
            self._inspeccion = self._api().inspect_container(self.nombre)
        except docker.errors.NotFound:
            pass  # se queda la última inspección (o None)
        return self._inspeccion

    def causa_muerte(self) -> str | None:
        codigo = self._esperar_salida_del_proceso()
        estado = self._inspeccionar()
        if estado is None:  # Docker nunca creó el contenedor: no es culpa del bot
            detalle = "sigue sin crearlo" if codigo is None else f"terminó con código {codigo}"
            raise FallaInfraestructura(f"`docker run` {detalle}: el contenedor no existe.")
        if not estado["State"]["Running"] and estado["State"].get("StartedAt", "").startswith(
            "0001-"
        ):
            raise FallaInfraestructura(f"Docker creó {self.nombre} pero no pudo iniciarlo.")
        if estado["State"].get("OOMKilled"):
            return "memoria"
        if estado["State"]["Running"]:
            return None
        return "error"

    def terminar(self) -> None:
        if self._proc is None:
            return
        import docker

        self._inspeccionar()
        try:
            self._api().remove_container(self.nombre, force=True)
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as e:
            raise FallaInfraestructura(f"Docker no pudo borrar {self.nombre}: {e}") from e
        finally:
            try:
                self._proc.wait(timeout=_ESPERA_MUERTE_S)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._esperar_hilos()
