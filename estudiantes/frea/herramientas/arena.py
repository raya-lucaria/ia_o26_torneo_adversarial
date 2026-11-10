"""Match rápido entre dos carpetas de bot, con dados pareados como en la liga.

    python herramientas/arena.py <carpeta_a> <carpeta_b> --pares 100 --reloj 3 --hilos 8

Cada bot corre en su propio proceso, que se reutiliza entre partidas. Cada par juega los mismos dados
con los colores invertidos. Los dados los tira este script: no importa nada del árbitro.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from motor import Estado, Jugada


def proceso_bot(carpeta):
    """Lado hijo: sirve preparar/jugar por stdin y stdout en JSON."""
    salida = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # los print del bot no ensucian el protocolo
    sys.path.insert(0, carpeta)
    import strategy

    class Config:
        pass

    for linea in sys.stdin:
        m = json.loads(linea)
        inicio = time.perf_counter()
        try:
            if m["t"] == "preparar":
                c = Config()
                c.color, c.semilla, c.reloj_s = m["color"], m["semilla"], m["reloj"]
                c.carpeta_datos = Path(carpeta) / "data"
                strategy.preparar(c)
                r = {}
            else:
                r = {"j": str(strategy.jugar(Estado.desde_dict(m["estado"]), m["restante"]))}
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        r["dt"] = time.perf_counter() - inicio
        salida.write(json.dumps(r) + "\n")


class Bot:
    def __init__(self, carpeta):
        self.p = subprocess.Popen(
            [sys.executable, __file__, "--hijo", carpeta],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def pedir(self, **m):
        self.p.stdin.write(json.dumps(m) + "\n")
        linea = self.p.stdout.readline()
        return json.loads(linea) if linea else {"error": "murió", "dt": 0.0}


def partida(blancas, negras, reloj, semilla, juego):
    """Regresa (color ganador, motivo, jugadas, reloj usado, texto de la partida)."""
    bots = {"blancas": blancas, "negras": negras}
    gens = {c: np.random.default_rng([semilla, juego, i]) for i, c in enumerate(bots)}
    usado = dict.fromkeys(bots, 0.0)
    for i, (color, bot) in enumerate(bots.items()):
        bot.pedir(t="preparar", color=color, semilla=semilla * 2 + i, reloj=reloj)
    e, n, log = Estado.inicial(), 0, []
    while (g := e.ganador()) is None:
        c = e.turno
        otro = "negras" if c == "blancas" else "blancas"
        r = bots[c].pedir(t="jugar", estado=e.a_dict(), restante=max(0.0, reloj - usado[c]))
        usado[c] += r["dt"]
        if "error" in r:
            return otro, "error: " + r["error"], n, usado, log
        if usado[c] > reloj:
            return otro, "reloj", n, usado, log
        try:
            j = Jugada.desde_texto(r["j"])
            if j not in e.jugadas():
                raise ValueError
        except ValueError:
            return otro, "ilegal", n, usado, log
        if e.es_ataque(j):
            p = e.prob_ataque(j)
            gano = gens[c].random() < p
            log.append(f"{j} p={p:.2f} {'gana' if gano else 'pierde'}")
            e = e.aplicar(j, gana_ataque=gano)
        else:
            log.append(str(j))
            e = e.aplicar(j)
        n += 1
    return g, "juego", n, usado, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pares", type=int, default=50)
    ap.add_argument("--reloj", type=float, default=3.0)
    ap.add_argument("--hilos", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--registro", help="archivo donde guardar las partidas que perdió a")
    args = ap.parse_args()

    pendientes = list(range(args.pares))
    candado = threading.Lock()
    gana_a, total, fallas, n_jugadas, max_reloj = [0], [0], [], [], [0.0]
    perdidas = []

    def trabajar():
        a, b = Bot(os.path.abspath(args.a)), Bot(os.path.abspath(args.b))
        while True:
            with candado:
                if not pendientes:
                    break
                par = pendientes.pop()
            for a_blancas in (True, False):
                bl, ne = (a, b) if a_blancas else (b, a)
                g, motivo, n, usado, log = partida(bl, ne, args.reloj, args.semilla, par)
                color_a = "blancas" if a_blancas else "negras"
                with candado:
                    total[0] += 1
                    gana_a[0] += g == color_a
                    n_jugadas.append(n)
                    max_reloj[0] = max(max_reloj[0], usado[color_a])
                    if motivo != "juego":
                        fallas.append(f"par {par}: ganó {g}, {motivo}")
                    if g != color_a:
                        perdidas.append((color_a, " ".join(f"[{x}]" for x in log)))
        for bot in (a, b):
            bot.p.kill()

    inicio = time.time()
    hilos = [threading.Thread(target=trabajar) for _ in range(args.hilos)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    t, w = total[0], gana_a[0]
    p = w / max(t, 1)
    margen = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(t, 1))
    print(f"{Path(args.a).name} {w} - {t - w} {Path(args.b).name}: {100 * p:.1f} % ± {100 * margen:.1f}")
    print(f"jugadas media {np.mean(n_jugadas):.0f} · reloj máx de a {max_reloj[0]:.2f}/{args.reloj} s"
          f" · {time.time() - inicio:.0f} s de pared")
    for x in fallas[:10]:
        print("✗", x)
    if args.registro:
        with open(args.registro, "w") as arch:
            for color, texto in perdidas:
                arch.write(f"Perdió con {color}: {texto}\n\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--hijo":
        proceso_bot(sys.argv[2])
    else:
        main()
