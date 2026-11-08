"""SPSA sobre los pesos de combate: ajusta 8 pesos de data/pesos.json por partidas ganadas.

En cada vuelta mueve esos pesos al azar ±c (θ+ y θ−), juega con las mismas semillas θ+ y θ−
contra cada rival y además θ+ contra θ−, y empuja los pesos hacia el lado que ganó más
(Spall, 1992). Nosotros jugamos a nodos fijos; los rivales, con su strategy.py y reloj real.
Con esto salieron los pesos sc400 (r69 + 400 vueltas, rivales: pamplinotas y torneo_breakthrough,
con el acuerdo de esos equipos).

python estudiantes/alondra_bernardo/herramientas/spsa_combate.py \\
    estudiantes/alondra_bernardo/herramientas/pesos/r69.json \\
    --rivales CARPETA_RIVAL_1 CARPETA_RIVAL_2 --vueltas 400 --salida sc
"""
import argparse, json, random, sys, time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EQUIPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reloj_arena  # noqa: E402  (carga los rivales con sus propios módulos)
from motor import Estado  # noqa: E402

# Apoyos (2), amenazas (2), ataques posibles, colgados, falange y bloqueados.
GRUPO = np.array([40, 41, 42, 43, 52, 57, 58, 59])
AJ = {}


def iniciar(rivales, nodos, reloj):
    for r in rivales:
        AJ[r] = reloj_arena.cargar(r, "rival_" + Path(r).name)
    sys.path.insert(0, str(EQUIPO))
    import rapido
    AJ.update(rapido=rapido, nodos=nodos, reloj=reloj)
    for k in "ab":
        AJ[k] = rapido.Contexto(np.zeros(rapido.RASGOS))
        AJ[k].control[5] = 1  # LMR, como en strategy.py


def mover(ctx, e):
    rapido = AJ["rapido"]
    ctx.limpiar()
    mejor = None
    for prof in range(1, rapido.MAX_PLY):
        v, cod = rapido.buscar(ctx, e._tablero, e._puntos, e._color, prof, AJ["nodos"])
        if cod is None:
            break
        mejor = cod
        if abs(v) > 0.999:
            break
    js = e.jugadas()
    return js[0] if mejor is None else next(j for j in js if str(j) == rapido.texto(mejor))


def partida(arg):
    """+1 si gana el lado A (pesos th), −1 si pierde. El otro lado es th2 o un rival."""
    th, th2, rival, semilla, a_blancas = arg
    rng = random.Random(semilla)
    AJ["a"].w[:] = th
    color_a = "blancas" if a_blancas else "negras"
    color_b = "negras" if a_blancas else "blancas"
    if rival is None:
        AJ["b"].w[:] = th2
    else:
        AJ[rival].preparar(SimpleNamespace(color=color_b, reloj_s=AJ["reloj"], preparar_s=10.0,
                                           semilla=semilla, carpeta_datos=Path(rival) / "data"))
    resto = AJ["reloj"]
    e = Estado.inicial()
    n = 0
    while e.ganador() is None:
        if n < 2:  # apertura al azar: las mismas semillas no repiten partidas
            j = rng.choice(e.jugadas())
        elif e.turno == color_a:
            j = mover(AJ["a"], e)
        elif rival is None:
            j = mover(AJ["b"], e)
        else:
            t = time.perf_counter()
            r = AJ[rival].jugar(e, max(resto, 0.05))
            resto -= time.perf_counter() - t
            j = next(x for x in e.jugadas() if str(x) == str(r))
        probs, hijos = zip(*e.siguiente(j))
        e = rng.choices(hijos, weights=probs)[0]
        n += 1
    return 1.0 if e.ganador() == color_a else -1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pesos", type=Path)
    p.add_argument("--rivales", nargs="+", required=True, help="carpetas de los bots rivales")
    p.add_argument("--vueltas", type=int, default=400)
    p.add_argument("--nodos", type=int, default=60000)
    p.add_argument("--reloj", type=float, default=6.0, help="reloj de los rivales por partida")
    p.add_argument("--procesos", type=int, default=8)
    p.add_argument("--a", type=float, default=0.5)
    p.add_argument("--c", type=float, default=0.1)
    p.add_argument("--semilla", type=int, default=11)
    p.add_argument("--salida", required=True, help="prefijo de los pesos guardados cada 25 vueltas")
    x = p.parse_args()
    th = np.array(json.loads(x.pesos.read_text())["pesos"], float)
    escala = np.maximum(np.abs(th), 0.1)  # cada peso se mueve en proporción a su tamaño
    rng = np.random.default_rng(x.semilla)
    with Pool(x.procesos, iniciar, (x.rivales, x.nodos, x.reloj)) as pool:
        for k in range(1, x.vueltas + 1):
            ck = x.c / k ** 0.101
            ak = x.a / (k + 50) ** 0.602
            d = np.zeros(th.size)
            d[GRUPO] = rng.choice([-1.0, 1.0], GRUPO.size)
            tp, tm = th + ck * escala * d, th - ck * escala * d
            trabajos, signo = [], []
            for i, rival in enumerate(x.rivales):
                for j in range(4):
                    s = k * 100000 + i * 100 + j
                    trabajos += [(tp, None, rival, s, j % 2 == 0), (tm, None, rival, s, j % 2 == 0)]
                    signo += [1.0, -1.0]
            for j in range(8):
                trabajos.append((tp, tm, None, k * 100000 + 900 + j // 2, j % 2 == 0))
                signo.append(2.0)  # un duelo directo mide θ+ contra θ−: cuenta como un par
            r = pool.map(partida, trabajos, chunksize=1)
            g = float(np.dot(r, signo)) / len(trabajos)
            th = th + ak * escala * g * d
            if k % 25 == 0 or k == x.vueltas:
                Path(f"{x.salida}_{k}.json").write_text(json.dumps({"pesos": th.tolist(), "rasgos": th.size}))
                print(f"vuelta {k}", flush=True)


if __name__ == "__main__":
    main()
