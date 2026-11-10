"""Genera data/aperturas.json: el libro de aperturas (offline, no corre en la liga).

Para cada color, recorre TODAS las respuestas posibles del rival y guarda nuestra mejor jugada,
buscada mucho más hondo que en partida:
  - con blancas: nuestras jugadas 1, 2 y 3 (1 + ~25 + ~625 posiciones);
  - con negras: nuestras jugadas 1 y 2 (~25 + ~625 posiciones).
Más adelante casi todas las posiciones son únicas (en 500 partidas contra montecarlo, desde
nuestra 4a jugada ninguna se repitió), así que un libro más largo no se usaría.

La búsqueda es la misma de strategy.py, con los pesos de data/pesos.json y la ventaja asumida
a favor de quien mueve (que en el libro siempre somos nosotros).

Uso (desde estudiantes/<equipo>/, con el .venv del torneo activo):
    python herramientas/generar_aperturas.py --procs 2 --t1 6 --t2 2 --t3 1.5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARPETA))

bot = None


def _iniciar(carpeta_datos):
    global bot
    import strategy

    bot = strategy
    bot._cargar_datos(carpeta_datos)
    bot._nuevo_contexto()
    bot._calentar()          # compilar antes de medir tiempos de búsqueda
    bot._nuevo_contexto()


def _texto(o, d):
    c = "abcdefghi"
    return f"{c[o % 9]}{o // 9 + 1}-{c[d % 9]}{d // 9 + 1}"


def _mejor(args):
    """Mejor jugada de quien mueve (que somos nosotros) con `segundos` de búsqueda."""
    tab_l, pts_l, lado, segundos = args
    tab = np.array(tab_l, np.int8)
    pts = np.array(pts_l, np.int64)
    bot.C.w[bot.NFEAT - 1] = bot.VENTAJA if lado == 0 else -bot.VENTAJA
    o, d, v, prof, nodos = bot._buscar(tab, pts, lado, segundos, duro=segundos * 2.5)
    return int(o), int(d), float(v), int(prof)


def _aplicar(tab, pts, lado, o, d, gana):
    t2, p2 = tab.copy(), pts.copy()
    bot.hacer(t2, p2, lado, o, d, gana, np.uint64(0), bot.C.zp, bot.C.zt, bot.C.zs)
    return t2, p2


def _respuestas(tab, pts, lado):
    """Posiciones tras cada jugada legal de `lado` (los ataques, con sus dos resultados)."""
    n = bot.generar(tab, lado, bot.C.mo, bot.C.md, 0)
    jugadas = [(int(bot.C.mo[0, i]), int(bot.C.md[0, i])) for i in range(n)]
    out = []
    for o, d in jugadas:
        for gana in ((True, False) if tab[d] != 0 else (True,)):
            out.append(_aplicar(tab, pts, lado, o, d, gana))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--t1", type=float, default=6.0, help="segundos por posición, nivel 1")
    ap.add_argument("--t2", type=float, default=2.0, help="segundos por posición, nivel 2")
    ap.add_argument("--t3", type=float, default=1.5, help="segundos por posición, nivel 3 (blancas)")
    ap.add_argument("--salida", default=str(CARPETA / "data" / "aperturas.json"))
    a = ap.parse_args()
    _iniciar(CARPETA / "data")
    libro, profs = {}, []

    inicial = np.zeros(81, np.int8)
    inicial[:18] = 1
    inicial[63:] = 2
    cero = np.zeros(2, np.int64)

    def buscar_nivel(posiciones, segundos, pool):
        """posiciones: dict clave -> (tab, pts, lado). Regresa dict clave -> (o, d)."""
        claves = [k for k in posiciones if k not in libro]
        tareas = [(posiciones[k][0].tolist(), posiciones[k][1].tolist(), posiciones[k][2], segundos) for k in claves]
        res = pool.map(_mejor, tareas, chunksize=1)
        for k, (o, d, v, prof) in zip(claves, res):
            libro[k] = _texto(o, d)
            profs.append(prof)
        return {k: posiciones[k] for k in claves}, dict(zip(claves, res))

    def siguientes(posiciones, mejores):
        """Aplica nuestra jugada del libro y expande todas las respuestas del rival."""
        nuevas = {}
        for k, (tab, pts, lado) in posiciones.items():
            o, d = mejores[k][0], mejores[k][1]
            # nuestra jugada (en la apertura no hay ataques; si lo hubiera, ambos resultados)
            for gana in ((True, False) if tab[d] != 0 else (True,)):
                t2, p2 = _aplicar(tab, pts, lado, o, d, gana)
                for t3, p3 in _respuestas(t2, p2, 1 - lado):
                    nuevas[bot.clave_posicion(t3, p3, lado)] = (t3, p3, lado)
        return nuevas

    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, initializer=_iniciar, initargs=(CARPETA / "data",)) as pool:
        # nivel 1: blancas en la posición inicial; negras tras cada primera jugada de blancas
        n1 = {bot.clave_posicion(inicial, cero, 0): (inicial, cero, 0)}
        for t, p in _respuestas(inicial, cero, 0):
            n1[bot.clave_posicion(t, p, 1)] = (t, p, 1)
        pos1, mej1 = buscar_nivel(n1, a.t1, pool)
        print(f"nivel 1: {len(pos1)} posiciones ({time.time() - t0:.0f} s)", flush=True)
        # nivel 2
        n2 = siguientes(pos1, mej1)
        pos2, mej2 = buscar_nivel(n2, a.t2, pool)
        print(f"nivel 2: {len(pos2)} posiciones ({time.time() - t0:.0f} s)", flush=True)
        # nivel 3: solo con blancas
        n3 = siguientes({k: v for k, v in pos2.items() if v[2] == 0}, mej2)
        pos3, _ = buscar_nivel(n3, a.t3, pool)
        print(f"nivel 3: {len(pos3)} posiciones ({time.time() - t0:.0f} s)", flush=True)

    datos = {
        "descripcion": ("Libro de aperturas: clave = 81 casillas (0 vacía, 1 blanca, 2 negra; a1..i1, "
                        "a2..i2, ...) + quién mueve (b/n) + puntos de blancas y negras; valor = nuestra "
                        "jugada. Cubre todas las respuestas del rival en nuestras jugadas 1-3 con "
                        "blancas y 1-2 con negras."),
        "origen": (f"herramientas/generar_aperturas.py con los pesos de data/pesos.json: {a.t1} s por "
                   f"posición en el nivel 1, {a.t2} s en el 2 y {a.t3} s en el 3 (profundidad media "
                   f"{np.mean(profs):.1f}, mínima {min(profs)})."),
        "posiciones": libro,
    }
    Path(a.salida).write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    print(f"{len(libro)} posiciones en {a.salida} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
