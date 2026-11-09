"""Genera experiencia del participante contra un único rival UCT fuerte.

El proceso es deliberadamente independiente del entrenamiento: solo lee el
modelo activo, escribe partidas/checkpoints y nunca sustituye pesos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


def atomico(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cargar_modulos(root: Path):
    sistema = root / "sistema"
    equipo = root / "estudiantes/iacasitegano"
    herramientas = equipo / "herramientas"
    sys.path[:0] = [str(sistema), str(equipo), str(herramientas)]
    from motor import Estado, Jugada
    import busqueda_eval3 as busqueda
    from fase5_rivales import uct
    return Estado, Jugada, busqueda, uct


def tablero_estado(estado):
    from motor.jugada import casilla_de_texto
    tablero = np.zeros(81, dtype=np.int8)
    for color_i, color in enumerate(("blancas", "negras"), 1):
        for casilla in estado.piezas(color):
            tablero[casilla_de_texto(casilla)] = color_i
    return tablero


def muestra(estado, ply):
    return {
        "tablero": tablero_estado(estado).tolist(),
        "puntos": [int(estado.puntos["blancas"]), int(estado.puntos["negras"])],
        "turno": int(estado.turno == "negras"),
        "ply": int(ply),
    }


def resultado_combate(estado, jugada, dados):
    if not estado.es_ataque(jugada):
        return None
    prob = estado.prob_ataque(jugada)
    u = dados[estado.turno].random()
    return {"p": prob, "u": u, "gano": bool(u < prob)}


def jugar(args):
    root = Path(args.root).resolve()
    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)
    for nombre in ("partidas", "shards", "parciales"):
        (salida / nombre).mkdir(parents=True, exist_ok=True)
    Estado, Jugada, busqueda, uct = cargar_modulos(root)
    pesos_path = root / "estudiantes/iacasitegano/data/pesos_eval3_ligero.npz"
    with np.load(pesos_path, allow_pickle=False) as z:
        pesos = z["pesos"].copy()
    busqueda.cargar_pesos(pesos)
    if busqueda.GANANCIA <= 0:
        raise ValueError("El buscador activo no cargó una ganancia terminal válida")
    contrato = {
        "participante": "P activo congelado",
        "rival_unico": "uct",
        "uct_iteraciones": args.uct_iteraciones,
        "limite_jugada_s": args.limite_jugada_s,
        "minutos": args.minutos,
        "semilla_base": args.semilla,
        "sha_busqueda": sha(root / "estudiantes/iacasitegano/busqueda_eval3.py"),
        "sha_pesos": sha(pesos_path),
        "sha_eval3": sha(root / "estudiantes/iacasitegano/eval3.py"),
        "sha_motor": sha(root / "sistema/motor/nucleo.py"),
        "nota": "Experiencia offline; no es prueba oficial de fuerza ni promoción.",
    }
    mf = salida / "manifest.json"
    if mf.exists() and json.loads(mf.read_text()) != contrato:
        raise ValueError("Contrato distinto en la salida; no mezclar corridas")
    atomico(mf, contrato)
    comienzo = time.monotonic()
    completas = 0
    for juego_i in range(args.max_partidas):
        if time.monotonic() - comienzo >= args.minutos * 60:
            break
        nombre = f"{juego_i:06d}_uct_{'blancas' if juego_i % 2 == 0 else 'negras'}"
        partida = salida / "partidas" / f"{nombre}.json"
        shard = salida / "shards" / f"{nombre}.npz"
        parcial = salida / "parciales" / f"{nombre}.json"
        if partida.exists() and shard.exists():
            completas += 1
            continue
        nuestro_color = "blancas" if juego_i % 2 == 0 else "negras"
        meta = {
            "juego": juego_i,
            "nuestro_color": nuestro_color,
            "rival": "uct",
            "semilla": args.semilla + juego_i,
        }
        registros = []
        estados = []
        estado = Estado.inicial()
        dados = {
            "blancas": random.Random(f"uct14h_dado_{args.semilla + juego_i}_blancas"),
            "negras": random.Random(f"uct14h_dado_{args.semilla + juego_i}_negras"),
        }
        if parcial.exists():
            viejo = json.loads(parcial.read_text())
            registros = viejo["jugadas"]
            for registro in registros:
                estados.append(muestra(estado, len(estados)))
                jugada = Jugada.desde_texto(registro["jugada"])
                combate = registro["combate"]
                if combate is not None:
                    assert dados[estado.turno].random() == combate["u"]
                estado = estado.aplicar(jugada, combate["gano"] if combate else None)
        while not estado.terminado():
            if time.monotonic() - comienzo >= args.minutos * 60:
                atomico(parcial, {"meta": meta, "jugadas": registros})
                atomico(salida / "progreso.json", {"estado": "pausa_por_presupuesto", "completadas": completas})
                return
            ply = len(registros)
            estados.append(muestra(estado, ply))
            politica_seed = f"uct14h_politica_{args.semilla + juego_i}_{ply}"
            if estado.turno == nuestro_color:
                limite = time.perf_counter() + args.limite_jugada_s
                jugada, valor, profundidad = busqueda.elegir(estado, nuestro_color, limite)
                info = {
                    "actor": "participante",
                    "valor": float(valor),
                    "profundidad": int(profundidad),
                    "nodos": int(busqueda.ULTIMA_BUSQUEDA["nodos"]),
                    "abortos": int(busqueda.ULTIMA_BUSQUEDA["abortos"]),
                }
            else:
                jugada, info = uct(estado, politica_seed, args.uct_iteraciones, filtrar=True)
                info = {
                    "actor": "uct",
                    "iteraciones": int(args.uct_iteraciones),
                    "aristas": len(info.get("aristas", [])),
                    "victoria_inmediata": bool(info.get("victoria_inmediata", False)),
                }
            if jugada not in estado.jugadas():
                raise ValueError(f"Jugada ilegal en ply {ply}: {jugada}")
            combate = resultado_combate(estado, jugada, dados)
            registros.append({
                "n": ply + 1,
                "color": estado.turno,
                "jugada": str(jugada),
                "combate": combate,
                "info": info,
            })
            estado = estado.aplicar(jugada, combate["gano"] if combate else None)
            atomico(parcial, {"meta": meta, "jugadas": registros})
            if len(registros) > 1000:
                raise RuntimeError("Partida imposible: más de 1000 plies")
        ganador = estado.ganador()
        n = len(estados)
        with shard.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(
                f,
                tablero=np.asarray([x["tablero"] for x in estados], dtype=np.int8),
                puntos=np.asarray([x["puntos"] for x in estados], dtype=np.int8),
                turno=np.asarray([x["turno"] for x in estados], dtype=np.int8),
                ply=np.asarray([x["ply"] for x in estados], dtype=np.int32),
                ganador_turno=np.asarray([
                    int((ganador == ("negras" if x["turno"] else "blancas"))) for x in estados
                ], dtype=np.int8),
            )
        shard.with_suffix(".tmp").replace(shard)
        atomico(partida, {
            "version": 1,
            "meta": meta,
            "jugadas": registros,
            "ganador": ganador,
            "puntos_finales": dict(estado.puntos),
            "shard_sha256": sha(shard),
        })
        completas += 1
        atomico(salida / "progreso.json", {"estado": "generando", "completadas": completas, "max_partidas": args.max_partidas})
        print(json.dumps({"completadas": completas, "juego": juego_i, "ganador": ganador}, ensure_ascii=False), flush=True)
    atomico(salida / "progreso.json", {"estado": "completo", "completadas": completas, "max_partidas": args.max_partidas})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(os.environ.get("BREAKTHROUGH_ROOT", "/workspace")))
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--minutos", type=float, default=840)
    p.add_argument("--max-partidas", type=int, default=100000)
    p.add_argument("--uct-iteraciones", type=int, default=4096)
    p.add_argument("--limite-jugada-s", type=float, default=4.0)
    p.add_argument("--semilla", type=int, required=True)
    jugar(p.parse_args())


if __name__ == "__main__":
    main()
