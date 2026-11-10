"""Consolida una corrida UCT14h, analiza resultados y ajusta candidatos offline.

La selección usa una validación por partidas separadas; el test interno se lee
una sola vez después de seleccionar. Nada de esta herramienta toca los pesos
activos ni declara fuerza oficial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def metrics(y, prob, groups):
    y = np.asarray(y, float)
    prob = np.clip(np.asarray(prob, float), 1e-12, 1 - 1e-12)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    brier = (y - prob) ** 2
    logloss = -(y * np.log(prob) + (1 - y) * np.log1p(-prob))
    return {
        "n": int(len(y)),
        "grupos": int(len(counts)),
        "brier": float(np.mean(brier)),
        "logloss": float(np.mean(logloss)),
        "brier_igual_peso_grupo": float(weights @ brier),
        "logloss_igual_peso_grupo": float(weights @ logloss),
    }


def main(args):
    root = Path(args.root).resolve()
    salida = Path(args.salida).resolve()
    team = root / "estudiantes/iacasitegano"
    tools = team / "herramientas"
    sys.path[:0] = [str(root / "sistema"), str(team), str(tools)]
    from fase5_features import features, probabilidades
    from fase5_modelos import logistica, pesos_grupos

    partidas = []
    boards = []
    points = []
    turns = []
    labels = []
    groups = []
    analysis = []
    for worker in sorted(salida.glob("worker_*")):
        for game in sorted((worker / "partidas").glob("*.json")):
            shard = worker / "shards" / (game.stem + ".npz")
            if not shard.exists():
                continue
            meta = json.loads(game.read_text())
            with np.load(shard, allow_pickle=False) as z:
                b, p, t, y = z["tablero"], z["puntos"], z["turno"], z["ganador_turno"]
            if not (len(b) == len(p) == len(t) == len(y) and len(b)):
                raise ValueError(f"Shard vacío o inconsistente: {shard}")
            gid = f"{meta['meta']['semilla']}:{meta['meta']['juego']}"
            boards.append(b); points.append(p); turns.append(t); labels.append(y)
            groups.append(np.full(len(b), gid))
            jugadas = meta["jugadas"]
            nuestro = meta["meta"]["nuestro_color"]
            own = [j for j in jugadas if j["color"] == nuestro]
            rival = [j for j in jugadas if j["color"] != nuestro]
            ataques_own = [j for j in own if j["combate"] is not None]
            ataques_rival = [j for j in rival if j["combate"] is not None]
            victorias_own = sum(bool(j["combate"]["gano"]) for j in ataques_own)
            victorias_rival = sum(bool(j["combate"]["gano"]) for j in ataques_rival)
            ganador = meta["ganador"]
            analysis.append({
                "grupo": gid,
                "juego": meta["meta"]["juego"],
                "nuestro_color": nuestro,
                "ganador": ganador,
                "resultado": "victoria" if ganador == nuestro else "derrota",
                "puntos_finales": meta["puntos_finales"],
                "plies": len(jugadas),
                "ataques_propios": len(ataques_own),
                "combates_propios_ganados": victorias_own,
                "ataques_uct": len(ataques_rival),
                "combates_uct_ganados": victorias_rival,
                "abortos_busqueda": sum(j["info"].get("abortos", 0) for j in own),
                "profundidad_media": float(np.mean([j["info"].get("profundidad", 0) for j in own])) if own else 0.0,
            })
            partidas.append(gid)
    if not boards:
        raise ValueError("No hay partidas completas para consolidar")
    board = np.concatenate(boards)
    point = np.concatenate(points)
    turn = np.concatenate(turns)
    y = np.concatenate(labels).astype(float)
    group = np.concatenate(groups)
    X = features(board, point, turn)
    # 70/15/15 por partida, nunca por fila: no se mezclan posiciones de un juego.
    hashes = np.array([int(hashlib.sha256((str(g) + "_uct14h").encode()).hexdigest()[:8], 16) / 2**32 for g in group])
    fold = np.where(hashes < .70, 0, np.where(hashes < .85, 1, 2)).astype(np.int8)
    if any(len(np.unique(group[fold == i])) < 2 for i in range(3)):
        raise ValueError("La partición por partidas quedó demasiado pequeña")
    np.savez_compressed(salida / "dataset.npz", X=X, y=y, grupo=group, fold=fold, partida=group)
    write(salida / "dataset_manifest.json", {
        "filas": int(len(y)), "partidas": int(len(set(group))),
        "filas_por_fold": {str(i): int(np.sum(fold == i)) for i in range(3)},
        "partidas_por_fold": {str(i): int(len(set(group[fold == i]))) for i in range(3)},
        "rival_unico": "uct_4096", "split": "70/15/15 por partida", "dataset_sha256": sha(salida / "dataset.npz"),
    })
    write(salida / "ANALISIS_DECISIONES.json", {
        "partidas": analysis,
        "resumen": {
            "total": len(analysis),
            "victorias": sum(x["resultado"] == "victoria" for x in analysis),
            "derrotas": sum(x["resultado"] == "derrota" for x in analysis),
            "promedio_plies": float(np.mean([x["plies"] for x in analysis])),
        },
        "limite": "Asociación descriptiva; no prueba causalidad de una decisión individual.",
    })
    peso = pesos_grupos(group[fold == 0])
    candidatos = {}
    for lam in (10.0, 100.0, 1000.0):
        w = logistica(X[fold == 0], y[fold == 0], lam, peso, iteraciones=80)
        p = probabilidades(X @ w)
        candidatos[f"logistica_lambda{lam:g}"] = {
            "lambda": lam,
            "w": w,
            "validacion": metrics(y[fold == 1], p[fold == 1], group[fold == 1]),
        }
    elegido = min(candidatos, key=lambda k: (candidatos[k]["validacion"]["brier_igual_peso_grupo"], k))
    best = candidatos[elegido]
    pbest = probabilidades(X @ best["w"])
    best_weights = np.r_[best["w"], [1, 0, 0, 0, 0, 0, 1, 0]]
    candidate_path = salida / "candidato_uct14h.npz"
    np.savez_compressed(candidate_path, pesos=best_weights)
    for c in candidatos.values():
        c.pop("w")
    report = {
        "estado": "seleccion_interna_completa",
        "rival_unico": "UCT4096",
        "candidatos": candidatos,
        "seleccionado": elegido,
        "seleccion": best["validacion"],
        "test_interno_una_vez": metrics(y[fold == 2], pbest[fold == 2], group[fold == 2]),
        "candidato_sha256": sha(candidate_path),
        "dataset_sha256": sha(salida / "dataset.npz"),
        "modelo_activo_tocado": False,
        "fuerza_oficial_medida": False,
        "nota": "Holdout del mismo experimento; requiere controles funcionales y partidas oficiales antes de aceptar.",
    }
    write(salida / "VALIDACION_INTERNA.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"candidatos"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    main(p.parse_args())
