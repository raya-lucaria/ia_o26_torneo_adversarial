"""Resume resultados JSON del torneo en una tabla compacta por color."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


FALLAS = {"ilegal", "error", "reloj", "memoria", "preparar"}


def resumir(carpeta, bot):
    grupos = defaultdict(list)
    for archivo in Path(carpeta).rglob("*.json"):
        try:
            registro = json.loads(archivo.read_text(encoding="utf-8"))
            if not {"meta", "ganador", "jugadas", "reloj_usado", "puntos"} <= registro.keys():
                continue
        except (OSError, ValueError):
            continue
        meta = registro["meta"]
        for color in ("blancas", "negras"):
            if meta.get(color) != bot:
                continue
            gano = registro["ganador"] == color
            fallo = registro.get("motivo") in FALLAS and not gano
            grupos[color].append(
                {
                    "gano": gano,
                    "puntos": registro["puntos"].get(color, 0),
                    "jugadas": len(registro["jugadas"]),
                    "reloj": registro["reloj_usado"].get(color, 0.0),
                    "decision": max(
                        (j.get("tiempo", 0.0) for j in registro["jugadas"] if j["color"] == color),
                        default=0.0,
                    ),
                    "profundidad": max(
                        (
                            (j.get("info") or {}).get("prof", 0)
                            for j in registro["jugadas"]
                            if j["color"] == color
                        ),
                        default=0,
                    ),
                    "nps": max(
                        (
                            (j.get("info") or {}).get("nps", 0)
                            for j in registro["jugadas"]
                            if j["color"] == color
                        ),
                        default=0,
                    ),
                    "ilegales": int(fallo and registro.get("motivo") == "ilegal"),
                    "otras_fallas": int(fallo and registro.get("motivo") != "ilegal"),
                }
            )

    print(f"Bot: {bot}")
    print(
        "Color   W-L   puntos   jugadas_prom   prof_max   nps_max   reloj_max_s "
        " decision_max_s   ilegales   otras_fallas"
    )
    total = 0
    for color in ("blancas", "negras"):
        partidas = grupos[color]
        if not partidas:
            continue
        total += len(partidas)
        ganadas = sum(p["gano"] for p in partidas)
        perdidas = len(partidas) - ganadas
        puntos = sum(p["puntos"] for p in partidas)
        promedio = sum(p["jugadas"] for p in partidas) / len(partidas)
        reloj = max(p["reloj"] for p in partidas)
        decision = max(p["decision"] for p in partidas)
        profundidad = max(p["profundidad"] for p in partidas)
        nps = max(p["nps"] for p in partidas)
        ilegales = sum(p["ilegales"] for p in partidas)
        otras = sum(p["otras_fallas"] for p in partidas)
        print(
            f"{color:7} {ganadas:2}-{perdidas:<2} {puntos:7} {promedio:13.1f} "
            f"{profundidad:10} {nps:9} {reloj:12.3f} {decision:16.3f} {ilegales:9} {otras:13}"
        )
    print(f"Total de partidas resumidas: {total}")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carpeta", type=Path, help="carpeta que contiene los JSON de las partidas")
    parser.add_argument("--bot", default="frame_perfect_push")
    args = parser.parse_args()
    total = resumir(args.carpeta, args.bot)
    if not total:
        parser.error("no encontre registros de partidas para ese bot")


if __name__ == "__main__":
    main()