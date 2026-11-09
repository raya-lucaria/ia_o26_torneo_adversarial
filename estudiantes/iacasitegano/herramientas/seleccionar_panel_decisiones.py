"""Congela por metadatos 24 casos de desarrollo; no ejecuta búsqueda ni usa dados futuros."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def seleccionar(casos):
    reglas = (
        ("tempranos", lambda x: x["decision_propia"] <= 16 and "muestreo_fijo" in x["etiquetas"]),
        ("dos_puntos", lambda x: max(x["estado"]["puntos"].values()) == 2),
        ("tardios", lambda x: "dos_ultimas_decisiones_retrospectivas" in x["etiquetas"]),
    )
    grupos, fuentes = Counter(), Counter()
    usados = set()
    panel = []
    for categoria, predicado in reglas:
        for color in ("blancas", "negras"):
            for _ in range(4):
                opciones = [x for x in casos if predicado(x) and x["color_propio"] == color
                            and x["estado_sha256"] not in usados]
                if not opciones:
                    raise ValueError(f"Sin casos suficientes: {categoria}/{color}")
                x = min(opciones, key=lambda x: (grupos[x["grupo"]], fuentes[x["fuente"]], x["id"]))
                panel.append({**x, "categoria_panel": categoria})
                usados.add(x["estado_sha256"])
                grupos[x["grupo"]] += 1
                fuentes[x["fuente"]] += 1
    return panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("entrada", type=Path)
    p.add_argument("salida", type=Path)
    args = p.parse_args()
    raw = args.entrada.read_bytes()
    casos = [json.loads(line) for line in raw.splitlines() if line.strip()]
    panel = seleccionar(casos)
    contenido = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in panel)
    manifiesto = {
        "version": 1,
        "entrada": str(args.entrada),
        "entrada_sha256": hashlib.sha256(raw).hexdigest(),
        "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "regla": "tempranos <=16 con muestreo_fijo; dos_puntos max(marcador)=2; tardios dos_ultimas; "
                 "4 por color/categoria, orden indicado, deduplicar estado_sha256; "
                 "elegir menor uso de grupo, luego fuente, luego ID lexicografico",
        "advertencia": "Desarrollo visto. Tardios se identifican retrospectivamente; no implican errores.",
        "casos": len(panel),
        "grupos": len({x["grupo"] for x in panel}),
        "por_categoria_color": dict(Counter(x["categoria_panel"] + "/" + x["color_propio"] for x in panel)),
        "ids": [x["id"] for x in panel],
        "panel_sha256": hashlib.sha256(contenido.encode()).hexdigest(),
        "busquedas_ejecutadas": False,
    }
    archivos = {"panel.jsonl": contenido,
                "manifest.json": json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n"}
    for nombre, texto in archivos.items():
        destino = args.salida / nombre
        if destino.exists() and destino.read_text() != texto:
            raise FileExistsError(f"Panel congelado distinto: {destino}; usar otra carpeta")
    args.salida.mkdir(parents=True, exist_ok=True)
    for nombre, texto in archivos.items():
        (args.salida / nombre).write_text(texto)
    print(json.dumps(manifiesto, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
