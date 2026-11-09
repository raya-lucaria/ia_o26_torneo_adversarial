"""Reconstruye trazas completas y exporta posiciones previas a decisiones.

Herramienta offline: no busca, entrena ni estima qué jugada era mejor. La muestra
es de desarrollo y contiene selección retrospectiva; no sirve como validación
de fuerza. No exporta tiradas futuras al registro de una posición.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trazas", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--equipo", default="iacasitegano")
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("La salida ya existe; usa otra carpeta para conservar la evidencia.")
    # Se fija antes de importar el motor: reconstrucción ligera, sin compilación.
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    for nombre in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[nombre] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sistema"))
    from motor import Estado, Jugada

    archivos = sorted(args.trazas.glob("*.json"))
    if not archivos:
        parser.error("No hay JSON en la carpeta de trazas.")
    casos, fuentes = [], []
    plies = 0
    for ruta in archivos:
        contenido = ruta.read_bytes()
        datos = json.loads(contenido)
        meta = datos["meta"]
        colores = [c for c in ("blancas", "negras") if meta[c] == args.equipo]
        if len(colores) != 1:
            raise ValueError(f"{ruta}: el equipo debe aparecer exactamente una vez")
        propio = colores[0]
        rival = "negras" if propio == "blancas" else "blancas"
        estado = Estado.inicial()
        decisiones = []
        vio_dos = False
        for numero, registro in enumerate(datos["jugadas"], 1):
            if registro["n"] != numero or registro["color"] != estado.turno:
                raise ValueError(f"{ruta}: secuencia/turno incoherente en {numero}")
            jugada = Jugada.desde_texto(registro["jugada"])
            if jugada not in estado.jugadas():
                raise ValueError(f"{ruta}: jugada ilegal en {numero}")
            ataque = estado.es_ataque(jugada)
            combate = registro["combate"]
            if ataque != (combate is not None):
                raise ValueError(f"{ruta}: combate incoherente en {numero}")
            prob = estado.prob_ataque(jugada)
            if ataque and abs(prob - combate["p"]) > 1e-12:
                raise ValueError(f"{ruta}: probabilidad incoherente en {numero}")
            if estado.turno == propio:
                etiquetas = []
                indice = len(decisiones) + 1
                if indice in (8, 16, 24, 32):
                    etiquetas.append("muestreo_fijo")
                if estado.puntos[propio] == 2 and not vio_dos:
                    etiquetas.append("primera_decision_con_dos_puntos")
                    vio_dos = True
                if estado.puntos[rival] == 2:
                    etiquetas.append("rival_con_dos_puntos")
                legal = estado.jugadas()
                serial = estado.a_dict()
                huella = hashlib.sha256(json.dumps(
                    serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                decisiones.append({
                    "id": f"{ruta.stem}:ply:{numero}",
                    "fuente": ruta.name, "ply": numero,
                    "decision_propia": indice, "color_propio": propio,
                    "grupo": ":".join(str(meta[k]) for k in
                                        ("semilla_maestra", "pareja", "juego")),
                    "estado_sha256": huella, "estado": serial,
                    "jugada_observada": str(jugada),
                    "probabilidad_exito_observada": prob,
                    "jugadas_legales": [str(j) for j in legal],
                    "ataques_legales": sum(estado.es_ataque(j) for j in legal),
                    "info_busqueda_observada": registro.get("info", {}),
                    "etiquetas": etiquetas,
                    "diagnostico": "pendiente_contrafactual",
                })
            estado = estado.aplicar(jugada, combate["gano"] if ataque else None)
            plies += 1
        if estado.puntos != datos["puntos"] or estado.ganador() != datos["ganador"]:
            raise ValueError(f"{ruta}: marcador/ganador final no coincide con el motor")
        if len(decisiones) >= 2:
            for caso in decisiones[-2:]:
                caso["etiquetas"].append("dos_ultimas_decisiones_retrospectivas")
        casos.extend(c for c in decisiones if c["etiquetas"])
        fuentes.append({"archivo": ruta.name,
                        "sha256": hashlib.sha256(contenido).hexdigest(),
                        "ganador": datos["ganador"], "puntos": datos["puntos"]})

    # Las dos partidas de un par de dados tienen el mismo grupo. Una partición
    # futura debe mantener juntas todas sus posiciones y repeticiones.
    manifiesto = {
        "version": 1, "equipo": args.equipo,
        "trazas": str(args.trazas.resolve()), "partidas_reconstruidas": len(fuentes),
        "plies_reconstruidos": plies, "casos": len(casos),
        "grupos": len({c["grupo"] for c in casos}),
        "fuentes": fuentes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "alcance": "Desarrollo visto; reconstrucción, no diagnóstico causal ni fuerza.",
        "seleccion": "Decisiones propias 8/16/24/32, primera con 2 puntos, rival con 2, últimas 2.",
        "particion": "Agrupar por semilla/pareja/juego y deduplicar estados antes de entrenar.",
        "contrafactual": "El analizador debe usar solo estado y alternativas, sin dados futuros.",
    }
    args.salida.mkdir(parents=True)
    (args.salida / "casos.jsonl").write_text("".join(
        json.dumps(c, ensure_ascii=False) + "\n" for c in casos), encoding="utf-8")
    (args.salida / "manifest.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifiesto[k] for k in (
        "partidas_reconstruidas", "plies_reconstruidos", "casos", "grupos", "alcance")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
