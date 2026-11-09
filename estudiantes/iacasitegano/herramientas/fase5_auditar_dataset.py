"""Audita particiones históricas y calibración sin ajustar ni compilar modelos.

El dataset histórico es desarrollo: esta auditoría no crea una confirmación nueva.
Las métricas contra resultados observados dependen de las políticas que los generaron.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

for nombre in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[nombre] = "1"
import numpy as np
from fase5_features import features, metricas, particion_hash, probabilidades


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auditar(args):
    with np.load(args.dataset, allow_pickle=False) as z:
        datos = {k: z[k] for k in z.files}
    with np.load(args.pesos, allow_pickle=False) as z:
        w = z["pesos"][:267].copy()
    activas = np.r_[np.arange(0,253), np.arange(263,266)]
    if np.any(np.delete(w, activas)):
        raise ValueError("Este auditor solo soporta familias ligeras")
    ids = datos["partida_origen"].astype(str)
    val = particion_hash(ids, sal=args.sal_principal)
    y = (datos["ganador_blancas"] == (datos["turno"] == 0)).astype(float)
    grupos_por_id = {}
    fuentes_faltantes = []
    for ident in np.unique(ids):
        ruta = args.fuentes / str(ident)
        if ruta.is_file():
            d = json.loads(ruta.read_text())
            m = d["meta"]
            # El dado se identifica por semilla maestra/pareja/juego, compartido por colores.
            grupos_por_id[ident] = f"{m['semilla_maestra']}:{m['pareja']}:{m['juego']}"
        else:
            fuentes_faltantes.append(str(ident))
            grupos_por_id[ident] = "desconocido:" + str(ident)
    grupos = np.array([grupos_por_id[x] for x in ids])
    lados = defaultdict(set)
    for ident in np.unique(ids):
        lados[grupos_por_id[ident]].add(bool(particion_hash([ident], sal=args.sal_principal)[0]))
    cruzados = {g for g, lados_g in lados.items() if len(lados_g) == 2}
    estado_lados = defaultdict(set)
    estado_filas = defaultdict(list)
    llaves = []
    for i in range(len(y)):
        raw = datos["tablero"][i].tobytes() + datos["puntos"][i].tobytes() + bytes([int(datos["turno"][i])])
        llave = hashlib.sha256(raw).hexdigest()
        llaves.append(llave)
        estado_lados[llave].add(bool(val[i]))
        estado_filas[llave].append(i)
    duplicados_cruzados = {k for k, lados_k in estado_lados.items() if len(lados_k) == 2}
    pred = np.empty(len(y))
    suma = np.zeros(267)
    cuadrados = np.zeros(267)
    gradiente = np.zeros(267)
    n_aj = int((~val).sum())
    for inicio in range(0, len(y), 1024):
        s = slice(inicio, inicio + 1024)
        X = features(datos["tablero"][s], datos["puntos"][s], datos["turno"][s])
        pred[s] = probabilidades(X @ w)
        mascara = ~val[s]
        Xa = X[mascara]
        suma += Xa.sum(axis=0)
        cuadrados += (Xa*Xa).sum(axis=0)
        gradiente += Xa.T @ (y[s][mascara] - pred[s][mascara])
        if args.pausa:
            time.sleep(args.pausa)
    esc = np.sqrt(np.maximum(0, cuadrados / n_aj - (suma / n_aj)**2))
    esc[esc < 1e-12] = 1
    we = w * esc
    ge = gradiente / esc
    lam = float(ge @ we / (we @ we))
    residuo = float(np.linalg.norm(ge - lam*we) / max(np.linalg.norm(ge), 1e-15))
    perfil = {}
    for nombre, mascara in {
        "ajuste_historico": ~val, "validacion_historica_vista": val,
        "validacion_sin_estado_repetido_en_ajuste": val & ~np.isin(llaves, list(duplicados_cruzados)),
    }.items():
        perfil[nombre] = metricas(y[mascara], pred[mascara], grupos[mascara])
    for rival in np.unique(datos["rival"]):
        mascara = val & (datos["rival"] == rival)
        perfil["rival:"+str(rival)] = metricas(y[mascara], pred[mascara], grupos[mascara])
    for nombre, mascara_fase in {
        "temprana_ply0_20": datos["ply"] < 20,
        "media_ply20_60": (datos["ply"] >= 20) & (datos["ply"] < 60),
        "tardia_ply60mas": datos["ply"] >= 60,
        "alguien_dos_puntos": np.max(datos["puntos"], axis=1) == 2,
        "ventaja_marcador": datos["puntos"][np.arange(len(y)), datos["turno"]] > datos["puntos"][np.arange(len(y)), 1-datos["turno"]],
    }.items():
        mascara = val & mascara_fase
        perfil["fase:"+nombre] = metricas(y[mascara], pred[mascara], grupos[mascara])
    bandas = []
    for lo in np.arange(0, 1, .1):
        mascara = val & (pred >= lo) & (pred < lo + .1)
        bandas.append({"desde": float(lo), "n": int(mascara.sum()),
                       "prediccion_media": float(pred[mascara].mean()) if mascara.any() else None,
                       "victoria_observada": float(y[mascara].mean()) if mascara.any() else None})
    res = {
        "dataset_sha256": sha(args.dataset), "pesos_sha256": sha(args.pesos),
        "sal_particion": args.sal_principal,
        "fuentes": str(args.fuentes), "n": len(y), "partidas": len(np.unique(ids)),
        "ajuste": n_aj, "validacion": int(val.sum()), "grupos_dados": len(lados),
        "grupos_dados_en_ambas_particiones": len(cruzados),
        "posiciones_validacion_de_grupo_compartido": int((val & np.isin(grupos, list(cruzados))).sum()),
        "fuentes_faltantes": fuentes_faltantes,
        "estados_distintos": len(estado_lados), "estados_repetidos_entre_particiones": len(duplicados_cruzados),
        "filas_validacion_repetidas_en_ajuste": sum(bool(v) and k in duplicados_cruzados for v,k in zip(val,llaves)),
        "estados_con_resultados_distintos": sum(len(set(y[ix])) > 1 for ix in estado_filas.values()),
        "estacionariedad_modelo": {"lambda_inferida": lam, "residuo_relativo": residuo,
                                   "advertencia": "Diagnostico de ecuaciones IRLS, no prueba unica de receta"},
        "metricas": perfil, "bandas_validacion": bandas,
        "limites": ["Validacion historica ya usada para elegir modelos; no confirmacion nueva",
                    "Estados repetidos con resultados distintos pueden reflejar azar/politicas, no son por si solos errores de etiqueta",
                    "No se ajustaron pesos ni se midio fuerza"],
        "herramientas_sha256": {x: sha(Path(__file__).with_name(x)) for x in [Path(__file__).name, "fase5_features.py"]},
    }
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "auditoria.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(args.salida / "predicciones_y_grupos.npz", prob=pred, y=y, grupo=grupos,
                        validacion_historica=val, estado_sha256=np.array(llaves), partida=ids)
    print(json.dumps({k: res[k] for k in ["n", "partidas", "ajuste", "validacion", "grupos_dados",
                     "grupos_dados_en_ambas_particiones", "estados_repetidos_entre_particiones",
                     "filas_validacion_repetidas_en_ajuste", "estacionariedad_modelo", "metricas"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--pesos", type=Path, required=True)
    ap.add_argument("--fuentes", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--pausa", type=float, default=.03)
    ap.add_argument("--sal-principal", default="_principal")
    auditar(ap.parse_args())
