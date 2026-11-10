"""Diagnóstico descriptivo de dos modelos congelados; no ajusta ni abre test."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

FAMILIAS = {
    "puntos": [0], "patrones": list(range(1, 244)),
    "filas": list(range(244, 253)), "marcador": list(range(263, 266)),
    "apagadas": list(range(253, 263)) + [266],
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -z))


def pesos_grupo(g):
    _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    w = 1. / counts[inv]
    return w / w.sum()


def media_grupos(values, groups):
    ids, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return ids, np.bincount(inv, weights=values) / counts


def metricas(y, z, groups):
    p = sigmoid(z); w = pesos_grupo(groups)
    return {"filas": len(y), "grupos": len(np.unique(groups)),
            "brier": float(w @ ((p-y)**2)),
            "logloss": float(w @ (np.logaddexp(0., z)-y*z)),
            "prediccion_media": float(w @ p), "etiqueta_media": float(w @ y)}


def resumen(d, z0, z1):
    y, g, ply, turn = (d[k] for k in ("y", "grupo", "ply", "turno"))
    masks = {"global": np.ones(len(y), bool), "blancas": turn == 0,
             "negras": turn == 1, "inicio": ply < 20,
             "medio": (ply >= 20) & (ply < 60), "tardio": ply >= 60,
             "marcador_igual": d['X'][:, 0] == 0,
             "marcador_favorable": d['X'][:, 0] > 0,
             "marcador_adverso": d['X'][:, 0] < 0}
    out = {}
    for name, mask in masks.items():
        if not mask.any(): continue
        a = metricas(y[mask], z0[mask], g[mask]); b = metricas(y[mask], z1[mask], g[mask])
        out[name] = {"base": a, "candidato": b,
                     "delta_brier": b['brier']-a['brier'],
                     "delta_logloss": b['logloss']-a['logloss']}
    return out


def diagnosticar(lote, salida):
    lote, salida = Path(lote).resolve(), Path(salida).resolve()
    if salida.exists(): raise FileExistsError(salida)
    gate = lote / 'validacion'
    c = json.loads((gate/'contrato.json').read_text())
    sel = json.loads((gate/'seleccion_replay/RESULTADO.json').read_text())
    # Lista cerrada: ni datos combinados ni test.npz se cargan aquí.
    datasets, hashes = {}, {}
    for name in ('train', 'replay', 'retencion', 'validation'):
        path = gate/(name+'.npz'); hashes[str(path)] = sha(path)
        if hashes[str(path)] != c['archivos'][path.name]: raise ValueError('Partición cambió: '+name)
        with np.load(path, allow_pickle=False) as z:
            datasets[name] = {k: z[k] for k in ('X', 'y', 'grupo', 'ply', 'turno')}
    base_path, cand_path = gate/'base.npz', lote/'candidato/replay.npz'
    with np.load(base_path, allow_pickle=False) as z: base = z['pesos'].copy()
    with np.load(cand_path, allow_pickle=False) as z: candidate = z['pesos'].copy()
    if sha(base_path) != c['archivos']['base.npz'] or sha(cand_path) != sel['candidato_sha256']:
        raise ValueError('Modelos cambiados')
    if not np.array_equal(base[267:], candidate[267:]): raise ValueError('Banderas distintas')
    hashes.update({str(base_path): sha(base_path), str(cand_path): sha(cand_path)})
    w0, w1 = base[:267], candidate[:267]; delta = w1-w0
    results, grupos, atribuciones, calibracion = {}, {}, {}, {}
    for name, d in datasets.items():
        X, y, g = d['X'], d['y'], d['grupo']; weight = pesos_grupo(g)
        z0, z1 = X @ w0, X @ w1
        p0, p1 = sigmoid(z0), sigmoid(z1)
        results[name] = resumen(d, z0, z1)
        ids, diff = media_grupos((p1-y)**2-(p0-y)**2, g)
        rng = np.random.default_rng(4823)
        boot = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
        grupos[name] = {
            "peores": int((diff > 1e-12).sum()), "mejores": int((diff < -1e-12).sum()),
            "iguales": int((abs(diff) <= 1e-12).sum()), "total": len(diff),
            "mediana_delta": float(np.median(diff)),
            "intervalo_bootstrap_descriptivo_95": np.quantile(boot, [.025, .975]).tolist(),
            "por_grupo": [{"grupo": str(i), "delta_brier": float(v)} for i, v in zip(ids, diff)],
        }
        # Atribución exacta a lo largo del segmento de logits entre ambos modelos.
        # No crea/evalúa modelos alternativos ni identifica un efecto causal.
        dz = z1-z0; de = (p1-y)**2-(p0-y)**2
        slope = 2*(p0-y)*p0*(1-p0)
        np.divide(de, dz, out=slope, where=abs(dz) > 1e-7)
        attrs = {family: float(weight @ (slope*(X[:, cols] @ delta[cols])))
                 for family, cols in FAMILIAS.items()}
        if abs(sum(attrs.values())-float(weight @ de)) > 1e-9:
            raise ValueError('La atribución no suma al cambio de Brier')
        atribuciones[name] = attrs
        # Bins prefijados por predicción BASE: la población de cada bin no cambia.
        bins = np.minimum((p0*10).astype(int), 9); rows = []
        for i in range(10):
            mask = bins == i
            if not mask.any(): continue
            sub = weight[mask]/weight[mask].sum()
            rows.append({"bin_base": [i/10, (i+1)/10], "filas": int(mask.sum()),
                         "grupos": len(np.unique(g[mask])),
                         "pred_base": float(sub @ p0[mask]), "pred_candidato": float(sub @ p1[mask]),
                         "observado": float(sub @ y[mask])})
        calibracion[name] = rows
    for part, stored in [('validation', 'validation'), ('retencion', 'retencion')]:
        for model in ('base', 'candidato'):
            got = results[part]['global'][model]
            ref = sel[stored]['global_'][model]
            for key in ('brier', 'logloss'):
                if abs(got[key]-ref[key+'_igual_peso_grupo']) > 1e-11:
                    raise ValueError('No reproduce métricas certificadas')
    h, n = datasets['replay'], datasets['train']
    X = np.concatenate([h['X'], n['X']]); y = np.r_[h['y'], n['y']]
    weight = np.r_[.8*pesos_grupo(h['grupo']), .2*pesos_grupo(n['grupo'])]*len(y)
    scale = X.std(axis=0); scale[scale < 1e-12] = 1
    standardized = X/scale; center = w0*scale; change = delta*scale
    gradient_base, gradient_final = {}, {}
    for origin, sl in [('replay', slice(0, len(h['y']))), ('nuevo', slice(len(h['y']), len(y)))]:
        A = standardized[sl]; yy = y[sl]; ww = weight[sl]
        g0 = A.T @ (ww*(sigmoid(A @ center)-yy))
        g1 = A.T @ (ww*(sigmoid(A @ (center+change))-yy))
        gradient_base[origin] = g0; gradient_final[origin] = g1
    total_final = gradient_final['replay']+gradient_final['nuevo']+100*change
    support_h = np.any(abs(h['X']) > 1e-12, axis=0)
    support_n = np.any(abs(n['X']) > 1e-12, axis=0)
    novel = np.flatnonzero(~support_h & support_n)
    grad0 = gradient_base['replay']+gradient_base['nuevo']
    grad_norm = float(np.max(abs(total_final)))
    if grad_norm > 1e-5: raise ValueError('Candidato no estacionario en el objetivo congelado')
    optimization = {
        'max_gradiente_final': grad_norm,
        'producto_gradiente_base_con_desplazamiento': {
            k: float(v @ change) for k, v in gradient_base.items()},
        'penalizacion_prior': float(50*(change @ change)),
        'gradiente_base_norma': float(np.linalg.norm(grad0)),
        'columnas_solo_nuevo_train': novel.tolist(),
        'familias': {f: {'norma_delta_escalada': float(np.linalg.norm(change[cols])),
                         'grad_replay_dot_delta': float(gradient_base['replay'][cols] @ change[cols]),
                         'grad_nuevo_dot_delta': float(gradient_base['nuevo'][cols] @ change[cols])}
                     for f, cols in FAMILIAS.items()},
    }
    result = {'fecha': datetime.datetime.now().astimezone().isoformat(),
              'alcance': 'Diagnóstico posterior descriptivo; solo dos modelos congelados, sin entrenamiento ni test',
              'fuentes': hashes, 'script_sha256': sha(__file__),
              'perfiles': results, 'grupos': grupos, 'calibracion_bins_base': calibracion,
              'atribucion_brier_segmento_logit': atribuciones,
              'advertencia_atribucion': 'Depende del segmento entre logits, no prueba causal ni selección de familias',
              'optimizacion': optimization,
              'verificaciones': {'reproduce_metricas': True, 'atribucion_suma': True,
                                  'gradiente_objetivo': True, 'test_cargado': False,
                                  'modelos_alternativos_evaluados': 0}}
    salida.mkdir(parents=True)
    (salida/'DIAGNOSTICO.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'perfiles_globales': {k:v['global'] for k,v in results.items()},
                      'grupos': {k:{a:b for a,b in v.items() if a!='por_grupo'} for k,v in grupos.items()},
                      'atribuciones': atribuciones, 'optimizacion': optimization}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lote', type=Path, required=True)
    p.add_argument('--salida', type=Path, required=True)
    args = p.parse_args(); diagnosticar(args.lote, args.salida)
