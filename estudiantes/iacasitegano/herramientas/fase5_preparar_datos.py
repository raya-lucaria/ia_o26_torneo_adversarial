"""Dataset ligero con grupos de dados y purga de estados/espejos entre particiones.

Las tres particiones siguen siendo desarrollo historico; no son examen final nuevo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from fase5_features import features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--auditoria", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--rival-prueba", default="noche_minimax_fijo")
    p.add_argument("--pausa", type=float, default=.08)
    args = p.parse_args()
    metadata = json.loads((args.auditoria / "auditoria.json").read_text())
    if sha(args.dataset) != metadata["dataset_sha256"]:
        raise ValueError("Dataset diferente al auditado")
    if (args.salida / "datos.npz").exists():
        raise FileExistsError("Salida ya preparada; usarla o elegir otra ruta")
    with np.load(args.dataset, allow_pickle=False) as z:
        d = {k:z[k] for k in z.files}
    with np.load(args.auditoria / "predicciones_y_grupos.npz", allow_pickle=False) as z:
        grupos, y, partidas = z["grupo"], z["y"], z["partida"]
    if not np.array_equal(partidas, d["partida_origen"]):
        raise ValueError("Orden de filas distinto")
    ids = np.unique(grupos)
    h = {g:int(hashlib.sha256((str(g)+"_fase5_20261028").encode()).hexdigest()[:8],16)/2**32 for g in ids}
    # 0 ajuste, 1 seleccion, 2 prueba interna historica. Todo un grupo queda junto.
    fold = np.array([2 if h[g]<.2 else 1 if h[g]<.4 else 0 for g in grupos],np.int8)
    familia = set(grupos[d["rival"] == args.rival_prueba])
    fold[np.isin(grupos,list(familia))] = 2
    estados = []
    for t,pt,c in zip(d["tablero"],d["puntos"],d["turno"]):
        sufijo = pt.tobytes()+bytes([int(c)])
        # El espejo conserva puntos y turno.
        raw = min(t.tobytes(),t.reshape(9,9)[:,::-1].tobytes()) + sufijo
        estados.append(hashlib.sha256(raw).hexdigest())
    estados = np.array(estados)
    exam = set(estados[fold==2])
    sel = set(estados[fold==1])
    antes = {str(i):int((fold==i).sum()) for i in range(3)}
    purgar = ((fold==0) & np.isin(estados,list(exam|sel))) | ((fold==1) & np.isin(estados,list(exam)))
    fold[purgar] = -1
    usados = fold >= 0
    indices = np.flatnonzero(usados)
    X = np.empty((len(indices),267),np.float32)
    for inicio in range(0,len(indices),1024):
        ix = indices[inicio:inicio+1024]
        X[inicio:inicio+len(ix)] = features(d["tablero"][ix],d["puntos"][ix],d["turno"][ix])
        if args.pausa:
            time.sleep(args.pausa)
    args.salida.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.salida/"datos.npz",X=X,y=y[usados],grupo=grupos[usados],
                        partida=partidas[usados],fold=fold[usados],estado=estados[usados],
                        rival=d["rival"][usados],indice_original=indices)
    resumen = {"dataset_original_sha256":sha(args.dataset),"auditoria_sha256":sha(args.auditoria/"auditoria.json"),
               "herramienta_sha256":sha(__file__),"features_sha256":sha(Path(__file__).with_name("fase5_features.py")),
               "datos_sha256":sha(args.salida/"datos.npz"),"sal_grupos":"_fase5_20261028",
               "folds":{"0":"ajuste","1":"seleccion","2":"prueba_interna_historica"},
               "regla":"grupo entero por hash 60/20/20; grupos del rival prueba a fold2; purgar estado/espejo de ajuste si esta en 1/2, y de seleccion si esta en 2",
               "rival_prueba":args.rival_prueba,"filas_antes_purga":antes,"purgadas":int(purgar.sum()),
               "filas_despues":{str(i):int((fold==i).sum()) for i in range(3)},
               "grupos_despues":{str(i):len(set(grupos[fold==i])) for i in range(3)},
               "advertencia":"Prueba interna historica: estos datos ya participaron en desarrollo anterior. Confirmacion requiere partidas nuevas"}
    # Contratos de separacion sobre los datos realmente exportados.
    for i in range(3):
        if not (fold==i).any():
            raise ValueError(f"Particion vacia: {i}")
        for j in range(i):
            assert not set(grupos[fold==i]) & set(grupos[fold==j])
            assert not set(estados[fold==i]) & set(estados[fold==j])
    (args.salida/"manifest.json").write_text(json.dumps(resumen,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(resumen,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
