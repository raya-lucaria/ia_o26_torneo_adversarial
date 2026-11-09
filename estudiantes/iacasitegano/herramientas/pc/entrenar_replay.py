"""Un candidato: logística λ100, prior activo y mezcla80%histórico/20%nuevo."""
import argparse
import json
from pathlib import Path
from validacion import cargar, contrato, pesos, sha
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fase5_features import probabilidades
from fase5_modelos import pesos_grupos, escala

def ajustar(X,y,peso,referencia,lam=100):
    scale=escala(X); A=X/scale; center=referencia*scale; w=center.copy()
    def loss(v):
        z=A@v
        return float(peso@(np.logaddexp(0,z)-y*z)+.5*lam*np.sum((v-center)**2))
    for _ in range(50):
        prob=probabilidades(A@w)
        grad=A.T@(peso*(y-prob))-lam*(w-center)
        H=A.T@(A*(peso*np.maximum(prob*(1-prob),1e-8))[:,None])+lam*np.eye(A.shape[1])
        step=np.linalg.solve(H,grad); factor=1.; previous=loss(w)
        while factor>1e-7 and loss(w+factor*step)>previous:factor*=.5
        if loss(w+factor*step)>previous+1e-9:raise ValueError('Ajuste no desciende')
        w+=factor*step
        if max(abs(factor*step))<1e-8:break
    grad=A.T@(peso*(y-probabilidades(A@w)))-lam*(w-center)
    if max(abs(grad))>1e-5:raise ValueError('Ajuste no convergente')
    return w/scale

def entrenar(root,salida):
    root=Path(root);salida=Path(salida);contrato(root)
    if salida.exists():raise FileExistsError(salida)
    if (root/'EXAMEN_ABIERTO.json').exists():raise ValueError('Examen ya abierto; no entrenar para rescatarlo')
    h,n=cargar(root/'replay.npz'),cargar(root/'train.npz');base=pesos(root/'base.npz')
    X=np.concatenate([h['X'],n['X']]);y=np.r_[h['y'],n['y']]
    # La masa relativa de cada origen es fija; dentro de cada uno pesa igual cada grupo.
    wh=pesos_grupos(h['grupo']);wn=pesos_grupos(n['grupo'])
    weight=np.r_[.8*wh/wh.sum(),.2*wn/wn.sum()]*len(y)
    w=ajustar(X,y,weight,base[:267])
    salida.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(salida,pesos=np.r_[w,base[267:]],crudo=w,modelo='replay80_20_lambda100')
    info=dict(candidato_sha256=sha(salida),train_sha256=sha(root/'train.npz'),historico_sha256=sha(root/'replay.npz'),lambda_=100,masa_historica=.8,masa_nueva=.2,script_sha256=sha(__file__),promovido=False)
    salida.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n');return info

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directorio',type=Path,required=True);p.add_argument('--salida',type=Path,required=True)
    a=p.parse_args();print(json.dumps(entrenar(a.directorio,a.salida),indent=2))
