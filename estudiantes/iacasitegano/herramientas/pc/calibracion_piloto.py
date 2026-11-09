"""Calibración descriptiva del modelo congelado, sin entrenar ni seleccionar."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fase5_features import features, metricas, probabilidades
from validacion import cargar, pesos
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def medir(auditoria, modelo, salida):
    salida=Path(salida)
    if salida.exists():raise FileExistsError(salida)
    audit=json.loads(Path(auditoria).read_text());parts=[];groups=[]
    for f in audit['fuentes']:
        path=Path(f['partida'])
        if sha(path)!=f['sha256']:raise ValueError('Partida cambió tras auditoría')
        record=json.loads(path.read_text());shard=path.parent.parent/'shards'/(path.stem+'.npz')
        if sha(shard)!=record['shard_sha256']:raise ValueError('Shard modificado')
        d=cargar(shard);parts.append(d);groups.extend([f['grupo']]*len(d['turno']))
    d={k:np.concatenate([p[k] for p in parts]) for k in ['tablero','puntos','turno','ply','rival','ganador_blancas']}
    group=np.array(groups);w=pesos(modelo)
    pred=probabilidades(features(d['tablero'],d['puntos'],d['turno'])@w[:267])
    y=(d['ganador_blancas']==(d['turno']==0)).astype(float)
    difference=np.where(d['turno']==0,d['puntos'][:,0]-d['puntos'][:,1],d['puntos'][:,1]-d['puntos'][:,0])
    masks=dict(global_=np.ones(len(y),bool),blancas=d['turno']==0,negras=d['turno']==1,
        inicio=d['ply']<20,medio=(d['ply']>=20)&(d['ply']<60),tardio=d['ply']>=60,
        ventaja=difference>0,igualdad=difference==0,desventaja=difference<0)
    masks.update({str(r):d['rival']==r for r in np.unique(d['rival'])})
    result=dict(modelo_sha256=sha(modelo),auditoria_sha256=sha(auditoria),features_sha256=sha(Path(__file__).resolve().parents[1]/'fase5_features.py'),
        perfiles={name:metricas(y[m],pred[m],group[m]) for name,m in masks.items()},
        alcance='Diagnóstico piloto de políticas observadas. No valores óptimos, selección de modelo ni confirmación independiente. Reportar grupos y no extrapolar a rivales nuevos.')
    salida.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result['perfiles']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['auditoria','modelo','salida']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(medir(a.auditoria,a.modelo,a.salida),indent=2))
