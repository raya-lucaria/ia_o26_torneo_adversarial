"""Reúne lotes nuevos completos, reserva UCT y purga estados/espejos."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validacion import cargar, guardar, sha
from fase5_features import features
import numpy as np

def preparar(lotes, salida):
    salida = Path(salida)
    if salida.exists(): raise FileExistsError(salida)
    if salida.suffix!='.npz':raise ValueError('La salida debe terminar en .npz')
    rows=[]; groups=[]; sources=[]; ids=set(); expected_contract=None
    for folder in map(Path, lotes):
        state=json.loads((folder/'progreso.json').read_text())
        if state['estado']!='completo': raise ValueError('Lote incompleto: '+str(folder))
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve().parents[1]/'fase5_verificar_episodios.py'),'--lote',str(folder)],check=True)
        manifest=json.loads((folder/'manifest.json').read_text())
        if 'montecarlo' not in manifest['rivales']: raise ValueError('No es el lote reforzado')
        actual_contract={k:v for k,v in manifest.items() if k!='semilla'}
        if expected_contract is None:expected_contract=actual_contract
        elif actual_contract!=expected_contract:raise ValueError('Fuentes o presupuestos distintos entre trabajadores')
        complete_files=list((folder/'partidas').glob('*.json'))
        if len(complete_files)!=manifest['partidas'] or state['completadas']!=manifest['partidas']:raise ValueError('Lote incompleto aunque progreso diga completo')
        for path in sorted((folder/'partidas').glob('*.json')):
            d=json.loads(path.read_text()); m=d['meta']; shard=folder/'shards'/(path.stem+'.npz')
            if sha(shard)!=d['shard_sha256']: raise ValueError('Shard alterado')
            group=f"{m['semilla_maestra']}:{m['pareja']}:{m['juego']}"
            ident=(group,m['blancas'])
            if ident in ids: raise ValueError('Semillas/partidas duplicadas entre trabajadores')
            ids.add(ident); z=cargar(shard); rows.append(z)
            groups.extend([group]*len(z['turno']))
            sources.append({'json_sha256':sha(path),'shard_sha256':sha(shard),'group':group})
    merged={k:np.concatenate([d[k] for d in rows]) for k in ['tablero','puntos','turno','ply','ganador_blancas','rival']}
    if set(merged['rival'])!={'offline_montecarlo','offline_agresivo','offline_carrera','offline_uct'}:raise ValueError('Faltan familias de rivales previstas')
    group=np.array(groups)
    holdout=set(group[merged['rival']=='offline_uct'])
    fold=np.array([2 if g in holdout else 1 if int(hashlib.sha256((g+'_pc_validacion_20261029').encode()).hexdigest()[:8],16)/2**32<.25 else 0 for g in group],np.int8)
    states=[]
    for board, pts, turn in zip(merged['tablero'],merged['puntos'],merged['turno']):
        raw=min(board.tobytes(),board.reshape(9,9)[:,::-1].tobytes())+pts.tobytes()+bytes([int(turn)])
        states.append(hashlib.sha256(raw).hexdigest())
    states=np.array(states); test=set(states[fold==2]); val=set(states[fold==1])
    purge=((fold==0)&np.isin(states,list(test|val)))|((fold==1)&np.isin(states,list(test)))
    fold[purge]=-1; keep=fold>=0
    salida.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(salida,X=features(merged['tablero'][keep],merged['puntos'][keep],merged['turno'][keep]),
                        y=(merged['ganador_blancas'][keep]==(merged['turno'][keep]==0)).astype(float),
                        grupo=group[keep],estado=states[keep],fold=fold[keep],turno=merged['turno'][keep],ply=merged['ply'][keep])
    info=dict(datos_sha256=sha(salida),sources=sources,purgadas=int(purge.sum()),
              groups={str(i):len(set(group[fold==i])) for i in [0,1,2]},
              rows={str(i):int((fold==i).sum()) for i in [0,1,2]},
              scope='UCT completo reservado; otros grupos75/25 train/validation. Retención histórica y examen deportivo se evalúan aparte.')
    guardar(salida.with_suffix('.json'),info)
    return info

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lotes',type=Path,nargs='+',required=True);p.add_argument('--salida',type=Path,required=True)
    a=p.parse_args();print(json.dumps(preparar(a.lotes,a.salida),ensure_ascii=False,indent=2))
