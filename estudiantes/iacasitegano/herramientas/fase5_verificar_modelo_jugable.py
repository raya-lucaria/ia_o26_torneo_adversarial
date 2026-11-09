"""Comprueba representación→NPZ→evaluador→utilidad sin entrenar ni medir reloj.

Usar trazas piloto ya inspeccionadas, nunca el test de selección en curso.
Las sondas de pesos son controles sintéticos en memoria, no candidatos de fuerza.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def comprobar_vector(w):
    import numpy as np
    if w.shape!=(275,) or w.dtype!=np.dtype('float64') or not np.isfinite(w).all():
        raise ValueError('Se requieren275pesos float64 finitos, no el formato antiguo267')
    if not np.array_equal(w[267:],np.array([1,0,0,0,0,0,1,0])):
        raise ValueError('Banderas fuera del contrato ligero')
    active=np.zeros(267,bool);active[:253]=True;active[263:266]=True
    if np.any(w[:267][~active]!=0):
        raise ValueError('Coeficientes no nulos en familias apagadas')
    return active

def verificar(buscador,modelo,lotes,salida,jit=False):
    if Path(salida).exists():raise FileExistsError(salida)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
    os.environ['NUMBA_DISABLE_JIT']='0' if jit else '1'
    os.environ['NUMBA_CACHE_DIR']=str(Path(salida).resolve().parent/('cache_modelo_jit' if jit else 'cache_modelo_py'))
    import numpy as np
    from fase5_features import features
    equipo=Path(__file__).resolve().parents[1]
    if sha(equipo/'eval3.py')!=sha(Path(buscador).parent/'eval3.py'):raise ValueError('Representaciones distintas')
    sys.path[:0]=[str(Path(buscador).resolve().parent),str(equipo.parents[1]/'sistema')]
    import eval3
    spec=importlib.util.spec_from_file_location('fase5_contrato_modelo',Path(buscador).resolve())
    search=importlib.util.module_from_spec(spec);sys.modules[spec.name]=search;spec.loader.exec_module(search)
    with np.load(modelo,allow_pickle=False) as z:w=z['pesos'].copy()
    active=comprobar_vector(w)
    boards=[];points=[];sources=[]
    for folder in map(Path,lotes):
        progress=json.loads((folder/'progreso.json').read_text())
        if progress['estado']!='completo':raise ValueError('Usar solo el piloto completo, no generación en curso')
        for file in sorted((folder/'partidas').glob('*.json')):
            game=json.loads(file.read_text());shard=folder/'shards'/(file.stem+'.npz')
            if sha(shard)!=game['shard_sha256']:raise ValueError('Shard alterado')
            with np.load(shard,allow_pickle=False) as z:
                n=len(z['tablero'])
                for i in sorted({0,n//2,n-1}):boards.append(z['tablero'][i]);points.append(z['puntos'][i])
            sources.append(dict(partida=str(file),sha256=sha(file),shard_sha256=sha(shard)))
    if not boards:raise ValueError('Piloto vacío')
    boards=np.asarray(boards,np.int8);points=np.asarray(points,np.int64)
    probe=w.copy();probe[:267]=0
    probe[:267][active]=np.random.default_rng(20261029).normal(0,.05,int(active.sum()))
    comparisons=0;logit_error=0.;utility_error=0.;switched=0
    for label,vector in [('referencia',w),('sonda_sintetica',probe),('referencia_restituida',w)]:
        search.cargar_pesos(vector)
        loaded,gain=search._pesos_y_ganancia()
        if gain!=1 or not np.array_equal(loaded,vector):raise ValueError('Carga de pesos o escala incorrecta')
        for color in (0,1):
            expected=features(boards,points,np.full(len(boards),color))@vector[:267]
            for board,pt,value in zip(boards,points,expected):
                logit=float(eval3.evaluar_nb(board,pt,color,loaded))
                utility=float(search._evaluar_numba(board,pt,color,loaded))
                logit_error=max(logit_error,abs(logit-value))
                utility_error=max(utility_error,abs(utility-float(np.tanh(value/2))))
                if label=='sonda_sintetica' and abs(value)>1e-6:switched+=1
                comparisons+=1
    if logit_error>1e-10 or utility_error>1e-10 or switched==0:raise ValueError('Predicción offline y runtime no coinciden')
    result=dict(correcto=True,jit=jit,comparaciones=comparisons,estados_muestreados=len(boards),
        max_error_logit=logit_error,max_error_utilidad=utility_error,cambio_de_pesos_y_restitucion_correctos=True,
        pesos_sha256=sha(modelo),buscador_sha256=sha(buscador),eval3_sha256=sha(equipo/'eval3.py'),fuentes=sources,
        limite='Equivalencia funcional, no calibración, reloj, fuerza ni promoción. Usar trazas ya vistas.')
    Path(salida).parent.mkdir(parents=True,exist_ok=True);Path(salida).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='fuentes'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--buscador',type=Path,required=True);p.add_argument('--pesos',type=Path,required=True)
    p.add_argument('--lotes',type=Path,nargs='+',required=True);p.add_argument('--salida',type=Path,required=True);p.add_argument('--jit',action='store_true')
    a=p.parse_args();print(json.dumps(verificar(a.buscador,a.pesos,a.lotes,a.salida,a.jit),indent=2))
