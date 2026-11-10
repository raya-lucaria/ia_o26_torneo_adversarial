"""Una hipótesis nueva: 13 coeficientes ajustables y patrones activos congelados.

No usa test ni selecciona lambda/mezclas. Las métricas antiguas son desarrollo;
no autorizan aceptación sin validación prospectiva independiente.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fase5_diagnosticar_replay256 import sigmoid, pesos_grupo, resumen

FREE=np.array([0,*range(244,253),*range(263,266)])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(lote, salida):
    lote=Path(lote).resolve();salida=Path(salida).resolve()
    if salida.exists():raise FileExistsError(salida)
    gate=lote/'validacion';contract=json.loads((gate/'contrato.json').read_text())
    data={};inputs={}
    for name in ('train','replay'):
        p=gate/(name+'.npz');inputs[str(p)]=sha(p)
        if inputs[str(p)]!=contract['archivos'][p.name]:raise ValueError('Datos cambiados')
        with np.load(p,allow_pickle=False) as z:data[name]={k:z[k] for k in ('X','y','grupo','turno','ply')}
    p=gate/'base.npz';inputs[str(p)]=sha(p)
    if inputs[str(p)]!=contract['archivos']['base.npz']:raise ValueError('Base cambiada')
    with np.load(p,allow_pickle=False) as z:base=z['pesos'].copy()
    h,n=data['replay'],data['train']
    X=np.concatenate([h['X'],n['X']]);y=np.r_[h['y'],n['y']]
    weight=np.r_[.8*pesos_grupo(h['grupo']),.2*pesos_grupo(n['grupo'])]*len(y)
    offset=X@base[:267];scale=X[:,FREE].std(axis=0);scale[scale<1e-12]=1
    A=X[:,FREE]/scale;delta=np.zeros(len(FREE));lam=100
    def loss(d):
        z=offset+A@d
        return float(weight@(np.logaddexp(0.,z)-y*z)+.5*lam*(d@d))
    history=[]
    for iteration in range(50):
        prob=sigmoid(offset+A@delta)
        gradient=A.T@(weight*(prob-y))+lam*delta
        H=A.T@(A*(weight*np.maximum(prob*(1-prob),1e-8))[:,None])+lam*np.eye(len(FREE))
        step=np.linalg.solve(H,-gradient);factor=1.;before=loss(delta)
        while factor>1e-7 and loss(delta+factor*step)>before:factor*=.5
        after=loss(delta+factor*step)
        if after>before+1e-9:raise ValueError('No desciende')
        delta+=factor*step;history.append({'iteracion':iteration,'objetivo':after,'factor':factor})
        if max(abs(factor*step))<1e-8:break
    grad=A.T@(weight*(sigmoid(offset+A@delta)-y))+lam*delta
    if max(abs(grad))>1e-5:raise ValueError('No convergente')
    w=base.copy();w[FREE]+=delta/scale
    fixed=np.ones(275,bool);fixed[FREE]=False
    if not np.array_equal(w[fixed],base[fixed]):raise ValueError('Cambió coeficiente congelado')
    salida.mkdir(parents=True)
    model=salida/'candidato.npz'
    np.savez_compressed(model,pesos=w,crudo=w[:267],modelo='subespacio13_patrones_fijos_lambda100')
    info={'candidato_sha256':sha(model),'fuentes_train':inputs,'coeficientes_libres':FREE.tolist(),
          'lambda':100,'mezcla_historico_nuevo':[.8,.2],'ajustes':1,'script_sha256':sha(__file__),
          'gradiente_maximo':float(max(abs(grad))),'coeficientes_congelados_identicos':True,
          'historia':history,'modelo_aceptado':False,'test_evaluado':False}
    (salida/'ENTRENAMIENTO.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    # El NPZ está fijado antes de cargar cualquier evaluación. No se reajusta.
    for name in ('validation','retencion'):
        p=gate/(name+'.npz');inputs[str(p)]=sha(p)
        if inputs[str(p)]!=contract['archivos'][p.name]:raise ValueError('Evaluación cambió')
        with np.load(p,allow_pickle=False) as z:data[name]={k:z[k] for k in ('X','y','grupo','turno','ply')}
    profiles={name:resumen(d,d['X']@base[:267],d['X']@w[:267]) for name,d in data.items()}
    v=profiles['validation']['global'];r=profiles['retencion']['global']
    checks={'mejora_desarrollo_antiguo':v['delta_brier']<=-.002,
            'logloss_desarrollo_antiguo':v['delta_logloss']<=0,
            'retencion_global':r['delta_brier']<=.005,
            'retencion_estratos':all(x['delta_brier']<=.01 for k,x in profiles['retencion'].items()
                                     if k in ('blancas','negras','inicio','medio','tardio')),
            'colores_desarrollo_antiguo':all(profiles['validation'][k]['delta_brier']<=.005 for k in ('blancas','negras'))}
    result={'candidato_sha256':sha(model),'perfiles':profiles,'checks_desarrollo_visto':checks,
            'continuar_a_validacion_nueva':all(checks.values()),'modelo_aceptado':False,
            'advertencia':'Validation antigua y retención ya influyeron en formular la hipótesis; este pase NO es validación prospectiva ni fuerza.',
            'fuentes_evaluacion':{str(gate/(x+'.npz')):inputs[str(gate/(x+'.npz'))] for x in ('validation','retencion')},
            'test_cargado':False}
    (salida/'DESARROLLO_VISTO.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'candidato':info['candidato_sha256'],'gradiente_maximo':info['gradiente_maximo'],
                     'perfiles_globales':{k:v['global'] for k,v in profiles.items()},'checks':checks,
                     'continuar_a_validacion_nueva':all(checks.values()),'modelo_aceptado':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lote',type=Path,required=True);p.add_argument('--salida',type=Path,required=True)
    a=p.parse_args();run(a.lote,a.salida)
