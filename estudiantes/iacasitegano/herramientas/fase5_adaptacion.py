"""Actualizacion con datos nuevos: compara prior cero y prior modelo activo.

Se ejecuta solo con los 60 episodios completos. UCT queda fuera del ajuste.
El piloto inicial se usa para ajuste/diagnostico, nunca como examen nuevo.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[n]='1'
import numpy as np
from fase5_features import features,metricas,probabilidades
from fase5_modelos import escala,pesos_grupos


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ajustar(X,y,grupos,referencia,lam=100):
    s=escala(X);A=X/s;centro=referencia*s;w=centro.copy();peso=pesos_grupos(grupos)
    def funcion(v):
        z=A@v
        return float(peso@(np.logaddexp(0,z)-y*z)+.5*lam*np.sum((v-centro)**2))
    for _ in range(50):
        prob=probabilidades(A@w)
        H=A.T@(A*(peso*np.maximum(prob*(1-prob),1e-8))[:,None])+lam*np.eye(A.shape[1])
        grad=A.T@(peso*(y-prob))-lam*(w-centro)
        paso=np.linalg.solve(H,grad);valor=funcion(w);factor=1.
        while factor>1e-7 and funcion(w+factor*paso)>valor:factor*=.5
        if funcion(w+factor*paso)>valor+1e-9:raise ValueError('Paso no decreciente')
        w+=factor*paso
        if np.max(abs(factor*paso))<1e-8:break
    grad=A.T@(peso*(y-probabilidades(A@w)))-lam*(w-centro)
    if np.max(abs(grad))>1e-5:raise ValueError('Ajuste no convergente')
    return w/s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lote',type=Path,required=True)
    p.add_argument('--datos',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    args=p.parse_args()
    progreso=json.loads((args.lote/'progreso.json').read_text())
    if progreso['estado']!='completo' or progreso['completadas']!=60:
        p.error('Se necesitan 60 episodios completos del protocolo, no un subconjunto')
    if args.salida.exists():raise FileExistsError(args.salida)
    combinado=json.loads(args.datos.with_suffix('.json').read_text())
    if combinado['partidas']!=60 or combinado['dataset_sha256']!=sha(args.datos):
        raise ValueError('Dataset combinado no corresponde a su manifiesto')
    if combinado['contrato_lote_sha256']!=sha(args.lote/'manifest.json'):
        raise ValueError('Contrato de episodios distinto')
    for fuente in combinado['fuentes']:
        if sha(args.lote/fuente['archivo'])!=fuente['sha256']:
            raise ValueError('Fuente modificada despues de combinar')
    with np.load(args.datos,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(args.base,allow_pickle=False) as z:base=z['pesos'].copy()
    if len(np.unique(d['partida_origen']))!=60:raise ValueError('Dataset distinto al lote completo')
    if set(d['partida_origen'])!={x['archivo'] for x in combinado['fuentes']}:
        raise ValueError('IDs del dataset no corresponden a las fuentes')
    grupos=[];fold=[];metas={};fuentes=[]
    for ident in np.unique(d['partida_origen']):
        ruta=args.lote/str(ident);j=json.loads(ruta.read_text());m=j['meta']
        shard=args.lote/'shards'/(ruta.stem+'.npz')
        if sha(shard)!=j['shard_sha256']:raise ValueError('Shard invalido')
        g=f"{m['semilla_maestra']}:{m['pareja']}:{m['juego']}"
        par=int(m['pareja']);uct=any(str(m[c])=='offline_uct' for c in ['blancas','negras'])
        h=int(hashlib.sha256((g+'_adaptacion_centrada_20261028').encode()).hexdigest()[:8],16)/2**32
        f=-1 if par==2 else 2 if uct else 0 if par in [0,1] else 1 if h<.25 else 0
        metas[str(ident)]=(g,f);fuentes.append({'archivo':str(ident),'sha256':sha(ruta)})
    for ident in d['partida_origen']:
        g,f=metas[str(ident)];grupos.append(g);fold.append(f)
    grupos=np.array(grupos);fold=np.array(fold,np.int8)
    estados=[]
    for t,pt,c in zip(d['tablero'],d['puntos'],d['turno']):
        raw=min(t.tobytes(),t.reshape(9,9)[:,::-1].tobytes())+pt.tobytes()+bytes([int(c)])
        estados.append(hashlib.sha256(raw).hexdigest())
    estados=np.array(estados);examen=set(estados[fold==2]);seleccion=set(estados[fold==1])
    purga=((fold==0)&np.isin(estados,list(examen|seleccion)))|((fold==1)&np.isin(estados,list(examen)))
    fold[purga]=-1
    for i in range(3):
        if len(set(grupos[fold==i]))<2:raise ValueError('Particion sin grupos suficientes')
        for j in range(i):
            assert not set(grupos[fold==i])&set(grupos[fold==j])
            assert not set(estados[fold==i])&set(estados[fold==j])
    args.salida.mkdir(parents=True)
    contrato={'datos_sha256':sha(args.datos),'base_sha256':sha(args.base),'script_sha256':sha(__file__),
              'lambda':100,'controles':['sin_actualizar','prior_cero','prior_activo'],
              'particion':'UCT a prueba excepto piloto par2 excluido; piloto pares0/1 ajuste; resto hash25% seleccion; purga espejo',
              'fuentes':fuentes,'filas':{str(i):int((fold==i).sum()) for i in [-1,0,1,2]},
              'grupos':{str(i):len(set(grupos[fold==i])) for i in [0,1,2]},
              'filtros':{'seleccion_mejora_brier':.002,'seleccion_logloss_no_empeora':True,
                         'prueba_brier_no_empeora':True,'prueba_regresion_color_max':.005}}
    (args.salida/'contrato.json').write_text(json.dumps(contrato,ensure_ascii=False,indent=2)+'\n')
    X=features(d['tablero'],d['puntos'],d['turno']);y=(d['ganador_blancas']==(d['turno']==0)).astype(float)
    modelos={'sin_actualizar':base[:267]}
    for nombre,ref in [('prior_cero',np.zeros(267)),('prior_activo',base[:267])]:
        modelos[nombre]=ajustar(X[fold==0],y[fold==0],grupos[fold==0],ref)
        np.savez_compressed(args.salida/(nombre+'.npz'),crudo=modelos[nombre],modelo='logistica')
    sel={};m=fold==1
    for nombre,w in modelos.items():sel[nombre]=metricas(y[m],probabilidades(X[m]@w),grupos[m])
    nombre=min(['prior_cero','prior_activo'],key=lambda x:sel[x]['brier_igual_peso_grupo'])
    a=sel['sin_actualizar'];b=sel[nombre]
    pasa=b['brier_igual_peso_grupo']<=a['brier_igual_peso_grupo']-.002 and b['logloss_igual_peso_grupo']<=a['logloss_igual_peso_grupo']
    resultado={'seleccion':sel,'candidato':nombre,'pasa_seleccion':pasa,'prueba_interna':None,'promovido':False}
    if pasa:
        perfiles={}
        for color in ['ambos',0,1]:
            m=(fold==2) if color=='ambos' else (fold==2)&(d['turno']==color)
            perfiles[str(color)]={n:metricas(y[m],probabilidades(X[m]@modelos[n]),grupos[m]) for n in ['sin_actualizar',nombre]}
        pasa_prueba=all(v[nombre]['brier_igual_peso_grupo']<=v['sin_actualizar']['brier_igual_peso_grupo']+(0 if c=='ambos' else .005) for c,v in perfiles.items())
        resultado.update(prueba_interna=perfiles,pasa_prueba=pasa_prueba)
        np.savez_compressed(args.salida/'candidato_laboratorio.npz',pesos=np.r_[modelos[nombre],base[267:]],crudo=modelos[nombre],modelo='logistica')
    resultado['advertencia']='Desarrollo offline con nuevas politicas; requiere costo, legalidad y partidas oficiales nuevas. No ajustar tras abrir prueba.'
    (args.salida/'RESULTADO.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(resultado,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
