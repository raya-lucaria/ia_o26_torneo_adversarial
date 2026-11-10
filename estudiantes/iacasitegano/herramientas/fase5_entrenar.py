"""Compara ridge/logistica y ajuste GA en datos agrupados, con checkpoints por ajuste.

Produce candidatos de laboratorio; requiere pruebas de busqueda y partidas nuevas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for nombre in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[nombre] = "1"
import numpy as np
from fase5_features import metricas,probabilidades
from fase5_modelos import (ridge,logistica,pesos_grupos,prob_modelo,contribuciones,
                           genetico,aplicar_genoma)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomico(path,datos):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(path)


def ejecutar(args):
    with np.load(args.datos,allow_pickle=False) as z:
        X,y,g,f = z['X'].astype(float),z['y'],z['grupo'],z['fold']
    if any(not np.any(f==i) for i in range(3)):
        raise ValueError("Faltan particiones")
    # Confirmar contratos aunque los datos procedan de otra herramienta.
    with np.load(args.datos,allow_pickle=False) as z:
        for i in range(3):
            for j in range(i):
                assert not set(g[f==i])&set(g[f==j])
                assert not set(z['estado'][f==i])&set(z['estado'][f==j])
    args.salida.mkdir(parents=True,exist_ok=True)
    lambdas = [float(x) for x in args.lambdas.split(',')]
    contrato = {"datos_sha256":sha(args.datos),"modelo_base_sha256":sha(args.base),
                "fuentes_sha256":{x:sha(Path(__file__).with_name(x)) for x in
                                   ['fase5_entrenar.py','fase5_modelos.py','fase5_features.py']},
                "lambdas":lambdas,"objetivos":{"ridge":"2*y-1","logistica":"y"},
                "seleccion":"Brier con igual peso por grupo en fold1",
                "GA":{"poblacion":12,"generaciones":5,"semilla":20261028,
                      "parametros":["puntos","filas","marcador"],"rango":[.75,1.25],
                      "fitness":"Brier igual peso grupo en fold0; cruce aritmetico, mutacion sigma .05 p=1/3; elitismo2"},
                "advertencia":"Todo es desarrollo historico; no demuestra fuerza ni generalizacion nueva"}
    mf=args.salida/'contrato.json'
    if mf.exists() and json.loads(mf.read_text())!=contrato:
        raise ValueError("Contrato diferente; elegir otra salida")
    atomico(mf,contrato)
    peso = pesos_grupos(g[f==0])
    resultados=[]
    ajustes=0
    for tipo in ('logistica','ridge'):
        for lam in lambdas:
            nombre=f'{tipo}_lambda{lam:g}'
            pesos_path=args.salida/(nombre+'.npz')
            info_path=args.salida/(nombre+'.json')
            if not info_path.exists():
                objetivo=y[f==0] if tipo=='logistica' else 2*y[f==0]-1
                w=(logistica if tipo=='logistica' else ridge)(X[f==0],objetivo,lam,peso)
                np.savez_compressed(pesos_path,crudo=w,modelo=tipo)
                info={"nombre":nombre,"modelo":tipo,"lambda":lam,"archivo":pesos_path.name,
                      "pesos_sha256":sha(pesos_path)}
                for fold,et in [(0,'ajuste'),(1,'seleccion')]:
                    m=f==fold
                    info[et]=metricas(y[m],prob_modelo(X[m],w,tipo),g[m])
                atomico(info_path,info)
                ajustes+=1
            info=json.loads(info_path.read_text())
            if sha(pesos_path)!=info['pesos_sha256']:
                raise ValueError('Checkpoint de pesos modificado')
            resultados.append(info)
            print(json.dumps({"ajuste":nombre,"brier_seleccion":info['seleccion']['brier_igual_peso_grupo']}),flush=True)
            if args.max_ajustes and ajustes>=args.max_ajustes:
                atomico(args.salida/'progreso.json',{"estado":"ajustes_pendientes","completados":len(resultados)})
                return
    ganador=min(resultados,key=lambda r:(r['seleccion']['brier_igual_peso_grupo'],r['nombre']))
    with np.load(args.salida/ganador['archivo'],allow_pickle=False) as z:w=z['crudo']
    tipo=ganador['modelo']
    ga_path=args.salida/'genetico.json'
    if ga_path.exists():
        ga=json.loads(ga_path.read_text())
    else:
        ga=genetico(contribuciones(X[f==0],w),y[f==0],peso,tipo)
        atomico(ga_path,ga)
    variantes={'sin_ajuste':w,'GA':aplicar_genoma(w,ga['mejor_genoma']),
               'aleatorio':aplicar_genoma(w,ga['aleatorio_genoma'])}
    seleccion={}
    for nombre,v in variantes.items():
        m=f==1
        seleccion[nombre]=metricas(y[m],prob_modelo(X[m],v,tipo),g[m])
    nombre_final=min(variantes,key=lambda k:(seleccion[k]['brier_igual_peso_grupo'],k))
    elegido=variantes[nombre_final]
    informe={"estado":"desarrollo_offline_completo","modelos":resultados,"ganador_modelo":ganador['nombre'],
             "comparacion_genetico":seleccion,"variante_elegida":nombre_final,"modelo":tipo,
             "prueba_interna_historica":{},"fuerza_medida":False,
             "advertencia":"La comparacion con base reutiliza datos historicos que la base vio; se requieren partidas nuevas"}
    with np.load(args.base,allow_pickle=False) as z:wbase=z['pesos'][:267]
    m=f==2
    informe['prueba_interna_historica']['candidato']=metricas(y[m],prob_modelo(X[m],elegido,tipo),g[m])
    informe['prueba_interna_historica']['base']=metricas(y[m],probabilidades(X[m]@wbase),g[m])
    empacado=np.r_[elegido,[1,0,0,0,0,0,1,0]]
    np.savez_compressed(args.salida/'candidato_laboratorio.npz',pesos=empacado,crudo=elegido,modelo=tipo)
    informe['candidato_sha256']=sha(args.salida/'candidato_laboratorio.npz')
    informe['enlace_requerido']=('sigmoide -> utilidad 2p-1, o logits en control especifico' if tipo=='logistica'
                                else 'clip(retorno,-1,1), terminales +-1; no cargar como logit')
    atomico(args.salida/'RESULTADO.json',informe)
    atomico(args.salida/'progreso.json',{"estado":"offline_completo_partidas_pendientes"})
    print(json.dumps({k:informe[k] for k in ['estado','ganador_modelo','variante_elegida','prueba_interna_historica']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datos',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--lambdas',default='1,10,100,1000')
    p.add_argument('--max-ajustes',type=int,default=0,help='detener tras N ajustes nuevos; reanudar con misma orden')
    ejecutar(p.parse_args())
