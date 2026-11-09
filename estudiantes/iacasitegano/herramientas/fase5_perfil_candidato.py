"""Perfiles posteriores del candidato ya elegido, sin reajustar ni reseleccionar."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from fase5_features import metricas,probabilidades
from fase5_modelos import prob_modelo


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ajustes',type=Path,required=True)
    p.add_argument('--datos',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    args=p.parse_args()
    resultado=json.loads((args.ajustes/'RESULTADO.json').read_text())
    contrato=json.loads((args.ajustes/'contrato.json').read_text())
    manifest=json.loads(args.datos.with_name('manifest.json').read_text())
    if sha(args.original)!=manifest['dataset_original_sha256'] or sha(args.datos)!=contrato['datos_sha256']:
        raise ValueError('Datos distintos a los congelados')
    if sha(args.base)!=contrato['modelo_base_sha256'] or sha(args.ajustes/'candidato_laboratorio.npz')!=resultado['candidato_sha256']:
        raise ValueError('Modelo distinto al seleccionado')
    with np.load(args.datos,allow_pickle=False) as z:
        X,y,g,f,ix,rival=z['X'].astype(float),z['y'],z['grupo'],z['fold'],z['indice_original'],z['rival']
    with np.load(args.original,allow_pickle=False) as z:
        turno,ply,puntos=z['turno'][ix],z['ply'][ix],z['puntos'][ix]
    with np.load(args.base,allow_pickle=False) as z:wbase=z['pesos'][:267]
    with np.load(args.ajustes/'candidato_laboratorio.npz',allow_pickle=False) as z:w=z['crudo']
    base=probabilidades(X@wbase); candidato=prob_modelo(X,w,resultado['modelo'])
    m=f==2
    perfiles={}
    mascaras={'global':np.ones(len(y),bool),'blancas':turno==0,'negras':turno==1,
              'temprana':ply<20,'media':(ply>=20)&(ply<60),'tardia':ply>=60,
              'alguien_dos_puntos':np.max(puntos,axis=1)==2}
    mascaras.update({'rival:'+str(r):rival==r for r in np.unique(rival)})
    for nombre,ms in mascaras.items():
        sel=m&ms
        a=metricas(y[sel],base[sel],g[sel]);b=metricas(y[sel],candidato[sel],g[sel])
        delta=None if not sel.any() else b['brier_igual_peso_grupo']-a['brier_igual_peso_grupo']
        perfiles[nombre]={'base':a,'candidato':b,'delta_brier_agrupado':delta}
    # Intervalo descriptivo agrupado: cada grupo aporta su media, no cada posicion una replica.
    grupos,inversa=np.unique(g[m],return_inverse=True)
    delta=(candidato[m]-y[m])**2-(base[m]-y[m])**2
    medias=np.bincount(inversa,weights=delta)/np.bincount(inversa)
    rng=np.random.default_rng(20261028)
    boot=np.mean(medias[rng.integers(0,len(grupos),size=(400,len(grupos)))],axis=1)
    salida={'modelo':resultado['modelo'],'variante':resultado['variante_elegida'],
            'perfiles_prueba_interna':perfiles,
            'delta_brier_grupo_bootstrap':{'media':float(medias.mean()),'IC95_descriptivo':np.quantile(boot,[.025,.975]).tolist(),'grupos':len(grupos),'replicas':400},
            'advertencia':'Desarrollo historico visto; perfil e intervalo no sustituyen partidas nuevas. No se reselecciona candidato con estos cortes.',
            'candidato_sha256':resultado['candidato_sha256'],'script_sha256':sha(__file__)}
    (args.ajustes/'PERFIL.json').write_text(json.dumps(salida,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(salida,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
