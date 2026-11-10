"""Regenera el eval3 ligero historico con IRLS centrado en la base antigua.

Receta recuperada: hash _principal, columnas puntos/patrones/filas/marcador,
escalado sin centrar, lambda100, regularizador hacia puntos/filas de pesos.npy.
Las tolerancias comparan arrays, no metadatos ZIP del NPZ.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[n]='1'
import numpy as np
from fase5_features import features,particion_hash,probabilidades,metricas


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--pesos-antiguos',type=Path,required=True)
    p.add_argument('--referencia',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--tolerancia',type=float,default=1e-8)
    args=p.parse_args()
    if args.salida.exists():raise FileExistsError(args.salida)
    with np.load(args.dataset,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    antiguo=np.load(args.pesos_antiguos,allow_pickle=False)
    if antiguo.shape!=(6,):raise ValueError('Se requieren los seis pesos historicos')
    with np.load(args.referencia,allow_pickle=False) as z:original=z['pesos'].copy()
    X=features(d['tablero'],d['puntos'],d['turno'])
    y=(d['ganador_blancas']==(d['turno']==0)).astype(float)
    validacion=particion_hash(d['partida_origen'],sal='_principal')
    columnas=np.r_[np.arange(253),np.arange(263,266)]
    Xa=X[~validacion][:,columnas];ya=y[~validacion]
    esc=Xa.std(axis=0);esc[esc<1e-12]=1.
    A=Xa/esc
    ref=np.zeros(267);ref[0]=antiguo[0];ref[244:253]=antiguo[1]*np.arange(9)
    centro=ref[columnas]*esc;w=centro.copy();I=100*np.eye(len(columnas));historia=[]
    for i in range(25):
        z=A@w;prob=1/(1+np.exp(-np.clip(z,-30,30)));s=np.clip(prob*(1-prob),1e-6,None)
        H=A.T@(A*s[:,None])+I;g=A.T@(ya-prob)-100*(w-centro)
        paso=np.linalg.solve(H,g);w+=paso
        historia.append({'iteracion':i+1,'paso_max':float(np.max(abs(paso)))})
        if np.max(abs(paso))<1e-8:break
    crudo=np.zeros(267);crudo[columnas]=w/esc
    empaquetado=np.r_[crudo,[1,0,0,0,0,0,1,0]]
    error=float(np.max(abs(empaquetado-original)))
    pred=X@crudo;pred_original=X@original[:267]
    max_pred=float(np.max(abs(pred-pred_original)))
    args.salida.mkdir(parents=True)
    np.savez(args.salida/'pesos_reproducidos.npz',pesos=empaquetado,crudo=crudo,
             modelo='ligero5_solo_puntos_patrones_filas_marcador')
    resultado={'receta':{'objetivo':'ganador real visto por quien mueve','particion':'SHA256(id+_principal) primeros32bits <.2*2**32 es validacion',
                         'columnas':columnas.tolist(),'lambda':100,'escalado':'std del ajuste, sin centrar',
                         'centro_puntos':float(antiguo[0]),'centro_filas':(antiguo[1]*np.arange(9)).tolist(),
                         'centro_patrones_marcador':0,'IRLS_max_iteraciones':25,'tolerancia_paso':1e-8,
                         'clip_logit':[-30,30],'piso_curvatura':1e-6},
               'ajuste':int((~validacion).sum()),'validacion':int(validacion.sum()),
               'max_diferencia_pesos':error,'max_diferencia_logits_dataset':max_pred,
               'tolerancia':args.tolerancia,'coincide':error<=args.tolerancia and max_pred<=args.tolerancia,
               'metricas_validacion':metricas(y[validacion],probabilidades(pred[validacion]),d['partida_origen'][validacion]),
               'historia':historia,'versiones':{'python':sys.version,'numpy':np.__version__},
               'hashes':{'dataset':sha(args.dataset),'pesos_antiguos':sha(args.pesos_antiguos),'referencia':sha(args.referencia),
                         'regenerado':sha(args.salida/'pesos_reproducidos.npz'),'script':sha(__file__),
                         'features':sha(Path(__file__).with_name('fase5_features.py'))},
               'advertencia':'Reproduccion historica, no prueba de independencia de validacion ni de fuerza'}
    (args.salida/'RESULTADO.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in resultado.items() if k not in ['receta','historia']},ensure_ascii=False,indent=2))
    if not resultado['coincide']:raise SystemExit(1)


if __name__=='__main__':main()
