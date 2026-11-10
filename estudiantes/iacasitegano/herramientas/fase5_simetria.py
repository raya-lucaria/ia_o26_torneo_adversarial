"""Proyeccion sin ajuste de pesos de patrones a simetria izquierda/derecha."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from fase5_features import features,metricas,probabilidades


def espejo(k):
    d=[(k//3**i)%3 for i in range(5)]
    return d[2]+3*d[1]+9*d[0]+27*d[4]+81*d[3]


def proyectar(w):
    w=np.array(w,dtype=float,copy=True)
    indices=np.array([espejo(k) for k in range(243)])
    w[1:244]=.5*(w[1:244]+w[1:244][indices])
    return w


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--auditoria',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    args=p.parse_args()
    if args.salida.exists():raise FileExistsError(args.salida)
    with np.load(args.base,allow_pickle=False) as z:w=z['pesos'].copy()
    ws=proyectar(w)
    ix=np.array([espejo(k) for k in range(243)])
    assert np.array_equal(ix[ix],np.arange(243))
    np.testing.assert_allclose(proyectar(ws),ws,atol=0)
    with np.load(args.dataset,allow_pickle=False) as z:t,pt,c=z['tablero'],z['puntos'],z['turno']
    with np.load(args.auditoria/'predicciones_y_grupos.npz',allow_pickle=False) as z:y,g,val=z['y'],z['grupo'],z['validacion_historica']
    metadata=json.loads((args.auditoria/'auditoria.json').read_text())
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest()!=metadata['dataset_sha256']:
        raise ValueError('Dataset distinto')
    pred=np.empty(len(y));ref=np.empty(len(y));max_delta=0.
    for inicio in range(0,len(y),1024):
        s=slice(inicio,inicio+1024);X=features(t[s],pt[s],c[s])
        pred[s]=probabilidades(X@ws[:267]);ref[s]=probabilidades(X@w[:267])
        Xm=features(t[s].reshape(-1,9,9)[:,:,::-1].reshape(-1,81),pt[s],c[s])
        diferencia=np.max(abs(X@ws[:267]-Xm@ws[:267]));max_delta=max(max_delta,float(diferencia))
    a=metricas(y[val],ref[val],g[val]);b=metricas(y[val],pred[val],g[val])
    args.salida.mkdir(parents=True)
    np.savez_compressed(args.salida/'pesos_simetria.npz',pesos=ws,crudo=ws[:267],modelo='simetria_lateral_sin_ajuste')
    out={'base':a,'simetria':b,'delta_logloss':b['logloss']-a['logloss'],
         'max_delta_logit_espejo_proyectado':max_delta,'filtro_historico':b['logloss']-a['logloss']<=.01,
         'filtro_episodios_nuevos':'pendiente','fuerza_medida':False,
         'candidato_sha256':hashlib.sha256((args.salida/'pesos_simetria.npz').read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.salida/'RESULTADO.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(out,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
