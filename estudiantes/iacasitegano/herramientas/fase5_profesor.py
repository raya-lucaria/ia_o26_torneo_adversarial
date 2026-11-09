"""Profesor de busqueda offline: compara acciones seleccionadas a mayor horizonte.

Sus valores NO son verdad optima del juego: usa el evaluador del profesor.
No se entrena ni se mide fuerza por acuerdo. Un aborto no produce etiqueta.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path

for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[n]='1'
os.environ['NUMBA_DISABLE_JIT']='0'
sys.dont_write_bytecode=True


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--seleccion-acciones',type=Path,required=True)
    ap.add_argument('--buscador',type=Path,required=True)
    ap.add_argument('--salida',type=Path,required=True)
    ap.add_argument('--profundidades',default='5,7')
    ap.add_argument('--tope-nodos',type=int,default=1000000)
    ap.add_argument('--max-casos',type=int,default=6)
    args=ap.parse_args()
    equipo=Path(__file__).resolve().parents[1]
    args.salida.mkdir(parents=True,exist_ok=True)
    os.environ['NUMBA_CACHE_DIR']=str((args.salida/'cache_numba').resolve())
    sys.path[:0]=[str(equipo),str(equipo.parents[1]/'sistema')]
    import numpy as np
    import eval3
    from motor import Estado,Jugada
    from motor.jugada import casilla_de_texto
    from motor.nucleo import MAX_JUGADAS
    if (args.buscador.parent/'eval3.py').read_bytes()!=(equipo/'eval3.py').read_bytes():
        ap.error('eval3 distinto')
    spec=importlib.util.spec_from_file_location('fase5_teacher',args.buscador.resolve())
    mod=importlib.util.module_from_spec(spec);sys.modules['fase5_teacher']=mod;spec.loader.exec_module(mod)
    with np.load(equipo/'data/pesos_eval3_ligero.npz',allow_pickle=False) as z:pesos=z['pesos'].copy()
    mod.cargar_pesos(pesos)
    if mod.GANANCIA!=1:
        ap.error('Este protocolo requiere profesor en escala utilidad y terminales +-1')
    profundidades=[int(x) for x in args.profundidades.split(',')]
    if min(profundidades)<1 or args.tope_nodos<1:ap.error('Presupuesto invalido')
    contrato={'panel_sha256':sha(args.panel),'seleccion_sha256':sha(args.seleccion_acciones),
              'buscador_sha256':sha(args.buscador),'eval3_sha256':sha(equipo/'eval3.py'),
              'pesos_sha256':sha(equipo/'data/pesos_eval3_ligero.npz'),
              'herramienta_sha256':sha(__file__),'profundidades':profundidades,
              'tope_nodos_por_rama':args.tope_nodos,'max_casos':args.max_casos,
              'escala':'utilidad [-1,1], terminales +-1',
              'limite':'Profesor comparte modelo; etiquetas aproximadas y seleccion parcial de acciones. No es optimo global ni prueba de fuerza.'}
    mf=args.salida/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=contrato:raise ValueError('Contrato distinto')
    mf.write_text(json.dumps(contrato,ensure_ascii=False,indent=2)+'\n')
    casos={x['id']:x for x in map(json.loads,args.panel.read_text().splitlines())}
    seleccion=json.loads(args.seleccion_acciones.read_text())
    # Orden e IDs ya publicados por el selector, sin elegir por resultado del profesor.
    for entrada in seleccion['casos'][:args.max_casos]:
        caso=casos[entrada['id']];estado=Estado.desde_dict(caso['estado'])
        objetivo=int(estado.turno=='negras');ident=hashlib.sha256(caso['id'].encode()).hexdigest()[:16]
        for d in profundidades:
            archivo=args.salida/f'{ident}_d{d}.json'
            if archivo.exists():continue
            etiquetas=[];inicio=time.monotonic()
            for texto in entrada['acciones']:
                j=Jugada.desde_texto(texto)
                if j not in estado.jugadas():raise ValueError('Accion ilegal del selector')
                ramas=[]
                for prob,hijo in estado.siguiente(j):
                    t=np.zeros(81,np.int8)
                    for c,nombre in enumerate(('blancas','negras')):
                        for pos in hijo.piezas(nombre):t[casilla_de_texto(pos)]=c+1
                    p=np.array([hijo.puntos[c] for c in ('blancas','negras')],np.int64)
                    contador=np.zeros(2,np.int64)
                    valor=mod._valor_numba(t,p,int(hijo.turno=='negras'),objetivo,d-1,-1.,1.,pesos,1.,
                        np.empty(MAX_JUGADAS,np.int64),np.empty(MAX_JUGADAS,np.int64),
                        np.zeros(mod.TT_TAM,np.uint64),np.zeros(mod.TT_TAM),contador,args.tope_nodos)
                    ramas.append({'p':float(prob),'valor':None if contador[1] else float(valor),
                                  'abortado':bool(contador[1]),'nodos':int(contador[0])})
                completo=not any(x['abortado'] for x in ramas)
                q=sum(x['p']*x['valor'] for x in ramas) if completo else None
                etiquetas.append({'jugada':texto,'ramas':ramas,'completo':completo,'utilidad':q,
                                   'probabilidad_estimada':.5*(1+q) if completo else None})
            r={'id':caso['id'],'profundidad':d,'etiquetas':etiquetas,'segundos':time.monotonic()-inicio,
               'advertencia':contrato['limite']}
            if sha(args.buscador)!=contrato['buscador_sha256']:raise RuntimeError('Profesor modificado')
            tmp=archivo.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');tmp.replace(archivo)
            print(json.dumps({'id':caso['id'],'profundidad':d,'acciones_completas':sum(x['completo'] for x in etiquetas),
                              'acciones':len(etiquetas),'segundos':r['segundos']}),flush=True)


if __name__=='__main__':main()
