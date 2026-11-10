"""Lote nuevo con Monte Carlo fijo y UCT ampliado.

Episodios offline recuperables: profesor de busqueda frente a rivales diversos.

No es liga oficial ni evidencia de fuerza. Dados privados simulados; etiquetas por ganador.
Cada ply queda guardado. Reanuda por replay y RNG privado, sin escribir en la estrategia.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,sys,time
from pathlib import Path
import importlib.metadata

for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[n]='1'
os.environ['NUMBA_DISABLE_JIT']='0'
sys.dont_write_bytecode=True


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomico(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--buscador',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--partidas',type=int,default=60)
    p.add_argument('--minutos',type=float,default=60)
    p.add_argument('--semilla',type=int,default=52000000)
    p.add_argument('--profundidad',type=int,default=7)
    p.add_argument('--tope-nodos',type=int,default=3000000)
    p.add_argument('--uct-iteraciones',type=int,default=4096)
    p.add_argument('--max-juegos-nuevos',type=int,default=0,help='cuota por sesion; 0 completa el lote')
    p.add_argument('--mc-muestras',type=int,default=128)
    p.add_argument('--mc-rollouts',type=int,default=20)
    args=p.parse_args()
    if min(args.partidas,args.profundidad,args.tope_nodos,args.uct_iteraciones,args.mc_muestras,args.mc_rollouts)<1 or args.minutos<=0:p.error('Presupuestos positivos requeridos')
    equipo=Path(__file__).resolve().parents[1]
    args.salida.mkdir(parents=True,exist_ok=True)
    (args.salida/'partidas').mkdir(exist_ok=True);(args.salida/'shards').mkdir(exist_ok=True)
    os.environ['NUMBA_CACHE_DIR']=str((args.salida/'cache_numba').resolve())
    sys.path[:0]=[str(equipo),str(equipo.parents[1]/'sistema')]
    import numpy as np
    from motor import Estado,Jugada
    from motor.jugada import casilla_de_texto
    from fase5_rivales import uct,estilo
    from fase5_rivales_robustos import montecarlo
    if (args.buscador.parent/'eval3.py').read_bytes()!=(equipo/'eval3.py').read_bytes():p.error('eval3 distinto')
    spec=importlib.util.spec_from_file_location('fase5_generador_busqueda',args.buscador.resolve())
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    with np.load(equipo/'data/pesos_eval3_ligero.npz',allow_pickle=False) as z:w=z['pesos'].copy()
    mod.cargar_pesos(w)
    if mod.GANANCIA!=1:p.error('Generador requiere escala utilidad')
    contrato={k:getattr(args,k) for k in ['partidas','semilla','profundidad','tope_nodos','uct_iteraciones','mc_muestras','mc_rollouts']}
    contrato.update({'versiones':{'python':sys.version,'numpy':np.__version__,'numba':importlib.metadata.version('numba')},'buscador_sha256':sha(args.buscador),'pesos_sha256':sha(equipo/'data/pesos_eval3_ligero.npz'),
                     'eval3_sha256':sha(equipo/'eval3.py'),'motor_sha256':sha(equipo.parents[1]/'sistema/motor/nucleo.py'),
                     'script_sha256':sha(__file__),'rivales_sha256':sha(Path(__file__).with_name('fase5_rivales.py')),
                     'rivales_robustos_sha256':sha(Path(__file__).with_name('fase5_rivales_robustos.py')),
                     'rivales':['montecarlo','agresivo','carrera','uct'],'orden':'parejas por rival ciclico, profesor blancas/negras; dado privado por color compartido en el par',
                     'limite':'Datos de desarrollo offline. No son partidas oficiales ni prueba de fuerza del bot entregado.'})
    mf=args.salida/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=contrato:raise ValueError('Contrato diferente: usar otra salida')
    atomico(mf,contrato)
    comienzo=time.monotonic()
    completadas=0
    nuevas=0
    for i in range(args.partidas):
        par=i//2;rival=contrato['rivales'][par%len(contrato['rivales'])];color='blancas' if i%2==0 else 'negras'
        otro='negras' if color=='blancas' else 'blancas';semilla=args.semilla+par
        nombre=f'{i:04d}_{rival}_{color}'
        final=args.salida/'partidas'/f'{nombre}.json';shard=args.salida/'shards'/f'{nombre}.npz'
        parcial=args.salida/'partidas'/f'{nombre}.parcial'
        if final.exists():
            d=json.loads(final.read_text());assert sha(shard)==d['shard_sha256'];completadas+=1;continue
        if time.monotonic()-comienzo>=args.minutos*60:break
        if args.max_juegos_nuevos and nuevas>=args.max_juegos_nuevos:break
        meta={'blancas':'profesor_offline' if color=='blancas' else 'offline_'+rival,
              'negras':'profesor_offline' if color=='negras' else 'offline_'+rival,
              'semilla_maestra':semilla,'pareja':par,'juego':0}
        jugadas=json.loads(parcial.read_text())['jugadas'] if parcial.exists() else []
        estado=Estado.inicial();dados={c:random.Random(f'fase5_dado_{semilla}_{c}') for c in ('blancas','negras')}
        muestras=[]

        def muestrear(e,ply):
            t=np.zeros(81,np.int8)
            for c,n in enumerate(('blancas','negras')):
                for pos in e.piezas(n):t[casilla_de_texto(pos)]=c+1
            return (t,np.array([e.puntos[c] for c in ('blancas','negras')],np.int8),int(e.turno=='negras'),ply)

        for registro in jugadas:
            muestras.append(muestrear(estado,registro['n']-1));j=Jugada.desde_texto(registro['jugada'])
            combate=registro['combate']
            assert j in estado.jugadas() and estado.turno==registro['color']
            if combate is not None:assert dados[estado.turno].random()==combate['u']
            estado=estado.aplicar(j,combate['gano'] if combate else None)
        while not estado.terminado():
            if time.monotonic()-comienzo>=args.minutos*60:
                atomico(args.salida/'progreso.json',{'estado':'pausa_por_presupuesto','completadas':completadas,'parcial':str(parcial)})
                return
            ply=len(jugadas);muestras.append(muestrear(estado,ply));seed_politica=f'fase5_politica_{semilla}_{color}_{ply}'
            if estado.turno==color:
                t,pt,c,_=muestras[-1];pt=pt.astype(np.int64);elegida=estado.jugadas()[0];info={}
                for profundidad in range(1,args.profundidad+1):
                    contador=np.zeros(2,np.int64)
                    o,d,v,aborto,_=mod._mejor_jugada_numba(t,pt,c,profundidad,w,1.,-1,-1,contador,args.tope_nodos)
                    if aborto:break
                    elegida=Jugada(int(o),int(d));info={'prof':profundidad,'eval':float(v),'nodos':int(contador[0])}
                j=elegida
            elif rival=='montecarlo':j,info=montecarlo(estado,seed_politica,args.mc_muestras,args.mc_rollouts)
            elif rival=='uct':j,info=uct(estado,seed_politica,args.uct_iteraciones)
            else:j,info=estilo(estado,rival,seed_politica)
            if j not in estado.jugadas():raise ValueError('Politica produjo jugada ilegal')
            combate=None
            if estado.es_ataque(j):
                prob=estado.prob_ataque(j);u=dados[estado.turno].random();combate={'p':prob,'u':u,'gano':u<prob}
            jugadas.append({'n':ply+1,'color':estado.turno,'jugada':str(j),'combate':combate,'info':info})
            estado=estado.aplicar(j,combate['gano'] if combate else None)
            atomico(parcial,{'meta':meta,'jugadas':jugadas})
            if len(jugadas)>1000:raise RuntimeError('Secuencia imposible o motor no monotono')
        ganador=estado.ganador();n=len(muestras)
        tmp=shard.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,tablero=np.array([x[0] for x in muestras]),puntos=np.array([x[1] for x in muestras]),
                                turno=np.array([x[2] for x in muestras],np.int8),ply=np.array([x[3] for x in muestras],np.int32),
                                ganador_blancas=np.full(n,int(ganador=='blancas'),np.int8),
                                partida_origen=np.full(n,'partidas/'+final.name),rival=np.full(n,'offline_'+rival))
        tmp.replace(shard)
        atomico(final,{'version':1,'meta':meta,'jugadas':jugadas,'ganador':ganador,'puntos':estado.puntos,
                       'motivo':'puntos' if max(estado.puntos.values())>=3 else 'sin_jugadas','shard_sha256':sha(shard)})
        if sha(args.buscador)!=contrato['buscador_sha256']:raise RuntimeError('Profesor modificado durante lote')
        completadas+=1;nuevas+=1;atomico(args.salida/'progreso.json',{'estado':'generando','completadas':completadas,'total':args.partidas})
        print(json.dumps({'completadas':completadas,'rival':rival,'color_profesor':color,'ganador':ganador,'posiciones':n}),flush=True)
    atomico(args.salida/'progreso.json',{'estado':'completo' if completadas==args.partidas else 'pausa_por_presupuesto',
                                       'completadas':completadas,'total':args.partidas})


if __name__=='__main__':main()
