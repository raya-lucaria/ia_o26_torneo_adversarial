"""Cola offline recuperable. Sin Kubernetes, sin escritura de pesos activos."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
TOOLS=HERE.parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[n]='1'

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def commands(root,salida,workers,games,minutes,seed):
    result=[]
    for i in range(workers):
        result.append([sys.executable,'-B',str(root/'estudiantes/iacasitegano/herramientas/fase5_generar_robusto.py'),
                       '--buscador',str(root/'material/profesor_utilidad/busqueda_eval3.py'),
                       '--salida',str(salida/f'worker_{i}'),'--partidas',str(games),
                       '--minutos',str(minutes),'--semilla',str(seed+i*1000000)])
    return result

def run(root,salida,workers=4,games=64,minutes=60,seed=53000000,dry_run=False):
    root=Path(root).resolve();salida=Path(salida).resolve()
    if not 1<=workers<=4 or games<8 or games%8 or games>=1000000 or minutes<=0:
        raise ValueError('1..4 trabajadores; partidas múltiplo8 y <1000000; minutos positivos')
    cmd=commands(root,salida,workers,games,minutes,seed)
    if dry_run:return {'commands':cmd,'started':False}
    if not (root/'material/profesor_utilidad/busqueda_eval3.py').is_file():
        raise ValueError('Usar el paquete portable: falta material/profesor_utilidad')
    salida.mkdir(parents=True,exist_ok=True)
    contract={'workers':workers,'games_per_worker':games,'seed_base':seed,'seed_stride':1000000,
              'generation':'fase5_generar_robusto.py','depth':7,'node_budget':3000000,
              'uct_iterations':4096,'mc_samples':128,'mc_rollouts':20}
    mf=salida/'job.json'
    if mf.exists() and json.loads(mf.read_text())!=contract:
        raise ValueError('Otro trabajo en esa salida; conservar semillas y configuración')
    write(mf,contract)
    # Lock de ejecución: evita dos lanzadores sobre los mismos checkpoints.
    lock=salida/'ACTIVO.json'
    with lock.open('x') as f:json.dump({'pid':os.getpid(),'host':platform.node()},f)
    children=[];logs=[];interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    previous={s:signal.signal(s,stop) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        for i,args in enumerate(cmd):
            log=(salida/f'worker_{i}.log').open('a');logs.append(log)
            proc=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
            children.append(proc)
        while any(p.poll() is None for p in children):
            progress=[]
            for i,p in enumerate(children):
                file=salida/f'worker_{i}/progreso.json'
                detail=json.loads(file.read_text()) if file.exists() else {'estado':'iniciando_o_compilando'}
                progress.append({'worker':i,'pid':p.pid,'exit':p.poll(),'detail':detail})
            write(salida/'progreso_pc.json',{'at':datetime.datetime.now().astimezone().isoformat(),'workers':progress})
            if interrupted:
                for p in children:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                break
            time.sleep(3)
        for p in children:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
        result={'exit_codes':[p.returncode for p in children],'interrupted':interrupted,'complete':False}
        if not interrupted and all(p.returncode==0 for p in children):
            result['complete']=all(json.loads((salida/f'worker_{i}/progreso.json').read_text())['estado']=='completo' for i in range(workers))
            if result['complete']:
                for i in range(workers):
                    subprocess.run([sys.executable,'-B',str(TOOLS/'fase5_verificar_episodios.py'),'--lote',str(salida/f'worker_{i}')],cwd=root,check=True)
        final_progress=[]
        for i,p in enumerate(children):
            file=salida/f'worker_{i}/progreso.json'
            detail=json.loads(file.read_text()) if file.exists() else {'estado':'sin_checkpoint'}
            final_progress.append({'worker':i,'pid':p.pid,'exit':p.returncode,'detail':detail})
        write(salida/'progreso_pc.json',{'at':datetime.datetime.now().astimezone().isoformat(),'workers':final_progress,'supervisor_terminado':True})
        write(salida/'RESULTADO_PC.json',result)
        return result
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in logs:f.close()
        for s,handler in previous.items():signal.signal(s,handler)
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--partidas',type=int,default=64)
    p.add_argument('--minutos',type=float,default=60);p.add_argument('--semilla',type=int,default=53000000)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();result=run(a.root,a.salida,a.workers,a.partidas,a.minutos,a.semilla,a.dry_run)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('interrupted'):raise SystemExit(130)
    if any(code!=0 for code in result.get('exit_codes',[])):raise SystemExit(1)
