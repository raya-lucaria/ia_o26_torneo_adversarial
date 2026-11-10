"""Espera sesión actual y reanuda una sola vez si pausa por datos incompletos.

Nunca reinicia errores, rechazos o interrupciones manuales. No cambia muestras,
fuentes, semillas ni puertas. La única variación permitida es tiempo de sesión.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def guardar(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def comando(root,out,minutos):
    return [sys.executable,'-B',str(Path(__file__).with_name('continuar_fase5.py')),
            '--root',str(root),'--salida',str(out),'--minutos',str(minutos)]

def supervisar(root,out,minutos=360,intervalo=10):
    root=Path(root).resolve();out=Path(out).resolve()
    protocol=out/'PROTOCOLO.json';expected=hashlib.sha256(protocol.read_bytes()).hexdigest()
    lock=out/'AUTO_ACTIVO.json'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),protocolo_sha256=expected),f)
    child=None;stop=False
    def detener(signum,frame):
        nonlocal stop
        stop=True
    previous={s:signal.signal(s,detener) for s in [signal.SIGINT,signal.SIGTERM]}
    def estado(fase,**extra):
        r=dict(fase=fase,fecha=datetime.datetime.now().astimezone().isoformat(),promovido=False,**extra)
        guardar(out/'CONTINUACION.json',r);print(json.dumps(r,ensure_ascii=False),flush=True);return r
    try:
        estado('esperando_sesion_actual',minutos_adicionales_maximos=minutos)
        while (out/'PIPELINE_ACTIVO.json').exists() and not stop:time.sleep(intervalo)
        if stop:return estado('detenido_sin_reanudar')
        if hashlib.sha256(protocol.read_bytes()).hexdigest()!=expected:raise ValueError('Protocolo cambió; no reanudar')
        current=json.loads((out/'ESTADO.json').read_text())
        if current['fase']!='pausa_datos_incompletos':
            return estado('sin_reinicio',estado_flujo=current['fase'])
        args=comando(root,out,minutos)
        with (out/'continuacion.log').open('a') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            estado('continuando',pid=child.pid,comando=args)
            while child.poll() is None:
                if stop:
                    os.killpg(child.pid,signal.SIGTERM);break
                time.sleep(min(intervalo,2))
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        current=json.loads((out/'ESTADO.json').read_text())
        return estado('sesion_adicional_terminada',exit=code,estado_flujo=current['fase'])
    except BaseException as exc:
        estado('error_sin_reintentar',error=str(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for s,handler in previous.items():signal.signal(s,handler)
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True);p.add_argument('--minutos',type=float,default=360)
    a=p.parse_args()
    if not 0<a.minutos<=360:p.error('Una continuación de hasta360min; nunca indefinida')
    supervisar(a.root,a.salida,a.minutos)
