"""Fase5 recuperable: datos→contrato→un ajuste→selección→test; nunca promueve."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import sys
from lanzar import run, write
from validacion import sha, preparar, contrato, seleccionar, examinar, LIMITES
from entrenar_replay import entrenar
from preparar_lotes import preparar as combinar

MAIN=Path(__file__).resolve().parents[4]
HIST=MAIN/'resultados/medicion/infra_pc_20261029/historico_preparado.npz'

def protocolo(root):
    sources=[root/'material/profesor_utilidad/busqueda_eval3.py',
             root/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz',
             root/'estudiantes/iacasitegano/herramientas/fase5_generar_robusto.py',
             root/'estudiantes/iacasitegano/herramientas/fase5_rivales.py',
             root/'estudiantes/iacasitegano/herramientas/fase5_rivales_robustos.py',
             root/'estudiantes/iacasitegano/eval3.py',root/'sistema/motor/nucleo.py',
             MAIN/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz',HIST]
    sources += [Path(__file__).with_name(n) for n in ['continuar_fase5.py','lanzar.py','validacion.py','entrenar_replay.py','preparar_lotes.py']]
    sources += [Path(__file__).resolve().parents[1]/n for n in ['fase5_features.py','fase5_modelos.py','fase5_verificar_episodios.py']]
    return dict(workers=4,partidas_por_worker=64,semillas=[94000000,95000000,96000000,97000000],
        grupos_previstos=128,limites=LIMITES,lambda_=100,replay_historico=.8,replay_nuevo=.2,
        profesor='U máximo7/3M por iteración; guardar profundidad efectiva; no óptimo',
        rivales='MC128x20,agresivo,carrera,UCT4096',test='FamiliaUCT completa; una exposición tras selección',
        fuentes={str(f):sha(f) for f in sources},python=sys.version,arquitectura=platform.machine())

def continuar(root,salida,minutos=60):
    root=Path(root).resolve();out=Path(salida).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'PIPELINE_ACTIVO.json'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),host=platform.node()),f)
    def estado(fase,**extra):
        r=dict(fase=fase,fecha=datetime.datetime.now().astimezone().isoformat(),promovido=False,**extra)
        write(out/'ESTADO.json',r);print(json.dumps(r,ensure_ascii=False),flush=True);return r
    try:
        current=protocolo(root);file=out/'PROTOCOLO.json'
        if file.exists() and json.loads(file.read_text())!=current:raise ValueError('Protocolo modificado; conservar salida y revisar fuentes')
        if not file.exists():write(file,current)
        estado('generando',episodios_previstos=256,minutos_sesion=minutos)
        result=run(root,out/'lotes',4,64,minutos,94000000)
        if result['interrupted']:return estado('interrumpido',generacion=result)
        if any(result['exit_codes']):raise ValueError('Un generador terminó con error; revisar logs')
        if not result['complete']:return estado('pausa_datos_incompletos',generacion=result)
        if protocolo(root)!=current:raise ValueError('Fuentes cambiaron durante generación; no entrenar')
        datos=out/'datos.npz'
        if not datos.exists():combinar([out/f'lotes/worker_{i}' for i in range(4)],datos)
        meta=json.loads(datos.with_suffix('.json').read_text())
        if sha(datos)!=meta['datos_sha256']:raise ValueError('Datos modificados o incompletos')
        gate=out/'validacion';base=MAIN/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz'
        if not gate.exists():preparar(datos,HIST,base,gate,'nuevo')
        c=contrato(gate)
        if c['datos_sha256']!=sha(datos) or c['historico_sha256']!=sha(HIST) or c['archivos']['base.npz']!=sha(base):raise ValueError('Contrato pertenece a otros datos/base')
        candidate=out/'candidato/replay.npz'
        estado('entrenamiento',grupos=meta['groups'],purga_historico=c['purga_historico'])
        if not candidate.exists():entrenar(gate,candidate)
        info=json.loads(candidate.with_suffix('.json').read_text())
        if sha(candidate)!=info['candidato_sha256']:raise ValueError('Candidato modificado o incompleto')
        selected=gate/'seleccion_replay/RESULTADO.json'
        s=json.loads(selected.read_text()) if selected.exists() else seleccionar(gate,candidate,'replay')
        if sha(candidate)!=s['candidato_sha256']:raise ValueError('Selección de otro candidato')
        if not s['pasa']:return estado('rechazado_seleccion',checks=s['checks'])
        estado('examen_offline')
        exam=gate/'seleccion_replay/EXAMEN.json'
        e=json.loads(exam.read_text()) if exam.exists() else examinar(gate,'replay')
        return estado('pendiente_tecnica_y_partidas_oficiales' if e['apto_para_puerta_tecnica'] else 'rechazado_test',examen=e)
    except BaseException as exc:
        estado('error_revisar',error=str(exc));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);p.add_argument('--minutos',type=float,default=60)
    a=p.parse_args();r=continuar(a.root,a.salida,a.minutos)
    if r['fase']=='interrumpido':raise SystemExit(130)
