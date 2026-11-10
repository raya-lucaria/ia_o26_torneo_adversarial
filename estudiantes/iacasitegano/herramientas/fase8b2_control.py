"""Preparación y guardia de ventana8B.2. Nunca selecciona modelos o mata carga ajena."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from fase8a_soporte import ROOT, TEAM, Budget, sha, save

OUT = ROOT/'resultados/medicion/fase8B2_runtime_20261029'
PREV = ROOT/'resultados/medicion/fase8B1_paquete_20261029'
IMAGE = 'sha256:79983872dd22263467bfb3f1e19a8619edb15cacfb79f2d9aef4f03b2d19d188'


def command(args, destination, cwd=ROOT, env=None, timeout=120):
    destination.mkdir(parents=True, exist_ok=False)
    save(destination/'comando.json', {'args':args,'cwd':str(cwd),'env_overrides':env or {},'at':datetime.now().astimezone().isoformat()})
    with (destination/'stdout.log').open('w') as stdout, (destination/'stderr.log').open('w') as stderr:
        result = subprocess.run(args, cwd=cwd, env={**os.environ, **(env or {})}, stdout=stdout, stderr=stderr, timeout=timeout)
    save(destination/'resultado.json', {'exit':result.returncode,'finished':datetime.now().astimezone().isoformat()})
    return result.returncode


def prepare():
    budget = Budget()
    if (OUT/'CONGELADO.json').exists():
        raise FileExistsError('Ya preparado; no sobrescribir')
    decision = json.loads((OUT/'DECISION_TECNICA.json').read_text())
    assert decision['estado']=='seleccion_explicita' and decision['escala']=='eval3_raw_logit' and not decision['promovido']
    decision_hash = sha(OUT/'DECISION_TECNICA.json')
    for name, info in decision['archivos'].items():
        assert sha(Path(info['path']),budget)==info['sha256']
    previous = json.loads((PREV/'manifiesto_inicial.json').read_text())
    runtime = {p:sha(Path(p),budget) for p in previous['runtime']}
    phase5 = json.loads((ROOT/'resultados/medicion/fase5_lote256_mac_20261029/PROTOCOLO.json').read_text())['fuentes']
    assert len(phase5)==17 and all(sha(Path(p),budget)==expected for p,expected in phase5.items())
    cache = {str(p):sha(p,budget) for p in sorted((TEAM/'data/numba_cache').rglob('*')) if p.is_file()}
    destination = OUT/'staging_sin_cache_01/estudiantes/iacasitegano'
    destination.mkdir(parents=True,exist_ok=False)
    for name, info in decision['archivos'].items():
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(info['path'],target)
        assert sha(target)==info['sha256']
    regression = OUT/'regresion_01/estudiantes/iacasitegano'
    shutil.copytree(destination,regression)
    (regression/'herramientas').mkdir()
    original_suite = ROOT/'resultados/medicion/fase8A_preintegracion_20261029/origenes/verificar_cotas_tt.py'
    shutil.copy2(original_suite,regression/'herramientas/verificar_cotas_tt.py')
    baseline = OUT/'baseline_P_01/estudiantes/iacasitegano'
    shutil.copytree(ROOT/'resultados/medicion/fase8A_preintegracion_20261029/copias/P/estudiantes/iacasitegano',baseline)
    frozen = {'at':datetime.now().astimezone().isoformat(),'decision_sha256':decision_hash,'runtime':runtime,'phase5_sources':phase5,
              'active_cache':cache,'decision_files':decision['archivos'],'stage':str(destination),'regression_root':str(regression),
              'baseline_root':str(baseline),'suite_sha256':sha(original_suite),'protocol_sha256':sha(OUT/'PROTOCOLO.md'),
              'image_expected':IMAGE,'source_origins':{str(p):sha(p,budget) for p in [ROOT/'sistema/arbitro/procesos.py',ROOT/'sistema/arbitro/partida.py',
                ROOT/'sistema/arbitro/runner.py',ROOT/'sistema/validador/validar.py',TEAM/'herramientas/cache_numba_build.py',
                TEAM/'herramientas/fase8a_soporte.py',TEAM/'herramientas/fase8a_verificar.py',
                PREV/'CIERRE_8B1.md',PREV/'8B2_PENDIENTE.md']}}
    save(OUT/'CONGELADO.json',frozen)
    save(OUT/'progreso.json',{'status':'prepared_pending_exclusive_window','regressions':'pending','cold':'pending','smoke':'pending','technical_games':'pending'})
    budget.wait()
    print(json.dumps({'stage':str(destination),'decision_sha256':decision_hash,'phase5_verified':17},indent=2))


def gate(label):
    folder=OUT/'ventanas'/label
    folder.mkdir(parents=True,exist_ok=False)
    top=subprocess.check_output(['rtk','proxy','top','-l','3','-s','3','-n','0'],text=True)
    (folder/'top.log').write_text(top)
    idle=[float(v) for v in re.findall(r'([0-9.]+)% idle',top)]
    samples=[]
    for index in range(2):
        raw=subprocess.check_output(['ps','-axo','pid,ppid,%cpu,etime,command'],text=True)
        (folder/f'procesos_{index}.log').write_text(raw)
        rows=[]
        for line in raw.splitlines()[1:]:
            parts=line.split(None,4)
            if len(parts)!=5:
                continue
            pid,ppid,cpu,elapsed,cmd=parts
            # Nuestro controlador y shells envolventes no son carga ajena.
            if int(pid) in [os.getpid(),os.getppid()] or 'fase8b2_' in cmd or 'ps -axo' in cmd:
                continue
            rows.append({'pid':int(pid),'cpu':float(cpu),'elapsed':elapsed,'command':cmd})
        samples.append(rows)
        if index==0:
            time.sleep(3)
    first={row['pid']:row for row in samples[0]}
    sustained=[r for r in samples[1] if r['pid'] in first and min(r['cpu'],first[r['pid']]['cpu'])>=50
               and not any(name in r['command'] for name in ['WindowServer','Codex (Renderer)','Codex (Service)','Terminal.app'])]
    compute=[r for rows in samples for r in rows if any(name in r['command'] for name in
        ['fase5_generar','continuar_fase5.py','fase5_entrenar.py','entrenar_replay.py','fase5_diagnosticar','fase5_s13'])
        and 'python' in r['command'].lower()]
    docker_ps=subprocess.run(['rtk','proxy','docker','ps','--no-trunc','--format','{{json .}}'],text=True,capture_output=True)
    (folder/'docker_ps.log').write_text(docker_ps.stdout+docker_ps.stderr)
    kube=subprocess.run(['/Users/josema/.kube/bin/kubectl-desktop','--context','docker-desktop','get','pods','-A','-o',
        'custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,PHASE:.status.phase','--request-timeout=10s'],text=True,capture_output=True)
    (folder/'kubernetes.log').write_text(kube.stdout+kube.stderr)
    running_jobs=[line for line in kube.stdout.splitlines()[1:] if line.startswith('default') and 'Running' in line]
    accepted=len(idle)>=3 and all(v>=70 for v in idle[-2:]) and not sustained and not compute and not running_jobs and docker_ps.returncode==0
    report={'at':datetime.now().astimezone().isoformat(),'idle_top_all':idle,'idle_last_two':idle[-2:],
            'other_sustained_ge50':sustained,'compute_jobs':compute,'kubernetes_default_running_jobs':running_jobs,
            'top_processes_last':sorted(samples[-1],key=lambda r:r['cpu'],reverse=True)[:12],
            'accepted':accepted,'rule':'last2 top idle>=70%, no sustained>=50% noninteractive other CPU, no foreign compute/job',
            'control_plane_not_stopped':True}
    save(folder/'ventana.json',report)
    print(json.dumps({k:report[k] for k in ['at','idle_last_two','other_sustained_ge50','compute_jobs','kubernetes_default_running_jobs','accepted']},indent=2))
    return accepted


def require_gate(label):
    p=OUT/'ventanas'/label/'ventana.json'
    report=json.loads(p.read_text())
    elapsed=(datetime.now().astimezone()-datetime.fromisoformat(report['at'])).total_seconds()
    if not report['accepted'] or elapsed>900:
        raise RuntimeError('Ventana no aceptada o antigua; no iniciar carga')


def regressions(label):
    require_gate(label)
    frozen=json.loads((OUT/'CONGELADO.json').read_text())
    suite=Path(frozen['regression_root'])/'herramientas/verificar_cotas_tt.py'
    assert sha(suite)==frozen['suite_sha256']
    for mode in ['python','jit']:
        dest=OUT/'regresiones'/mode
        args=['rtk','proxy',str(ROOT/'.venv/bin/python'),'-B',str(suite),'--baseline',str(Path(frozen['baseline_root'])/'busqueda_eval3.py'),'--extendido','--tt-bits','20']
        if mode=='jit':
            args.append('--jit')
        exitcode=command(args,dest,env={'PYTHONPATH':str(ROOT/'sistema'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'},timeout=300)
        if exitcode:
            raise RuntimeError(f'Regresión {mode} falló; conservar intento, detener rama')
        print(json.dumps({'regression':mode,'passed':True}),flush=True)
    progress=json.loads((OUT/'progreso.json').read_text());progress['regressions']='passed_python_and_JIT';save(OUT/'progreso.json',progress)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['preparar','ventana','regresiones'])
    parser.add_argument('--ventana',default=None)
    args=parser.parse_args()
    if args.step=='preparar':prepare()
    elif args.step=='ventana':
        if not args.ventana:parser.error('Falta nombre nuevo --ventana')
        gate(args.ventana)
    else:
        if not args.ventana:parser.error('Falta --ventana aceptada')
        regressions(args.ventana)


if __name__=='__main__':main()
