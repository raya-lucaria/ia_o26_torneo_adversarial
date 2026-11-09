"""Exporta resultados de pods finalizados sin mostrar blobs en la terminal."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

def recoger(kubeconfig, salida, job='fase5-validacion-20261029', esperadas=4):
    salida=Path(salida);salida.mkdir(parents=True,exist_ok=True)
    base=['kubectl','--kubeconfig',str(kubeconfig),'-n','breakthrough-lab']
    def command(*args):
        return subprocess.check_output([*base,*args],text=True)
    pods=json.loads(command('get','pods','-l','job-name='+job,'-o','json'))
    (salida/'pods.json').write_text(json.dumps(pods,indent=2)+'\n')
    results=[]
    for pod in pods['items']:
        name=pod['metadata']['name'];annotations=pod['metadata'].get('annotations',{})
        index=annotations.get('lab-tarea',annotations.get('batch.kubernetes.io/job-completion-index'))
        phase=pod['status']['phase'];entry=dict(indice=index,pod=name,fase=phase)
        if phase in ['Succeeded','Failed']:
            text=command('logs',name);folder=salida/('tarea_'+index);folder.mkdir(exist_ok=True)
            lines=text.splitlines();blob=next((s.split('=',1)[1] for s in lines if s.startswith('ARTEFACTOS_BASE64=')),None)
            (folder/'pod.log').write_text('\n'.join(s for s in lines if not s.startswith('ARTEFACTOS_BASE64='))+'\n')
            if blob:
                raw=base64.b64decode(blob,validate=True)
                expected=next(s.split('=',1)[1] for s in lines if s.startswith('ARCHIVO_SHA256='))
                assert hashlib.sha256(raw).hexdigest()==expected
                (folder/'artefactos.tar.gz').write_bytes(raw)
                with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                    tar.extractall(folder/'archivos',filter='data')
                entry['resultado']=json.loads((folder/'archivos/RESULTADO.json').read_text())
            else:entry['error']='Sin archivo exportado; consultar pod.log'
        results.append(entry)
    result=dict(tareas=results,completo=len(results)==esperadas and all(r['fase']=='Succeeded' and 'resultado' in r for r in results))
    (salida/'ESTADO.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kubeconfig',type=Path,required=True);p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--job',default='fase5-validacion-20261029');p.add_argument('--esperadas',type=int,default=4)
    a=p.parse_args();print(json.dumps(recoger(a.kubeconfig,a.salida,a.job,a.esperadas),indent=2))
