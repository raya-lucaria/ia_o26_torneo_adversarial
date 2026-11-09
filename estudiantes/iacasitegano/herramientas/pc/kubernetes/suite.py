"""Batería Linux offline: cuatro tareas; ningún resultado acredita fuerza."""
import hashlib
import base64
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

ROOT = Path('/lab')
OUT = Path('/salida')
TOOLS = ROOT/'estudiantes/iacasitegano/herramientas'
sys.path[:0] = [str(TOOLS/'pc'), str(TOOLS), str(ROOT/'sistema')]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(name, args):
    p = subprocess.run([sys.executable, '-B', *map(str, args)], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT/(name+'.log')).write_text(p.stdout)
    print(json.dumps({'paso':name, 'exit':p.returncode}), flush=True)
    if p.returncode:
        print(p.stdout[-6000:], flush=True)
        raise RuntimeError(name)

def main():
    task = int(os.environ['JOB_COMPLETION_INDEX'])
    OUT.mkdir(parents=True, exist_ok=True)
    run('integridad', [ROOT/'verificar_paquete.py'])
    if task == 0:
        run('contratos_validacion', [TOOLS/'pc/test_validacion.py'])
        run('modelos_motor_y_rivales', [TOOLS/'test_fase5.py'])
    elif task == 1:
        run('receta_activa', [TOOLS/'fase5_reproducir_activo.py',
            '--dataset', ROOT/'entrada/posiciones_historicas.npz',
            '--pesos-antiguos', ROOT/'estudiantes/iacasitegano/data/pesos.npy',
            '--referencia', ROOT/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz',
            '--salida', OUT/'receta'])
        r=json.loads((OUT/'receta/RESULTADO.json').read_text())
        assert r['coincide'] and r['max_diferencia_pesos'] <= 1e-8
    elif task == 2:
        args=[TOOLS/'fase5_generar_robusto.py',
            '--buscador', ROOT/'material/profesor_utilidad/busqueda_eval3.py',
            '--salida', OUT/'episodios', '--partidas','8','--minutos','8',
            '--semilla','75000000','--profundidad','2','--tope-nodos','10000',
            '--uct-iteraciones','32','--mc-muestras','2','--mc-rollouts','2']
        run('generar_primer_par', [*args,'--max-juegos-nuevos','2'])
        first={str(p.relative_to(OUT)):sha(p) for p in (OUT/'episodios/partidas').glob('*.json')}
        assert len(first)==2
        run('reanudar_cuatro_familias', args)
        assert all(sha(OUT/p)==s for p,s in first.items())
        assert json.loads((OUT/'episodios/progreso.json').read_text())['estado']=='completo'
        run('replay_independiente_nuevo', [TOOLS/'fase5_verificar_episodios.py','--lote',OUT/'episodios'])
        run('preparar_particiones', [TOOLS/'pc/preparar_lotes.py','--lotes',OUT/'episodios','--salida',OUT/'datos.npz'])
        from validacion import cargar
        d=cargar(OUT/'datos.npz')
        for i in range(3):
            for j in range(i):
                for key in ['grupo','estado']:
                    assert not set(d[key][d['fold']==i]) & set(d[key][d['fold']==j])
        (OUT/'REANUDACION.json').write_text(json.dumps(dict(originales_intactos=first,
            partidas=8, familias=4, alcance='Presupuestos mínimos funcionales; no datos de entrenamiento cualificados.'),indent=2))
    elif task == 3:
        from validacion import preparar, seleccionar
        preparar(ROOT/'entrada/historico_preparado.npz',ROOT/'entrada/historico_preparado.npz',
                 ROOT/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz',OUT/'auditoria','historico')
        r=seleccionar(OUT/'auditoria',ROOT/'auxiliar/N_archivado.npz','N_archivado')
        assert not r['pasa'] and not r['test_evaluado']
        assert not (OUT/'auditoria/EXAMEN_ABIERTO.json').exists()
        historical=Path('/tmp/episodios_historicos')
        shutil.copytree(ROOT/'resultados/medicion/fase5_20261028/episodios_60',historical)
        run('replay_historico_60', [TOOLS/'fase5_verificar_episodios.py','--lote',historical])
        shutil.copyfile(historical/'VERIFICACION.json',OUT/'REPLAY_HISTORICO.json')
    else:
        raise ValueError(task)
    import numpy, numba
    result=dict(tarea=task, correcta=True, python=sys.version, arquitectura=platform.machine(),
                numpy=numpy.__version__,numba=numba.__version__,
                pesos_sha256=sha(ROOT/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz'),
                limite='Pruebas funcionales Linux; no reloj oficial, fuerza ni promoción.')
    (OUT/'RESULTADO.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)
    # Exportación acotada por stdout: permite conservar resultados cuando termina
    # el pod sin montar el checkout ni mantener contenedores de espera vivos.
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for path in sorted(OUT.rglob('*')):
            if not path.is_file() or 'cache_numba' in path.parts:
                continue
            if path.suffix in ['.json','.log'] or (path.suffix=='.npz' and 'auditoria' not in path.parts):
                tar.add(path, arcname=str(path.relative_to(OUT)),recursive=False)
    data=archive.getvalue()
    assert len(data)<2*1024*1024, 'Resultados demasiado grandes para logs; conservar pod y revisar'
    print('ARCHIVO_SHA256='+hashlib.sha256(data).hexdigest(), flush=True)
    print('ARTEFACTOS_BASE64='+base64.b64encode(data).decode(), flush=True)

if __name__=='__main__':
    main()
