"""Laboratorio portable: fuentes publicas del motor, herramientas propias y checkpoints.

Solo empaqueta una generacion detenida/completa, nunca copia caches o credenciales.
"""
import argparse,hashlib,io,json,tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lote',type=Path,required=True)
    p.add_argument('--dataset-historico',type=Path,required=True)
    p.add_argument('--profesor',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    args=p.parse_args()
    if args.salida.exists():raise FileExistsError(args.salida)
    progreso=json.loads((args.lote/'progreso.json').read_text())
    if progreso['estado']=='generando':raise ValueError('Primero detener por checkpoint o esperar finalizacion')
    equipo=Path(__file__).resolve().parents[1];raiz=equipo.parents[1]
    destino_lote='resultados/medicion/fase5_20261028/episodios_60'
    archivos={}
    for f in sorted((raiz/'sistema/motor').rglob('*.py')):
        archivos[str(f.relative_to(raiz))]=f
    for f in sorted((equipo/'herramientas').glob('fase5_*.py')):
        archivos['estudiantes/iacasitegano/herramientas/'+f.name]=f
    for nombre in ['test_fase5.py','extraer_casos_decision.py']:
        archivos['estudiantes/iacasitegano/herramientas/'+nombre]=equipo/'herramientas'/nombre
    archivos['estudiantes/iacasitegano/eval3.py']=equipo/'eval3.py'
    for nombre in ['pesos.npy','pesos_eval3_ligero.npz']:
        archivos['estudiantes/iacasitegano/data/'+nombre]=equipo/'data'/nombre
    for nombre,fuente in [('profesor_utilidad',args.profesor),('control_logits',args.control)]:
        archivos[f'material/{nombre}/busqueda_eval3.py']=fuente
        archivos[f'material/{nombre}/eval3.py']=fuente.parent/'eval3.py'
    archivos['entrada/posiciones_historicas.npz']=args.dataset_historico
    for sub in ['partidas','shards']:
        for f in sorted((args.lote/sub).iterdir()):
            if f.is_file() and f.suffix in ['.json','.npz','.parcial']:
                archivos[f'{destino_lote}/{sub}/{f.name}']=f
    for nombre in ['manifest.json','progreso.json','PRUEBA_REANUDACION_ANTES.json','VERIFICACION.json']:
        f=args.lote/nombre
        if f.exists():archivos[f'{destino_lote}/{nombre}']=f
    hashes={n:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for n,f in archivos.items()}
    leeme='''# Laboratorio fase 5 portable

Fuentes del motor oficial sin modificar (https://github.com/raya-lucaria/ia_o26_torneo_adversarial), herramientas propias y datos privados de desarrollo. NO es una entrega de liga ni una prueba de fuerza. No contiene Git, credenciales, entornos ni cache Numba.

Extraer en una carpeta nueva. En Windows se recomienda Ubuntu/WSL2 siguiendo el curso. Preparar Python con NumPy y Numba en un entorno offline. No hace falta GPU ni red durante los trabajos. El cache se compila de nuevo en este host; no se comparte entre sistemas.

Desde la raiz extraida, verificar hashes con python verificar_paquete.py. Comandos (anteponer rtk proxy si esta instalado):

Reanudar episodios, hasta siete horas y mismos presupuestos/semillas:
python estudiantes/iacasitegano/herramientas/fase5_generar_partidas.py --buscador material/profesor_utilidad/busqueda_eval3.py --salida resultados/medicion/fase5_20261028/episodios_60 --partidas 60 --minutos 420

Verificar todas las filas y transiciones al terminar:
python estudiantes/iacasitegano/herramientas/fase5_verificar_episodios.py --lote resultados/medicion/fase5_20261028/episodios_60

Combinar los 60 completos, usando archivo nuevo si ya existia uno:
python estudiantes/iacasitegano/herramientas/fase5_unir_episodios.py --lote resultados/medicion/fase5_20261028/episodios_60 --salida resultados/medicion/fase5_20261028/episodios_60/dataset_060.npz

Adaptacion con lambda100 y particiones ya fijadas, solo tras completar el lote:
python estudiantes/iacasitegano/herramientas/fase5_adaptacion.py --lote resultados/medicion/fase5_20261028/episodios_60 --datos resultados/medicion/fase5_20261028/episodios_60/dataset_060.npz --base estudiantes/iacasitegano/data/pesos_eval3_ligero.npz --salida resultados/medicion/fase5_20261028/adaptacion_nueva

Reproducir el modelo actualmente desplegado, sin reemplazarlo:
python estudiantes/iacasitegano/herramientas/fase5_reproducir_activo.py --dataset entrada/posiciones_historicas.npz --pesos-antiguos estudiantes/iacasitegano/data/pesos.npy --referencia estudiantes/iacasitegano/data/pesos_eval3_ligero.npz --salida resultados/medicion/reproduccion_host

No sobrescribir modelos activos. Devolver resultados, manifiestos y versiones del host. Un candidato que pase los filtros offline todavia requiere pruebas oficiales con limites y rivales distintos. No cambiar umbrales tras ver la prueba interna UCT. No comparar relojes entre hosts como A/B.
'''
    verificar='''import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent
for name,info in json.loads((root/'MANIFEST.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==info['sha256'],name
print('Integridad del paquete OK')
'''
    args.salida.parent.mkdir(parents=True,exist_ok=True);tmp=args.salida.with_suffix('.tmp')
    with tarfile.open(tmp,'w') as tar:
        for n,f in archivos.items():tar.add(f,arcname=n,recursive=False)
        for n,texto in [('MANIFEST.json',json.dumps(hashes,ensure_ascii=False,indent=2)+'\n'),('LEEME.md',leeme),('verificar_paquete.py',verificar)]:
            data=texto.encode();info=tarfile.TarInfo(n);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
    tmp.replace(args.salida)
    print(json.dumps({'paquete':str(args.salida),'bytes':args.salida.stat().st_size,
                      'sha256':hashlib.sha256(args.salida.read_bytes()).hexdigest(),'archivos':len(archivos)+3,
                      'episodios_completados':progreso['completadas']},indent=2))


if __name__=='__main__':main()
