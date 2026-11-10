"""Paquete offline explicito para otro host; excluye Git, credenciales y caches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--auditoria',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--pesos-antiguos',type=Path)
    p.add_argument('--salida',type=Path,required=True)
    args=p.parse_args()
    if args.salida.exists():
        raise FileExistsError(args.salida)
    herramientas=Path(__file__).resolve().parent
    archivos={f'herramientas/{n}':herramientas/n for n in
              ['fase5_features.py','fase5_modelos.py','fase5_preparar_datos.py','fase5_entrenar.py',
               'fase5_reproducir_activo.py','fase5_perfil_candidato.py']}
    archivos.update({'entrada/posiciones_resultado.npz':args.dataset,
                     'entrada/pesos_base.npz':args.base,
                     'entrada/auditoria/auditoria.json':args.auditoria/'auditoria.json',
                     'entrada/auditoria/predicciones_y_grupos.npz':args.auditoria/'predicciones_y_grupos.npz'})
    if args.pesos_antiguos is not None:
        archivos['entrada/pesos_antiguos.npy']=args.pesos_antiguos
    manifiesto={n:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
                for n,f in archivos.items()}
    metadata=json.loads((args.auditoria/'auditoria.json').read_text())
    if manifiesto['entrada/posiciones_resultado.npz']['sha256']!=metadata['dataset_sha256']:
        raise ValueError('Dataset no corresponde a auditoria')
    if manifiesto['entrada/pesos_base.npz']['sha256']!=metadata['pesos_sha256']:
        raise ValueError('Base no corresponde a auditoria')
    readme='''# Lote offline fase 5

Este paquete prepara y compara modelos sobre DESARROLLO HISTORICO. No demuestra fuerza.
Necesita Python con NumPy; ningun componente necesita GPU, motor o red para entrenar.
Trabajar desde la carpeta extraida, dentro del entorno Python preparado del proyecto.
Comprobar SHA-256 con MANIFEST.json. No medir partidas en el mismo host durante el ajuste.

0. Reproducir el modelo activo (si el paquete incluye pesos_antiguos.npy):

    python herramientas/fase5_reproducir_activo.py --dataset entrada/posiciones_resultado.npz --pesos-antiguos entrada/pesos_antiguos.npy --referencia entrada/pesos_base.npz --salida trabajo/reproduccion

La receta historica usa IRLS lambda100 centrado en la base antigua. Se comparan arrays, no el hash del ZIP.

1. Preparar datos (no repetir si trabajo/datos/manifest.json existe y sus hashes coinciden):

    python herramientas/fase5_preparar_datos.py --dataset entrada/posiciones_resultado.npz --auditoria entrada/auditoria --salida trabajo/datos

2. Piloto de un ajuste, con checkpoint:

    python herramientas/fase5_entrenar.py --datos trabajo/datos/datos.npz --base entrada/pesos_base.npz --salida trabajo/ajustes --max-ajustes 1

3. Reanudar y completar los ocho ajustes y GA, misma carpeta y contrato:

    python herramientas/fase5_entrenar.py --datos trabajo/datos/datos.npz --base entrada/pesos_base.npz --salida trabajo/ajustes

Si se interrumpe un ajuste sin JSON de checkpoint, se recalcula ese ajuste. Los ya completos se verifican y reutilizan.
El GA tiene 72 evaluaciones mas una referencia aleatoria de 72. Su coste usa contribuciones precalculadas.
Modelo/parametros se eligen con fold1; fold2 es prueba interna historica ya vista en desarrollo anterior.
El NPZ candidato es de laboratorio. Ridge requiere enlace de retorno acotado y terminales +-1;
no cargarlo como logit ni sobrescribir pesos activos. Para logistica, especificar control/escala de busqueda.

Devolver trabajo/ajustes, trabajo/datos/manifest.json y versiones Python/NumPy/SO/CPU.
Despues faltan pruebas del buscador, costo y partidas oficiales con datos nuevos.
'''
    args.salida.parent.mkdir(parents=True,exist_ok=True)
    temporal=args.salida.with_suffix('.tmp')
    with tarfile.open(temporal,'w') as tar:
        for nombre,f in archivos.items():
            tar.add(f,arcname=nombre,recursive=False)
        for nombre,contenido in [('MANIFEST.json',json.dumps(manifiesto,ensure_ascii=False,indent=2)+'\n'),('LEEME.md',readme)]:
            data=contenido.encode(); info=tarfile.TarInfo(nombre); info.size=len(data);info.mode=0o644
            tar.addfile(info,io.BytesIO(data))
    temporal.replace(args.salida)
    print(json.dumps({'paquete':str(args.salida),'sha256':hashlib.sha256(args.salida.read_bytes()).hexdigest(),
                      'bytes':args.salida.stat().st_size,'archivos':len(archivos)+2},indent=2))


if __name__=='__main__':main()
