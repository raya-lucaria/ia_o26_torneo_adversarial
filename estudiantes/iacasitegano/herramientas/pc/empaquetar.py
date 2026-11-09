"""Amplía el laboratorio verificado, sin Git, credenciales, entornos ni cachés."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['base','historico','salida']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.salida.exists():raise FileExistsError(a.salida)
    files={}
    with tarfile.open(a.base) as t:
        for m in t.getmembers():
            if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('Miembro no permitido')
            if m.name not in ['MANIFEST.json','LEEME.md','verificar_paquete.py']:
                files[m.name]=t.extractfile(m).read()
    tools=Path(__file__).resolve().parents[1]
    for f in tools.glob('fase5_*.py'):
        files['estudiantes/iacasitegano/herramientas/'+f.name]=f.read_bytes()
    for f in Path(__file__).resolve().parent.iterdir():
        if f.is_file() and f.suffix in ['.py','.sh','.ps1','.txt','.md']:
            files['estudiantes/iacasitegano/herramientas/pc/'+f.name]=f.read_bytes()
    files['entrada/historico_preparado.npz']=a.historico.read_bytes()
    files['LEEME_PC.md']=(tools/'pc/LEEME_PC.md').read_bytes()
    files['LEEME.md']=b'Laboratorio PC: empezar por LEEME_PC.md. No reemplazar pesos activos. No es entrega de liga. Windows aun no probado.\n'
    verify='''import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
for name,info in json.loads((root/'MANIFEST.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==info['sha256'],name
print('Integridad del paquete PC verificada')
'''
    files['verificar_paquete.py']=verify.encode()
    manifest={n:{'sha256':digest(data),'bytes':len(data)} for n,data in files.items()}
    files['MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    a.salida.parent.mkdir(parents=True,exist_ok=True);tmp=a.salida.with_suffix('.tmp')
    with tarfile.open(tmp,'w') as t:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
    tmp.replace(a.salida)
    result={'path':str(a.salida),'sha256':digest(a.salida.read_bytes()),'bytes':a.salida.stat().st_size,
            'input_files':len(manifest),'base_archive_sha256':digest(a.base.read_bytes()),'windows_tested':False}
    a.salida.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
