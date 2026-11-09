"""Combina solo parejas completas de episodios offline, verificando cada shard."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lote',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    args=p.parse_args()
    if args.salida.exists():raise FileExistsError(args.salida)
    grupos={}
    for f in sorted((args.lote/'partidas').glob('*.json')):
        d=json.loads(f.read_text());m=d['meta'];g=(m['semilla_maestra'],m['pareja'],m['juego'])
        grupos.setdefault(g,[]).append((f,d))
    fuentes=[];cols={};excluidas=[]
    for g,pareja in grupos.items():
        if len(pareja)!=2:
            excluidas.extend(str(f) for f,_ in pareja);continue
        colores={next(c for c in ['blancas','negras'] if d['meta'][c]=='profesor_offline') for _,d in pareja}
        if colores!={'blancas','negras'}:raise ValueError('Pareja sin ambos colores')
        for f,d in pareja:
            shard=args.lote/'shards'/(f.stem+'.npz')
            if sha(shard)!=d['shard_sha256']:raise ValueError('Shard modificado')
            with np.load(shard,allow_pickle=False) as z:
                for k in z.files:cols.setdefault(k,[]).append(z[k])
            fuentes.append({'archivo':str(f.relative_to(args.lote)),'sha256':sha(f),'shard_sha256':sha(shard)})
    if not fuentes:raise ValueError('No hay parejas completas')
    args.salida.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.salida,**{k:np.concatenate(v) for k,v in cols.items()})
    info={'partidas':len(fuentes),'posiciones':sum(len(x) for x in cols['turno']),
          'fuentes':fuentes,'parejas_parciales_excluidas_por_diseño':excluidas,
          'dataset_sha256':sha(args.salida),'contrato_lote_sha256':sha(args.lote/'manifest.json'),
          'alcance':'Entrenamiento offline; no fuerza oficial ni confirmacion'}
    args.salida.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in info.items() if k!='fuentes'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
