"""Replay independiente: comprueba tambien cada fila del shard y su etiqueta."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ['NUMBA_DISABLE_JIT']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'sistema'))
import numpy as np
from motor import Estado,Jugada
from motor.jugada import casilla_de_texto


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lote',type=Path,required=True)
    args=p.parse_args();partidas=plies=0;fuentes=[]
    for ruta in sorted((args.lote/'partidas').glob('*.json')):
        d=json.loads(ruta.read_text());shard=args.lote/'shards'/(ruta.stem+'.npz')
        assert hashlib.sha256(shard.read_bytes()).hexdigest()==d['shard_sha256']
        with np.load(shard,allow_pickle=False) as z:
            assert len(z['turno'])==len(d['jugadas'])
            estado=Estado.inicial()
            for k,r in enumerate(d['jugadas']):
                t=np.zeros(81,np.int8)
                for c,n in enumerate(['blancas','negras']):
                    for pos in estado.piezas(n):t[casilla_de_texto(pos)]=c+1
                np.testing.assert_array_equal(z['tablero'][k],t)
                np.testing.assert_array_equal(z['puntos'][k],[estado.puntos[c] for c in ['blancas','negras']])
                assert z['turno'][k]==int(estado.turno=='negras') and z['ply'][k]==k
                assert z['ganador_blancas'][k]==int(d['ganador']=='blancas')
                assert z['partida_origen'][k]=='partidas/'+ruta.name
                assert r['n']==k+1 and r['color']==estado.turno
                j=Jugada.desde_texto(r['jugada']);assert j in estado.jugadas()
                ataque=estado.es_ataque(j);combate=r['combate'];assert ataque==(combate is not None)
                if ataque:
                    assert abs(combate['p']-estado.prob_ataque(j))<1e-12
                    assert 0<=combate['u']<1 and combate['gano']==(combate['u']<combate['p'])
                estado=estado.aplicar(j,combate['gano'] if ataque else None)
            assert estado.ganador()==d['ganador'] and estado.puntos==d['puntos']
            partidas+=1;plies+=len(d['jugadas'])
        fuentes.append({'archivo':ruta.name,'sha256':hashlib.sha256(ruta.read_bytes()).hexdigest()})
    res={'partidas':partidas,'filas_verificadas':plies,'errores':0,'fuentes':fuentes,
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'limite':'Replay correcto de episodios privados; no validacion de fuerza ni limites oficiales'}
    (args.lote/'VERIFICACION.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in res.items() if k!='fuentes'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
