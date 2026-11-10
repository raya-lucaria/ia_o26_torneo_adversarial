"""Describe episodios completos y profundidad efectiva; no selecciona modelos."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def auditar(lotes, salida):
    salida=Path(salida)
    if salida.exists():raise FileExistsError(salida)
    depths=Counter();families={};groups=set();roles=defaultdict(set);sources=[];nodes=[]
    phases={k:set() for k in ['inicio','medio','tardio']};fallback=0;teacher_moves=0
    total_games=0;total_rows=0
    for folder in map(Path,lotes):
        progress=json.loads((folder/'progreso.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        paths=sorted((folder/'partidas').glob('*.json'))
        if progress['estado']!='completo' or len(paths)!=manifest['partidas']:
            raise ValueError('Solo auditar lote completo: '+str(folder))
        verification=json.loads((folder/'VERIFICACION.json').read_text())
        if verification['errores'] or verification['partidas']!=len(paths):
            raise ValueError('Replay independiente incompleto')
        verified={r['archivo']:r['sha256'] for r in verification['fuentes']}
        for path in paths:
            if verified.get(path.name)!=sha(path):raise ValueError('Partida cambió tras replay')
            d=json.loads(path.read_text());m=d['meta'];group=f"{m['semilla_maestra']}:{m['pareja']}:{m['juego']}"
            color=next(c for c in ['blancas','negras'] if m[c]=='profesor_offline')
            if color in roles[group]:raise ValueError('Episodio duplicado')
            roles[group].add(color);groups.add(group)
            rival=m['negras' if color=='blancas' else 'blancas']
            family=families.setdefault(rival,dict(partidas=0,blancas=0,negras=0,grupos=set()))
            family['partidas']+=1;family[color]+=1;family['grupos'].add(group)
            total_games+=1;total_rows+=len(d['jugadas'])
            for j in d['jugadas']:
                if j['color']!=color:continue
                teacher_moves+=1;info=j.get('info',{})
                if 'prof' in info:depths[int(info['prof'])]+=1
                else:fallback+=1
                if 'nodos' in info:nodes.append(int(info['nodos']))
                ply=j['n']-1;phase='inicio' if ply<20 else 'medio' if ply<60 else 'tardio'
                phases[phase].add(group)
            sources.append(dict(partida=str(path),sha256=sha(path),grupo=group))
    if any(v!={'blancas','negras'} for v in roles.values()):raise ValueError('Pareja incompleta')
    for f in families.values():f['grupos']=len(f['grupos'])
    result=dict(partidas=total_games,filas=total_rows,grupos=len(groups),familias=families,
                decisiones_profesor=teacher_moves,profundidad_completa=dict(sorted(depths.items())),
                respaldo_sin_iteracion_completa=fallback,grupos_por_fase={k:len(v) for k,v in phases.items()},
                nodos_ultima_iteracion=dict(mediana=statistics.median(nodes) if nodes else None,maximo=max(nodes) if nodes else None),
                cobertura_suficiente_para_5V=False if len(groups)<24 else None,
                fuentes=sources,limite='Muestra piloto pequeña. No mide fuerza oficial ni prueba valor óptimo. Cotas de nodos por iteración; nodos de última iteración no son costo total. El ajuste exige particiones/estratos/retención de5V.')
    salida.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='fuentes'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lotes',nargs='+',type=Path,required=True);p.add_argument('--salida',type=Path,required=True)
    a=p.parse_args();print(json.dumps(auditar(a.lotes,a.salida),ensure_ascii=False,indent=2))
