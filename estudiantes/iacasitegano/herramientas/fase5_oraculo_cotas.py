"""Cotas racionales de utilidad terminal; hojas desconocidas=[-1,1].

No usa evaluador aprendido. Un horizonte o presupuesto agotado no recibe valor0.
Solo L(alternativa)>U(observada) demuestra dominancia bajo reglas del motor.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['NUMBA_DISABLE_JIT']='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sistema'))
from motor import Estado,Jugada

UNKNOWN=(F(-1),F(1))

def probabilidad(p):
    q=F(round(float(p)*20),20)
    if not 0<=q<=1 or abs(float(q)-float(p))>1e-12:
        raise ValueError('Probabilidad fuera de la retícula1/20 del motor actual')
    return q

class Oraculo:
    def __init__(self,objetivo,tope,cpu_fraction=1):
        self.objetivo=objetivo;self.tope=tope;self.expandidos=0;self.hojas_desconocidas=0
        self.cache={};self.cpu_fraction=cpu_fraction;self.cpu0=time.process_time();self.wall0=time.monotonic()
    def limitar_cpu(self):
        if self.expandidos%32==0 and self.cpu_fraction<1:
            falta=(time.process_time()-self.cpu0)/self.cpu_fraction-(time.monotonic()-self.wall0)
            if falta>0:time.sleep(falta)
    def key(self,e,d):
        return (tuple(sorted(e.piezas('blancas'))),tuple(sorted(e.piezas('negras'))),
                e.puntos['blancas'],e.puntos['negras'],e.turno,d)
    def accion(self,e,j,d):
        branches=list(e.siguiente(j));weighted=[];total=F(0)
        for p,h in branches:
            q=probabilidad(p);total+=q
            lo,hi=self.valor(h,d-1);weighted.append((q*lo,q*hi))
        if total!=1:raise ValueError('Ramas no suman1')
        return (sum(x[0] for x in weighted),sum(x[1] for x in weighted))
    def valor(self,e,d):
        if e.terminado():
            winner=e.ganador()
            if winner not in ['blancas','negras']:raise ValueError('Terminal sin ganador')
            v=F(1 if winner==self.objetivo else -1);return(v,v)
        if d<=0 or self.expandidos>=self.tope:
            self.hojas_desconocidas+=1;return UNKNOWN
        key=self.key(e,d)
        if key in self.cache:return self.cache[key]
        self.expandidos+=1;self.limitar_cpu()
        actions=sorted(e.jugadas(),key=str);values=[];maximize=e.turno==self.objetivo
        for i,j in enumerate(actions):
            values.append(self.accion(e,j,d))
            if maximize and max(v[0] for v in values)==1:
                result=(F(1),F(1));break
            if not maximize and min(v[1] for v in values)==-1:
                result=(F(-1),F(-1));break
            if self.expandidos>=self.tope and i+1<len(actions):
                values.append(UNKNOWN);break
        else:
            result=None
        if 'result' not in locals() or result is None:
            op=max if maximize else min
            result=(op(v[0] for v in values),op(v[1] for v in values))
        if result[0]==result[1]:self.cache[key]=result
        return result

def analizar_caso(case,depth=3,node_budget=8000,cpu_fraction=.2):
    e=Estado.desde_dict(case['estado']);objective=case['color_propio']
    if e.turno!=objective or e.terminado():raise ValueError('Raíz propia no terminal requerida')
    moves=sorted(e.jugadas(),key=str);observed=Jugada.desde_texto(case['jugada_observada'])
    if observed not in moves:raise ValueError('Observada ilegal')
    per_action=max(1,node_budget//len(moves));bounds={};counts={}
    cpu0=time.process_time();wall0=time.monotonic()
    for j in moves:
        oracle=Oraculo(objective,per_action,cpu_fraction)
        bounds[str(j)]=oracle.accion(e,j,depth)
        counts[str(j)]=dict(expandidos=oracle.expandidos,desconocidos=oracle.hojas_desconocidas)
        if cpu_fraction<1:
            wait=(time.process_time()-cpu0)/cpu_fraction-(time.monotonic()-wall0)
            if wait>0:time.sleep(wait)
    chosen=bounds[str(observed)];better=[j for j,b in bounds.items() if b[0]>chosen[1]]
    return dict(id=case['id'],estado=case['estado'],observada=str(observed),profundidad=depth,
        tope_por_accion=per_action,cotas={j:[str(v) for v in b] for j,b in bounds.items()},contadores=counts,
        alternativas_demostrablemente_mejores=better,dominancia_demostrada=bool(better),
        limite='Cotas de juego completo a partir de expansión finita. Si no separan, no demuestra corrección, equivalencia ni valor óptimo de acciones.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True);p.add_argument('--profundidad',type=int,default=3)
    p.add_argument('--tope-nodos',type=int,default=8000);p.add_argument('--cpu-fraction',type=float,default=.2)
    a=p.parse_args()
    if a.profundidad<1 or a.tope_nodos<1 or not 0<a.cpu_fraction<=1:p.error('Presupuestos inválidos')
    a.salida.mkdir(parents=True,exist_ok=True)
    panel=[json.loads(s) for s in a.panel.read_text().splitlines()]
    cases=[c for c in panel if c['categoria_panel']=='tardios']
    if len(cases)!=8:raise ValueError('Protocolo requiere exactamente8tardíos del panel24')
    source_paths=[a.panel,Path(__file__),*[ROOT/'sistema/motor'/n for n in ['nucleo.py','estado.py','jugada.py']]]
    contract=dict(ids=[c['id'] for c in cases],depth=a.profundidad,budget=a.tope_nodos,
                  fuentes={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths})
    mf=a.salida/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=contract:raise ValueError('Contrato distinto')
    if not mf.exists():mf.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    results=[]
    for i,case in enumerate(cases):
        file=a.salida/f'caso_{i}.json'
        if file.exists():r=json.loads(file.read_text())
        else:
            r=analizar_caso(case,a.profundidad,a.tope_nodos,a.cpu_fraction)
            tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');tmp.replace(file)
        results.append(r);print(json.dumps(dict(id=r['id'],dominancia=r['dominancia_demostrada'])),flush=True)
    summary=dict(casos=len(results),dominancias=sum(r['dominancia_demostrada'] for r in results),
                 usa_modelo=False,test_nuevo_leido=False,limite='Panel histórico de diagnóstico; no prueba de fuerza ni nuevo dataset independiente.')
    (a.salida/'RESUMEN.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
