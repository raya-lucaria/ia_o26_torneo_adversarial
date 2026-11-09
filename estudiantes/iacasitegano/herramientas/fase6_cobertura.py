"""Cobertura de finales, replay acotado en CPU. No búsqueda/benchmark ni integración."""
from __future__ import annotations
import argparse,collections,csv,datetime,hashlib,importlib.util,json,math,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'resultados/medicion/fase6_cobertura_20261029'
CODEX=Path('/Users/josema/vs code/BreakthroughBotIA-codex/resultados/medicion')
LEGACY=Path('/Users/josema/vs code/BreakthroughBotIA-claude')
ANTI=Path('/Users/josema/vs code/BreakthroughBotIA-antigravity')
MDP=Path('/Users/josema/vs code/BreakthroughBotIA-mdp')
P='ceba365326e5ba8dd102218843f2b92744738d9c8a18db000f9edbc188832b7f'
A='cc888a3c7fc7ba896f5145c36c56c77120bc6f0ef0b0ba45f077023bbeb7a177'
U='83997dd323509bbfc2edd92de46a81a3770a6f1d5603daa730352ed56303277e'
W='37678d0bd700c27a29d0084558158db1a65b01409dbf54828e89706441c2250b'
N='ab1a599270424a40ff323dceb3aad3b2358a9ff87979f72db0721ad1fd5b3718'
NORMAL={'puntos','sin_jugadas'}
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def cmd(args):return subprocess.check_output(['rtk','proxy',*args],cwd=ROOT,text=True)
def inventory():
 assert not (OUT/'inventario.json').exists(),'Inventario ya congelado; usar analizar/resumir'
 refs=[];excluded=[];metadata={}
 def meta(p):
  p=Path(p);metadata[str(p.resolve())]={'sha256':sha(p),'bytes':p.stat().st_size};return json.loads(p.read_text())
 def add(path,pop,version,kind,status,source=None,weights=W,expected=None,evidence=None):
  p=Path(path).resolve();d=json.loads(p.read_text());h=sha(p)
  if expected:assert h==expected,(str(p),'hash mismatch')
  if d.get('version')!=1 or d.get('ganador') not in ['blancas','negras'] or not isinstance(d.get('jugadas'),list):
   excluded.append({'path':str(p),'reason':'JSON sin desenlace completo válido'});return
  own=[c for c in ['blancas','negras'] if d['meta'][c] in ['iacasitegano','profesor_offline','iacasitegano_anytime2_instrumentado']]
  assert len(own)==1,(p,own)
  refs.append({'path':str(p),'sha256':h,'bytes':p.stat().st_size,'population':pop,'version':version,'kind':kind,'status':status,
   'own_color':own[0],'opponent':d['meta']['negras' if own[0]=='blancas' else 'blancas'],'seed':d['meta']['semilla_maestra'],
   'pair':d['meta']['pareja'],'game':d['meta']['juego'],'moves':len(d['jugadas']),'reason':d['motivo'],
   'source_sha256':source,'weights_sha256':weights,'identity_evidence':str(evidence) if evidence else None,
   'source_identity_limit':None if source else 'Fuente histórica no fijada por SHA en esta traza; no inferirla del nombre actual'})
 baseline=meta(ROOT/'resultados/medicion/diagnostico_decisiones_20261028_baseline40/manifest.json')
 for x in baseline['fuentes']:add(Path(baseline['trazas'])/x['archivo'],'baseline_P_40','P','oficial','desarrollo',P,expected=x['sha256'],evidence=ROOT/'obsidian/Experimentos/2026-10-27-Analisis-baseline-40.md')
 aa=meta(CODEX/'aa_control_20261027_1806/progreso.json')
 for r in aa['runs']:
  for g in r['games']:add(g['file'],'AA_P_valido','P','oficial','desarrollo',P,expected=g['sha256'],evidence=CODEX/'aa_control_20261027_1806/progreso.json')
 ex=meta(CODEX/'aa_control_20261027_1806/intento_interrumpido_excluido.json')
 for p in sorted(Path(ex['folder']).glob('*.json')):add(p,'AA_P_excluido','P','oficial','excluida',P,evidence=CODEX/'aa_control_20261027_1806/intento_interrumpido_excluido.json')
 for name,base,labels in [('utilidad',CODEX/'utilidad_victoria_20261028_codex',{'A':('A',A,W),'B':('U',U,W)}),
   ('fase5',ROOT/'resultados/medicion/fase5_oficial_20261028',{'A':('U',U,W),'B':('N',U,N)})]:
  s=meta(base/'progreso.json');meta(base/'manifiesto.json')
  for r in s['runs']:
   v,source,weights=labels[r['variant']];status='excluida' if r['status']!='complete' else 'tecnica' if r['technical'] else 'desarrollo'
   for g in r['games']:add(g['path'],f'{name}_{v}_{status}',v,'oficial',status,source,weights,g['sha256'],base/'manifiesto.json')
 s=meta(ROOT/'resultados/medicion/random_principal_20261029/resumen.json')
 for x in s['canonical_json']:add(x['path'],'random_P_tecnica','P','oficial','tecnica',s['source_sha256'],s['model_sha256'],x['sha256'],ROOT/'resultados/medicion/random_principal_20261029/manifiesto.json')
 for p in sorted(Path('/Users/josema/.codex/worktrees/tt-cotas-exactas/BreakthroughBotIA/resultados/2026-10-27_22-47-44').glob('*.json')):
  add(p,'TT_A_tecnica','A','oficial','tecnica',A,evidence='/Users/josema/.codex/worktrees/tt-cotas-exactas/BreakthroughBotIA/resultados/medicion/tt_validacion_20261027/RESULTADOS.md')
 for p in sorted((LEGACY/'resultados/2026-10-26_13-59-15').glob('*.json')):
  add(p,'historico_detector_80','legacy_anytime2','oficial','historica',source=None,weights=None,evidence='/Users/josema/noche_salida/lab/verificacion_carrera_antigravity.md')
 for label,dirs,expected in [('privado60',[ROOT/'resultados/medicion/fase5_20261028/episodios_60'],60),
  ('piloto16',sorted((ROOT/'resultados/medicion/fase5_mac_k8s_20261029/piloto_mac').glob('worker_*')),16)]:
  total=0
  for folder in dirs:
   if not folder.is_dir():continue
   m=meta(folder/'manifest.json');prog=meta(folder/'progreso.json')
   assert prog.get('estado')=='completo',folder
   for p in sorted((folder/'partidas').glob('*.json')):
    add(p,f'{label}_U_profesor','U_profesor','privada','desarrollo_privado',m['buscador_sha256'],m['pesos_sha256'],evidence=folder/'manifest.json');total+=1
   excluded.extend({'path':str(p.resolve()),'reason':'checkpoint .parcial, excluido aunque exista JSON final'} for p in (folder/'partidas').glob('*.parcial'))
  assert total==expected,(label,total)
 canonical={}
 for r in refs:
  if r['sha256'] in canonical:canonical[r['sha256']]['aliases'].append(r)
  else:canonical[r['sha256']]={**r,'aliases':[]}
 detector=LEGACY/'estudiantes/iacasitegano_anytime2_instrumentado/carrera_check.py'
 paths=[Path(__file__),ROOT/'sistema/motor/estado.py',ROOT/'sistema/motor/nucleo.py',detector,ANTI/'carrera_antigravity.py',
  LEGACY/'estudiantes/iacasitegano/herramientas/finales_1v1.py',LEGACY/'estudiantes/iacasitegano/herramientas/finales_2v1.py',
  LEGACY/'estudiantes/iacasitegano/strategy.py',MDP/'estudiantes/iacasitegano/herramientas/generar_tabla_carreras.py',MDP/'estudiantes/iacasitegano/busqueda_eval3.py']
 paths += [LEGACY/'estudiantes/iacasitegano/data'/f'finales_{x}.npy' for x in ['1v1','2v1','1v2']]+[MDP/'estudiantes/iacasitegano/data/tabla_carreras.npy']
 paths += [ROOT/'estudiantes/iacasitegano'/x for x in ['strategy.py','busqueda_eval3.py','eval3.py','data/pesos_eval3_ligero.npz','data/libro.json'] if (ROOT/'estudiantes/iacasitegano'/x).exists()]
 sources={str(p.resolve()):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
 snap={'at':now(),'branch':cmd(['git','branch','--show-current']).strip(),'head':cmd(['git','rev-parse','HEAD']).strip(),
 'status':cmd(['git','status','--short','--branch']),'processes':cmd(['ps','-ww','-axo','pid,ppid,pcpu,stat,etime,command']),
 'sources':sources,'metadata':metadata,'protocol_sha256':sha(OUT/'PROTOCOLO.md'),'detector':str(detector),
 'excluded_live_root':str(ROOT/'resultados/medicion/fase5_lote256_mac_20261029'),
 'scope_limit':'Series nombradas en PROTOCOLO.md; no inventario exhaustivo de todos los prototipos históricos',
 'trace_counts':dict(collections.Counter(r['population'] for r in canonical.values()))}
 save(OUT/'manifiesto.json',snap);save(OUT/'inventario.json',{'at':now(),'traces':list(canonical.values()),'excluded_files':excluded})
 save(OUT/'progreso.json',{'status':'inventory_frozen','traces':len(canonical),'completed':0,'protocol_sha256':snap['protocol_sha256']})
 print(json.dumps(snap['trace_counts'],indent=2),flush=True)

class Budget:
 def __init__(self,ratio):self.ratio=ratio;self.wall=time.monotonic();self.cpu=time.process_time()
 def wait(self):
  target=(time.process_time()-self.cpu)/self.ratio-(time.monotonic()-self.wall)
  if target>0:time.sleep(min(target,0.5))

def engine():
 os.environ['NUMBA_DISABLE_JIT']='1';os.environ['NUMBA_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
 sys.path.insert(0,str(ROOT/'sistema'))
 from motor import Estado,Jugada
 return Estado,Jugada

def threats(e,color,Estado):
 idx=0 if color=='blancas' else 1
 if e.puntos[color]!=2:return {'any':False,'sure_third':False,'attack_chance':False,'max_p':0.0,'moves':[]}
 view=e if e.turno==color else Estado(e._tablero.copy(),idx,e._puntos.copy())
 if view.terminado():return {'any':False,'sure_third':False,'attack_chance':False,'max_p':0.0,'moves':[]}
 rows=[];goal=8 if idx==0 else 0
 for j in view.jugadas():
  if j.destino//9!=goal:continue
  children=view.siguiente(j)
  p=sum(prob for prob,s in children if s.puntos[color]==3)
  rows.append({'move':str(j),'attack':view.es_ataque(j),'p_third':p,'sure_third':all(s.puntos[color]==3 for _,s in children),
   'sure_game_win':all(s.ganador()==color for _,s in children),
   'branches':[{'p':prob,'points':s.puntos,'winner':s.ganador()} for prob,s in children]})
 return {'any':any(x['p_third']>0 for x in rows),'sure_third':any(x['sure_third'] for x in rows),
 'attack_chance':any(x['attack'] and 0<x['p_third']<1 for x in rows),'max_p':max([x['p_third'] for x in rows],default=0.0),'moves':rows}

def analyze(cpu_fraction):
 manifest=json.loads((OUT/'manifiesto.json').read_text());inv=json.loads((OUT/'inventario.json').read_text())
 assert sha(OUT/'PROTOCOLO.md')==manifest['protocol_sha256']
 for p,v in manifest['sources'].items():assert sha(p)==v['sha256'],p
 Estado,Jugada=engine();spec=importlib.util.spec_from_file_location('fase6_detector_historico',manifest['detector']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 budget=Budget(cpu_fraction);directory=OUT/'por_traza';directory.mkdir(exist_ok=True)
 done=0
 for ref in inv['traces']:
  target=directory/(ref['sha256']+'.json')
  if target.exists():done+=1;continue
  assert sha(ref['path'])==ref['sha256']
  d=json.loads(Path(ref['path']).read_text());e=Estado.inicial();records=[];errors=[]
  try:
   for ply,move in enumerate(d['jugadas'],1):
    assert move['n']==ply and move['color']==e.turno and not e.terminado()
    if e.turno==ref['own_color']:
     state=e.a_dict();nb=len(state['blancas']);nn=len(state['negras']);pt=state['puntos'];color=e.turno;other='negras' if color=='blancas' else 'blancas'
     own=threats(e,color,Estado);opp=threats(e,other,Estado)
     geometry=all(w//9>=b//9 or abs(w%9-b%9)>b//9-w//9 for w in [i for i,v in enumerate(e._tablero) if v==1] for b in [i for i,v in enumerate(e._tablero) if v==2])
     pure=bool(m.es_carrera_pura(e));info=move.get('info') or {}
     rec={'ply':move['n'],'state':state,'state_sha256':hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),
       'color':color,'white_pawns':nb,'black_pawns':nn,'total_pawns':nb+nn,'points_white':pt['blancas'],'points_black':pt['negras'],
       'distance_sum':sum(8-i//9 if v==1 else i//9 for i,v in enumerate(e._tablero) if v),
       'old_small_table':(nb,nn) in [(1,1),(2,1),(1,2)],'old_pure_race':pure,'geometry_only':geometry,
       'old_MDP_gate':pt['blancas']==pt['negras']==2 and nb>0 and nn>0,
       'expanded_le6':0<nb and 0<nn and nb+nn<=6,
       'third_own':own,'third_opponent_hypothetical':opp,
       'recorded_detector':info.get('carrera_pura'),'recorded_detector_available':'carrera_pura' in info,
       'old_small_field':info.get('final_exacto'), 'expanded_nontrivial_le6':0<nb and 0<nn and nb+nn<=6 and not own['sure_third']}
     records.append(rec)
    j=Jugada.desde_texto(move['jugada']);combat=move['combate'];attack=e.es_ataque(j)
    assert attack==(combat is not None)
    if attack:
     assert abs(e.prob_ataque(j)-combat['p'])<=1e-12
     assert isinstance(combat['gano'],bool)
     if 'u' in combat:assert combat['gano']==(combat['u']<combat['p'])
    e=e.aplicar(j,None if combat is None else combat['gano'])
    budget.wait()
   assert dict(e.puntos)==d['puntos']
   if d['motivo'] in NORMAL:assert e.ganador()==d['ganador']
   else:assert ref['status'] in ['excluida','tecnica'] or d['motivo'] not in NORMAL
  except Exception as ex:errors.append({'type':type(ex).__name__,'message':str(ex),'after_recorded_roots':len(records)})
  save(target,{'trace_sha256':ref['sha256'],'reference':ref,'replay_valid':not errors,'errors':errors,'decisions':records,
    'final_points':e.puntos,'final_winner_motor':e.ganador(),'adjudication':d['motivo'] not in NORMAL,
    'denominator':'pre-root of every own recorded legal move; no fabricated root for a failed response'})
  done+=1;save(OUT/'progreso.json',{'status':'replaying','completed':done,'traces':len(inv['traces']),'cpu_fraction_limit':cpu_fraction,'last_trace':ref['path']})
  if done%10==0:print(f'replay {done}/{len(inv["traces"])}',flush=True)
 save(OUT/'progreso.json',{'status':'replay_complete','completed':done,'traces':len(inv['traces']),'cpu_fraction_limit':cpu_fraction})
 print('REPLAY COMPLETE',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['inventario','analizar','resumir']);p.add_argument('--cpu-fraction',type=float,default=0.12);args=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 if args.command=='inventario':inventory()
 elif args.command=='analizar':analyze(args.cpu_fraction)
 else:
  from fase6_resumir import summarize
  summarize()

if __name__=='__main__':main()
