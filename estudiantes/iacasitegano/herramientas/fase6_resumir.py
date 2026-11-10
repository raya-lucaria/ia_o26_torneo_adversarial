"""Agregación de cobertura congelada, sin motor ni búsquedas."""
import collections,csv,datetime,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'resultados/medicion/fase6_cobertura_20261029'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def csvout(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize():
 inv=json.loads((OUT/'inventario.json').read_text());manifest=json.loads((OUT/'manifiesto.json').read_text())
 assert json.loads((OUT/'progreso.json').read_text())['status'] in ['replay_complete','closed_rejected','candidate_pending_correctness']
 pops=collections.defaultdict(list);invalid=[];allroots={};provenance=[];game_rows=[];recorded_conflicts=[]
 for ref in inv['traces']:
  p=OUT/'por_traza'/(ref['sha256']+'.json');d=json.loads(p.read_text())
  assert d['trace_sha256']==ref['sha256'] and sha(ref['path'])==ref['sha256']
  if not d['replay_valid']:invalid.append({'trace':ref,'errors':d['errors']});continue
  for row in d['decisions']:
   key=row['state_sha256'];allroots.setdefault(key,row)
   provenance.append({'state_sha256':key,'trace_sha256':ref['sha256'],'path':ref['path'],'population':ref['population'],'version':ref['version'],'kind':ref['kind'],'status':ref['status'],'ply':row['ply'],'color':row['color']})
   pops[ref['population']].append((ref,row))
   if row['recorded_detector_available'] and bool(row['recorded_detector'])!=row['old_pure_race']:
    recorded_conflicts.append({'path':ref['path'],'ply':row['ply'],'recorded':row['recorded_detector'],'replayed':row['old_pure_race']})
  game_rows.append({'sha256':ref['sha256'],'population':ref['population'],'version':ref['version'],'status':ref['status'],'kind':ref['kind'],'color':ref['own_color'],
   'decisions':len(d['decisions']),'natural_terminal':not d['adjudication'],'adjudicated':d['adjudication'],
   'old_small_hits':sum(x['old_small_table'] for x in d['decisions']),'old_race_hits':sum(x['old_pure_race'] for x in d['decisions']),
   'expanded_le6_hits':sum(x['expanded_nontrivial_le6'] for x in d['decisions'])})
 assert not recorded_conflicts,recorded_conflicts
 summaries={};rows=[];distributions=[];bycolor=[]
 keys=['old_small_table','old_pure_race','geometry_only','old_MDP_gate','expanded_le6','expanded_nontrivial_le6']
 def aggregate(items):
  n=len(items);unique={x['state_sha256']:x for _,x in items}
  counts={k:sum(bool(x[k]) for _,x in items) for k in keys}
  counts.update(third_own_any=sum(x['third_own']['any'] for _,x in items),third_own_sure=sum(x['third_own']['sure_third'] for _,x in items),
   third_own_probabilistic_attack=sum(x['third_own']['attack_chance'] for _,x in items),
   third_opponent_any=sum(x['third_opponent_hypothetical']['any'] for _,x in items),
   third_opponent_sure=sum(x['third_opponent_hypothetical']['sure_third'] for _,x in items),
   third_opponent_probabilistic_attack=sum(x['third_opponent_hypothetical']['attack_chance'] for _,x in items),
   recorded_detector_available=sum(x['recorded_detector_available'] for _,x in items),
   recorded_detector_true=sum(bool(x['recorded_detector']) for _,x in items if x['recorded_detector_available']))
  return {'decisions':n,'unique_states':len(unique),'traces':len({r['sha256'] for r,_ in items}),
   'counts':counts,'unique_counts':{k:sum(x[k] for x in unique.values()) for k in keys},
   'min_material':min((x['total_pawns'] for _,x in items),default=None),
   'max_material':max((x['total_pawns'] for _,x in items),default=None),
   'zero_coverage_is_descriptive_not_binomial_independence':True,
   'hypothetical_root_coverage_only':True}
 for pop in sorted(pops):
  items=pops[pop];s=aggregate(items);summaries[pop]=s
  rows.append({'population':pop,'traces':s['traces'],'decisions':s['decisions'],'unique_states':s['unique_states'],'min_material':s['min_material'],'max_material':s['max_material'],**s['counts']})
  for color in ['blancas','negras']:
   c=aggregate([(r,x) for r,x in items if x['color']==color]);bycolor.append({'population':pop,'color':color,'decisions':c['decisions'],'unique_states':c['unique_states'],**c['counts']})
  for dimension in ['white_pawns','black_pawns','total_pawns','points_white','points_black','ply','color','distance_sum']:
   for val,cnt in sorted(collections.Counter(str(x[dimension]) for _,x in items).items()):distributions.append({'population':pop,'dimension':dimension,'value':val,'decisions':cnt})
  for cap in [2,3,4,6,8]:
   cnt=sum(x['total_pawns']<=cap for _,x in items);distributions.append({'population':pop,'dimension':'total_pawns_le','value':str(cap),'decisions':cnt})
  for cap in [6,8,12,16,24]:
   cnt=sum(x['distance_sum']<=cap for _,x in items);distributions.append({'population':pop,'dimension':'distance_sum_le','value':str(cap),'decisions':cnt})
 historical=summaries.get('historico_detector_80',{})
 histmatch=historical.get('decisions')==3323 and historical.get('counts',{}).get('recorded_detector_true')==0
 eligible=[]
 for pop,s in summaries.items():
  refs=[r for r,x in pops[pop]];relevant=all(r['version'] in ['P','A','U','U_profesor'] and r['status'] not in ['excluida','tecnica'] for r in refs)
  hits=s['counts']['expanded_nontrivial_le6'];ngames=len({r['sha256'] for r,x in pops[pop] if x['expanded_nontrivial_le6']})
  if relevant and s['decisions'] and hits/s['decisions']>=0.01 and ngames>=5:eligible.append({'population':pop,'hits':hits,'games':ngames})
 rejected=not eligible
 runtime={p:v for p,v in manifest['sources'].items() if '/BreakthroughBotIA/estudiantes/iacasitegano/' in p and '/herramientas/' not in p}
 changed=[p for p,v in runtime.items() if sha(p)!=v['sha256']]
 assert not changed,changed
 result={'at':datetime.datetime.now().astimezone().isoformat(),'trace_count':len(inv['traces']),'valid_replays':len(game_rows),'invalid_replays':invalid,
  'populations':summaries,'total_recorded_own_decisions':len(provenance),'unique_root_states':len(allroots),'historical_3323_reproduced':histmatch,
  'detector_recorded_vs_replay_conflicts':recorded_conflicts,'expanded_domain':'full board <=6 total pawns, both sides present, with turn/points; terminal descendants included',
  'expanded_frequency_filter':{'minimum_fraction':0.01,'minimum_complete_games':5,'discount_already_sure_third_win':True,'eligible_populations':eligible},
  'decision':'reject_new_table_for_coverage' if rejected else 'candidate_pending_correctness',
  'old_detector_math_root_hits':sum(s['counts']['old_pure_race'] for s in summaries.values()),
  'old_small_table_root_hits':sum(s['counts']['old_small_table'] for s in summaries.values()),
  'principal_has_no_solver_hook':True,'internal_leaf_coverage':'sin medir','runtime_changed':changed,
  'new_games':0,'new_benchmarks':0,'new_solver_built':False,'resource_caps_if_future_solver':{'table_max_bytes':10*1024*1024,'max_states_per_query':250000,'exhaustion':'unknown'},
  'source_hash':sha(ROOT/'estudiantes/iacasitegano/busqueda_eval3.py'),'weights_hash':sha(ROOT/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz'),
  'limits':['Only own roots with recorded moves; failed responses without a move are not fabricated.',
   'Opponent threat assumes opponent to move on same board; not a forced loss after own reply.',
   'Probabilistic attacks are not sure wins; observed outcomes are not action labels.',
   'Per-population exposures and unique-state counts are descriptive; games and states are dependent.',
   'No leaf activation instrumentation; no league/clock/preparation/force claim.',
   'Live 256-episode generation excluded regardless of later completion; other legacy prototypes not exhaustive.']}
 dump(OUT/'cobertura.json',result);csvout('cobertura.csv',rows);csvout('cobertura_por_color.csv',bycolor);csvout('distribuciones.csv',distributions);csvout('trazas_replay.csv',game_rows)
 with (OUT/'procedencia.jsonl').open('w') as f:
  for x in provenance:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 with (OUT/'estados_unicos.jsonl').open('w') as f:
  for key,x in sorted(allroots.items()):f.write(json.dumps({'state_sha256':key,**x},ensure_ascii=False)+'\n')
 dump(OUT/'progreso.json',{'status':'closed_rejected' if rejected else 'candidate_pending_correctness','completed':len(game_rows),'invalid':len(invalid),'traces':len(inv['traces']),'cpu_benchmark_performed':False})
 print(json.dumps({'valid_replays':len(game_rows),'invalid':len(invalid),'decisions':len(provenance),'unique':len(allroots),
  'historical_3323':histmatch,'old_hits':result['old_detector_math_root_hits'],'small_hits':result['old_small_table_root_hits'],'decision':result['decision'],'eligible':eligible},indent=2))

if __name__=='__main__':summarize()
