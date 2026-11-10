"""Certificado descriptivo de fase6. No construye tablas ni mide rendimiento."""
import ast,collections,csv,datetime,hashlib,importlib.util,json,math,struct,sys,time
from pathlib import Path
import fase6_cobertura as coverage
R=coverage.ROOT;O=coverage.OUT
def dump(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def link(text,path):return f'[{text}](<{Path(path).resolve()}>)'
def npy_header(path):
 with Path(path).open('rb') as f:
  assert f.read(6)==b'\x93NUMPY';version=tuple(f.read(2));n=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
  h=ast.literal_eval(f.read(n).decode('latin1').strip())
 return {'path':str(path),'sha256':coverage.sha(path),'bytes':Path(path).stat().st_size,'shape':h['shape'],'dtype':h['descr'],'numpy_file_version':version}

def main():
 result=json.loads((O/'cobertura.json').read_text());inv=json.loads((O/'inventario.json').read_text());m=json.loads((O/'manifiesto.json').read_text())
 assert result['valid_replays']==890 and not result['invalid_replays'] and not result['expanded_frequency_filter']['eligible_populations']
 assert result['old_small_table_root_hits']==0 and result['decision']=='reject_new_table_for_coverage'
 assert all(coverage.sha(p)==v['sha256'] for p,v in m['sources'].items()),'Frozen source changed'
 Estado,Jugada=coverage.engine()
 spec=importlib.util.spec_from_file_location('fase6_legacy_check',m['detector']);detector=importlib.util.module_from_spec(spec);spec.loader.exec_module(detector)
 fixtures=[]
 # Analytic expectations independent of the coverage aggregation/denominators.
 for color in ['blancas','negras']:
  e=Estado.desde_dict({'blancas':['a8'] if color=='blancas' else ['a3'],
    'negras':['i7'] if color=='blancas' else ['i2'],'turno':color,
    'puntos':{'blancas':2 if color=='blancas' else 0,'negras':2 if color=='negras' else 0}})
  v=coverage.threats(e,color,Estado);assert v['any'] and v['sure_third'] and v['max_p']==1 and not v['attack_chance']
  j=next(j for j in e.jugadas() if j.destino//9==(8 if color=='blancas' else 0));child=e.siguiente(j)[0][1]
  assert child.puntos[color]==3 and child._tablero[j.origen]==child._tablero[j.destino]==0
  fixtures.append({'case':color+'_quiet_third_and_removal','ok':True})
  time.sleep(.05)
 e=Estado.desde_dict({'blancas':['a8','e4'],'negras':['b9','i7'],'turno':'blancas','puntos':{'blancas':2,'negras':0}})
 v=coverage.threats(e,'blancas',Estado);attack=next(x for x in v['moves'] if x['move']=='a8-b9')
 assert attack['attack'] and attack['p_third']==0.5 and not attack['sure_third']
 children=e.siguiente(Jugada.desde_texto('a8-b9'));assert [c.puntos['blancas'] for _,c in children]==[3,2]
 assert children[1][1]._tablero[Jugada.desde_texto('a8-b9').destino]==2
 fixtures.append({'case':'attack_third_two_distinct_outcomes_half_probability','ok':True})
 e=Estado.desde_dict({'blancas':['a8','e4'],'negras':['b9','a9','i7'],'turno':'blancas','puntos':{'blancas':2,'negras':0}})
 v=coverage.threats(e,'blancas',Estado);assert v['any'] and v['attack_chance'] and not v['sure_third'] and v['max_p']<1
 fixtures.append({'case':'probabilistic_only_is_not_sure_win','ok':True})
 e=Estado.desde_dict({'blancas':['a8'],'negras':['i7'],'turno':'blancas','puntos':{'blancas':1,'negras':0}})
 assert not coverage.threats(e,'blancas',Estado)['any'];fixtures.append({'case':'second_point_is_not_third','ok':True})
 e=Estado.desde_dict({'blancas':['a3'],'negras':['i7'],'turno':'blancas','puntos':{'blancas':3,'negras':0}})
 assert e.ganador()=='blancas' and not detector.es_carrera_pura(e);fixtures.append({'case':'terminal_rejected_by_detector','ok':True})
 e=Estado.desde_dict({'blancas':[],'negras':['i7'],'turno':'blancas'});assert e.ganador()=='negras' and not detector.es_carrera_pura(e)
 fixtures.append({'case':'no_moves_terminal_is_loss_for_side_to_move','ok':True})
 e=Estado.desde_dict({'blancas':['a8'],'negras':['i2'],'turno':'blancas','puntos':{'blancas':2,'negras':2}})
 assert detector.es_carrera_pura(e);fixtures.append({'case':'geometric_race_positive_with_enough_material','ok':True})
 e=Estado.desde_dict({'blancas':['a8'],'negras':['i2'],'turno':'blancas'});assert not detector.es_carrera_pura(e)
 fixtures.append({'case':'same_geometry_insufficient_material_rejected','ok':True})
 dump(O/'REGRESIONES_COBERTURA.json',{'at':coverage.now(),'JIT_disabled':True,'solver_constructed':False,'cases':fixtures,
   'motor_sha256':coverage.sha(R/'sistema/motor/nucleo.py'),'tests_are_not_timing_measurements':True})

 hooks={'es_carrera_pura','resolver_carrera','_consultar_carrera_reducida','_valor_final','_tabla_de_final','cargar_tabla_carreras'}
 audit=[]
 for name in ['strategy.py','busqueda_eval3.py','eval3.py']:
  p=R/'estudiantes/iacasitegano'/name;t=ast.parse(p.read_text())
  found=sorted({n.id for n in ast.walk(t) if isinstance(n,ast.Name) and n.id in hooks}|{n.attr for n in ast.walk(t) if isinstance(n,ast.Attribute) and n.attr in hooks})
  imports=[ast.unparse(n) for n in ast.walk(t) if isinstance(n,(ast.Import,ast.ImportFrom))]
  audit.append({'path':str(p),'sha256':coverage.sha(p),'legacy_hooks':found,'imports':imports})
 assert not any(x['legacy_hooks'] for x in audit)
 strategy=(R/'estudiantes/iacasitegano/strategy.py').read_text()
 assert 'all(hijo.ganador() == color for _, hijo in estado.siguiente(j))' in strategy
 dump(O/'conexion_actual.json',{'files':audit,'legacy_solver_connected':False,'internal_leaves_sampled':False,
  'existing_immediate_safe_win_guard':True,'declared_actual_solver_calls':'none by current source, not a sampled leaf activation rate'})

 race=[];bookpath=R/'estudiantes/iacasitegano/data/libro.json';book=json.loads(bookpath.read_text()) if bookpath.exists() else {}
 for ref in inv['traces']:
  data=json.loads((O/'por_traza'/(ref['sha256']+'.json')).read_text())
  for row in data['decisions']:
   if not row['old_pure_race']:continue
   e=Estado.desde_dict(row['state']);entry=book.get(str(e.clave));book_hit=entry is not None and Jugada(*entry) in e.jugadas()
   assert row['third_own']['sure_third'] and not book_hit
   race.append({'trace':ref['path'],'population':ref['population'],'state_sha256':row['state_sha256'],'ply':row['ply'],
    'state':row['state'],'own_safe_third':True,'current_book_override':False,'winning_moves':[x for x in row['third_own']['moves'] if x['sure_third']]})
 assert len(race)==25
 dump(O/'carreras_puras_observadas.json',{'positions':25,'traces':len({x['trace'] for x in race}),
  'already_covered_by_current_safe_win_guard':25,'current_book_overrides':0,'records':race,
  'limit':'Hypothetical P policy at these private-professor roots; not a counterfactual full-game result'})

 tables=[]
 for shape in ['1v1','2v1','1v2']:tables.append(npy_header(coverage.LEGACY/f'estudiantes/iacasitegano/data/finales_{shape}.npy'))
 tables.append(npy_header(coverage.MDP/'estudiantes/iacasitegano/data/tabla_carreras.npy'))
 domains={'at':coverage.now(),'tables':tables,
  'exact_old_domain':{'exported_root_material':[[1,1],[2,1],[1,2]],'state_key':'all pawn squares, turn, both points; no shared squares or pawn left on own goal row',
   'table_entries':'valid nonterminal points 0..2 each, both turns; float32 probability white wins, sentinel -1 outside valid entries',
   'mathematical_solver':'untruncated memoized recursion over full board; points terminal first, no-legal-move loss, real dynamic support, both combat branches, scoring pawn removed',
   'scope':'exported tables restricted by material; underlying recursive function accepts more pieces but was not invoked here'},
  'pure_detector':{'source':m['detector'],'terminal':False,'material_guard':'n_white>=3-p_white AND n_black>=3-p_black',
   'geometry':'for every white/black pair: rw>=rb OR abs(cw-cb)>rb-rw; conservative exclusion of future interaction',
   'turn_points':'turn sets race tie winner; both points determine how many pawns must score',
   'resolver':'sum smallest 3-p distances per side; equal sums favor current turn; no attacks in admitted geometry',
   'known_limit':'historical 1v1 exhaustive audit found 72 conservative false negatives for frontal block; historical proof scope not all material'},
  'old_MDP':{'gate_nonterminal':'both scores 2, at least one pawn per side; choose most advanced racers and current support',
   'math_domain':'synthetic two-racer board with fixed support parameters 0/1/2; real turn/points/terminal/block rules in reduced model',
   'not_full_board_exact':'Extra pawns change support, blocking and future actions; gate frequency is not exact solver coverage',
   'status':'previously rejected, not revived or integrated'},
  'new_full_board_domain_proposed_before_replay':result['expanded_domain'],
  'observed_frequency':0,'reason_no_build':'No root with <=6 pieces; minimum even in private episodes is 11; no nontrivial opportunity passes the published filter.',
  'cost_not_benchmarked':True,'caps':result['resource_caps_if_future_solver'],
  'dense_float32_2v2_conservative_lower_bound_bytes':math.comb(36,2)**2*18*4,
  'lower_bound_limit':'Only a disjoint-half-board subset already exceeds 10MiB for dense float32; not an impossibility theorem for sparse/compressed representations'}
 dump(O/'dominios.json',domains)
 domain_text='''# Dominios históricos y alcance real

## Tablas 1v1, 2v1 y 1v2

Los archivos/sizes/SHAs y encabezados NPY reales están en dominios.json. Los generadores se conservan en el worktree BreakthroughBotIA-claude; no están conectados al principal actual. Exportan casillas de todos los peones, turno, puntos 0..2 y P(blancas ganan), con sentinel -1 para entradas inválidas. La recursión matemática usa el tablero completo: puntos terminales primero, derrota por falta de jugadas, ambas ramas con probabilidad real y apoyo recalculado, retirada del peón al anotar. Es un árbol completo a precisión numérica; la tabla float32 introduce su redondeo, no se afirma aritmética racional exacta. No se reconstruyeron ni revalidaron exhaustivamente ahora.

La guardia antigua de consulta en strategy era exactamente material (1,1)/(2,1)/(1,2), tras libro/forzadas/victoria inmediata. Un solver genérico capaz de aceptar cualquier tablero no significa una tabla exportada para material arbitrario. El antiguo worktree de fase4 también pasaba 1v1 a una feature de búsqueda; eso no existe en el P activo ni mide cobertura de sus hojas.

## Detector de carrera pura

Se ejecutó la copia histórica carrera_check.py sin modificarla. Rechaza terminales y material insuficiente para que AMBOS lados completen sus 3-puntos; exige conos de todos los pares rivales disjuntos (o que ya se hayan rebasado). La geometría sola se publica por separado: no equivale al detector completo. El resolver analítico suma las menores distancias de los 3-p peones necesarios y el turno deshace empates. El informe histórico aporta pruebas finitas y 72 falsos negativos conservadores de bloqueo frontal en 1v1; no se amplía esa prueba a todo tablero.

## MDP reducida histórica

El gatillo de ambos puntos=2 con un racer avanzado por lado puede aparecer con mucho material. Su tabla mantiene el apoyo FIJO como parámetro y omite otras acciones/piezas: no es un final exacto del tablero completo con escoltas. Contar ese gatillo no autoriza reutilizar sus valores como exactos. Continúa rechazada y desconectada; no se reabre.

## Única ampliación contemplada y decisión

Tablero completo con hasta seis peones, turno/puntos y terminales completos. Se fijó antes del replay un mínimo práctico de 1% de raíces y cinco partidas relevantes, descontando victorias seguras inmediatas. No aparece ninguna posición con <=6 peones, ni siquiera <=8: mínimo global11. No se construye tabla ni solver. Como límite descriptivo de costo, solo las configuraciones 2v2 separadas en dos mitades de36casillas ya ocupan28,576,800bytes en una tabla densa float32 (18combinaciones de turno/puntos), mayor que10MiB. No excluye compresión ni un dominio diferente futuro.

Las25carreras puras privadas observadas tienen victoria segura inmediata y ninguna entrada del libro actual que la preceda. La guardia actual strategy.jugar ya enumera esa jugada; un solver de finales no añade una decisión nueva en esas raíces. Esto no simula el resultado deportivo de P desde las trayectorias privadas ni permite extrapolar a hojas internas o rivales no observados.
'''
 (O/'DOMINIOS.md').write_text(domain_text)

 pops=result['populations'];table=[];threattable=[]
 for pop,s in pops.items():
  c=s['counts'];table.append(f"| {pop} | {s['traces']} | {s['decisions']} | {s['unique_states']} | {s['min_material']} | {c['old_small_table']} | {c['old_pure_race']} | {c['expanded_nontrivial_le6']} |")
  threattable.append(f"| {pop} | {s['decisions']} | {c['third_own_any']} | {c['third_own_sure']} | {c['third_own_any']-c['third_own_sure']} | {c['third_opponent_any']} | {c['third_opponent_sure']} |")
 handoff=('Fase 6 cerrada por rechazo de integración en este alcance: 890 trazas completas y 36,730 raíces propias '
  '(35,094 estados únicos), incluidas las 16 del piloto Mac, con 0 errores de replay. Tabla 1v1/2v1/1v2: 0 oportunidades; '
  'detector histórico 0/3323 reproducido y 0 en poblaciones oficiales; 25 carreras puras privadas, todas con victoria segura inmediata ya cubierta por P. '
  'Dominio completo <=6 peones: 0 raíces (mínimo global 11), no se construye solver ni se integra nada. Cobertura de hojas sin medir; '
  'sin benchmark/liga mientras fase 5 genera, sin cambios a P/U/N ni corrección TT de fase 8. Evidencia: resultados/medicion/fase6_cobertura_20261029/CIERRE.md.')
 excluded_count=sum(r['status']=='excluida' for r in inv['traces'])
 text=f'''# Fase 6 — cobertura y decisión sobre finales

**Cerrado: rechazo de integrar las tablas históricas y de construir la ampliación examinada por falta de oportunidad adicional en las raíces observadas.** No es una mejora aceptada ni un rechazo de todos los finales posibles. No hay solver nuevo ni candidato deportivo pendiente.

## Pregunta y población

¿Aparecen los dominios de finales históricos, o un dominio exacto de material algo mayor, en decisiones reales? Se congelaron890JSON finales, por referencia y SHA-256: oficiales bajo límites del torneo, privados sin reloj oficial, técnicas y partidas completas de bloques excluidos. Las categorías y versiones no se mezclan para fuerza ni se calculan victorias agregadas. Fueron reproducidas890/890sin errores, desde el inicio y a través de todas las jugadas con el motor actual sin cambios y JIT desactivado. Conteo:36,730decisiones propias con jugada registrada,35,094estados completos únicos; todas las procedencias conservadas. Estado final no sustituye decisiones.

Se incluyeron los16episodios finales completos del piloto Mac (1495plies,750decisiones del profesor), no su resumen de la primera sesión. Se excluyeron76archivos .parcial. El lote activo fase5_lote256_mac_20261029 no se leyó ni se incorpora aunque termine después. El marco de series queda fijado en PROTOCOLO.md; no es un censo de cada prototipo antiguo del disco. Las réplicas comparten posiciones y grupos: son frecuencias descriptivas, no ensayos independientes.

## Identidad y dominios

Principal P: búsqueda `{result['source_hash']}`, NPZ `{result['weights_hash']}`, rama `{m['branch']}`, HEAD `{m['head']}`. P conserva su código original: corrección TT independiente de A pendiente de fase8. Inventario separa P, A/logits-TT corregida, U/utilidad, N/NPZ rechazado, profesor privado U y legacy_anytime2. Las atribuciones históricas P de baseline/A-A usan commit/documentos de esas series; el histórico anytime2 no tiene un SHA contemporáneo de participante en cada traza y se marca desconocido, sin inventarlo. Hashes de U/N proceden de manifiestos cerrados.

{link('DOMINIOS.md',O/'DOMINIOS.md')} y {link('dominios.json',O/'dominios.json')} especifican turno/puntos, material, ataques, bloqueo, precisión y tamaños reales. Las tablas se localizaron en BreakthroughBotIA-claude y la MDP en BreakthroughBotIA-mdp; no se cambiaron. El detector es la copia histórica que se instrumentó. El análisis estático de estrategia/búsqueda/evaluación actuales no encuentra conexiones a esos solvers: {link('conexion_actual.json',O/'conexion_actual.json')}.

## Cobertura por población

Numeradores sobre decisiones propias registradas. Small=guardia1v1/2v1/1v2; pura=detector completo; <=6=ampliación no trivial descontando victoria segura inmediata. Ceros explícitos.

| Población | Trazas | Decisiones | Estados únicos | Mín. peones | Small | Pura | <=6no trivial |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

**Tabla pequeña:0/36,730. Dominio<=6:0/36,730; incluso<=8:0.** Mínimo global11peones (privado60), mínimo del piloto14. La guardia geométrica sin material y la guardia aproximada2–2 de MDP se publican en JSON/CSV; no son cobertura exacta de una tabla completa.

Se reproduce el histórico:3323decisiones/80partidas, campo carrera_pura presente en3323y0positivos, sin conflictos con replay. Las poblaciones oficiales observadas tienen0detecciones de carrera pura. Las25positivas están solo en privados:24/3222en episodios60y1/750en piloto16, repartidas en6partidas; **25/25ya tienen jugada segura del tercer punto**, sin entrada legal del libro que anule la guardia actual. {link('Casos completos y transiciones',O/'carreras_puras_observadas.json')}. Son oportunidades matemáticas del detector, no uso real de solver ni mejora nueva para P. No se ocultan para repetir una afirmación universal de cero.

## Amenazas inmediatas del tercer punto

Se enumera la lista legal de jugadas a meta con dos puntos y Estado.siguiente evalúa ambas ramas. Una anotación probable por ataque no se cuenta como segura. Rival significa turno hipotético del rival sobre el mismo tablero: no es derrota forzada después de responder. No se evalúan acciones por el dado futuro ni se usan fallos como etiquetas. Las columnas “solo probable” y “segura” son disjuntas; “posible” es su suma.

| Población | Denominador | Propio posible | Propio seguro | Propio solo probable | Rival posible | Rival seguro |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(threattable)}

Distribuciones completas de peones por lado/total, puntos, ply, color y distancia agregada: distribuciones.csv. Numeradores por color y población: cobertura_por_color.csv. Los archivos por_traza y estados_unicos/procedencia permiten auditar cada conteo. Las oportunidades propias inmediatas ya están cubiertas por la guardia existente de strategy; no se atribuye beneficio deportivo por contarlas.

## Decisión de recursos y corrección

La única ampliación de material prefijada (tablero completo<=6, turno/puntos, terminales) no supera su filtro de oportunidad:0raíces frente a mínimo1%y5partidas relevantes. **No construir**. No se ejecutó un solver grande, un benchmark, liga o entrenamiento. El límite de10MiB/250000estados por consulta permanece como contrato para una futura hipótesis justificada; agotar presupuesto debe devolver desconocido, nunca exacto. Ni un apoyo congelado ni una búsqueda truncada se consideran solución del tablero real.

Los generadores/solvers históricos fueron leídos y sus fuentes/tablas/encabezados se registran por SHA. Sus verificaciones exhaustivas anteriores son evidencia histórica referenciada; no se repitieron aquí. La cobertura usa el detector real y890replays completos verificando legalidad, turno, probabilidades, resultados registrados y puntos finales. Además9regresiones analíticas ligeras sin JIT comprueban tercer punto por ambos colores, retirada del anotador, dos ramas de ataque0.5, amenaza solo probabilística, segundo punto, terminales/bloqueo y guardia de material del detector: {link('REGRESIONES_COBERTURA.json',O/'REGRESIONES_COBERTURA.json')}. No constituyen un solver nuevo ni tiempos de rendimiento.

## Límites y pendientes

- Cobertura medida solo en raíces propias; activación/frecuencia de hojas internas: **sin medir**. No se extrapola desde el estado final. El principal no tiene hook de solver por inspección de fuente, no porque se haya observado una tasa de consultas interna.
- Privados profesor/U y rivales estilizados no son P a reloj oficial. Sus25estados podrían decidirse con la guardia de P sobre ese tablero; no se simula la partida contrafactual ni se promete fuerza frente a tiers privados.
- Hay {excluded_count} trazas completas de bloques excluidos (20 A/A + 59 utilidad + 23 fase 5); las categorías técnicas se mantienen separadas. El inventario/CSV identifica cada población. Las exclusiones no se rescatan como fuerza. Solo existen decisiones con jugada registrada: respuestas fallidas sin jugada no se inventan. Ningún replay produjo error.
- Memoria pico/preparación/reloj de un integrado nuevo y rendimiento: **sin medir**, porque no se integró ni se construyó candidato. El RSS o tiempo de este análisis no se presenta como métrica del bot.
- No se requiere ventana para este rechazo descriptivo. Una hipótesis futura con frecuencia útil debe congelar dominio/oráculo y pedir ventana exclusiva antes de medir tiempo o jugar. La generación activa de fase5 no se detuvo ni se modificó.
- No hay trabajo obligatorio pendiente para la decisión de esta tabla/dominio. Mantener finales fuera del principal; otras ampliaciones o instrumentación de hojas requieren otro encargo. Fase5 y corrección TT de fase8 continúan por sus contratos, sin reabrir fase2 o N.

## Evidencia, recuperación y preservación

{link('Protocolo previo',O/'PROTOCOLO.md')} · {link('Inventario congelado',O/'inventario.json')} · {link('Manifiesto de fuentes y versiones',O/'manifiesto.json')} · {link('Cobertura JSON',O/'cobertura.json')} · {link('Cobertura CSV',O/'cobertura.csv')} · {link('Procedencia',O/'procedencia.jsonl')}. Código únicamente en herramientas/fase6_cobertura.py, fase6_resumir.py y fase6_cerrar.py; el último genera este certificado sin modificar el análisis congelado.

Reproducir desde la raíz: `rtk proxy nice -n 19 .venv/bin/python -B estudiantes/iacasitegano/herramientas/fase6_cobertura.py analizar --cpu-fraction 0.12` reutiliza checkpoints sin sumar; `... fase6_cobertura.py resumir` recalcula CSV/JSON; `... fase6_cerrar.py` regenera el cierre. El inventario no se amplía. CPU limitada para convivencia, sin JIT; los contadores de CPU se usaron solo para limitar carga, no como benchmark.

Fuentes de P, evaluación, estrategia, pesos, libro y motor conservan sus hashes iniciales. No commit/merge/push/promoción ni port de TT; sin alterar fases2/5 ni contenedores ajenos. Solo se crea nota específica de fase6, no se editan 01-Estado/03-Bitacora/07-Plan-definitivo.

## Párrafo de traspaso

{handoff}
'''
 replacements={'congelaron890JSON':'congelaron 890 JSON','reproducidas890/890sin':'reproducidas 890/890 sin','Conteo:36,730decisiones':'Conteo: 36,730 decisiones',
  'registrada,35,094':'registrada, 35,094','los16episodios':'los 16 episodios','excluyeron76archivos':'excluyeron 76 archivos',
  'plies,750':'plies, 750','Tabla pequeña:0':'Tabla pequeña: 0','Dominio<=6:0':'Dominio <=6: 0','incluso<=8:0':'incluso <=8: 0',
  'histórico:3323':'histórico: 3323','tienen0detecciones':'tienen 0 detecciones','privados:24':'privados: 24',
  '35,094estados':'35,094 estados','16episodios':'16 episodios','1495plies':'1495 plies','750decisiones':'750 decisiones',
  '76archivos':'76 archivos','Small=guardia1v1/2v1/1v2':'Small = guardia 1v1/2v1/1v2',
  '3323decisiones/80partidas':'3323 decisiones / 80 partidas','presente en3323y0positivos':'presente en 3323 y 0 positivos',
  'Las25positivas':'Las 25 positivas','24/3222en':'24/3222 en','1/750en':'1/750 en','en episodios60y':'en episodios 60 y',
  'en piloto16':'en piloto 16','en6partidas':'en 6 partidas','25/25ya':'25/25 ya','global11peones':'global 11 peones',
  'piloto14':'piloto 14','<=6no trivial':'<=6 no trivial','guardia aproximada2–2':'guardia aproximada 2–2',
  'nunca un racer':'nunca un racer','0raíces':'0 raíces','mínimo1%y5partidas':'mínimo 1% y 5 partidas',
  'de10MiB/250000estados':'de 10 MiB / 250000 estados','y890replays':'y 890 replays','Además9regresiones':'Además 9 regresiones',
  'ataque0.5':'ataque 0.5','Sus25estados':'Sus 25 estados','a fase8':'a fase 8','fases2/5':'fases 2/5','de fase5':'de fase 5'}
 for a,b in replacements.items():text=text.replace(a,b)
 (O/'CIERRE.md').write_text(text)
 final={'at':coverage.now(),'decision':'closed_rejected_coverage','traces':890,'decisions':36730,'unique_states':35094,
  'all_replays_valid':True,'roots_small_table':0,'roots_full_le6':0,'roots_pure_detector':25,'pure_roots_already_safe_third':25,
  'domain_oracle_new_not_needed':True,'coverage_sanity_regressions':len(fixtures),'search_leaf_coverage':'unmeasured',
  'protocol_unchanged':coverage.sha(O/'PROTOCOLO.md')==m['protocol_sha256'],
  'original_sources_unchanged':{p:coverage.sha(p)==v['sha256'] for p,v in m['sources'].items()},
  'closing_tool_sha256':coverage.sha(Path(__file__)),'motor_JIT_disabled':True,'new_games':0,'benchmarks':0,
  'principal_preserved':True,'handoff_paragraph':handoff,'excluded_completed_games':excluded_count,
  'generation_live_root_excluded':m['excluded_live_root'],'coverage_json_sha256':coverage.sha(O/'cobertura.json')}
 assert final['protocol_unchanged'] and all(final['original_sources_unchanged'].values())
 dump(O/'AUDITORIA_FINAL.json',final)
 progress=json.loads((O/'progreso.json').read_text());progress.update(status='closed_rejected',decision='no_integration_coverage',remaining_required_work=[],closing='CIERRE.md');dump(O/'progreso.json',progress)
 print(json.dumps({k:final[k] for k in ['decision','traces','decisions','roots_pure_detector','pure_roots_already_safe_third','coverage_sanity_regressions','principal_preserved']},indent=2))

if __name__=='__main__':main()
