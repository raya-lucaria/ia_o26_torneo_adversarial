"""Diagnostico posterior: ambos modelos sobre exactamente las mismas trazas oficiales.

No entrena ni ajusta filtros. Replay completo, solo bloques validos, sin JIT.
Etiquetas de resultados bajo politica observada; no son valores optimos de acciones.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
os.environ['NUMBA_DISABLE_JIT']='1';sys.dont_write_bytecode=True
import numpy as np
from fase5_features import features,metricas,probabilidades

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--serie',type=Path,required=True)
    ap.add_argument('--salida',type=Path,required=True)
    ap.add_argument('--historico',type=Path)
    ap.add_argument('--auditoria',type=Path)
    a=ap.parse_args()
    state=json.loads((a.serie/'progreso.json').read_text())
    if state['status']!='all_games_complete':raise ValueError('Esperar la muestra fijada completa')
    if a.salida.exists():raise FileExistsError(a.salida)
    manifest=json.loads((a.serie/'manifiesto.json').read_text())
    equipo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(equipo.parents[1]/'sistema'))
    from motor import Estado,Jugada
    from motor.jugada import casilla_de_texto
    weights={};hashes={}
    for v,root in manifest['roots'].items():
        path=Path(root)/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz'
        expected=manifest['files'][v]['iacasitegano']['data/pesos_eval3_ligero.npz']['sha256']
        assert sha(path)==expected
        with np.load(path,allow_pickle=False) as z:weights[v]=z['pesos'].copy()
        assert weights[v].shape==(275,) and np.isfinite(weights[v]).all()
        hashes[v]=expected
    assert np.array_equal(weights['A'][267:],weights['B'][267:])
    boards=[];points=[];turns=[];ys=[];groups=[];variants=[];colors=[];plies=[];sources=[];skipped=[]
    seen=set()
    normal={'puntos','sin_jugadas','sin_piezas','meta'}
    for run in state['runs']:
        if run['status']!='complete' or run['technical']:continue
        for entry in run['games']:
            path=Path(entry['path']);assert str(path) not in seen;seen.add(str(path))
            assert sha(path)==entry['sha256'];game=json.loads(path.read_text())
            if game['motivo'] not in normal:
                skipped.append({'path':str(path),'reason':game['motivo']});continue
            own='blancas' if game['meta']['blancas']=='iacasitegano' else 'negras'
            est=Estado.inicial();n=0
            for j in game['jugadas']:
                assert not est.terminado() and est.turno==j['color'] and j['n']==n+1
                move=Jugada.desde_texto(j['jugada']);assert move in est.jugadas()
                combat=j['combate'];attack=est.es_ataque(move);assert attack==(combat is not None)
                if attack:
                    assert abs(est.prob_ataque(move)-combat['p'])<1e-12
                    assert (combat['u']<combat['p'])==combat['gano']
                if est.turno==own:
                    board=np.zeros(81,np.int8)
                    for c,name in enumerate(['blancas','negras']):
                        for sq in est.piezas(name):board[casilla_de_texto(sq)]=c+1
                    boards.append(board);points.append([est.puntos['blancas'],est.puntos['negras']]);turns.append(int(est.turno=='negras'))
                    ys.append(int(game['ganador']==own));variants.append(run['variant']);colors.append(own);plies.append(n)
                    groups.append(f"{run['seed']}:{game['meta']['pareja']}:{game['meta']['juego']}")
                est=est.aplicar(move,combat['gano'] if combat else None);n+=1
            assert est.ganador()==game['ganador'] and est.puntos==game['puntos']
            sources.append({'path':str(path),'sha256':entry['sha256'],'variant':run['variant'],'plies':n})
    assert len(seen)==160
    X=features(np.array(boards),np.array(points,np.int8),np.array(turns,np.int8));y=np.array(ys,float)
    groups=np.array(groups);variants=np.array(variants);colors=np.array(colors);plies=np.array(plies);points=np.array(points)
    logits={v:X@w[:267] for v,w in weights.items()}
    probs={v:probabilidades(z) for v,z in logits.items()}
    def extremes(z):
        return dict(min=float(z.min()),max=float(z.max()),fraction_abs_logit_10=float((abs(z)>=10).mean()),fraction_abs_logit_20=float((abs(z)>=20).mean()),fraction_exact_saturated_numpy_utility=float((abs(np.tanh(.5*z))==1).mean()))
    profiles={}
    for v in ['A','B']:
        masks={'todos':variants==v,'blancas':(variants==v)&(colors=='blancas'),'negras':(variants==v)&(colors=='negras'),
               'inicio':(variants==v)&(plies<20),'medio':(variants==v)&(plies>=20)&(plies<60),'tardio':(variants==v)&(plies>=60),
               'alguien_dos_puntos':(variants==v)&(points.max(axis=1)==2)}
        profiles[v]={}
        for name,mask in masks.items():
            profiles[v][name]={'states':int(mask.sum()),'wins_observed':float(y[mask].mean()) if mask.any() else None,
                'models':{model:metricas(y[mask],prob[mask],groups[mask]) for model,prob in probs.items()}}
    family={'puntos':[0],'patrones':list(range(1,244)),'filas':list(range(244,253)),'marcador':list(range(263,266))}
    norms={f:{v:dict(l2=float(np.linalg.norm(w[indices])),min=float(w[indices].min()),max=float(w[indices].max())) for v,w in weights.items()} for f,indices in family.items()}
    result=dict(status='diagnostico_sin_ajuste',source_series_sha256=sha(a.serie/'progreso.json'),models_sha256=hashes,
        script_sha256=sha(__file__),features_sha256=sha(Path(__file__).with_name('fase5_features.py')),
        games=len(sources),states=len(y),skipped_fault_games=skipped,profiles=profiles,weight_families=norms,sources=sources,logit_extremes={v:{model:extremes(z[variants==v]) for model,z in logits.items()} for v in ['A','B']},
        scope='Ambos modelos en los mismos estados por politica, etiquetas por resultado observado. No prueba valor optimo ni causa unica de derrota. No se usa para cambiar este candidato o criterio.')
    if bool(a.historico)!=bool(a.auditoria):raise ValueError('Historico y auditoria deben darse juntos')
    if a.historico:
        audit=json.loads((a.auditoria/'auditoria.json').read_text())
        assert sha(a.historico)==audit['dataset_sha256'] and hashes['A']==audit['pesos_sha256']
        with np.load(a.historico,allow_pickle=False) as z:hist={k:z[k] for k in z.files}
        with np.load(a.auditoria/'predicciones_y_grupos.npz',allow_pickle=False) as z:
            hy,hg,hv,hp,ids=z['y'].copy(),z['grupo'].copy(),z['validacion_historica'].copy(),z['prob'].copy(),z['partida'].copy()
        assert np.array_equal(hist['partida_origen'],ids)
        assert np.array_equal(hy,(hist['ganador_blancas']==(hist['turno']==0)).astype(float))
        HX=features(hist['tablero'],hist['puntos'],hist['turno'])
        hlogits={v:HX@w[:267] for v,w in weights.items()}
        predictions={v:probabilidades(z) for v,z in hlogits.items()}
        assert np.max(abs(predictions['A']-hp))<1e-10
        masks={'ajuste_visto':~hv,'validacion_historica_vista':hv,'validacion_inicio':hv&(hist['ply']<20),'validacion_medio':hv&(hist['ply']>=20)&(hist['ply']<60),'validacion_tardia':hv&(hist['ply']>=60),'validacion_alguien_dos_puntos':hv&(hist['puntos'].max(axis=1)==2)}
        retention={name:{v:metricas(hy[mask],pred[mask],hg[mask]) for v,pred in predictions.items()} for name,mask in masks.items()}
        bands={}
        for v,pred in predictions.items():
            bands[v]=[]
            for lo in np.arange(0,1,.1):
                mask=hv&(pred>=lo)&(pred<min(1.000000001,lo+.1))
                bands[v].append({'from':float(lo),'n':int(mask.sum()),'groups':len(np.unique(hg[mask])),'predicted_mean':float(pred[mask].mean()) if mask.any() else None,'observed_win_rate':float(hy[mask].mean()) if mask.any() else None})
        result['retencion_historica']=dict(dataset_sha256=sha(a.historico),audit_sha256=sha(a.auditoria/'auditoria.json'),profiles=retention,calibration_bands_validation=bands,logit_extremes_validation={v:extremes(z[hv]) for v,z in hlogits.items()},scope='Datos historicos vistos por el modelo activo. Diagnostico de retencion, no confirmacion independiente ni filtro retroactivo.')
    a.salida.mkdir(parents=True)
    (a.salida/'RESULTADO.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(a.salida/'predicciones.npz',y=y,old=probs['A'],new=probs['B'],groups=groups,variant=variants,color=colors,ply=plies)
    lines=['# Transferencia de la calibracion a Montecarlo','',result['scope'],'','| Politica que genero estados | Estados propios | Modelo evaluado | Brier por grupo | Logloss por grupo |','|---|---:|---|---:|---:|']
    for v in ['A','B']:
        p=profiles[v]['todos']
        for model,metrics_ in p['models'].items():
            lines.append(f"| {v} | {p['states']} | {model} | {metrics_.get('brier_igual_peso_grupo')} | {metrics_.get('logloss_igual_peso_grupo')} |")
    if 'retencion_historica' in result:
        for subset,models in result['retencion_historica']['profiles'].items():
            lines.append(f"Retencion {subset}: {json.dumps(models,ensure_ascii=False)}")
    lines+=['','A=U con pesos originales; B=N adaptado. Solo partidas de desarrollo completas y no excluidas. Tecnicas y fallos no entran en calibracion; los fallos si cuentan en fuerza.','Modelos evaluados sobre estados identicos dentro de cada fila de politica. Los resultados dependen de continuaciones observadas y rivales; no son etiquetas contrafactuales de jugadas alternativas.','No se reentrena ni se modifican filtros. El diagnostico sirve para decidir la siguiente hipotesis, con datos nuevos para confirmarla.','',f'Normas por familia: {json.dumps(norms,ensure_ascii=False)}']
    (a.salida/'DIAGNOSTICO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'games':len(sources),'states':len(y),'profiles':{v:profiles[v]['todos'] for v in ['A','B']}},ensure_ascii=False))

if __name__=='__main__':main()
