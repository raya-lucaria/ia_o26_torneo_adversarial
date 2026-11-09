"""Validación prospectiva S13: un modelo fijo, 64 episodios y una exposición.

Requiere liberar explícitamente la ventana de 8B.2. No entrena, no promueve ni
abre test256; UCT nuevo se identifica por nombre y no se carga al evaluar S13.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
from lanzar import run, write
from validacion import sha
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fase5_features import features, metricas, probabilidades
import numpy as np

MAIN = Path(__file__).resolve().parents[4]
BASE_SHA = '37678d0bd700c27a29d0084558158db1a65b01409dbf54828e89706441c2250b'
S13_SHA = 'aba46e5595e442473ce5672abd5a6c3141e153761110e90600075b0e1300227d'
FAMILIAS = ('montecarlo', 'agresivo', 'carrera', 'uct')
LIMITES = dict(grupos_total=12, grupos_familia=4, grupos_color_fase=4,
               mejora_brier=.002, delta_logloss_max=0., regresion_color=.005)


def congelar(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    file = out/'CONTRATO.json'
    if file.exists(): raise FileExistsError(file)
    audit = out/'AUDITORIA_SEMILLAS.json'
    if not json.loads(audit.read_text())['apto']: raise ValueError('Semillas sin auditoría apta')
    old = MAIN/'resultados/medicion/fase5_lote256_mac_20261029/validacion'
    candidate = MAIN/'resultados/medicion/fase5_subespacio13_20261029/ajuste/candidato.npz'
    base = old/'base.npz'
    if sha(candidate) != S13_SHA or sha(base) != BASE_SHA: raise ValueError('Modelo inesperado')
    sources = [Path(__file__), Path(__file__).with_name('lanzar.py'), Path(__file__).with_name('validacion.py'),
               Path(__file__).resolve().parents[1]/'fase5_features.py',
               Path(__file__).resolve().parents[1]/'fase5_verificar_episodios.py',
               root/'material/profesor_utilidad/busqueda_eval3.py',
               root/'material/profesor_utilidad/eval3.py',
               root/'estudiantes/iacasitegano/eval3.py',
               root/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz',
               candidate, base, audit]
    sources += [root/'estudiantes/iacasitegano/herramientas'/n for n in
                ('fase5_generar_robusto.py', 'fase5_rivales.py', 'fase5_rivales_robustos.py')]
    sources += sorted((root/'sistema/motor').glob('*.py'))
    sources += [old/(n+'.npz') for n in ('train', 'replay', 'retencion', 'validation')]
    protocol = dict(root=str(root), base=str(base), candidato=str(candidate),
                    fuentes={str(p): sha(p) for p in sources}, limites=LIMITES,
                    workers=4, episodios_por_worker=16, minutos_sesion=90,
                    semillas_base=[98000000, 99000000, 100000000, 101000000],
                    grupos_previstos=dict(validacion=24, uct_reserva=8),
                    exclusiones=[str(old/(n+'.npz')) for n in ('train','replay','retencion','validation')],
                    purga='estado canónico horizontal con puntos int8 y turno; excluir desarrollo previo; eliminar estados presentes en varias familias nuevas',
                    exposiciones_maximas=1, entrenamiento_permitido=False,
                    nota='El replay estructural del generador valida consistencia sin evaluar S13 contra UCT. El evaluador de este flujo no carga las partidas/shards UCT.')
    write(file, protocol)
    (out/'CONTRATO.sha256').write_text(sha(file)+'\n')
    return protocol


def contrato(out):
    out = Path(out); p = out/'CONTRATO.json'
    if sha(p) != (out/'CONTRATO.sha256').read_text().strip(): raise ValueError('Contrato modificado')
    c = json.loads(p.read_text())
    wrong = [name for name, h in c['fuentes'].items() if sha(name) != h]
    if wrong: raise ValueError('Fuentes cambiadas: '+str(wrong))
    return c


def estados(tableros, puntos, turnos):
    import hashlib
    result = []
    for b, p, t in zip(tableros, puntos, turnos):
        b = np.asarray(b, np.int8); p = np.asarray(p, np.int8)
        raw = min(b.tobytes(), b.reshape(9,9)[:,::-1].tobytes())+p.tobytes()+bytes([int(t)])
        result.append(hashlib.sha256(raw).hexdigest())
    return np.array(result)


def purgar(ids, familias, prohibidos):
    seen = {}
    for ident, family in zip(ids, familias): seen.setdefault(ident, set()).add(str(family))
    compartidos = {ident for ident, fs in seen.items() if len(fs) > 1}
    previo = np.isin(ids, list(prohibidos))
    multi = np.isin(ids, list(compartidos))
    return ~(previo | multi), dict(previo=int(previo.sum()), multifamilia=int(multi.sum()),
                                  union=int((previo | multi).sum()))


def cobertura(d, limits):
    g, turn, ply, family = (d[k] for k in ('grupo','turno','ply','familia'))
    counts = {'global': len(set(g))}
    checks = {'global': counts['global'] >= limits['grupos_total']}
    for f in FAMILIAS[:3]:
        counts[f] = len(set(g[family == f])); checks[f] = counts[f] >= limits['grupos_familia']
    for color in (0,1):
        for phase, mask in [('inicio',ply<20),('medio',(ply>=20)&(ply<60)),('tardio',ply>=60)]:
            name = str(color)+'_'+phase
            counts[name] = len(set(g[(turn == color) & mask]))
            checks[name] = counts[name] >= limits['grupos_color_fase']
    return dict(pasa=all(checks.values()), grupos=counts, checks=checks)


def preparar(out):
    out = Path(out); c = contrato(out); target = out/'validacion_nueva.npz'
    ready = out/'VALIDACION_LISTA.json'
    if ready.exists():
        if sha(ready) != (out/'VALIDACION_LISTA.sha256').read_text().strip():
            raise ValueError('Manifiesto de validación cambió')
        r = json.loads(ready.read_text())
        if sha(target) != r['datos_sha256']: raise ValueError('Validación preparada cambió')
        return r
    if target.exists(): raise ValueError('NPZ sin manifiesto final: preservar y revisar')
    rows, groups, families, sources, reserve = [], [], [], [], []
    manifests = []
    for i, seed in enumerate(c['semillas_base']):
        folder = out/f'lotes/worker_{i}'
        progress = json.loads((folder/'progreso.json').read_text())
        manifest = json.loads((folder/'manifest.json').read_text())
        if progress['estado'] != 'completo' or progress['completadas'] != 16:
            raise ValueError('Trabajador incompleto')
        if manifest['semilla'] != seed or manifest['partidas'] != 16:
            raise ValueError('Población/semillas distintas')
        expected = dict(profundidad=7,tope_nodos=3000000,uct_iteraciones=4096,
                        mc_muestras=128,mc_rollouts=20)
        if any(manifest[k] != v for k,v in expected.items()):
            raise ValueError('Presupuestos distintos del protocolo')
        if len(list((folder/'partidas').glob('*.json'))) != 16:
            raise ValueError('Cantidad de partidas inesperada')
        manifests.append({k:v for k,v in manifest.items() if k!='semilla'})
        for game in range(16):
            family = FAMILIAS[(game//2)%4]; color = 'blancas' if game%2 == 0 else 'negras'
            stem = f'{game:04d}_{family}_{color}'
            file, shard = folder/'partidas'/(stem+'.json'), folder/'shards'/(stem+'.npz')
            rec = dict(partida=str(file), partida_sha256=sha(file), shard_sha256=sha(shard),
                       grupo=f'{seed+game//2}:{game//2}:0', familia=family)
            if family == 'uct':
                # Solo bytes/hash; no JSON ni etiquetas/features de UCT en este consumidor.
                reserve.append(rec); continue
            meta = json.loads(file.read_text())
            if meta['meta']['semilla_maestra'] != seed+game//2 or meta['shard_sha256'] != sha(shard):
                raise ValueError('Partida/shard no corresponde')
            if meta['meta']['pareja'] != game//2 or meta['meta']['juego'] != 0 or meta['meta'][color] != 'profesor_offline':
                raise ValueError('Grupo/color profesor distinto')
            with np.load(shard,allow_pickle=False) as z:
                d = {k:z[k] for k in ('tablero','puntos','turno','ply','ganador_blancas')}
            rows.append(d); groups.extend([rec['grupo']]*len(d['turno']))
            families.extend([family]*len(d['turno'])); sources.append(rec)
    if any(m != manifests[0] for m in manifests): raise ValueError('Presupuestos/fuentes entre trabajadores difieren')
    teacher = Path(c['root'])/'material/profesor_utilidad/busqueda_eval3.py'
    if manifests[0]['buscador_sha256'] != sha(teacher): raise ValueError('Profesor distinto')
    expected_sources = {
        'script_sha256':Path(c['root'])/'estudiantes/iacasitegano/herramientas/fase5_generar_robusto.py',
        'pesos_sha256':Path(c['root'])/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz',
        'eval3_sha256':Path(c['root'])/'estudiantes/iacasitegano/eval3.py',
        'motor_sha256':Path(c['root'])/'sistema/motor/nucleo.py',
        'rivales_sha256':Path(c['root'])/'estudiantes/iacasitegano/herramientas/fase5_rivales.py',
        'rivales_robustos_sha256':Path(c['root'])/'estudiantes/iacasitegano/herramientas/fase5_rivales_robustos.py'}
    if any(manifests[0][k] != sha(p) for k,p in expected_sources.items()):
        raise ValueError('Fuentes reales de generación distintas')
    merged = {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    group, family = np.array(groups), np.array(families)
    ids = estados(merged['tablero'], merged['puntos'], merged['turno'])
    forbidden = set()
    for path in c['exclusiones']:
        with np.load(path,allow_pickle=False) as z: forbidden.update(z['estado'].tolist())
    keep, dropped = purgar(ids, family, forbidden)
    d = dict(X=features(merged['tablero'][keep], merged['puntos'][keep], merged['turno'][keep]),
             y=(merged['ganador_blancas'][keep] == (merged['turno'][keep] == 0)).astype(float),
             grupo=group[keep], estado=ids[keep], familia=family[keep],
             turno=merged['turno'][keep], ply=merged['ply'][keep])
    if not len(d['y']): raise ValueError('Purga vació población')
    temp = target.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f,**d)
    temp.replace(target)
    result = dict(datos_sha256=sha(target), contrato_sha256=sha(out/'CONTRATO.json'),
                  fuentes=sources, reserva_uct=reserve, reserva_uct_evaluada=False,
                  purga=dropped, filas=len(d['y']), cobertura=cobertura(d,c['limites']))
    write(ready,result)
    (out/'VALIDACION_LISTA.sha256').write_text(sha(ready)+'\n')
    return result


def evaluar(out):
    out = Path(out); c = contrato(out); prepared = json.loads((out/'VALIDACION_LISTA.json').read_text())
    if sha(out/'VALIDACION_LISTA.json') != (out/'VALIDACION_LISTA.sha256').read_text().strip():
        raise ValueError('Manifiesto de validación cambió')
    if prepared['contrato_sha256'] != sha(out/'CONTRATO.json'):
        raise ValueError('Datos preparados para otro contrato')
    if sha(out/'validacion_nueva.npz') != prepared['datos_sha256']:
        raise ValueError('Datos cambiados antes de evaluar')
    if not prepared['cobertura']['pasa']:
        return dict(fase='inconcluso_cobertura', cobertura=prepared['cobertura'], test_evaluado=False)
    result_file = out/'RESULTADO_VALIDACION.json'
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if result['candidato_sha256'] != S13_SHA or result['datos_sha256'] != prepared['datos_sha256'] or result['contrato_sha256'] != sha(out/'CONTRATO.json'):
            raise ValueError('Resultado previo de otro modelo/dataset/contrato')
        return result
    with (out/'VALIDACION_ABIERTA.json').open('x') as f:
        json.dump({'candidato_sha256':S13_SHA,'datos_sha256':prepared['datos_sha256']},f)
    with np.load(out/'validacion_nueva.npz',allow_pickle=False) as z: d = {k:z[k] for k in z.files}
    with np.load(c['base'],allow_pickle=False) as z: base = z['pesos']
    with np.load(c['candidato'],allow_pickle=False) as z: candidate = z['pesos']
    p0,p1 = probabilidades(d['X']@base[:267]), probabilidades(d['X']@candidate[:267])
    masks = dict(global_=np.ones(len(p0),bool), blancas=d['turno']==0, negras=d['turno']==1,
                 inicio=d['ply']<20, medio=(d['ply']>=20)&(d['ply']<60), tardio=d['ply']>=60)
    masks.update({f:d['familia']==f for f in FAMILIAS[:3]})
    profiles = {name:dict(base=metricas(d['y'][m],p0[m],d['grupo'][m]),
                          candidato=metricas(d['y'][m],p1[m],d['grupo'][m])) for name,m in masks.items()}
    a,b = profiles['global_']['base'],profiles['global_']['candidato'];l = c['limites']
    checks = dict(cobertura=prepared['cobertura']['pasa'],
                  mejora_brier=b['brier_igual_peso_grupo']<=a['brier_igual_peso_grupo']-l['mejora_brier'],
                  logloss=b['logloss_igual_peso_grupo']<=a['logloss_igual_peso_grupo'],
                  colores=all(profiles[x]['candidato']['brier_igual_peso_grupo'] <=
                              profiles[x]['base']['brier_igual_peso_grupo']+l['regresion_color'] for x in ('blancas','negras')))
    result = dict(fase='validacion_nueva_pasada' if all(checks.values()) else 'rechazado_validacion_nueva',
                  checks=checks, perfiles=profiles, candidato_sha256=S13_SHA,
                  contrato_sha256=sha(out/'CONTRATO.json'), datos_sha256=prepared['datos_sha256'],
                  modelo_aceptado=False, test256_abierto=False, uct_nuevo_evaluado=False,
                  nota='Siguiente puerta requiere protocolo de examen y controles funcionales/deportivos; no promoción automática.')
    write(result_file,result); return result


def continuar(out, liberacion):
    out = Path(out).resolve();c = contrato(out);gate = json.loads(Path(liberacion).read_text())
    if gate['estado']!='ventana_8B2_terminada' or sha(gate['evidencia_cierre'])!=gate['sha256']:
        raise ValueError('Falta liberación verificada de ventana8B2')
    lock = out/'PIPELINE_S13_ACTIVO.json'
    with lock.open('x') as f: json.dump({'pid':os.getpid()},f)
    def state(phase,**extra):
        r = dict(fase=phase,fecha=datetime.datetime.now().astimezone().isoformat(),**extra)
        write(out/'ESTADO.json',r);print(json.dumps(r,ensure_ascii=False),flush=True);return r
    try:
        if (out/'RESULTADO_VALIDACION.json').exists():
            return state('finalizado_previamente',resultado=evaluar(out))
        state('generando',episodios=64,modelo_fijo=S13_SHA)
        result = run(c['root'],out/'lotes',4,16,c['minutos_sesion'],c['semillas_base'][0])
        if result['interrupted']:return state('interrumpido_por_usuario')
        if any(result['exit_codes']):raise ValueError('Generación falló; revisar sin reintento automático')
        if not result['complete']:return state('pausa_datos_incompletos')
        contrato(out);state('preparando_validacion_sin_uct')
        prepared = preparar(out)
        if not prepared['cobertura']['pasa']:return state('inconcluso_cobertura',cobertura=prepared['cobertura'])
        state('evaluando_una_vez');result = evaluar(out)
        return state(result['fase'],resultado=result)
    except BaseException as e:
        state('error_revisar',error=str(e));raise
    finally:lock.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);sub = p.add_subparsers(dest='cmd',required=True)
    q = sub.add_parser('congelar');q.add_argument('--root',type=Path,required=True);q.add_argument('--salida',type=Path,required=True)
    q = sub.add_parser('continuar');q.add_argument('--salida',type=Path,required=True);q.add_argument('--liberacion',type=Path,required=True)
    a = p.parse_args()
    if a.cmd=='congelar': print(json.dumps(congelar(a.root,a.salida),ensure_ascii=False,indent=2))
    else: continuar(a.salida,a.liberacion)
