"""Puertas offline prospectivas. No promueve pesos ni ejecuta partidas de liga."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fase5_features import metricas, probabilidades

LIMITES = dict(min_grupos_train=24, min_grupos_validacion=12, min_grupos_test=12,
               min_grupos_estrato=4, mejora_brier_validacion=.002,
               regresion_brier_retencion=.005, regresion_brier_estrato=.01,
               regresion_brier_color=.005, candidatos_maximos=2)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guardar(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def cargar(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def comprobar(d):
    required = {'X', 'y', 'grupo', 'estado', 'turno', 'ply'}
    if not required <= d.keys():
        raise ValueError('Faltan columnas: ' + str(required - d.keys()))
    n = len(d['y'])
    if d['X'].shape != (n, 267) or any(len(d[k]) != n for k in required):
        raise ValueError('Dimensiones inválidas')
    if not np.isfinite(d['X']).all() or not np.isin(d['y'], [0, 1]).all():
        raise ValueError('Features no finitas o etiquetas fuera de 0/1')
    if not np.isin(d['turno'], [0, 1]).all() or np.any(d['ply'] < 0):
        raise ValueError('Turno o ply inválidos')

def pesos(path):
    w = cargar(path)['pesos']
    if w.shape != (275,) or not np.isfinite(w).all():
        raise ValueError('Modelo debe contener 275 pesos finitos')
    if not np.array_equal(w[267:], np.array([1,0,0,0,0,0,1,0])):
        raise ValueError('Banderas distintas del eval3 ligero de este protocolo')
    return w

def preparar(datos, historico, base, salida, tipo='historico'):
    salida = Path(salida)
    if salida.exists():
        raise FileExistsError(salida)
    d, h = cargar(datos), cargar(historico)
    comprobar(d); comprobar(h); pesos(base)
    if 'fold' not in d or not np.isin(d['fold'], [0, 1, 2]).all():
        raise ValueError('Se requieren folds 0/1/2 ya purgados')
    if 'fold' not in h:
        raise ValueError('El histórico debe identificar ajuste0 y retención1')
    for i in range(3):
        for j in range(i):
            for key in ['grupo', 'estado']:
                if set(d[key][d['fold'] == i]) & set(d[key][d['fold'] == j]):
                    raise ValueError('Fuga entre particiones: ' + key)
    # El replay también forma parte del entrenamiento. En datos nuevos, sus
    # estados no pueden reaparecer en validation/test. La retención histórica
    # tampoco puede recibir entrenamiento indirecto a través de datos nuevos.
    purga_historico = dict(train=0, validation=0, test=0)
    if tipo == 'nuevo':
        h_train = h['estado'][h['fold'] == 0]
        h_ret = h['estado'][h['fold'] == 1]
        keep = np.ones(len(d['y']), bool)
        for fold, name in enumerate(['train', 'validation', 'test']):
            prohibidos = h_ret if fold == 0 else np.r_[h_train, h_ret]
            drop = (d['fold'] == fold) & np.isin(d['estado'], prohibidos)
            purga_historico[name] = int(drop.sum())
            keep &= ~drop
        d = {k: v[keep] for k, v in d.items()}
    salida.mkdir(parents=True)
    fs = {}
    for name, dataset, mask in [('train', d, d['fold'] == 0),
                                ('validation', d, d['fold'] == 1),
                                ('test', d, d['fold'] == 2),
                                ('replay', h, h['fold'] == 0),
                                ('retencion', h, h['fold'] == 1)]:
        part = {k: dataset[k][mask] for k in ['X', 'y', 'grupo', 'estado', 'turno', 'ply']}
        if not len(part['y']):
            raise ValueError('Partición vacía: ' + name)
        np.savez_compressed(salida / (name + '.npz'), **part)
        fs[name + '.npz'] = sha(salida / (name + '.npz'))
    # El replay y la retención no pueden compartir grupos/estados.
    for key in ['grupo', 'estado']:
        if set(h[key][h['fold'] == 0]) & set(h[key][h['fold'] == 1]):
            raise ValueError('Histórico con fuga replay/retención: ' + key)
    shutil.copyfile(base, salida / 'base.npz'); fs['base.npz'] = sha(base)
    contrato = dict(tipo=tipo, limites=LIMITES, archivos=fs, datos_sha256=sha(datos),
                    purga_historico=purga_historico,
                    entrenamiento={'lambda':100,'masa_historica':.8,'masa_nueva':.2},
                    trainer_sha256=sha(Path(__file__).with_name('entrenar_replay.py')),
                    features_sha256=sha(Path(__file__).resolve().parents[1]/'fase5_features.py'),
                    historico_sha256=sha(historico), script_sha256=sha(__file__),
                    advertencia='Retención histórica vista; no es confirmación nueva. Ninguna puerta offline acepta fuerza.')
    guardar(salida / 'contrato.json', contrato)
    (salida / 'contrato.sha256').write_text(sha(salida / 'contrato.json') + '\n')
    return contrato

def contrato(root):
    root = Path(root)
    if sha(root / 'contrato.json') != (root / 'contrato.sha256').read_text().strip():
        raise ValueError('Contrato modificado')
    c = json.loads((root / 'contrato.json').read_text())
    if c['script_sha256'] != sha(__file__):
        raise ValueError('Herramienta cambiada tras congelar')
    if c['trainer_sha256'] != sha(Path(__file__).with_name('entrenar_replay.py')) or c['features_sha256'] != sha(Path(__file__).resolve().parents[1]/'fase5_features.py'):
        raise ValueError('Entrenador o representación cambiaron tras congelar')
    for name, expected in c['archivos'].items():
        if sha(root / name) != expected:
            raise ValueError('Archivo modificado: ' + name)
    return c

def perfiles(d, w, base, limites, minimo):
    comprobar(d)
    p, b = probabilidades(d['X'] @ w[:267]), probabilidades(d['X'] @ base[:267])
    masks = dict(global_=np.ones(len(p), bool), blancas=d['turno'] == 0, negras=d['turno'] == 1,
                 inicio=d['ply'] < 20, medio=(d['ply'] >= 20) & (d['ply'] < 60), tardio=d['ply'] >= 60)
    result = {}
    for name, mask in masks.items():
        result[name] = dict(base=metricas(d['y'][mask], b[mask], d['grupo'][mask]),
                            candidato=metricas(d['y'][mask], p[mask], d['grupo'][mask]))
    cobertura = all(r['base']['grupos'] >= (minimo if name == 'global_' else limites['min_grupos_estrato'])
                    for name, r in result.items())
    return result, cobertura

def no_regresion(perfil, tolerancia):
    for name, r in perfil.items():
        if r['base']['n'] == 0:
            return False
        tol = tolerancia if name == 'global_' else LIMITES['regresion_brier_estrato']
        if r['candidato']['brier_igual_peso_grupo'] > r['base']['brier_igual_peso_grupo'] + tol:
            return False
    return True

def seleccionar(root, candidato, ident):
    root = Path(root); c = contrato(root)
    if not ident.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Identificador inválido')
    if (root / 'EXAMEN_ABIERTO.json').exists():
        raise ValueError('Examen ya abierto; no ajustar nuevos candidatos con ese test')
    selected = root / ('seleccion_' + ident)
    if selected.exists():
        raise FileExistsError(selected)
    if len(list(root.glob('seleccion_*'))) >= c['limites']['candidatos_maximos']:
        raise ValueError('Presupuesto de candidatos agotado')
    w, base = pesos(candidato), pesos(root / 'base.npz')
    if not np.array_equal(w[267:], base[267:]):
        raise ValueError('Este protocolo no cambia representación ni banderas')
    validation = cargar(root / 'validation.npz')
    historical = cargar(root / 'retencion.npz')
    v, cobertura = perfiles(validation, w, base, c['limites'], c['limites']['min_grupos_validacion'])
    h, cobertura_h = perfiles(historical, w, base, c['limites'], c['limites']['min_grupos_validacion'])
    train = cargar(root / 'train.npz')
    train_ok = len(np.unique(train['grupo'])) >= c['limites']['min_grupos_train']
    a, b = v['global_']['base'], v['global_']['candidato']
    mejora = b['brier_igual_peso_grupo'] <= a['brier_igual_peso_grupo'] - c['limites']['mejora_brier_validacion']
    ll = b['logloss_igual_peso_grupo'] <= a['logloss_igual_peso_grupo']
    retention = no_regresion(h, c['limites']['regresion_brier_retencion'])
    color_ok = all(v[x]['candidato'].get('brier_igual_peso_grupo', float('inf')) <=
                   v[x]['base'].get('brier_igual_peso_grupo', -float('inf')) + c['limites']['regresion_brier_color']
                   for x in ['blancas', 'negras'])
    checks = dict(cobertura_validacion=cobertura, cobertura_retencion=cobertura_h,
                  grupos_train=train_ok, mejora_validacion=mejora, logloss=ll,
                  retencion=retention, colores=color_ok)
    selected.mkdir(); shutil.copyfile(candidato, selected / 'candidato.npz')
    result = dict(pasa=all(checks.values()), checks=checks, validation=v, retencion=h,
                  candidato_sha256=sha(candidato), contrato_sha256=sha(root / 'contrato.json'),
                  test_evaluado=False, promovido=False)
    guardar(selected / 'RESULTADO.json', result)
    return result

def examinar(root, ident):
    root = Path(root); c = contrato(root)
    selected = root / ('seleccion_' + ident)
    s = json.loads((selected / 'RESULTADO.json').read_text())
    if not s['pasa']:
        raise ValueError('Selección rechazada; prohibido evaluar el test')
    candidates = [json.loads(p.read_text()) for p in root.glob('seleccion_*/RESULTADO.json')]
    best = min((r for r in candidates if r['pasa']), key=lambda r:r['validation']['global_']['candidato']['brier_igual_peso_grupo'])
    if best['candidato_sha256'] != s['candidato_sha256']:
        raise ValueError('No es el mejor candidato por validación')
    if sha(selected / 'candidato.npz') != s['candidato_sha256']:
        raise ValueError('Candidato cambió después de seleccionarse')
    flag = root / 'EXAMEN_ABIERTO.json'
    with flag.open('x') as f:
        json.dump(dict(candidato_sha256=s['candidato_sha256'], estado='abierto_una_vez'), f)
    p, cobertura = perfiles(cargar(root / 'test.npz'), pesos(selected / 'candidato.npz'),
                            pesos(root / 'base.npz'), c['limites'], c['limites']['min_grupos_test'])
    pasa = cobertura and no_regresion(p, 0)
    pasa = pasa and all(p[x]['candidato'].get('brier_igual_peso_grupo', float('inf')) <=
                       p[x]['base'].get('brier_igual_peso_grupo', -float('inf')) + c['limites']['regresion_brier_color']
                       for x in ['blancas', 'negras'])
    result = dict(pasa=bool(pasa), test=p, cobertura=cobertura, tipo=c['tipo'],
                  apto_para_puerta_tecnica=bool(pasa and c['tipo'] == 'nuevo'),
                  promovido=False, fuerza_confirmada=False)
    guardar(selected / 'EXAMEN.json', result)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('preparar')
    for name in ['datos', 'historico', 'base', 'salida']: q.add_argument('--' + name, type=Path, required=True)
    q.add_argument('--tipo', choices=['nuevo', 'historico'], default='historico')
    for name in ['seleccionar', 'examinar']:
        q = sub.add_parser(name); q.add_argument('--directorio', type=Path, required=True); q.add_argument('--id', required=True)
        if name == 'seleccionar': q.add_argument('--candidato', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'preparar': result = preparar(a.datos, a.historico, a.base, a.salida, a.tipo)
    elif a.cmd == 'seleccionar': result = seleccionar(a.directorio, a.candidato, a.id)
    else: result = examinar(a.directorio, a.id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if 'pasa' in result and not result['pasa']: raise SystemExit(2)

if __name__ == '__main__': main()
