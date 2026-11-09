"""Generador offline: horizontes completos, sin alterar el participante activo.

Solo publica propuestas calificadas a horizonte finito, nunca óptimos del juego.
NUMBA_DISABLE_JIT evita escribir/compilar cachés durante la convivencia con fase5.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TEAM = ROOT / 'estudiantes/iacasitegano'
POLICY = {'version': 1, 'margin_gain': 0.02, 'tolerance_gain': 1e-10,
          'horizons': 'D,D+1', 'abort': 'unknown', 'safe_win': 'must_preserve',
          'label': 'qualified_finite_horizon_not_game_optimal'}
RESERVED = set(range(20261102, 20261105)) | set(range(20261111, 20261115)) | set(range(20261211, 20261217))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('No se admiten enlaces en archivos de salida')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Budget:
    """Contabilidad solo para limitar CPU, nunca para publicar rendimiento."""
    def __init__(self, fraction=0.12):
        if not 0 < fraction <= 0.12:
            raise ValueError('CPU debe estar en (0,0.12]')
        self.fraction = fraction
        self.wall = time.monotonic()
        self.cpu = time.process_time()

    def wait(self):
        while True:
            remaining = (time.process_time() - self.cpu) / self.fraction - (time.monotonic() - self.wall)
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.25))


def engine():
    for name in ['NUMBA_DISABLE_JIT', 'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']:
        os.environ[name] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'sistema'))
    from motor import Estado, Jugada
    return Estado, Jugada


def validate_book(book):
    if not isinstance(book, dict):
        raise ValueError('Libro debe ser objeto JSON')
    for key, move in book.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key or not 0 <= int(key) < 2**64:
            raise ValueError(f'Clave no canónica uint64: {key!r}')
        if not isinstance(move, list) or len(move) != 2 or any(type(x) is not int or not 0 <= x < 81 for x in move):
            raise ValueError(f'Par inválido: {move!r}')
        a, b = move
        if abs(a // 9 - b // 9) != 1 or abs(a % 9 - b % 9) > 1:
            raise ValueError(f'Geometría inválida: {move!r}')
    return True


def transitions(state, move):
    children = state.siguiente(move)
    if not math.isclose(sum(p for p, _ in children), 1.0, abs_tol=1e-12):
        raise ValueError('Probabilidades inválidas del motor')
    return children


def safe_moves(state):
    return [[j.origen, j.destino] for j in state.jugadas()
            if all(s.ganador() == state.turno for _, s in transitions(state, j))]


def safe_selection(state, move):
    safe = safe_moves(state)
    return not safe or move in safe


def load_search(search_path, weights_path):
    """Adaptador estricto para las dos transformaciones eval3 existentes."""
    engine()
    import numpy as np
    path = Path(search_path).resolve()
    evaluator = path.with_name('eval3.py')
    if not evaluator.is_file():
        raise ValueError('Falta eval3 junto al buscador')
    tree = ast.parse(path.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_evaluar_numba')
    returned = next(x.value for x in function.body if isinstance(x, ast.Return))
    expression = ast.unparse(returned)
    if expression == 'e3.evaluar_nb(tablero, puntos, color, pesos)':
        scale = 'eval3_raw_logit'
    elif expression == 'np.tanh(0.5 * logit)' and any(isinstance(x, ast.Assign) and ast.unparse(x.value) == 'e3.evaluar_nb(tablero, puntos, color, pesos)' for x in function.body):
        scale = '2_sigmoid_logit_minus_1'
    else:
        raise ValueError('Transformación no reconocida: requiere adaptador explícito')
    # Un proceso admite una sola fuente eval3; evita contaminación entre profesores.
    previous = sys.modules.get('eval3')
    if previous is not None and Path(previous.__file__).resolve() != evaluator:
        raise ValueError('Otra fuente eval3 ya importada; usar proceso aislado')
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('_fase7a_search_' + sha(path)[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(weights_path, allow_pickle=False) as data:
        w = data['pesos']
        if w.shape != (275,) or not np.isfinite(w).all():
            raise ValueError('Se requiere NPZ con pesos finitos275')
        module.cargar_pesos(w.copy())
    weights, gain = module._pesos_y_ganancia()
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('Ganancia inválida')
    sources = {str(p): sha(p) for p in [path, evaluator, Path(weights_path).resolve(),
              ROOT / 'sistema/motor/nucleo.py', ROOT / 'sistema/motor/estado.py', ROOT / 'sistema/motor/jugada.py', Path(__file__)]}
    return module, {'sources': sources, 'scale': scale, 'terminal_gain': float(gain),
                    'evaluation_expression': expression, 'weights_shape': list(weights.shape),
                    'python': sys.version, 'numpy': np.__version__, 'jit_disabled': True}


class Unknown(Exception):
    pass


class Teacher:
    def __init__(self, module, nodes, budget):
        self.module, self.limit, self.budget = module, nodes, budget
        self.used = 0
        self.weights, self.gain = module._pesos_y_ganancia()

    def value(self, state, perspective, depth):
        import numpy as np
        self.budget.wait()
        self.used += 1
        if self.used > self.limit:
            raise Unknown('node_budget')
        winner = state.ganador()
        if winner is not None:
            return float(self.gain if winner == perspective else -self.gain), {'terminal': winner}
        tab, pts, color = self.module._tablero_desde_estado(state)
        if depth == 0:
            value = float(self.module._evaluar_numba(tab, pts, 0 if perspective == 'blancas' else 1, self.weights))
            record = {'leaf': True}
        else:
            count = np.zeros(2, np.int64)
            result = self.module._mejor_jugada_numba(tab, pts, color, depth, self.weights, self.gain, -1, -1,
                                                    count, max(0, self.limit - self.used))
            self.used += int(count[0])
            if not isinstance(result, tuple) or len(result) != 5:
                raise ValueError('Se requiere raíz actual de cinco retornos')
            origin, destination, value, aborted, completed = result
            if aborted or count[1] or self.used > self.limit or completed != len(state.jugadas()):
                raise Unknown('aborted_or_incomplete_root')
            _, Jugada = engine()
            if Jugada(int(origin), int(destination)) not in state.jugadas():
                raise ValueError('Raíz retornó jugada ilegal')
            value = float(value) * (1 if state.turno == perspective else -1)
            record = {'root_move': [int(origin), int(destination)], 'root_completed': int(completed),
                      'aborted': False, 'nodes': int(count[0]), 'depth': depth}
        if not math.isfinite(value) or abs(value) > self.gain + 1e-10 * max(1, self.gain):
            raise Unknown('nonfinite_or_out_of_scale')
        return value, record

    def alternatives(self, state, depth):
        alternatives = []
        for move in state.jugadas():
            branches, value = [], 0.0
            for probability, child in transitions(state, move):
                child_value, evidence = self.value(child, state.turno, depth - 1)
                value += probability * child_value
                branches.append({'probability': probability, 'state': child.a_dict(),
                                 'value': child_value, 'continuation': evidence})
            alternatives.append({'move': [move.origen, move.destino], 'value': value,
                                 'attack': state.es_ataque(move), 'branches': branches})
        return sorted(alternatives, key=lambda a: (-a['value'], a['move']))


def qualify(state, horizons, gain):
    if len(horizons) != 2 or any(not h['alternatives'] for h in horizons):
        return None, 'incomplete'
    best = horizons[0]['alternatives'][0]['move']
    for horizon in horizons:
        rows = horizon['alternatives']
        if rows[0]['move'] != best:
            return None, 'horizon_disagreement'
        if len(rows) > 1 and rows[0]['value'] - rows[1]['value'] < POLICY['margin_gain'] * gain + POLICY['tolerance_gain'] * max(1, gain):
            return None, 'uncertain_margin'
    if not safe_selection(state, best):
        return None, 'would_displace_safe_win'
    return best, POLICY['label']


def evaluate(state, module, depth, nodes, budget):
    teacher = Teacher(module, nodes, budget)
    horizons = []
    try:
        if state.terminado():
            raise Unknown('terminal_input')
        for d in (depth, depth + 1):
            horizons.append({'depth': d, 'alternatives': teacher.alternatives(state, d)})
        move, reason = qualify(state, horizons, teacher.gain)
    except Unknown as exc:
        move, reason = None, str(exc)
    return {'move': move, 'reason': reason, 'nodes_accounted': teacher.used, 'horizons': horizons,
            'status': 'qualified' if move is not None else 'unknown'}


def identity_matches(previous, current):
    if previous != current:
        raise ValueError('Reanudación rechazada: fuentes/modelo/configuración/población distintos')


def safe_output(output):
    output = Path(output).resolve()
    # Más restrictivo que proteger solo libro.json: no escribir en ningún data del equipo.
    if (TEAM / 'data').resolve() == output or (TEAM / 'data').resolve() in output.parents:
        raise ValueError('Salida dentro de data activo prohibida')
    if output == ROOT or output == TEAM:
        raise ValueError('Salida debe ser carpeta experimental nueva')
    return output


def run(args):
    Estado, _ = engine()
    budget = Budget(args.cpu_fraction)
    if args.profundidad < 1 or args.nodos < 0 or args.max_posiciones < 1:
        raise ValueError('Presupuestos inválidos')
    seeds = [int(s) for s in args.semillas.split(',')]
    if any(s in RESERVED for s in seeds):
        raise ValueError('Semilla reservada')
    population = json.loads(args.posiciones.read_text())
    if population.get('usage') not in ['synthetic_fixture', 'opening_design'] or not population.get('provenance'):
        raise ValueError('Solo fixtures o población de diseño con procedencia explícita; no test5/examen9')
    rows = population['positions']
    if not rows or any(not r.get('provenance') for r in rows):
        raise ValueError('Cada raíz necesita procedencia')
    order = list(range(len(rows)))
    for seed in seeds:
        random.Random(seed).shuffle(order)
    order = order[:args.max_posiciones]
    output = safe_output(args.salida)
    if not args.reanudar and output.exists():
        raise FileExistsError('Salida ya existe; nunca sobrescribir una ejecución')
    module, model = load_search(args.buscador, args.pesos)
    identity = {'model': model, 'policy': POLICY, 'population_path': str(args.posiciones.resolve()),
                'population_sha256': sha(args.posiciones), 'population': args.poblacion, 'order': order,
                'seeds': seeds, 'depth': args.profundidad, 'nodes_per_position_total': args.nodos,
                'max_positions': args.max_posiciones, 'cpu_fraction': args.cpu_fraction}
    if args.reanudar:
        identity_matches(json.loads((output / 'manifiesto.json').read_text()), identity)
        # No admitir artefactos desconocidos, enlaces o modificaciones de checkpoints.
        checks = json.loads((output / 'progreso.json').read_text()).get('checkpoint_sha256', {})
        for name, expected in checks.items():
            if sha(output / name) != expected:
                raise ValueError('Checkpoint modificado')
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'manifiesto.json', identity)
        save(output / 'progreso.json', {'completed': 0, 'checkpoint_sha256': {}})
    lock = output / '.writer.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        book, provenance, state_keys, checks = {}, {}, {}, {}
        for i in order:
            row = rows[i]
            state = Estado.desde_dict(row['state'])
            state_id = fingerprint(state.a_dict())
            key = str(state.clave)
            if key in state_keys and state_keys[key] != state_id:
                raise ValueError('Colisión Zobrist entre estados completos')
            state_keys[key] = state_id
            path = output / f'posicion_{i:05d}.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record['state_sha256'] != state_id:
                    raise ValueError('Estado de checkpoint distinto')
            else:
                record = {'state': state.a_dict(), 'state_sha256': state_id, 'key': key,
                          'input_index': i, 'provenance': row['provenance'],
                          **evaluate(state, module, args.profundidad, args.nodos, budget)}
                save(path, record)
            checks[path.name] = sha(path)
            if record['move'] is not None:
                if key in book and book[key] != record['move']:
                    raise ValueError('Propuestas incompatibles para la misma clave')
                book[key] = record['move']
                provenance.setdefault(key, []).append({'input_index': i, 'state_sha256': state_id,
                                                       'provenance': row['provenance'], 'evidence': path.name})
            validate_book(book)
            save(output / 'libro.json', book)
            save(output / 'procedencia.json', provenance)
            save(output / 'progreso.json', {'completed': len(checks), 'total': len(order),
                                          'checkpoint_sha256': checks, 'entries': len(book)})
            budget.wait()
        return {'positions': len(order), 'entries': len(book), 'output': str(output)}
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--buscador', type=Path, required=True)
    parser.add_argument('--pesos', type=Path, required=True)
    parser.add_argument('--posiciones', type=Path, required=True)
    parser.add_argument('--poblacion', required=True)
    parser.add_argument('--profundidad', type=int, required=True)
    parser.add_argument('--nodos', type=int, required=True)
    parser.add_argument('--semillas', required=True, help='Solo ordenamiento; no semillas de partidas')
    parser.add_argument('--max-posiciones', type=int, required=True)
    parser.add_argument('--salida', type=Path, required=True, help='Directorio experimental NUEVO')
    parser.add_argument('--cpu-fraction', type=float, default=0.12)
    parser.add_argument('--reanudar', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
