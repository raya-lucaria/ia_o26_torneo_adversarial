"""Fixtures pequeños 8A, cargados por ruta absoluta. Sin JIT ni paquete oficial."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
import sys
import traceback

from fase8a_soporte import ROOT, OUT, Budget, engine, sha, save


@contextmanager
def variant(label, manifest, loaded):
    folder = Path(manifest['copies'][label])
    saved_path = sys.path.copy()
    aliases = ['eval3', 'busqueda_eval3', 'strategy']
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in aliases}
    created = []

    def load(stem, canonical):
        name = f'_fase8A_{label}_{stem}'
        if name in sys.modules:
            raise ValueError('Alias ya cargado')
        path = (folder / (stem + '.py')).resolve()
        assert sha(path) == manifest['copy_sha256'][label][stem + '.py']
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        created.append(name)
        sys.modules[canonical] = module
        spec.loader.exec_module(module)
        assert Path(module.__file__).resolve() == path
        functions = [getattr(module, name) for name in ['_valor_numba', '_evaluar_numba', 'jugar'] if hasattr(module, name)]
        code_paths = [f.__code__.co_filename for f in functions]
        assert all(Path(p).resolve() == path for p in code_paths)
        loaded.append({'variant': label, 'module_name': name, 'file': str(path), 'sha256': sha(path),
                       'spec_origin': module.__spec__.origin, 'function_code_files': code_paths})
        return module

    try:
        sys.path.insert(0, str(folder))
        evaluation = load('eval3', 'eval3')
        search = load('busqueda_eval3', 'busqueda_eval3')
        strategy = load('strategy', 'strategy')
        assert search.e3 is evaluation and strategy._busqueda is search
        yield search, strategy, evaluation
    finally:
        for name in created:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.path[:] = saved_path
        assert all(sys.modules.get(k, missing) is v for k, v in saved.items())


def tt_checks(manifest, budget, loaded):
    Estado, _ = engine()
    import numpy as np
    from motor.nucleo import MAX_JUGADAS
    from motor import nucleo
    evidence = {'mode': 'python_no_jit', 'TT_entries_fixture': 256, 'depth_max': 2,
                'counterexamples': [], 'windows': [], 'aborts': [], 'terminals': [], 'boundaries': []}
    fixture = Estado.desde_dict({'blancas': ['a3', 'e4'], 'negras': ['b7', 'g6'], 'turno': 'blancas'})
    evidence['original_state'] = fixture.a_dict()
    with variant('P', manifest, loaded) as (p, _, _), variant('TT_A', manifest, loaded) as (a, _, ev):
        with np.load(Path(manifest['copies']['P']) / 'data/pesos_eval3_ligero.npz', allow_pickle=False) as data:
            weights = data['pesos'].copy()
        saved_globals = [(m, m.PESOS, m.GANANCIA, m.TT_TAM, m.TT_MASCARA) for m in [p, a]]
        try:
            for m in [p, a]:
                assert m.QUIESCENCIA_MAX == 0
                m.cargar_pesos(weights.copy())
                m.TT_TAM, m.TT_MASCARA = 256, np.uint64(255)
                assert not hasattr(m._valor_numba, 'signatures'), 'No se permite JIT'
            gain = float(a._pesos_y_ganancia()[1])
            evidence['gain'] = gain
            oracle_nodes = 0

            def oracle(state, depth, objective):
                nonlocal oracle_nodes
                oracle_nodes += 1
                if oracle_nodes > 5000:
                    raise RuntimeError('Oráculo excede fixture pequeño; pendiente ventana exclusiva')
                budget.wait()
                winner = state.ganador()
                if winner:
                    return gain if winner == ['blancas', 'negras'][objective] else -gain
                if depth == 0:
                    return float(ev.evaluar_nb(state._tablero.copy(), state._puntos.copy(), objective, weights))
                values = [sum(prob * oracle(child, depth - 1, objective) for prob, child in state.siguiente(j)) for j in state.jugadas()]
                return (max if state.turno == ['blancas', 'negras'][objective] else min)(values)

            def table():
                return np.zeros(256, np.uint64), np.zeros(256)

            def root_key(state, depth):
                return np.uint64(state.clave) ^ np.uint64(depth)

            def query(module, state, objective, alpha, beta, keys, values, abort=False, cap=5000, depth=2):
                tab, points, color = module._tablero_desde_estado(state)
                initial_tab, initial_points = tab.copy(), points.copy()
                counter = np.array([0, int(abort)], np.int64)
                result = module._valor_numba(tab, points, color, objective, depth, alpha, beta, module.PESOS, gain,
                           np.empty(MAX_JUGADAS, np.int64), np.empty(MAX_JUGADAS, np.int64), keys, values, counter, cap)
                assert np.array_equal(tab, initial_tab) and np.array_equal(points, initial_points)
                budget.wait()
                return float(result), [int(x) for x in counter]

            for objective in [0, 1]:
                expected = oracle(fixture, 2, objective)
                # Ventanas originales intactas; no sustituir los contraejemplos.
                windows = [(0.1, 1.0), (1.0, 2.0), (-2.0, -1.0), (gain - 1, gain), (-gain, 1 - gain)]
                for alpha, beta in windows:
                    row = {'objective': objective, 'depth': 2, 'window': [alpha, beta], 'oracle': expected}
                    for label, module in [('P', p), ('TT_A', a)]:
                        keys, values = table()
                        narrow, count1 = query(module, fixture, objective, alpha, beta, keys, values)
                        key = root_key(fixture, 2)
                        stored = bool(keys[int(key & module.TT_MASCARA)] == key)
                        wide, count2 = query(module, fixture, objective, -gain, gain, keys, values)
                        clean, count3 = query(module, fixture, objective, -gain, gain, *table())
                        assert not any(c[1] for c in [count1, count2, count3])
                        assert abs(clean - expected) <= 1e-12
                        row[label] = {'narrow': narrow, 'wide_shared': wide, 'wide_clean': clean,
                                      'root_stored_after_narrow': stored, 'node_counters': [count1, count2, count3]}
                        if label == 'TT_A':
                            assert abs(wide - expected) <= 1e-12
                            if not alpha < narrow < beta:
                                assert not stored, 'Una cota se almacenó como exacta'
                        elif abs(wide - expected) > 1e-12:
                            detail = {'objective': objective, 'window': [alpha, beta], 'oracle': expected, 'P_wide': wide}
                            evidence['counterexamples'].append(detail)
                    evidence['windows'].append(row)
                for alpha, beta in [(expected, expected + 1), (expected - 1, expected), (expected - 0.1, expected + 0.1)]:
                    keys, values = table()
                    value, _ = query(a, fixture, objective, alpha, beta, keys, values)
                    key = root_key(fixture, 2)
                    stored = bool(keys[int(key & a.TT_MASCARA)] == key)
                    if not alpha < value < beta:
                        assert not stored
                    wide, _ = query(a, fixture, objective, -gain, gain, keys, values)
                    assert abs(wide - expected) <= 1e-10
                    if alpha < expected < beta:
                        assert stored, 'Control positivo: TT quedó deshabilitada'
                    evidence['boundaries'].append({'objective': objective, 'window': [alpha, beta], 'value': value,
                                                   'root_stored': stored, 'wide': wide, 'oracle': expected})
                for initial_abort, cap in [(True, 5000), (False, 1)]:
                    keys, values = table()
                    value, counter = query(a, fixture, objective, -gain, gain, keys, values, initial_abort, cap)
                    assert counter[1] == 1 and not np.any(keys) and not np.any(values)
                    wide, _ = query(a, fixture, objective, -gain, gain, keys, values)
                    assert abs(wide - expected) <= 1e-12
                    evidence['aborts'].append({'objective': objective, 'pre_aborted': initial_abort, 'cap': cap,
                                               'result_discarded': value, 'counter': counter,
                                               'stores_before_wide': 0, 'wide_after': wide})
            failures = evidence['counterexamples']
            assert len(failures) == 2 and {d['objective'] for d in failures} == {0, 1}
            assert all(abs(abs(d['P_wide']) - 210.20324999290867) <= 1e-12 and abs(d['oracle']) <= 1e-12 for d in failures)
            terminals = [Estado.desde_dict(d) for d in [
                {'blancas': ['a3'], 'negras': ['g6'], 'turno': 'negras', 'puntos': {'blancas': 3, 'negras': 1}},
                {'blancas': ['a3'], 'negras': ['g6'], 'turno': 'blancas', 'puntos': {'blancas': 1, 'negras': 3}},
                {'blancas': [], 'negras': ['a5'], 'turno': 'blancas'}]]
            for state in terminals:
                for objective in [0, 1]:
                    keys, values = table()
                    expected = oracle(state, 2, objective)
                    actual, counter = query(a, state, objective, -gain, gain, keys, values)
                    assert actual == expected and counter[1] == 0 and not np.any(keys)
                    evidence['terminals'].append({'state': state.a_dict(), 'objective': objective, 'oracle': expected, 'actual': actual})
            evidence.update(oracle_nodes=oracle_nodes, corrected_failures=0, passed=True,
                            oracle='independent exhaustive motor transitions, eval3 leaves, no TT/pruning',
                            unchanged_inputs=True, full_extended_suite='pending_exclusive_window')
        finally:
            for m, weights_before, gain_before, size_before, mask_before in saved_globals:
                m.PESOS, m.GANANCIA, m.TT_TAM, m.TT_MASCARA = weights_before, gain_before, size_before, mask_before
    evidence['temporary_search_globals_restored'] = True
    return evidence


def book_checks(manifest, budget, loaded):
    Estado, Jugada = engine()
    rows, fixtures = [], []

    def state(white, black, color, pw=0, pb=0):
        return Estado.desde_dict({'blancas': white, 'negras': black, 'turno': color, 'puntos': {'blancas': pw, 'negras': pb}})

    for color in ['blancas', 'negras']:
        if color == 'blancas':
            e = state(['a8'], ['b9', 'i8'], color, 2, 0)
            unsafe, safe = Jugada.desde_texto('a8-b9'), Jugada.desde_texto('a8-a9')
            probability_only = state(['b8'], ['a9', 'b9', 'c9', 'i8'], color, 2, 0)
            forced = state(['a2'], ['a3', 'i8'], color)
        else:
            e = state(['h1', 'a2'], ['i2'], color, 0, 2)
            unsafe, safe = Jugada.desde_texto('i2-h1'), Jugada.desde_texto('i2-i1')
            probability_only = state(['g1', 'h1', 'i1', 'a2'], ['h2'], color, 0, 2)
            forced = state(['i7', 'a2'], ['i8'], color)
        fixtures.extend([
            {'name': 'libro_desplaza_victoria_' + color, 'state': e, 'entry': unsafe, 'P_expected': unsafe,
             'corrected_expected': safe, 'search_calls': 0},
            {'name': 'solo_ataques_probabilisticos_' + color, 'state': probability_only,
             'entry': probability_only.jugadas()[-1], 'P_expected': probability_only.jugadas()[-1],
             'corrected_expected': probability_only.jugadas()[-1], 'search_calls': 0},
            {'name': 'forzada_' + color, 'state': forced, 'entry': Jugada(0, 80), 'P_expected': forced.jugadas()[0],
             'corrected_expected': forced.jugadas()[0], 'search_calls': 0}])
        initial_dict = Estado.inicial().a_dict()
        initial_dict['turno'] = color
        quiet = Estado.desde_dict(initial_dict)
        chosen = quiet.jugadas()[-1]
        fixtures.extend([
            {'name': 'libro_legal_sin_victoria_' + color, 'state': quiet, 'entry': chosen, 'P_expected': chosen,
             'corrected_expected': chosen, 'search_calls': 0},
            {'name': 'libro_ilegal_busqueda_simulada_' + color, 'state': quiet, 'entry': Jugada(0, 80), 'P_expected': chosen,
             'corrected_expected': chosen, 'search_calls': 1}])
    for label in ['P', 'PRIORIDAD_LIBRO']:
        with variant(label, manifest, loaded) as (search, strategy, _):
            for fixture in fixtures:
                e, entry = fixture['state'], fixture['entry']
                expected = fixture['P_expected' if label == 'P' else 'corrected_expected']
                missing = object()
                names = ['color', 'libro', 'rng', 'reserva', 'reportar']
                old = {name: strategy.__dict__.get(name, missing) for name in names}
                original_choose = search.elegir
                calls = []

                def simulated(s, color, limit):
                    assert s is e and color == e.turno and fixture['search_calls'] == 1
                    calls.append({'state_identity': True, 'color': color, 'return': [expected.origen, expected.destino]})
                    return expected, 0.0, 1

                try:
                    strategy.color, strategy.libro = e.turno, {str(e.clave): [entry.origen, entry.destino]}
                    strategy.rng, strategy.reserva = random.Random(8081001), 0.0
                    strategy.reportar = lambda **kwargs: None
                    search.elegir = simulated
                    actual = strategy.jugar(e, 30.0)
                    assert actual == expected and actual in e.jugadas() and len(calls) == fixture['search_calls']
                    transitions = [{'probability': p, 'state': child.a_dict(), 'winner': child.ganador()} for p, child in e.siguiente(actual)]
                    if fixture['name'].startswith('libro_desplaza'):
                        attack = [{'probability': p, 'state': child.a_dict(), 'winner': child.ganador()} for p, child in e.siguiente(entry)]
                        assert len(attack) == 2 and [t['probability'] for t in attack] == [0.5, 0.5]
                        assert attack[0]['winner'] == e.turno and attack[1]['winner'] != e.turno
                        assert (all(t['winner'] == e.turno for t in transitions)) == (label == 'PRIORIDAD_LIBRO')
                    else:
                        attack = None
                    if fixture['name'].startswith('solo_ataques'):
                        assert len(e.jugadas()) == 2 and not any(all(child.ganador() == e.turno for _, child in e.siguiente(j)) for j in e.jugadas())
                    if fixture['name'].startswith('forzada'):
                        assert len(e.jugadas()) == 1
                    row = {'variant': label, 'fixture': fixture['name'], 'state': e.a_dict(), 'key': str(e.clave),
                           'book_entry': [entry.origen, entry.destino], 'actual': [actual.origen, actual.destino],
                           'expected': [expected.origen, expected.destino], 'search_calls': calls,
                           'transitions_of_chosen': transitions, 'attack_branches': attack, 'passed': True}
                finally:
                    search.elegir = original_choose
                    for name, value in old.items():
                        if value is missing:
                            strategy.__dict__.pop(name, None)
                        else:
                            strategy.__dict__[name] = value
                    assert search.elegir is original_choose and all(strategy.__dict__.get(k, missing) is v for k, v in old.items())
                row['temporary_globals_restored'] = True
                rows.append(row)
                budget.wait()
    return {'passed': True, 'calls_checked': len(rows), 'fixtures': rows, 'no_preparar_calentar': True,
            'simulated_search_not_runtime_change': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bloque', choices=['TT', 'libro', 'ambos'], default='ambos')
    args = parser.parse_args()
    budget, loaded = Budget(), []
    manifest = json.loads((OUT / 'manifiesto_inicial.json').read_text())
    engine()
    results = []
    try:
        if args.bloque in ['TT', 'ambos']:
            path = OUT / 'TT_FUNCIONAL.json'
            if path.exists():
                raise FileExistsError('Resultado TT ya existe; conservarlo')
            data = tt_checks(manifest, budget, loaded)
            save(path, data)
            results.append('TT')
        if args.bloque in ['libro', 'ambos']:
            path = OUT / 'LIBRO_FUNCIONAL.json'
            if path.exists():
                raise FileExistsError('Resultado libro ya existe; conservarlo')
            data = book_checks(manifest, budget, loaded)
            save(path, data)
            results.append('libro')
        save(OUT / 'MODULOS_CARGADOS.json', loaded)
        save(OUT / 'progreso.json', {'status': 'functional_checks_complete',
              'TT_functional': 'passed' if (OUT/'TT_FUNCIONAL.json').exists() else 'pending',
              'book_functional': 'passed' if (OUT/'LIBRO_FUNCIONAL.json').exists() else 'pending',
              'extended_jit_linux': 'pending_exclusive_window_8B'})
        budget.wait()
        print(json.dumps({'completed_blocks': results, 'modules_loaded_by_absolute_path': len(loaded)}, indent=2))
    except Exception:
        save(OUT / 'FALLO_FUNCIONAL.json', {'blocks_completed': results, 'traceback': traceback.format_exc(), 'loaded_modules': loaded})
        raise


if __name__ == '__main__':
    main()
