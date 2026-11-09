"""Cuatro comprobaciones de composición I0; reutiliza fixtures/loader8A."""
from __future__ import annotations

import json
from pathlib import Path

from fase8a_soporte import ROOT, Budget, engine, sha, save
from fase8a_verificar import variant

OUT = ROOT/'resultados/medicion/fase8B1_paquete_20261029'
PREVIOUS = ROOT/'resultados/medicion/fase8A_preintegracion_20261029'


def main():
    budget = Budget()
    if (OUT/'COMPOSICION_FUNCIONAL.json').exists():
        raise FileExistsError('Resultado ya existe')
    initial = json.loads((OUT/'manifiesto_inicial.json').read_text())
    manifest = {'copies': {'I0': initial['I0_root']}, 'copy_sha256': {'I0': initial['I0_files']}}
    for path, expected in initial['phase8A_artifacts'].items():
        assert sha(Path(path), budget) == expected
    tt = json.loads((PREVIOUS/'TT_FUNCIONAL.json').read_text())
    book = json.loads((PREVIOUS/'LIBRO_FUNCIONAL.json').read_text())
    Estado, Jugada = engine()
    import numpy as np
    from motor.nucleo import MAX_JUGADAS
    loaded, cases = [], []
    with variant('I0', manifest, loaded) as (search, strategy, _):
        before = search.PESOS, search.GANANCIA, search.TT_TAM, search.TT_MASCARA
        try:
            with np.load(Path(initial['I0_root'])/'data/pesos_eval3_ligero.npz', allow_pickle=False) as data:
                search.cargar_pesos(data['pesos'].copy())
            search.TT_TAM, search.TT_MASCARA = 256, np.uint64(255)
            gain = float(search._pesos_y_ganancia()[1])
            assert gain == tt['gain'] and not hasattr(search._valor_numba, 'signatures')
            state = Estado.desde_dict(tt['original_state'])
            for counterexample in tt['counterexamples']:
                keys, values = np.zeros(256, np.uint64), np.zeros(256)
                observed = []
                for alpha, beta in [counterexample['window'], [-gain, gain]]:
                    tab, points, color = search._tablero_desde_estado(state)
                    count = np.zeros(2, np.int64)
                    value = float(search._valor_numba(tab, points, color, counterexample['objective'], 2, alpha, beta,
                           search.PESOS, gain, np.empty(MAX_JUGADAS, np.int64), np.empty(MAX_JUGADAS, np.int64),
                           keys, values, count, 5000))
                    assert count[1] == 0
                    observed.append({'window': [alpha, beta], 'value': value, 'nodes': int(count[0])})
                    budget.wait()
                assert abs(observed[-1]['value']-counterexample['oracle']) <= 1e-12
                cases.append({'kind': 'TT_shared_window', 'state': state.a_dict(), 'objective': counterexample['objective'],
                              'reference8A': counterexample, 'observed': observed, 'passed': True})
        finally:
            search.PESOS, search.GANANCIA, search.TT_TAM, search.TT_MASCARA = before
            assert search.PESOS is before[0]
        for fixture in [r for r in book['fixtures'] if r['variant'] == 'PRIORIDAD_LIBRO' and r['fixture'].startswith('libro_desplaza')]:
            state = Estado.desde_dict(fixture['state'])
            missing = object()
            saved = {name: strategy.__dict__.get(name, missing) for name in ['color','libro']}
            original_choose = search.elegir

            def forbidden_search(*args, **kwargs):
                raise AssertionError('El fixture debe resolverse por victoria segura sin búsqueda')

            try:
                strategy.color, strategy.libro = state.turno, {str(state.clave): fixture['book_entry']}
                search.elegir = forbidden_search
                actual = strategy.jugar(state, 30.0)
                assert [actual.origen, actual.destino] == fixture['expected']
                assert all(child.ganador() == state.turno for _, child in state.siguiente(actual))
                cases.append({'kind': 'safe_win_before_book', 'state': state.a_dict(), 'book': fixture['book_entry'],
                              'actual': [actual.origen, actual.destino], 'expected8A': fixture['expected'], 'passed': True})
            finally:
                search.elegir = original_choose
                for name, value in saved.items():
                    if value is missing:
                        strategy.__dict__.pop(name, None)
                    else:
                        strategy.__dict__[name] = value
                assert search.elegir is original_choose and all(strategy.__dict__.get(n, missing) is v for n,v in saved.items())
            budget.wait()
    assert len(cases) == 4 and len(loaded) == 3
    save(OUT/'COMPOSICION_FUNCIONAL.json', {'passed': 4, 'failures': 0, 'cases': cases, 'loaded': loaded,
             'no_JIT': True, 'globals_and_aliases_restored': True, 'full8A_suite_repeated': False,
             'oracle8A_reused_not_recomputed': True, 'no_preparar_calentar': True})
    save(OUT/'progreso.json', {'status': 'I0_fixture_passed', 'composition_fixture': 'passed4', 'package': 'pending', 'static': 'pending'})
    budget.wait()
    print(json.dumps({'passed': 4, 'failures': 0, 'actual_loaded_modules': loaded}, indent=2))


if __name__ == '__main__':
    main()
