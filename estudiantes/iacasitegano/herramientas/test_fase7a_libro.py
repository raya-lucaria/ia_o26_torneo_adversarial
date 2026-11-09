"""Regresiones funcionales pequeñas; no liga ni benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import fase7a_libro as lib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', required=True, type=Path)
    args = parser.parse_args()
    out = args.salida.resolve()
    out.mkdir(parents=True, exist_ok=False)
    budget = lib.Budget()
    Estado, Jugada = lib.engine()
    import numpy as np
    results = []

    def check(name, condition, evidence=None):
        assert condition, name
        results.append({'name': name, 'passed': True, 'evidence': evidence})
        budget.wait()

    def rejected(name, function):
        try:
            function()
        except (ValueError, FileExistsError):
            check(name, True)
        else:
            raise AssertionError(name)

    white = Estado.desde_dict({'blancas': ['a8'], 'negras': ['b9', 'i8'], 'turno': 'blancas',
                              'puntos': {'blancas': 2, 'negras': 0}})
    black = Estado.desde_dict({'blancas': ['h1', 'a2'], 'negras': ['i2'], 'turno': 'negras',
                              'puntos': {'blancas': 0, 'negras': 2}})
    positions = {'usage': 'synthetic_fixture', 'provenance': 'Regresiones analíticas; no datos5/test9',
                 'positions': [{'state': s.a_dict(), 'provenance': f'fixture_{s.turno}_tercer_punto'} for s in [white, black]]}
    lib.save(out / 'posiciones.json', positions)
    for state, quiet, attack in [(white, 'a8-a9', 'a8-b9'), (black, 'i2-i1', 'i2-h1')]:
        j, a = Jugada.desde_texto(quiet), Jugada.desde_texto(attack)
        win = lib.transitions(state, j)[0][1]
        branches = lib.transitions(state, a)
        check('anotacion_retirada_' + state.turno, win.puntos[state.turno] == 3 and not win.piezas(state.turno)
              and win.ganador() == state.turno, win.a_dict())
        check('ataque_dos_resultados_' + state.turno, len(branches) == 2 and [p for p, _ in branches] == [0.5, 0.5]
              and branches[0][1].puntos[state.turno] == 3 and branches[1][1].puntos[state.turno] == 2,
              [{'p': p, 'state': child.a_dict()} for p, child in branches])
        check('preservar_victoria_' + state.turno, not lib.safe_selection(state, [a.origen, a.destino])
              and lib.safe_selection(state, [j.origen, j.destino]))
    initial = Estado.inicial()
    variants = []
    for color in ['blancas', 'negras']:
        for a in range(3):
            for b in range(3):
                d = initial.a_dict()
                d.update(turno=color, puntos={'blancas': a, 'negras': b})
                variants.append(Estado.desde_dict(d).clave)
    check('clave_incluye_turno_y_ambos_puntos', len(set(variants)) == 18, variants)
    check('esquema_uint64_jugada', lib.validate_book({str(white.clave): [63, 72]}))
    for bad in [{'01': [63, 72]}, {str(2**64): [63, 72]}, {'1': [True, 1]}, {'1': [80, 81]}, {'1': [8, 18]}]:
        rejected('esquema_rechazado_' + str(bad), lambda bad=bad: lib.validate_book(bad))
    rejected('salida_data_activo_rechazada', lambda: lib.safe_output(lib.TEAM / 'data'))
    search = lib.TEAM / 'busqueda_eval3.py'
    weights = lib.TEAM / 'data/pesos_eval3_ligero.npz'
    module, model = lib.load_search(search, weights)
    check('ganancia_P_cargada_no_fijada_a_uno', model['terminal_gain'] > 1 and model['scale'] == 'eval3_raw_logit', model)
    aborted = lib.evaluate(initial, module, 2, 2, budget)
    check('raiz_abortada_desconocida', aborted['move'] is None and aborted['status'] == 'unknown'
          and aborted['reason'] == 'aborted_or_incomplete_root', aborted)
    lib.save(out / 'aborto.json', aborted)

    def independent(state, perspective, depth):
        winner = state.ganador()
        if winner:
            return model['terminal_gain'] * (1 if winner == perspective else -1)
        if depth == 0:
            tab, pts, _ = module._tablero_desde_estado(state)
            return float(module._evaluar_numba(tab, pts, 0 if perspective == 'blancas' else 1, module.PESOS))
        vals = [sum(p * independent(child, perspective, depth - 1) for p, child in state.siguiente(j))
                for j in state.jugadas()]
        return (max if state.turno == perspective else min)(vals)

    for state in [white, black]:
        teacher = lib.Teacher(module, 200, budget)
        for depth in [1, 2]:
            rows = teacher.alternatives(state, depth)
            diffs = []
            for row in rows:
                expected = sum(p * independent(child, state.turno, depth - 1)
                               for p, child in state.siguiente(Jugada(*row['move'])))
                diffs.append(abs(expected - row['value']))
            check(f'enumeracion_independiente_{state.turno}_D{depth}', max(diffs) <= 1e-10 * model['terminal_gain'], diffs)
    bad = [63, 73]
    fake = [{'alternatives': [{'move': bad, 'value': model['terminal_gain']}, {'move': [63, 72], 'value': 0}]}] * 2
    check('calificacion_no_desplaza_victoria_segura', lib.qualify(white, fake, model['terminal_gain'])[1] == 'would_displace_safe_win')
    tied = [{'alternatives': [{'move': [63, 72], 'value': 1.0}, {'move': bad, 'value': 1.0}]}] * 2
    check('empate_desconocido', lib.qualify(white, tied, model['terminal_gain'])[0] is None)
    # Defecto de orden existente: reproducción solo en memoria, sin preparar/calentar.
    import strategy
    strategy.libro = {str(white.clave): bad}
    strategy.color = 'blancas'
    chosen = strategy.jugar(white, 30.0)
    check('runtime_existente_libro_precede_victoria_segura_reproducido', [chosen.origen, chosen.destino] == bad,
          {'chosen': bad, 'safe_move': [63, 72], 'source_sha256': lib.sha(lib.TEAM / 'strategy.py'), 'source_modified': False})

    base = [sys.executable, '-B', str(Path(lib.__file__)), '--pesos', str(weights), '--posiciones', str(out / 'posiciones.json'),
            '--poblacion', 'fixtures_sinteticos_7A', '--profundidad', '1', '--nodos', '200',
            '--semillas', '7071001,7071002', '--max-posiciones', '2']

    def cli(command, expected=0):
        result = subprocess.run(command, capture_output=True, text=True)
        check('CLI_' + str(len(results)), result.returncode == expected if expected == 0 else result.returncode != 0,
              {'command': command, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-2500:]})
        return result

    command = base + ['--buscador', str(search), '--salida', str(out / 'funcional_P')]
    cli(command)
    generated = json.loads((out / 'funcional_P/libro.json').read_text())
    check('libro_funcional_dos_colores_legal', len(generated) == 2 and all(Jugada(*generated[str(s.clave)]) in s.jugadas() for s in [white, black]))
    hashes = {str(p): lib.sha(p) for p in (out / 'funcional_P').glob('*.json')}
    cli(command, expected=1)
    check('salida_existente_no_sobrescrita', all(lib.sha(Path(p)) == h for p, h in hashes.items()))
    cli(command + ['--reanudar'])
    check('reanudar_idempotente', all(lib.sha(Path(p)) == h for p, h in hashes.items()))
    with np.load(weights) as data:
        changed = data['pesos'].copy()
    changed[0] += 1e-8
    np.savez(out / 'modelo_fixture_cambiado.npz', pesos=changed)
    changed_command = command.copy()
    changed_command[changed_command.index('--pesos') + 1] = str(out / 'modelo_fixture_cambiado.npz')
    error = cli(changed_command + ['--reanudar'], expected=1)
    check('modelo_distinto_detectado_en_manifiesto', 'Reanudación rechazada' in error.stderr)
    check('rechazo_no_altera_salida', all(lib.sha(Path(p)) == h for p, h in hashes.items()))
    utility = Path('/Users/josema/.codex/worktrees/utilidad-victoria/BreakthroughBotIA/estudiantes/iacasitegano/busqueda_eval3.py')
    if not utility.is_file():
        raise FileNotFoundError('Falta fuente U existente para fixture de escala')
    cli(base + ['--buscador', str(utility), '--salida', str(out / 'funcional_U')])
    um = json.loads((out / 'funcional_U/manifiesto.json').read_text())
    check('ganancia_utilidad_y_transformacion_cargadas', um['model']['terminal_gain'] == 1.0
          and um['model']['scale'] == '2_sigmoid_logit_minus_1', um['model'])
    ub = json.loads((out / 'funcional_U/libro.json').read_text())
    ur = [json.loads(p.read_text()) for p in (out / 'funcional_U').glob('posicion_*.json')]
    check('utilidad_descarta_margen_incierto_sin_forzar_entradas', not ub and len(ur) == 2
          and all(r['status'] == 'unknown' and r['reason'] == 'uncertain_margin' for r in ur),
          [{'reason': r['reason'], 'move': r['move']} for r in ur])
    lib.save(out / 'REGRESIONES.json', {'passed': len(results), 'failures': 0, 'cases': results,
                                       'not_strength_or_performance': True, 'policy': lib.POLICY})
    print(json.dumps({'passed': len(results), 'failures': 0, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
