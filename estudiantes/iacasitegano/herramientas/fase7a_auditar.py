"""Cobertura POTENCIAL del libro sobre las trazas cerradas de fase6, sin búsqueda."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import fase7a_libro as common

ROOT = common.ROOT


def csv_file(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze(source, output, budget):
    checksum = {}
    for line in (source / 'SHA256SUMS').read_text().splitlines():
        value, name = line.split('  ', 1)
        checksum[name] = value
    inventory = json.loads((source / 'inventario.json').read_text())
    assert len(inventory['traces']) == 890
    referenced = ['inventario.json', 'manifiesto.json', 'CIERRE.md', 'progreso.json',
                  'estados_unicos.jsonl', 'procedencia.jsonl', 'MANIFIESTO_FINAL.json', 'AUDITORIA_FINAL.json']
    files, traces = {}, []
    for name in referenced:
        assert common.sha(source / name) == checksum[name], name
        files[str(source / name)] = checksum[name]
    for ref in inventory['traces']:
        path = Path(ref['path'])
        assert 'fase5_lote256_mac_20261029' not in str(path)
        assert common.sha(path) == ref['sha256'], path
        name = 'por_traza/' + ref['sha256'] + '.json'
        assert common.sha(source / name) == checksum[name], name
        files[str(source / name)] = checksum[name]
        traces.append({**ref, 'checkpoint': str(source / name), 'checkpoint_sha256': checksum[name]})
        budget.wait()
    original = json.loads((source / 'manifiesto.json').read_text())
    for name in ['sistema/motor/nucleo.py', 'sistema/motor/estado.py', 'sistema/motor/jugada.py']:
        path = ROOT / name
        # El manifiesto6 registra paths absolutos.
        if str(path) in original['sources']:
            assert common.sha(path) == original['sources'][str(path)]['sha256'], path
    book = ROOT / 'estudiantes/iacasitegano/data/libro.json'
    record = {'phase6': str(source), 'files': files, 'traces': traces,
              'book_path': str(book), 'book_sha256': common.sha(book),
              'source_files': {str(p): common.sha(p) for p in [Path(__file__), Path(common.__file__),
                    ROOT / 'sistema/motor/nucleo.py', ROOT / 'sistema/motor/estado.py', ROOT / 'sistema/motor/jugada.py']},
              'protocol_sha256': common.sha(output / 'PROTOCOLO.md'), 'population_fixed': 890,
              'scope': 'Potential lookup at own recorded roots only, not historical activation or strength'}
    target = output / 'manifiesto_cobertura.json'
    if target.exists():
        common.identity_matches(json.loads(target.read_text()), record)
    else:
        common.save(target, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fase6', type=Path, default=ROOT / 'resultados/medicion/fase6_cobertura_20261029')
    parser.add_argument('--salida', type=Path, default=ROOT / 'resultados/medicion/fase7A_aperturas_20261029')
    args = parser.parse_args()
    source, output = args.fase6.resolve(), args.salida.resolve()
    budget = common.Budget()
    frozen = freeze(source, output, budget)
    Estado, Jugada = common.engine()
    book = json.loads(Path(frozen['book_path']).read_text())
    common.validate_book(book)
    folder = output / 'por_traza'
    folder.mkdir(exist_ok=True)
    cache, key_states, matched, observed_keys = {}, {}, [], set()
    totals, roots, all_entries = defaultdict(list), [], {k: [] for k in book}
    for number, ref in enumerate(frozen['traces'], 1):
        data = json.loads(Path(ref['checkpoint']).read_text())
        assert data['replay_valid'] and data['trace_sha256'] == ref['sha256']
        trace_roots = []
        for row in data['decisions']:
            state_id = common.fingerprint(row['state'])
            assert state_id == row['state_sha256']
            if state_id not in cache:
                state = Estado.desde_dict(row['state'])
                key = str(state.clave)
                if key in key_states and key_states[key] != state_id:
                    raise ValueError('Colisión observada de estados completos en clave Zobrist')
                key_states[key] = state_id
                found = key in book
                summary = {'state_sha256': state_id, 'key': key, 'found': found,
                           'legal': False, 'forced': False, 'potential_use': False,
                           'safe_win_available': False, 'safe_win_preserved': False, 'safe_win_displaced': False}
                if found:
                    legal = state.jugadas()
                    move = Jugada(*book[key])
                    safe = common.safe_moves(state)
                    summary.update(legal=move in legal, forced=len(legal) == 1,
                                   potential_use=move in legal and len(legal) > 1,
                                   safe_win_available=bool(safe),
                                   safe_win_preserved=bool(safe) and book[key] in safe,
                                   safe_win_displaced=bool(safe) and move in legal and book[key] not in safe)
                    observed_keys.add(key)
                    all_entries[key].append({'state_sha256': state_id, 'state': state.a_dict(), 'legal': move in legal,
                                             'safe_moves': safe, 'move': book[key]})
                cache[state_id] = summary
            summary = cache[state_id]
            root = {**summary, 'population': ref['population'], 'color': row['color'], 'ply': row['ply'],
                    'trace_sha256': ref['sha256'], 'trace': ref['path'], 'version': ref['version'],
                    'kind': ref['kind'], 'status': ref['status']}
            roots.append(root)
            trace_roots.append(root)
            if root['found']:
                matched.append(root)
            budget.wait()
        common.save(folder / (ref['sha256'] + '.json'), {'reference': ref, 'roots': trace_roots})
        common.save(output / 'progreso.json', {'status': 'coverage_running', 'completed': number, 'total': 890,
                                             'decisions': len(roots), 'unique_states': len(cache)})
        budget.wait()
    assert len(roots) == 36730 and len(cache) == 35094
    fields = ['found', 'legal', 'forced', 'potential_use', 'safe_win_available', 'safe_win_preserved', 'safe_win_displaced']

    def summarize(rows, group_fields):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[f] for f in group_fields)].append(row)
        results = []
        for group, members in sorted(groups.items()):
            results.append({**dict(zip(group_fields, group)), 'decisions': len(members),
                            'unique_states': len({m['state_sha256'] for m in members}),
                            **{f: sum(m[f] for m in members) for f in fields},
                            'unique_found': len({m['state_sha256'] for m in members if m['found']})})
        return results

    population = summarize(roots, ['population'])
    colors = summarize(roots, ['population', 'color'])
    plies = summarize(roots, ['population', 'color', 'ply'])
    for name, rows in [('cobertura.csv', population), ('cobertura_color.csv', colors), ('cobertura_ply.csv', plies)]:
        csv_file(output / name, rows)
    common.save(output / 'coincidencias.json', matched)
    common.save(output / 'entradas_auditadas.json', {k: {'move': book[k], 'states_observed': v,
                 'coverage': 'observed_legal' if v and all(x['legal'] for x in v) else 'unknown_without_observed_state' if not v else 'observed_illegal'}
                 for k, v in all_entries.items()})
    result = {'traces': 890, 'decisions': len(roots), 'unique_states': len(cache), 'entries': len(book),
              'observed_entries': len(observed_keys), 'zobrist_collisions_observed': 0,
              'root_scope_only': True, 'historical_activation': 'not_measured',
              'population': population, 'colors': colors,
              'no_strength_inference': True, 'book_sha256': frozen['book_sha256']}
    common.save(output / 'cobertura.json', result)
    common.save(output / 'progreso.json', {'status': 'coverage_complete', 'completed': 890, 'total': 890,
                                         'decisions': len(roots), 'unique_states': len(cache)})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
