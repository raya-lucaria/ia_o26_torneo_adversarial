"""Crea copias y parches independientes. Nunca aplica un parche al runtime activo."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import shutil
import subprocess

from fase8a_soporte import ROOT, OUT, TEAM, Budget, sha, save

PATCH_SHA = '38591003cb9be5dc6513289310f8823310575aac9f88855e72610a2675842abd'
A_SHA = 'cc888a3c7fc7ba896f5145c36c56c77120bc6f0ef0b0ba45f077023bbeb7a177'
REGRESSION = Path('/Users/josema/.codex/worktrees/tt-cotas-exactas/BreakthroughBotIA/estudiantes/iacasitegano/herramientas/verificar_cotas_tt.py')


def main():
    budget = Budget()
    target = OUT / 'manifiesto_inicial.json'
    if target.exists():
        raise FileExistsError('Preparación ya congelada; no sobrescribir')
    protocol5 = ROOT / 'resultados/medicion/fase5_lote256_mac_20261029/PROTOCOLO.json'
    p5 = json.loads(protocol5.read_text())
    phase5 = {name: {'expected': expected, 'actual': sha(Path(name), budget)} for name, expected in p5['fuentes'].items()}
    assert all(x['actual'] == x['expected'] for x in phase5.values()), phase5
    relatives = ['strategy.py', 'busqueda_eval3.py', 'eval3.py', 'data/pesos_eval3_ligero.npz', 'data/libro.json']
    runtime = {str(TEAM / name): sha(TEAM / name, budget) for name in relatives}
    for p in sorted((ROOT / 'sistema/motor').glob('*.py')):
        runtime[str(p)] = sha(p, budget)
    cache = {str(p): sha(p, budget) for p in sorted((TEAM / 'data/numba_cache').rglob('*')) if p.is_file()}
    source_patch = ROOT / 'resultados/medicion/cierre_fase2_20261029/P_a_A_correccion_TT.patch'
    assert sha(source_patch) == PATCH_SHA
    assert sha(REGRESSION) == 'f5464d8fa46f81e6f432cd8219f92944a4b5b974f92019521f40691e1331dc62'
    a_path = REGRESSION.parent.parent / 'busqueda_eval3.py'
    assert sha(a_path) == A_SHA
    origins = [source_patch, REGRESSION, a_path, ROOT / 'resultados/medicion/cierre_fase2_20261029/CIERRE.md',
               ROOT / 'resultados/medicion/fase7A_aperturas_20261029/CIERRE_7A.md',
               ROOT / 'resultados/medicion/fase7A_aperturas_20261029/fixtures_02/REGRESIONES.json',
               ROOT / 'estudiantes/iacasitegano/herramientas/test_fase7a_libro.py', protocol5]
    origin_hashes = {str(p): sha(p, budget) for p in origins}
    (OUT / 'origenes').mkdir(exist_ok=False)
    shutil.copyfile(source_patch, OUT / 'origenes/P_a_A_correccion_TT.patch')
    shutil.copyfile(REGRESSION, OUT / 'origenes/verificar_cotas_tt.py')
    shutil.copyfile(origins[5], OUT / 'origenes/REGRESIONES_7A.json')
    copies = {}
    for label in ['P', 'TT_A', 'PRIORIDAD_LIBRO']:
        base = OUT / 'copias' / label
        team = base / 'estudiantes/iacasitegano'
        team.mkdir(parents=True, exist_ok=False)
        for relative in relatives:
            destination = team / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(TEAM / relative, destination)
        copies[label] = str(team)
    suite_copy = Path(copies['TT_A']) / 'herramientas/verificar_cotas_tt.py'
    suite_copy.parent.mkdir(exist_ok=False)
    shutil.copyfile(REGRESSION, suite_copy)
    patches = OUT / 'parches'
    patches.mkdir(exist_ok=False)
    shutil.copyfile(source_patch, patches / '01_TT_exacta_P_a_A.patch')
    apply = subprocess.run(['rtk', 'proxy', 'patch', '--batch', '--forward', '--fuzz=0', '-p1', '-i',
                            str(patches / '01_TT_exacta_P_a_A.patch')], cwd=OUT / 'copias/TT_A', capture_output=True, text=True)
    save(OUT / 'aplicacion_TT.json', {'command': apply.args, 'cwd': str(OUT / 'copias/TT_A'),
                                    'exit': apply.returncode, 'stdout': apply.stdout, 'stderr': apply.stderr})
    assert apply.returncode == 0
    corrected = Path(copies['TT_A']) / 'busqueda_eval3.py'
    assert sha(corrected) == A_SHA and corrected.read_bytes() == a_path.read_bytes()
    original = (TEAM / 'strategy.py').read_text()
    start = original.index('    entrada_libro = libro.get(str(estado.clave))\n')
    win_start = original.index('    # Si una jugada gana la partida sin importar el dado', start)
    end = original.index('    disponible = tiempo_restante - reserva\n', win_start)
    book_block, win_block = original[start:win_start], original[win_start:end]
    reordered = original[:start] + win_block + book_block + original[end:]
    priority = Path(copies['PRIORIDAD_LIBRO']) / 'strategy.py'
    priority.write_text(reordered)
    patch_text = ''.join(difflib.unified_diff(original.splitlines(keepends=True), reordered.splitlines(keepends=True),
                          fromfile='a/estudiantes/iacasitegano/strategy.py', tofile='b/estudiantes/iacasitegano/strategy.py'))
    (patches / '02_victoria_segura_antes_libro.patch').write_text(patch_text)
    # Verificar parche limpio y su inversa en una carpeta de verificación nueva.
    verification = OUT / 'verificacion_parche_libro/estudiantes/iacasitegano'
    verification.mkdir(parents=True)
    shutil.copyfile(TEAM / 'strategy.py', verification / 'strategy.py')
    patch_checks = []
    for reverse in [False, True]:
        command = ['rtk', 'proxy', 'patch', '--batch', '--fuzz=0', '-p1', '-i', str(patches / '02_victoria_segura_antes_libro.patch')]
        if reverse:
            command.append('-R')
        result = subprocess.run(command, cwd=verification.parents[1], capture_output=True, text=True)
        expected = original if reverse else reordered
        assert result.returncode == 0 and (verification / 'strategy.py').read_text() == expected
        patch_checks.append({'command': command, 'exit': result.returncode, 'stdout': result.stdout, 'reverse': reverse})
    save(OUT / 'aplicacion_prioridad_libro.json', patch_checks)
    copied_hashes = {label: {relative: sha(Path(folder) / relative) for relative in relatives} for label, folder in copies.items()}
    differences = {label: [name for name in relatives if copied_hashes[label][name] != copied_hashes['P'][name]] for label in copies}
    assert differences == {'P': [], 'TT_A': ['busqueda_eval3.py'], 'PRIORIDAD_LIBRO': ['strategy.py']}
    manifest = {'runtime': runtime, 'active_cache': cache, 'phase5_protocol': str(protocol5), 'phase5': phase5,
                'origins': origin_hashes, 'copies': copies, 'copy_sha256': copied_hashes, 'independent_differences': differences,
                'protocol_sha256': sha(OUT / 'PROTOCOLO.md'), 'patches': {str(p): sha(p) for p in patches.glob('*.patch')},
                'original_suite_copy': {'path': str(suite_copy), 'sha256': sha(suite_copy), 'executed': False},
                'TT_matches_A_bytes': True, 'expected_A_sha256': A_SHA,
                'no_N': True, 'read_new_test': False, 'active_sources_modified': False}
    save(target, manifest)
    save(OUT / 'progreso.json', {'status': 'copies_prepared', 'TT_functional': 'pending', 'book_functional': 'pending'})
    budget.wait()
    print(json.dumps({'phase5_sources_verified': len(phase5), 'active_cache_files_frozen': len(cache),
                      'differences': differences, 'TT_SHA': sha(corrected)}, indent=2))


if __name__ == '__main__':
    main()
