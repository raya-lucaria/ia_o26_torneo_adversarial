"""Composición I0 desde copias/parches8A, sin editar fuentes congeladas."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from fase8a_soporte import ROOT, TEAM, Budget, sha, save

OUT = ROOT/'resultados/medicion/fase8B1_paquete_20261029'
PREVIOUS = ROOT/'resultados/medicion/fase8A_preintegracion_20261029'


def main():
    budget = Budget()
    if (OUT/'manifiesto_inicial.json').exists():
        raise FileExistsError('I0 ya preparada; no sobrescribir')
    previous = json.loads((PREVIOUS/'manifiesto_inicial.json').read_text())
    hashes8a = {}
    for line in (PREVIOUS/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = PREVIOUS/name
        assert sha(path, budget) == expected, path
        hashes8a[str(path)] = expected
    protocol5 = ROOT/'resultados/medicion/fase5_lote256_mac_20261029/PROTOCOLO.json'
    phase5 = json.loads(protocol5.read_text())['fuentes']
    assert len(phase5) == 17 and all(sha(Path(p), budget) == expected for p, expected in phase5.items())
    runtime = {p: sha(Path(p), budget) for p in previous['runtime']}
    cache = {str(p): sha(p, budget) for p in sorted((TEAM/'data/numba_cache').rglob('*')) if p.is_file()}
    reviewed = [ROOT/'docs/reglas.md', ROOT/'docs/entregar.md', ROOT/'sistema/limites.toml', ROOT/'sistema/config.py',
                *sorted((ROOT/'sistema/validador').glob('*.py')), ROOT/'sistema/cli/comandos.py',
                ROOT/'sistema/herramientas/comparar_entrega.py',
                TEAM/'herramientas/fase5_empaquetar.py', TEAM/'herramientas/fase5_empaquetar_sesion.py', TEAM/'herramientas/pc/empaquetar.py',
                TEAM/'herramientas/fase8a_soporte.py', TEAM/'herramientas/fase8a_verificar.py',
                TEAM/'equipo.toml', TEAM/'requirements.txt', TEAM/'INFORME.md', protocol5]
    origins = {str(p): sha(p, budget) for p in reviewed}
    source = Path(previous['copies']['P'])
    destination = OUT/'I0/estudiantes/iacasitegano'
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in previous['copy_sha256']['P'].items():
        assert sha(source/relative) == expected
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, target)
    patch_files = list(sorted((PREVIOUS/'parches').glob('*.patch')))
    (OUT/'parches').mkdir(exist_ok=False)
    applied = []
    for patch in patch_files:
        local_patch = OUT/'parches'/patch.name
        shutil.copyfile(patch, local_patch)
        command = ['rtk','proxy','patch','--batch','--forward','--fuzz=0','-p1','-i',str(local_patch)]
        result = subprocess.run(command, cwd=OUT/'I0', capture_output=True, text=True)
        evidence = {'command': command, 'cwd': str(OUT/'I0'), 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
                    'patch_sha256': sha(local_patch)}
        applied.append(evidence)
        save(OUT/'aplicacion_parches.json', applied)
        assert result.returncode == 0 and 'offset' not in result.stdout.lower() and 'fuzz' not in result.stdout.lower()
    assert sha(destination/'busqueda_eval3.py') == 'cc888a3c7fc7ba896f5145c36c56c77120bc6f0ef0b0ba45f077023bbeb7a177'
    assert sha(destination/'strategy.py') == '88ce9d4718d73e1fbb4ae5901b9ea08b79fc9d6f314a498d64ad7d0b25de4d43'
    assert (destination/'busqueda_eval3.py').read_bytes() == (Path(previous['copies']['TT_A'])/'busqueda_eval3.py').read_bytes()
    assert (destination/'strategy.py').read_bytes() == (Path(previous['copies']['PRIORIDAD_LIBRO'])/'strategy.py').read_bytes()
    for relative in ['eval3.py','data/pesos_eval3_ligero.npz','data/libro.json']:
        assert (destination/relative).read_bytes() == (source/relative).read_bytes()
    for relative in ['equipo.toml','requirements.txt']:
        shutil.copyfile(TEAM/relative, destination/relative)
    report = '''# I0 — informe experimental de composición, no entrega aprobada

## 1. Integrantes

Equipo iacasitegano: Jose y Daphne, según equipo.toml copiado sin cambios. Este documento describe únicamente el ensayo I0; no es el informe final del modelo seleccionado.

## 2. Idea en un párrafo

I0 combina en una carpeta aislada dos correcciones mínimas desde el principal P: no guardar cotas TT como valores exactos y preferir una victoria inmediata garantizada sobre una entrada de libro. No sustituye al participante activo ni al control U de entrenamiento.

## 3. Búsqueda (algoritmo y por qué)

Expectiminimax con poda y transiciones probabilísticas del motor, con corrección TT equivalente al control A. Conserva horizonte, hojas, orden y reloj de P. La tabla guarda exactos solamente con resultado interior a la ventana de entrada y sin aborto/corte. No contiene pistas internas rechazadas.

## 4. Función de evaluación

Evaluación lineal eval3 y el NPZ275 de respaldo congelado, no un modelo nuevo seleccionado. Este buscador usa score crudo de eval3 y terminales con cota más margen; no se equipara a utilidad tanh de U. No hay red neuronal. Familias activas originales sin retocar pesos/banderas.

## 5. Precomputación (qué y cómo regenerarla)

Los dos archivos data/libro.json y data/pesos_eval3_ligero.npz son copias exactas de los datos de respaldo identificados en el manifiesto externo. No se generaron entradas ni modelos. El constructor y su receta fuera del paquete reconstruyen este árbol desde fuentes por SHA; la procedencia histórica del libro y la receta del modelo se conservan por referencia externa. Generar un libro nuevo es otro experimento pendiente de selección de modelo. No se incluye caché Mac; la caché del entorno objetivo y su costo quedan pendientes.

## 6. Manejo del reloj

Se conserva el reparto/reserva y respaldo legal de P; solo cambia la prioridad de una victoria segura respecto al libro. Preparación, reloj y memoria pico no se midieron en I0. Se requiere ventana exclusiva para verificar límites oficiales y arranque frío del paquete final.

## 7. Resultados vs random y montecarlo

No se jugaron partidas con I0 contra random, Montecarlo ni otros rivales. No tiene validación deportiva o de humo oficial, y no se declara mejora ni finalista. Los fixtures de composición son comprobaciones funcionales, no resultados de fuerza. Partidas y semillas requieren protocolo posterior.

## 8. Qué intentamos y no funcionó

P conserva en activo los dos contraejemplos TT y la prioridad antigua del libro; I0 solamente ensaya sus arreglos ya revisados. Las pistas TT de fase2 y finales de fase6 permanecen cerrados/rechazados en su alcance, sin reabrirlos. Modelo y libro definitivos siguen pendientes; no se rescata el modelo N rechazado.

## 9. Fuentes y uso de IA

Motor, reglas y validador originales del torneo; código propio de iacasitegano. Codex asistió en revisión, parches mínimos, empaquetado y fixtures. La evidencia de fases8A/8B.1, SHA, comandos y referencias históricas vive fuera del paquete de juego para evitar rutas de máquina y resultados de entrenamiento dentro de él. Este borrador debe revisarse y reemplazarse por informe final coherente con la decisión explícita posterior.
'''
    (destination/'INFORME.md').write_text(report)
    files = {str(p.relative_to(destination)): sha(p, budget) for p in sorted(destination.rglob('*')) if p.is_file()}
    save(OUT/'manifiesto_inicial.json', {'runtime': runtime, 'active_cache': cache, 'phase5_sources': phase5,
             'origins': origins, 'phase8A_artifacts': hashes8a, 'I0_root': str(destination), 'I0_files': files,
             'patches': {str(OUT/'parches'/p.name): sha(p) for p in patch_files}, 'protocol_sha256': sha(OUT/'PROTOCOLO.md'),
             'original_P_root': str(source), 'independent_hashes_composed': True, 'selected_model5': False,
             'report': 'new experimental report; active report preserved; final report pending'})
    save(OUT/'progreso.json', {'status': 'I0_composed', 'composition_fixture': 'pending', 'package': 'pending', 'static': 'pending'})
    budget.wait()
    print(json.dumps({'I0_root': str(destination), 'files': len(files), 'phase5_sources_verified': len(phase5)}, indent=2))


if __name__ == '__main__':
    main()
