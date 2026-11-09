"""Certifica preparación8A desde evidencia existente; no ejecuta búsquedas."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from fase8a_soporte import ROOT, OUT, TEAM, Budget, sha, save

NOTE = ROOT / 'obsidian/Experimentos/2026-10-29-Fase8A-preintegracion.md'


def link(label, path):
    return f'[{label}](<{path}>)'


def main():
    budget = Budget()
    before = json.loads((OUT / 'antes.json').read_text())
    initial = json.loads((OUT / 'manifiesto_inicial.json').read_text())
    tt = json.loads((OUT / 'TT_FUNCIONAL.json').read_text())
    book = json.loads((OUT / 'LIBRO_FUNCIONAL.json').read_text())
    loaded = json.loads((OUT / 'MODULOS_CARGADOS.json').read_text())
    checks = {'runtime': {p: sha(Path(p), budget) == expected for p, expected in initial['runtime'].items()},
              'phase5': {p: sha(Path(p), budget) == expected['expected'] == expected['actual'] for p, expected in initial['phase5'].items()},
              'active_cache': {p: sha(Path(p), budget) == expected for p, expected in initial['active_cache'].items()},
              'origins': {p: sha(Path(p), budget) == expected for p, expected in initial['origins'].items()},
              'patches': {p: sha(Path(p), budget) == expected for p, expected in initial['patches'].items()}}
    assert all(all(group.values()) for group in checks.values()), checks
    assert {str(p) for p in (TEAM / 'data/numba_cache').rglob('*') if p.is_file()} == set(initial['active_cache'])
    assert sha(OUT / 'PROTOCOLO.md') == initial['protocol_sha256']
    assert all(sha(Path(x['file'])) == x['sha256'] and Path(x['file']).resolve().is_relative_to((OUT/'copias').resolve()) for x in loaded)
    assert all(all(Path(c).resolve() == Path(x['file']).resolve() for c in x['function_code_files']) for x in loaded)
    assert len(loaded) == 12
    copy_checks = {label: {relative: sha(Path(folder) / relative, budget) == initial['copy_sha256'][label][relative]
                   for relative in initial['copy_sha256'][label]} for label, folder in initial['copies'].items()}
    assert all(all(x.values()) for x in copy_checks.values())
    corrected = Path(initial['copies']['TT_A']) / 'busqueda_eval3.py'
    original_a = next(Path(p) for p, h in initial['origins'].items() if h == initial['expected_A_sha256'])
    assert corrected.read_bytes() == original_a.read_bytes()
    suite = Path(initial['original_suite_copy']['path'])
    assert sha(suite) == initial['original_suite_copy']['sha256'] == sha(OUT/'origenes/verificar_cotas_tt.py')
    assert tt['passed'] and tt['corrected_failures'] == 0 and book['passed']
    assert tt['original_state'] == {'blancas': ['a3', 'e4'], 'negras': ['g6', 'b7'], 'turno': 'blancas', 'puntos': {'blancas': 0, 'negras': 0}}
    assert len(tt['counterexamples']) == 2 and {r['objective'] for r in tt['counterexamples']} == {0, 1}
    assert all(abs(abs(r['P_wide'])-210.20324999290867) <= 1e-12 and abs(r['oracle']) <= 1e-12 for r in tt['counterexamples'])
    assert len(tt['windows']) == 10 and len(tt['boundaries']) == 6 and len(tt['aborts']) == 4 and len(tt['terminals']) == 6
    assert all(abs(r['TT_A']['wide_shared']-r['oracle']) <= 1e-12 for r in tt['windows'])
    assert Counter(r['root_stored'] for r in tt['boundaries']) == {False: 4, True: 2}
    assert book['calls_checked'] == 20 and all(r['passed'] and r['temporary_globals_restored'] for r in book['fixtures'])
    assert tt['temporary_search_globals_restored']
    phase5_protocol = json.loads(Path(initial['phase5_protocol']).read_text())
    assert {p: v['expected'] for p, v in initial['phase5'].items()} == phase5_protocol['fuentes']
    priority_sha = initial['copy_sha256']['PRIORIDAD_LIBRO']['strategy.py']
    head = subprocess.check_output(['rtk', 'proxy', 'git', 'rev-parse', 'HEAD'], text=True).strip()
    assert head == before['HEAD']
    handoff = ('8A preparada, sin integración activa ni cierre8: dos parches independientes desde P. '
               'TT coincide byte a byte con A cc888a3c; P reproduce exactamente ±210.20324999290867 frente a oráculo≈0 en los dos casos originales. '
               'Copia corregida pasa10 ventanas originales,6 fronteras/interiores,4 abortos y6 terminales con oráculo sinTT; '
               'prioridad de victoria segura antes del libro pasa20 llamadas en ambos colores. Rutas/SHA/co_filename reales guardados y globals restaurados. '
               'Runtime/caché y17 fuentes congeladas de5 intactos, sin leer test nuevo ni usar N. '
               'Pendientes8B: decisión real de5, composición aislada, suite extendida/JIT, paquete Linux/ABI/caché, frío/costo/seguridad y protocolo oficial propio con ventana exclusiva. '
               'No insertar parches a mitad de U/V de5. Evidencia: resultados/medicion/fase8A_preintegracion_20261029/CIERRE_8A.md.')
    for old, new in {'cierre8':'cierre 8','pasa10':'pasa 10',',6 ':', 6 ', ',4 ':', 4 ','y6 ':'y 6 ','sinTT':'sin TT',
                     'pasa20':'pasa 20','y17':'y 17','de5':'de 5','Pendientes8B':'Pendientes 8B'}.items():
        handoff = handoff.replace(old, new)
    content = f'''# Certificado de preparación 8A

**8A completada en su alcance: dos correcciones mínimas independientes preparadas y comprobadas en copias. Fase8 permanece abierta; principal sin integrar ni promover.** No se elige modelo ni se atribuye mejora deportiva. Fases2/6 no se repiten; 7B espera modelo y ventana exclusiva.

## Identidad, diferencias y parches

Rama `{before['branch']}`, HEAD `{head}`. Snapshot inicial: antes.json; final: MANIFIESTO_FINAL.json. Cambios existentes conservados. Únicas fuentes nuevas de herramientas: fase8a_soporte.py, fase8a_preparar.py, fase8a_verificar.py y fase8a_cerrar.py, dentro de herramientas/. Solo se crean copias/artefactos propios y nota específica; Estado/Bitácora/Plan no se editan.

| Copia experimental (carpeta absoluta en manifiesto) | Búsqueda SHA256 | Estrategia SHA256 | Diferencia frente a P |
|---|---|---|---|
| P | `{initial['copy_sha256']['P']['busqueda_eval3.py']}` | `{initial['copy_sha256']['P']['strategy.py']}` | ninguna |
| TT_A | `{initial['expected_A_sha256']}` | `{initial['copy_sha256']['TT_A']['strategy.py']}` | solo busqueda_eval3.py |
| PRIORIDAD_LIBRO | `{initial['copy_sha256']['PRIORIDAD_LIBRO']['busqueda_eval3.py']}` | `{priority_sha}` | solo strategy.py |

En las tres copias eval3/NPZ/libro son idénticos a P. NPZ:37678d0b… (275 pesos); no se usa N. Rutas físicas:`{OUT}/copias/{{P,TT_A,PRIORIDAD_LIBRO}}/estudiantes/iacasitegano/`. No hay copia compuesta con ambos arreglos ni reemplazo activo.

{link('01_TT_exacta_P_a_A.patch', OUT/'parches/01_TT_exacta_P_a_A.patch')}, SHA `38591003cb9be5dc6513289310f8823310575aac9f88855e72610a2675842abd`, es el parche original preservado del cierre2. Solo añade ventana alfa/beta de entrada y exige resultado estrictamente interior, sin corte/aborto, para guardar exacto. Aplicado con patch/fuzz0 exclusivamente en TT_A; resultado byte a byte idéntico al A real esperado. No añade pistas internas rechazadas, utilidad, hojas, reloj, buffers o pesos. Las fuentes P del principal conservan su defecto TT.

{link('02_victoria_segura_antes_libro.patch', OUT/'parches/02_victoria_segura_antes_libro.patch')}, SHA `{sha(OUT/'parches/02_victoria_segura_antes_libro.patch')}`, solo intercambia los dos bloques existentes de strategy: tras forzada, victoria segura y luego libro legal. Mantiene `all(hijo.ganador()==color...)` sobre todas las ramas de Estado.siguiente. Sin reescritura del cargador, libro o búsqueda. PRIORIDAD_LIBRO conserva búsqueda P sin corregir TT. Aplicación limpia e inversa comprobadas en otra copia de verificación, nunca sobre runtime.

## Corrección TT: controles negativos y positivos

Se conserva verificar_cotas_tt.py original, SHA f5464d8fa46f81e6f432cd8219f92944a4b5b974f92019521f40691e1331dc62, byte a byte en origenes/ y una copia ejecutable bajo TT_A/herramientas/. No se ejecutó su suite extendida/JIT. Procedencia del parche, fuentes, certificado2 y regresión7A en manifiesto_inicial.json.

Estado original intacto: blancas a3/e4, negras b7/g6, turno blancas, puntos0–0, D2; solo orden canónico de casillas al serializar. Ganancia P={tt['gain']}. Oráculo independiente expectiminimax sin TT/poda, motor real para legalidad/terminales/ambas ramas y eval3 para hojas;80 visitas totales. No se sustituye el fixture ni los valores originales.

| Perspectiva | Ventana estrecha original | P después, amplia compartida | Oráculo y A corregido |
|---|---|---:|---:|
| blancas | [419.7904604088274,420.7904604088274] | +210.20324999290867 | -1.3877787807814457e-17 |
| negras | [-420.7904604088274,-419.7904604088274] | -210.20324999290867 | +1.3877787807814457e-17 |

{link('TT_FUNCIONAL.json', OUT/'TT_FUNCIONAL.json')}:10 ventanas originales (5 por perspectiva), consulta estrecha/amplia compartida y limpia;6 ventanas de frontera/interior;4 abortos (previo y al expandir, ambas perspectivas);6 terminales (puntos de ambos colores y bloqueo). Cero diferencias corregidas frente al oráculo fuera de tolerancia1e-12 en regresión original y1e-10 en adicionales. Cuatro fronteras no almacenan raíz exacta, dos interiores sí: control positivo de que la TT no quedó simplemente deshabilitada. Abortos diminutos no escriben exactos incompletos; una búsqueda general puede conservar exactos de subárboles ya completados. Entradas de búsqueda/marcadores intactas y globals restaurados.

Tabla de fixture256, D≤2,5000 nodos máximo por consulta; no es validación compilada de TT real de2^20 entradas ni prueba exhaustiva. La fuente con sus constantes originales no se editó para reducir tabla; globals temporales restaurados. No se repite panel cronometrado2 ni se reevalúa el candidato de pistas rechazado.

## Prioridad de victoria segura: estrategia real en copias

{link('LIBRO_FUNCIONAL.json', OUT/'LIBRO_FUNCIONAL.json')}:20 llamadas verificadas,10 por variante, ambos colores. En el contraejemplo7A y su espejo, P elige ataque de libro que gana solo en una rama p0,5 y corregido anota de forma segura. Se guardan estados, puntos, clave, entrada, elegido y los dos hijos/probabilidades. Incluye posiciones sin victoria garantizada con dos ataques que solo ganan al tener éxito: ambos conservan la decisión legal del libro; libro legal tranquilo; libro ilegal que invoca exactamente una vez búsqueda simulada; forzada que ignora libro inválido. No asumir que ataque exitoso equivale a victoria segura.

No preparar/calentar ni búsqueda real desde strategy. Stub de búsqueda solo para comprobar el fallback de entrada ilegal; se restauran elegir/reportar/color/libro/rng/reserva y alias temporales. Carga por rutas absolutas y módulos propios:{link('MODULOS_CARGADOS.json', OUT/'MODULOS_CARGADOS.json')} registra12 cargas con __file__, spec.origin, SHA y co_filename de funciones. La estrategia corregida enlaza realmente con búsqueda P de SU copia, no con A ni un módulo activo homónimo. Las aserciones confirman restauración de sys.modules/sys.path.

## Preservación, entorno y límites

Verificados antes y después todos los hashes runtime/motor/libro/pesos;30 archivos de caché activo sin cambios ni altas;17 fuentes enumeradas en PROTOCOLO.json de fase5 coinciden con sus SHA congelados, incluido NPZ histórico leído solo para hash. No se inspeccionaron estados/test nuevos, datos de entrenamiento, pesosN o locks. No se pausó/detuvo ningún proceso ajeno ni contenedor. El flujo5 sigue su propia continuación automática.

Python -B, NUMBA_DISABLE_JIT=1 antes de imports, hilos1, nice19 MÁS limitador adaptativo CPU≤12% de un núcleo. No se compila, no se escribe caché en activos, no se ejecuta árbitro/liga/benchmark/arranque frío/generación masiva. El reloj/preparación/memoria pico y fuerza oficiales están **sin medir**; llamadas internas a perf_counter solo crean un deadline del stub, no una medición de rendimiento publicada. Pruebas finitas de corrección no certifican toda futura composición/arquitectura.

## Pendientes y reproducción

{link('LEEME.md', OUT/'LEEME.md')} contiene comandos pequeños ejecutados, recuperación por bloque y protección contra sobrescritura. {link('CHECKLIST_8B.md', OUT/'CHECKLIST_8B.md')} deja comandos de suite extendida/JIT NO ejecutados y controles pendientes: decisión real5 (sin asumir V), composición aislada, no doble parche TT si U ya está corregido, paquete/ABI/caché Linux, frío, respaldo/legalidad, reloj/preparación/memoria/tamaño oficiales y protocolo previo contra P/control corregido. Requieren ventana exclusiva coordinada y candidato/semillas/criterios congelados antes de jugar. La comparación U/V de5 mantiene estrategia/libro idénticos; no insertar estos arreglos a mitad.

No falta trabajo obligatorio ligero de8A. La suite extendida/JIT y controles oficiales quedan explícitamente en8B: no se declara fase8 cerrada ni permiso para sustituir activos. Sin commit, merge, push o promoción. Evidencia recuperable por JSON de bloque, copias y parches, manifiesto final, catálogo SHA256SUMS y nota específica.

## Párrafo de traspaso

{handoff}
'''
    replacements = {'Fase8':'Fase 8','Fases2/6':'Fases 2/6','pendiente de5':'pendiente de 5','NPZ:376':'NPZ: 376','físicas:':'físicas: ',
                    'cierre2':'cierre 2','certificado2':'certificado 2','regresión7A':'regresión 7A','puntos0–0':'puntos 0–0','hojas;80':'hojas; 80',
                    '}:10':'}: 10', 'json)>:10':'json)>: 10', ':10 ventanas':': 10 ventanas', ';6 ventanas':'; 6 ventanas',';4 abortos':'; 4 abortos',';6 terminales':'; 6 terminales',
                    'tolerancia1e-12':'tolerancia 1e-12','y1e-10':'y 1e-10','fixture256':'fixture 256','D≤2,5000':'D≤2, 5000','de2^20':'de 2^20',
                    'cronometrado2':'cronometrado 2',':20 llamadas':': 20 llamadas',',10 por':', 10 por','contraejemplo7A':'contraejemplo 7A','rama p0,5':'rama p=0,5',
                    'propios:':'propios: ','registra12':'registra 12','pesos;30':'pesos; 30','cambios ni altas;17':'cambios ni altas; 17','pesosN':'pesos N','flujo5':'flujo 5',
                    'real5':'real de 5','U/V de5':'U/V de 5','de8A':'de 8A','en8B':'en 8B','declara fase8 cerrada':'declara fase 8 cerrada'}
    for old, new in replacements.items():
        content = content.replace(old, new)
    (OUT / 'CIERRE_8A.md').write_text(content)
    note = f'''---
tipo: experimento
estado: preparacion-8A-completa-integracion-8B-pendiente
fecha: 2026-10-29
baseline_commit: {head}
---

# Fase 8A — correcciones independientes en copias

8A preparada, fase8 abierta. Principal/runtime/caché intactos; no selección de modelo ni mejora deportiva declarada.

- Copia TT_A: parche original SHA38591003…, fuente idéntica byte a byte a A cc888a3c. P reproduce los dos errores originales ±210.20324999290867 frente a oráculo≈0. Copia corregida pasa10 ventanas originales,6 fronteras/interiores,4 abortos y6 terminales sin TT/poda en el oráculo. Suite extendida/JIT pendiente.
- Copia PRIORIDAD_LIBRO: solo mueve victoria garantizada antes del libro legal; conserva búsqueda P.20 llamadas verificadas en ambos colores, con ataque/fracaso, victoria no segura, libro legal/ilegal, fallback simulado y forzada. Globals restaurados, sin preparar/calentar.
- Rutas absolutas/SHA/co_filename reales guardados;17 fuentes congeladas de5 y30 archivos de caché principal verificados antes/después. Sin test nuevo, N, liga, benchmark, JIT, generación masiva, commits o promoción. Las fases2/6 no se reabren; 7B sigue pendiente.

{link('Certificado y traspaso', OUT/'CIERRE_8A.md')} · {link('TT', OUT/'TT_FUNCIONAL.json')} · {link('Libro', OUT/'LIBRO_FUNCIONAL.json')} · {link('Checklist 8B y comandos pendientes', OUT/'CHECKLIST_8B.md')} · {link('Reproducción', OUT/'LEEME.md')}.

8B: recibir decisión real de5, revisar composición (no aplicar TT dos veces si buscador ya corregido), conservar U/V de5 con mismas fuentes/libro, validar Linux/ABI/caché/frío/seguridad/costo/tamaño y protocolo contra P/control corregido en ventana exclusiva. No integrar desde este encargo ni editar notas globales.
'''
    for old, new in {'fase8 abierta':'fase 8 abierta','SHA385':'SHA 385','pasa10':'pasa 10',',6 ':', 6 ', ',4 ':', 4 ','y6 ':'y 6 ',
                     'P.20':'P. 20','guardados;17':'guardados; 17','de5':'de 5','y30':'y 30','fases2/6':'fases 2/6'}.items():
        note = note.replace(old, new)
    NOTE.write_text(note)
    environment = {'python': sys.version, 'architecture': platform.machine(), 'platform': platform.platform(),
                   'numpy': importlib.metadata.version('numpy'), 'numba': importlib.metadata.version('numba'),
                   'functional_mode': 'NUMBA_DISABLE_JIT=1, threads1, nice19, adaptive_CPU_fraction0.12',
                   'no_active_cache_writes_verified': True}
    processes = subprocess.check_output(['rtk','proxy','ps','-axo','pid,ppid,%cpu,command'], text=True)
    status = subprocess.check_output(['rtk','proxy','git','status','--short','--branch'], text=True)
    final = {'at': datetime.now().astimezone().isoformat(), 'branch': before['branch'], 'HEAD': head, 'git_status': status,
             'environment': environment, 'checks': checks, 'copies_checks': copy_checks,
             'actual_loaded_modules': loaded, 'source_identical_to_A': True, 'priority_patch_sha256': sha(OUT/'parches/02_victoria_segura_antes_libro.patch'),
             'TT_negative_controls_reproduced': 2, 'TT_corrected_failures': 0, 'TT_original_windows': 10, 'TT_boundaries': 6,
             'TT_aborts': 4, 'TT_terminals': 6, 'book_calls_passed': 20, 'handoff_paragraph': handoff,
             'tools': {str(p): sha(p) for p in sorted((TEAM/'herramientas').glob('fase8a_*.py'))},
             'note': str(NOTE), 'note_sha256': sha(NOTE), 'protocol_sha256': sha(OUT/'PROTOCOLO.md'),
             'initial_manifest_sha256': sha(OUT/'manifiesto_inicial.json'),
             'phase5_processes_observed': [line for line in processes.splitlines() if any(x in line for x in ['continuar_fase5.py','fase5_generar_robusto.py'])],
             'active_integration': False, 'phase8_closed': False, 'N_used': False, 'new_test_inspected': False,
             'official_games': 0, 'benchmark': False, 'JIT_compilation': False,
             'pending_8B': ['phase5_model_decision', 'isolated_composition', 'extended_JIT_regressions',
                           'Linux_ABI_cache_cold_start', 'official_cost_safety_package', 'prefixed_official_comparison_exclusive_window']}
    save(OUT/'MANIFIESTO_FINAL.json', final)
    save(OUT/'progreso.json', {'status': '8A_complete_8B_pending', 'TT_functional': 'passed', 'book_functional': 'passed',
                             'required_light_8A_work_remaining': [], 'pending_8B': final['pending_8B']})
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(sha(p, budget)+'  '+str(p.relative_to(OUT))+'\n' for p in files))
    budget.wait()
    print(json.dumps({'status': '8A_complete_8B_pending', 'files_hashed': len(files), 'phase5_hashes_verified': len(checks['phase5']),
                      'active_cache_hashes_verified': len(checks['active_cache']), 'runtime_intact': True,
                      'TT_matches_A': True, 'book_calls': 20, 'note': str(NOTE)}, indent=2))


if __name__ == '__main__':
    main()
