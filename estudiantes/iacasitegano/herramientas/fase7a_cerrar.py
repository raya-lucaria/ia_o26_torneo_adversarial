"""Certificado reproducible de 7A desde evidencia guardada, sin búsquedas."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import subprocess

import fase7a_libro as lib

ROOT = lib.ROOT
OUT = ROOT / 'resultados/medicion/fase7A_aperturas_20261029'
NOTE = ROOT / 'obsidian/Experimentos/2026-10-29-Fase7A-aperturas-auditoria-generador.md'


def link(label, path):
    return f'[{label}](<{path}>)'


def main():
    before = json.loads((OUT / 'antes.json').read_text())
    coverage = json.loads((OUT / 'cobertura.json').read_text())
    frozen = json.loads((OUT / 'manifiesto_cobertura.json').read_text())
    regression = json.loads((OUT / 'fixtures_02/REGRESIONES.json').read_text())
    book_audit = json.loads((OUT / 'AUDITORIA_LIBRO.json').read_text())
    preservation = {name: lib.sha(ROOT / name) == info['sha256'] for name, info in before['sources'].items() if info['exists']}
    assert all(preservation.values()), preservation
    assert coverage['traces'] == 890 and coverage['decisions'] == 36730 and coverage['unique_states'] == 35094
    assert regression['passed'] == 35 and regression['failures'] == 0
    assert lib.sha(OUT / 'PROTOCOLO.md') == frozen['protocol_sha256']
    assert all(lib.sha(Path(p)) == h for p, h in frozen['source_files'].items())
    assert not list(OUT.rglob('.writer.lock'))
    # Independiente del agregador: releer solo apariciones propias ya guardadas.
    records = [json.loads(p.read_text()) for p in (OUT / 'por_traza').glob('*.json')]
    roots = [r for record in records for r in record['roots']]
    matched = [r for r in roots if r['found']]
    assert len(records) == 890 and len(roots) == 36730 and len({r['state_sha256'] for r in roots}) == 35094
    assert len(matched) == 470 and len({r['state_sha256'] for r in matched}) == 3
    for pop in coverage['population']:
        rows = [r for r in roots if r['population'] == pop['population']]
        assert len(rows) == pop['decisions'] and sum(r['found'] for r in rows) == pop['found']
        assert all(r['legal'] and r['potential_use'] and not r['safe_win_displaced'] for r in rows if r['found'])
    p_manifest = json.loads((OUT / 'fixtures_02/funcional_P/manifiesto.json').read_text())
    u_manifest = json.loads((OUT / 'fixtures_02/funcional_U/manifiesto.json').read_text())
    # Cambio de SHA manteniendo la MISMA ruta: el contrato no depende solo del nombre.
    changed = copy.deepcopy(p_manifest)
    model_path = str(ROOT / 'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz')
    changed['model']['sources'][model_path] = '0' * 64
    try:
        lib.identity_matches(p_manifest, changed)
    except ValueError:
        sha_change_rejected = True
    else:
        raise AssertionError('Cambio SHA con misma ruta aceptado')
    population_table = []
    for row in coverage['population']:
        colors = {x['color']: x for x in coverage['colors'] if x['population'] == row['population']}
        population_table.append(f"| {row['population']} | {row['decisions']} | {row['found']} | {row['unique_found']} | "
                                f"{colors['blancas']['found']}/{colors['blancas']['decisions']} | {colors['negras']['found']}/{colors['negras']['decisions']} |")
    handoff = ('Fase7A terminada: libro actual verificado (8 entradas,261bytes,SHA00fd209f…); cobertura potencial sobre las mismas890 trazas '
               'y36.730 raíces de6:3 claves/estados observados,470 apariciones legales (455 blancas/15 negras),sin conflictos de victoria segura. '
               'Generador aislado eval3/NPZ275 con ganancia/transformación del buscador, alternativas y ramas completas, abortos=desconocido, '
               'salida nueva y manifiestos;35 regresiones y dos fixtures por P/U. No se genera libro deportivo ni se modifica runtime. '
               'Fase7 sigue ABIERTA:7B espera modelo seleccionado en5, fuentes sin defectoTT pendiente, semillas nuevas auditadas y ventana exclusiva; '
               'protocolo previo con/sin libro preparado. Evidencia:resultados/medicion/fase7A_aperturas_20261029/CIERRE_7A.md.')
    # Espaciado del párrafo que se copiará al plan por el chat director.
    for old, new in {'8 entradas,261bytes,SHA':'8 entradas, 261 bytes, SHA ', 'mismas890':'mismas 890',
                     'y36.730':'y 36.730', 'de6:3':'de 6: 3', 'observados,470':'observados, 470',
                     'negras),sin':'negras), sin', 'desconocido, salida':'desconocido, salida',
                     'manifiestos;35':'manifiestos; 35', 'ABIERTA:7B':'ABIERTA: 7B', 'en5,':'en 5,',
                     'defectoTT':'defecto TT', 'Evidencia:resultados':'Evidencia: resultados'}.items():
        handoff = handoff.replace(old, new)
    text = f'''# Cierre del subencargo fase 7A — aperturas

**7A completada: auditoría y generador funcional aislado. Fase7 permanece abierta, sin libro candidato aceptado ni promoción.** Esta tarea no repite fases2/6, no cambia el runtime ni usa datos nuevos de5. La pregunta es qué cubre potencialmente el libro actual y cómo generar propuestas coherentes con el próximo modelo.

## Versiones y preservación

Rama `{before['branch']}`, HEAD `{before['HEAD']}`. P búsqueda `{before['sources']['estudiantes/iacasitegano/busqueda_eval3.py']['sha256']}`, eval3 `{before['sources']['estudiantes/iacasitegano/eval3.py']['sha256']}`, NPZ275 `{before['sources']['estudiantes/iacasitegano/data/pesos_eval3_ligero.npz']['sha256']}`. Nombre real del modelo: `pesos_eval3_ligero.npz`; `pesos_eval3.npz` no existe y no se creó. Fuentes, pesos, libro, motor e informe conservan SHA inicial. Los cambios preexistentes y los procesos ajenos se preservaron. No commit, merge, push, integración TT/finales ni promoción. No se editaron Estado/Bitácora/Plan.

## Libro y procedencia real

`data/libro.json`:8 entradas,261 bytes,SHA `00fd209f4ad9973d422e6370e9799c1dd3e3b48992518dcaa5d834680dc5f4c0`, idéntico al blob promovido en commit `ad12b176101afa967d662dfe68412c605ecbfb96`. JSON sin claves duplicadas, claves decimales canónicas uint64 y pares enteros0..80 de avance de una fila. Cuatro movimientos blancos y cuatro negros por dirección; la dirección no decodifica un estado a partir del hash. Esquema: `{{str(estado.clave):[origen,destino]}}`.

INFORME.md:359 documenta mezcla de blancas regeneradas y negras del baseline, experimento de octubre con evaluación anterior/pesos.npy (SHA d5cc25ee…), protocolo declarado300 self-play, semillas4242/5151/6262, D4, cuatro jugadas propias, mínimo10 muestras y margen0,08. Es procedencia documental, no manifiesto contemporáneo por entrada. No se certifican fuente/modelo/semilla originales de las negras ni se atribuye el libro al NPZ275 actual. Contra el padre de la promoción solo cambia el par de la clave15717320468099553441 de[36,45] a[0,9]. Sus resultados antiguos no se suman a poblaciones recientes ni prueban fuerza con eval3. Detalle:{link('AUDITORIA_LIBRO.json', OUT/'AUDITORIA_LIBRO.json')}.

`Estado.clave` usa Zobrist64 determinista de piezas/color/casilla, turno y ambos marcadores0..3. No omite turno ni puntos; reloj/historia no son parte de Estado. Un hash64 puede colisionar: no es una serialización inyectiva. En35.094 estados únicos no se observan colisiones, sin afirmar imposibilidad.18 variantes del mismo tablero cambiando turno/puntos producen18 claves distintas en fixtures.

strategy carga JSON sin comprobar esquema/confianza/modelo. Antes de usarlo resuelve una jugada forzada; después construye Jugada y comprueba pertenencia a legales. No hay corte por ply: la consulta es en cualquier raíz presente. Libro legal precede a guardia de victoria segura. Se reprodujo en memoria un contraejemplo sintético (blancas a8, negras b9/i8,2–0): libro a8-b9 puede atacar conp0,5 mientras a8-a9 gana seguro. No se invocó preparar/calentar ni se cambió strategy. Es un riesgo de prioridad existente, no un conflicto observado en el libro actual. Guardar regresión para revisión de integración; no corregir runtime congelado aquí.

## Cobertura POTENCIAL, no activación histórica ni fuerza

Exactamente890 trazas,36.730 decisiones propias y35.094 estados completos únicos del inventario cerrado de6; incluidos16 episodios del piloto. Se verifican hashes de trazas/checkpoints y se reutiliza su replay válido. No se vuelven a reproducir las partidas, no se leen parciales ni lote256 activo. P/A/U/N, oficiales/privados/técnicos/excluidos permanecen separados. La procedencia original completa sigue por referencia; por_traza conserva cada aparición.

Solo3/8 claves aparecen:470 consultas potenciales legales a más de una jugada (455 blancas,15 negras), derivadas de solo3 estados únicos. Por ply:447 en1,15 en2,8 en3. Todas tienen0 victorias seguras disponibles,0 desplazamientos y0 ilegales observadas. Son recuentos descriptivos de exposición; las470 repeticiones NO son470 muestras independientes. Cinco entradas sin estado observado conservan legalidad contextual **desconocida**, no se declaran ilegales ni válidas para cualquier tablero. Baseline:21/862 raíces blancas y0/754 negras. A/A válido:123/5000 blancas y2/4734 negras.

| Población | Raíces | Libro encontrado/legal | Estados únicos encontrados | Blancas numerador/raíces | Negras numerador/raíces |
|---|---:|---:|---:|---:|---:|
{chr(10).join(population_table)}

{link('Cobertura JSON', OUT/'cobertura.json')} · {link('CSV por población', OUT/'cobertura.csv')} · {link('CSV color', OUT/'cobertura_color.csv')} · {link('CSV ply', OUT/'cobertura_ply.csv')} · {link('Coincidencias y procedencia', OUT/'coincidencias.json')} · {link('Entradas observadas/desconocidas', OUT/'entradas_auditadas.json')}. No se atribuye ahorro de reloj ni mejora deportiva a estos encuentros. Activación efectiva histórica y cobertura interna de búsqueda:sin medir.

## Generador nuevo y pruebas funcionales

El antiguo libro_apertura.py se leyó y congeló por SHA, nunca se ejecutó: usa busqueda_numba/eval/pesos.npy, resultados finales de self-play simétrico y escribe al libro activo. No sustituir importaciones a ciegas: la raíz eval3 actual devuelve5 resultados y acepta contador/tope.

`herramientas/fase7a_libro.py` exige buscador/NPZ275/población/procedencia/semillas de orden/profundidad/nodos/salida experimental nueva. Enumera con el motor todas las alternativas y ramas probabilísticas; evalúa continuaciones con el buscador elegido (invirtiendo perspectiva rival) y terminales con su GANANCIA. Reconoce las dos transformaciones existentes y rechaza otras sin adaptador. No mezcla logit con utilidad: P gain={p_manifest['model']['terminal_gain']}, escala={p_manifest['model']['scale']}; U gain={u_manifest['model']['terminal_gain']}, escala={u_manifest['model']['scale']}.

Regla prospectiva fijada antes de ejecutar: todas las ramas/alternativas a D yD+1 completas, mismo mejor movimiento y margen≥0,02×GANANCIA en ambos; tolerancia1e-10×max(1,GANANCIA). Aborto, empate, discrepancia, no finito o falta de margen=desconocido. Una entrada no puede desplazar una victoria segura conocida. Valores heurísticos de horizonte completo, **no exactos del juego**; el defecto TT deP sigue pendiente de8 y limita usar P como profesor definitivo. Sin ampliación del reloj.

Datos libro.json separados de manifiesto y procedencia/estados completos por entrada. Fuentes motor/eval/búsqueda/NPZ/herramienta y configuración con SHA. Checkpoints, escritor único, rechazo de salida existente y reanudación con cualquier cambio de fuentes/modelo/configuración. Cambiar SHA manteniendo la misma ruta también se rechaza. Los archivos de trabajo son nuevos; datos activos/pesos/cachés no se escriben.

{link('35 regresiones', OUT/'fixtures_02/REGRESIONES.json')}: esquema/rangos, turno/puntos, legalidad, dos colores, ataque0,5 con ambas ramas, anotación/retirada, comparación de valores D1/D2 contra enumeración independiente pequeña, raíz actual de5 retornos/aborto=desconocido, veto de victoria segura, empate incierto, salida no sobrescribible, resume idempotente, rechazo de modelo cambiado y prioridad antigua reproducida. Fixture funcional:2 raíces sintéticas×P/U, D1/D2,200 nodos totales por posición, JIT desactivado y CPU≤12%. P produce2 entradas experimentales legales; U produce0 por margen D1 insuficiente(0,0031924<0,02), aunque D2 distingue mejor. No se baja el umbral. Se conserva la primera prueba con expectativa incorrecta y su incidencia; el generador/política no cambió entre tandas. No son posiciones de apertura deportivas ni etiquetas de fuerza.

Comandos exactos y recuperación:{link('LEEME.md', OUT/'LEEME.md')}. Herramientas nuevas solo en herramientas/:fase7a_auditar.py, fase7a_libro.py, test_fase7a_libro.py y fase7a_cerrar.py. Todos los artefactos/hashes y referencias en manifiestos. Se usa nice19/Python sin JIT; contabilidad CPU/nodos solo para limitar carga. No se midieron tiempos de rendimiento, memoria pico o preparación oficial; no se compiló paquete ni se ejecutó árbitro, benchmark, entrenamiento, liga o generación masiva. No se consumieron reservas. La carga de5 no se detuvo ni modificó.

## Decisión, límites y pendientes 7B

7A satisface auditoría/herramientas. El potencial del libro es limitado y asimétrico en estas poblaciones; su valor deportivo con el próximo modelo sigue desconocido. No se propone reemplazo activo. Conservar libro/runtime actuales y evidencia histórica, sin confundirlo con validación en el modelo elegido.

{link('PROTOCOLO_7B.md', OUT/'PROTOCOLO_7B.md')} fija paquete64 raíces de diseño/ambos colores,D2/D3,50000 nodos por raíz, control sin libro y candidato con libro congelado, dos rivales, desarrollo160 partidas y confirmación160 condicional, puertas técnicas y criterio deportivo previo de+8 victorias sobre80 por versión y ninguna familia pierde>2/40. Este método es condicionado: falta selección5, fuentes aptas, población nueva, rivales exactos, semillas deportivas nuevas auditadas y ventana exclusiva. No hay liga ejecutada ni semillas deportivas asignadas. Comparación U/V de5 conserva mismo libro en ambos y no se mezcla con7B. Test5/examen9 no diseñan entradas. No se empieza fase8/9 automáticamente.

## Párrafo de traspaso

{handoff}
'''
    replacements = {'Fase7': 'Fase 7', 'fases2/6': 'fases 2/6', 'nuevos de5': 'nuevos de 5', ':8 entradas,261 bytes,SHA': ': 8 entradas, 261 bytes, SHA',
                    'enteros0..80': 'enteros 0..80', 'declarado300': 'declarado 300', 'semillas4242': 'semillas 4242',
                    'mínimo10': 'mínimo 10', 'margen0,08': 'margen 0,08', 'clave157': 'clave 157', 'de[36,45] a[0,9]': 'de [36,45] a [0,9]',
                    'Detalle:': 'Detalle: ', 'marcadores0..3': 'marcadores 0..3', 'En35.094': 'En 35.094', '.18 variantes': '. 18 variantes',
                    'producen18': 'producen 18', ',2–0': ', 2–0', 'conp0,5': 'con p=0,5', 'Exactamente890': 'Exactamente 890',
                    'trazas,36.730': 'trazas, 36.730', 'y35.094': 'y 35.094', 'de6;': 'de 6;', 'incluidos16': 'incluidos 16',
                    'Solo3/8': 'Solo 3/8', 'aparecen:470': 'aparecen: 470', 'solo3 estados': 'solo 3 estados', 'ply:447 en1,15 en2,8 en3': 'ply: 447 en 1, 15 en 2, 8 en 3',
                    'tienen0': 'tienen 0', ',0 ': ', 0 ', 'las470': 'las 470', 'son470': 'son 470', 'Baseline:21/862': 'Baseline: 21/862',
                    'y0/754': 'y 0/754', 'válido:123/5000': 'válido: 123/5000', 'y2/4734': 'y 2/4734', 'búsqueda:sin': 'búsqueda: sin',
                    'devuelve5': 'devuelve 5', 'yD+1': 'y D+1', 'tolerancia1e-10': 'tolerancia 1e-10', 'deP': 'de P', 'de8': 'de 8',
                    'ataque0,5': 'ataque 0,5', 'de5 retornos': 'de 5 retornos', 'funcional:2': 'funcional: 2', 'D1/D2,200': 'D1/D2, 200',
                    'produce2': 'produce 2', 'produce0': 'produce 0', 'insuficiente(0,0031924': 'insuficiente (0,0031924',
                    'recuperación:': 'recuperación: ', 'herramientas/:fase7a': 'herramientas/: fase7a', 'carga de5': 'carga de 5',
                    'paquete64': 'paquete 64', 'colores,D2/D3,50000': 'colores, D2/D3, 50000', 'desarrollo160': 'desarrollo 160',
                    'confirmación160': 'confirmación 160', 'de+8': 'de +8', 'sobre80': 'sobre 80', 'pierde>2/40': 'pierde >2/40',
                    'selección5': 'selección 5', 'U/V de5': 'U/V de 5', 'con7B': 'con 7B', 'Test5/examen9': 'Test 5/examen 9'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    lib.save(OUT / 'AUDITORIA_FINAL.json', {'decision': '7A_complete_7B_pending_not_strength_accepted',
             'sources_preserved': preservation, 'traces_recounted': len(records), 'decisions_recounted': len(roots),
             'matches': len(matched), 'unique_matching_states': 3, 'by_color': dict(Counter(r['color'] for r in matched)),
             'by_ply': dict(Counter(r['ply'] for r in matched)), 'regressions_passed': 35,
             'same_path_changed_sha_resume_rejected': sha_change_rejected, 'historical_activations': 'unmeasured',
             'official_games': 0, 'benchmarks': 0, 'mass_book_generation': False, 'runtime_changed': False,
             'protocol_preserved': True, 'handoff_paragraph': handoff})
    (OUT / 'CIERRE_7A.md').write_text(text)
    note = f'''---
tipo: experimento
estado: fase7A-completa-fase7B-pendiente
fecha: 2026-10-29
baseline_commit: {before['HEAD']}
---

# Fase 7A — auditoría de aperturas y generador compatible

7A completada; fase7 abierta, sin mejora aceptada ni cambio del libro activo. Hipótesis: el libro antiguo cubre pocas raíces y de forma asimétrica; un generador coherente con el buscador debe abstenerse ante incertidumbre.

Libro: 8 entradas/261 bytes, SHA00fd209f… idéntico a promoción ad12b176. Procedencia documental de mezcla con evaluación anterior; no atribuirlo a NPZ275 actual. Clave contempla piezas, turno y ambos puntos, con riesgo residual de colisión64.

Mismas890 trazas de6,36.730 raíces/35.094 estados únicos: solo3 claves observadas,470 coincidencias potenciales legales(455 blancas/15 negras),0 conflictos con victoria segura. Repeticiones no independientes, sin inferencia de fuerza ni activaciones históricas. Cinco claves sin estado observado: legalidad contextual desconocida.

Generador nuevo aislado: eval3/NPZ275, escala/ganancia del buscador, alternativas/ambas ramas y horizontes completos, abstención por aborto/margen/discrepancia, salida nueva y procedencia por entrada.35 regresiones funcionales; P genera2 entradas de fixtures, U0 por margen insuficiente. Riesgo existente reproducido: libro legal precede guardia de victoria segura; no se modifica strategy.

{link('Certificado y traspaso', OUT/'CIERRE_7A.md')} · {link('Cobertura', OUT/'cobertura.json')} · {link('Regresiones', OUT/'fixtures_02/REGRESIONES.json')} · {link('Protocolo 7B condicionado', OUT/'PROTOCOLO_7B.md')} · {link('Reproducción', OUT/'LEEME.md')}.

Pendiente: modelo/fuentes seleccionados en5, dependencia TT de8 sin integrar, raíces de diseño y semillas deportivas nuevas auditadas, ventana exclusiva, costo y comparación oficial con/sin libro. Comparación U/V de5 mantiene el mismo libro. Sin benchmarks/ligas/generación masiva, reservas, cambios a fase5 o runtime; no se editan notas globales. Fases2/6 no se repiten.
'''
    for old, new in {'fase7': 'fase 7', 'SHA00': 'SHA 00', 'Mismas890': 'Mismas 890', 'de6,36.730': 'de 6, 36.730',
                     'solo3': 'solo 3', 'observadas,470': 'observadas, 470', 'legales(455': 'legales (455', '),0': '), 0',
                     'entrada.35': 'entrada. 35', 'genera2': 'genera 2', 'U0': 'U 0', 'en5': 'en 5', 'de8': 'de 8',
                     'de5': 'de 5', 'a fase5': 'a fase 5', 'Fases2/6': 'Fases 2/6', 'colisión64': 'colisión 64'}.items():
        note = note.replace(old, new)
    NOTE.parent.mkdir(exist_ok=True)
    NOTE.write_text(note)
    lib.save(OUT / 'progreso.json', {'status': '7A_complete_7B_pending', 'coverage_traces': 890,
                                  'required_7A_work_remaining': [], 'pending_7B': ['phase5_selection', 'source_freeze',
                                  'new_population_and_seeds', 'exclusive_CPU_window', 'cost_and_official_comparison']})
    processes = subprocess.check_output(['ps', '-axo', 'pid,ppid,%cpu,command'], text=True)
    lib.save(OUT / 'MANIFIESTO_FINAL.json', {'branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
             'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'status': subprocess.check_output(['git', 'status', '--short', '--branch'], text=True),
             'sources_preserved': preservation, 'tool_sha256': {str(ROOT / 'estudiantes/iacasitegano/herramientas' / name):
             lib.sha(ROOT / 'estudiantes/iacasitegano/herramientas' / name) for name in
             ['fase7a_auditar.py', 'fase7a_libro.py', 'test_fase7a_libro.py', 'fase7a_cerrar.py']},
             'note': str(NOTE), 'note_sha256': lib.sha(NOTE), 'protocol_sha256': lib.sha(OUT / 'PROTOCOLO.md'),
             'protocol_7B_sha256': lib.sha(OUT / 'PROTOCOLO_7B.md'),
             'phase5_processes_observed': [line for line in processes.splitlines() if any(x in line for x in
             ['continuar_fase5.py', 'fase5_generar_robusto.py', 'supervisar_continuacion'])],
             'decision': '7A_complete_7B_pending', 'official_games': 0, 'benchmark': False})
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (OUT / 'SHA256SUMS').write_text(''.join(lib.sha(p) + '  ' + str(p.relative_to(OUT)) + '\n' for p in files))
    print(json.dumps({'decision': '7A_complete_7B_pending', 'hash_files': len(files),
                      'regressions': 35, 'source_preservation': all(preservation.values()), 'closure': str(OUT/'CIERRE_7A.md')}, indent=2))


if __name__ == '__main__':
    main()
