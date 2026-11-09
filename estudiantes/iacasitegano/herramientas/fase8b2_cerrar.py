"""Certificado técnico I0, verificación final y fin de ventana. No lanza nuevas pruebas."""
from __future__ import annotations

import csv
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

from fase8a_soporte import ROOT, TEAM, Budget, sha, save
from fase8b2_control import OUT, IMAGE

NOTE=ROOT/'obsidian/Experimentos/2026-10-29-Fase8B2-runtime.md'


def link(label,path):return f'[{label}](<{path}>)'


def main():
    budget=Budget()
    frozen=json.loads((OUT/'CONGELADO.json').read_text())
    decision=json.loads((OUT/'DECISION_TECNICA.json').read_text())
    assert sha(OUT/'DECISION_TECNICA.json')==frozen['decision_sha256']
    assert sha(OUT/'PROTOCOLO.md')==frozen['protocol_sha256']
    checks={name:{p:sha(Path(p),budget)==expected for p,expected in frozen[key].items()}
            for name,key in [('runtime','runtime'),('phase5','phase5_sources'),('active_cache','active_cache'),('origins','source_origins')]}
    assert all(all(group.values()) for group in checks.values())
    assert len(checks['phase5'])==17
    assert {str(p) for p in (TEAM/'data/numba_cache').rglob('*') if p.is_file()}==set(frozen['active_cache'])
    prior=json.loads((ROOT/'resultados/medicion/fase8B1_paquete_20261029/manifiesto_inicial.json').read_text())
    active_report=TEAM/'INFORME.md'
    assert sha(active_report)==prior['origins'][str(active_report)]
    runtime=json.loads((OUT/'RUNTIME_EN_PRUEBA.json').read_text());participant=Path(runtime['participant'])
    exported=OUT/'paquete_exportado_01'
    export_manifest=json.loads((exported/'manifiesto.json').read_text())
    assert len(runtime['files'])==8 and not runtime['cache_included']
    assert all(sha(participant/name,budget)==info['sha256']==frozen['decision_files'][name]['sha256'] for name,info in runtime['files'].items())
    assert all(sha(exported/'estudiantes/iacasitegano'/name,budget)==info['sha256'] for name,info in runtime['files'].items())
    assert sha(exported/'participante.tar',budget)==export_manifest['archive_sha256']=='085cfa179fc51fbecfacb372fc15aae9ed03342baf18c374eca22d792b30786e'
    regressions={mode:json.loads((OUT/'regresiones'/mode/'stdout.log').read_text()) for mode in ['python','jit']}
    for mode,row in regressions.items():
        assert row['baseline_sha256']=='ceba365326e5ba8dd102218843f2b92744738d9c8a18db000f9edbc188832b7f'
        assert row['candidate_sha256']==decision['archivos']['busqueda_eval3.py']['sha256']
        assert len(row['fallos_original'])==2 and row['fallos_corregido']==0 and row['tt_entradas']==1048576
        assert row['extendido']=={'ventanas':192,'raices':15,'fallos':0} and row['casos_ventana']==10
    abi=json.loads((OUT/'linux/abi_01/probe.json').read_text())['identity']
    cold=json.loads((OUT/'FRIO_FINAL.json').read_text())
    initial_cold=json.loads((OUT/'linux/frio_sin_cache_01/probe.json').read_text())
    functional=json.loads((OUT/'FUNCIONAL_LINUX.json').read_text())
    abort=json.loads((OUT/'FALLBACK_ABORTO_LINUX.json').read_text())
    smoke=json.loads((OUT/'humo_oficial_01/RESUMEN.json').read_text())
    validator=json.loads((OUT/'humo_oficial_01/VALIDADOR_OFICIAL.json').read_text())
    technical=json.loads((OUT/'tecnicas_01/RESUMEN.json').read_text())
    assert validator=={'ok':True,'errores':[],'avisos':[],'humo_jugado':True}
    assert cold['all_passed'] and cold['trials']==3 and cold['max_prepare_s']<10 and cold['max_cgroup_peak_bytes']<1024**3
    assert initial_cold['ok'] and functional['functional']['passed']==9 and abort['passed']==2
    assert smoke['count']==2 and technical['count']==4
    rows=[{'population':'cold_initial','index':1,'color':'blancas','prepare_s':initial_cold['total_prepare_s'],
           'clock_s':None,'cgroup_peak_MiB':int(initial_cold['cgroup']['/sys/fs/cgroup/memory.peak'])/1024**2,'own_error':False}]
    for index,data in enumerate(cold['data'],1):
        assert data['ok'] and data['total_prepare_s']<10
        assert data['loaded']['cache_stats']['_mejor_jugada_numba']['cache_misses']==1
        assert data['loaded']['gain']==decision['ganancia_observada_8A']
        rows.append({'population':'cold_final','index':index,'color':'blancas','prepare_s':data['total_prepare_s'],
                     'clock_s':None,'cgroup_peak_MiB':int(data['cgroup']['/sys/fs/cgroup/memory.peak'])/1024**2,'own_error':False})
    for label,summary in [('humo',smoke),('tecnicas',technical)]:
        folder=OUT/('humo_oficial_01' if label=='humo' else 'tecnicas_01')
        for game in summary['games']:
            assert not game['own_error'] and not game['infrastructure_failure']
            assert game['reason'] in ['puntos','sin_jugadas'] and game['prepare_s']<10 and game['clock_s']<30
            assert game['cgroup_peak_bytes'] is not None and 0<game['cgroup_peak_bytes']<1024**3
            raw=json.loads((folder/f"partida_{game['index']:02d}.json").read_text())
            assert raw['meta']['semilla_maestra']==(2026 if label=='humo' else 20261221)
            telemetry=json.loads((folder/f"telemetria_{game['index']:02d}.json").read_text())
            own=next(t for t in telemetry if Path(t['folder'])==participant)
            assert own['loaded_adios']['strategy_sha256']==decision['archivos']['strategy.py']['sha256']
            assert own['loaded_adios']['search_sha256']==decision['archivos']['busqueda_eval3.py']['sha256']
            assert own['loaded_adios']['strategy_file']==own['loaded_adios']['jugar_code_file']=='/bot/strategy.py'
            for bot in telemetry:
                inspection=bot['inspect_before_owned_remove'];host=inspection['HostConfig']
                assert inspection['Image']==IMAGE and host['NanoCpus']==10**9
                assert host['Memory']==host['MemorySwap']==1024**3 and host['NetworkMode']=='none' and host['ReadonlyRootfs']
                assert all(not mount['RW'] for mount in inspection['Mounts'])
            rows.append({'population':label,'index':game['index'],'color':game['own_color'],'prepare_s':game['prepare_s'],
                         'clock_s':game['clock_s'],'cgroup_peak_MiB':game['cgroup_peak_bytes']/1024**2,'own_error':game['own_error']})
    with (OUT/'metricas.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    measured_maxima={'prepare_s':max(initial_cold['total_prepare_s'],cold['max_prepare_s'],smoke['max_prepare_s'],technical['max_prepare_s']),
                    'clock_s':max(smoke['max_clock_s'],technical['max_clock_s']),
                    'cgroup_peak_MiB':max(int(initial_cold['cgroup']['/sys/fs/cgroup/memory.peak']),cold['max_cgroup_peak_bytes'],
                                          smoke['max_cgroup_peak_bytes'],technical['max_cgroup_peak_bytes'])/1024**2}
    save(OUT/'metricas.json',{'rows':rows,'initial_no_cache_probe':initial_cold,'cold_final':cold,
                            'smoke_separate':smoke,'technical':technical,'validator':validator,'measured_maxima':measured_maxima})
    save(OUT/'SEMILLA_20261221_CONSUMIDA.json',{'seed':20261221,'consumed':True,'games':4,
         'scope':'tanda tecnica I0 vs random, not strength','seed_audit_sha256':sha(OUT/'AUDITORIA_SEMILLA_20261221.json'),
         'records':{str(OUT/'tecnicas_01'/f"partida_{g['index']:02d}.json"):g['record_sha256'] for g in technical['games']},
         'do_not_reuse_as_new':True})
    owned=subprocess.check_output(['rtk','proxy','docker','ps','-a','--filter','label=codex.experimento=8B2-I0-20261029','--format','{{json .}}'],text=True)
    assert not owned.strip(),'Quedan contenedores propios; no declarar ventana terminada'
    now=datetime.now().astimezone().isoformat()
    end={'at':now,'window_ended':True,'own_containers_remaining':0,'heavy_work_finished':True,
         'no_next_phase_launched':True,'S13_independent_validation_can_be_scheduled_after_this_window':True}
    save(OUT/'VENTANA_TECNICA_TERMINADA.json',end)
    colors=[]
    for color in ['blancas','negras']:
        games=[g for g in technical['games'] if g['own_color']==color]
        colors.append({'color':color,'games':len(games),'wins':sum(g['winner']==color for g in games),'own_errors':sum(g['own_error'] for g in games),
                       'max_prepare_s':max(g['prepare_s'] for g in games),'max_clock_s':max(g['clock_s'] for g in games),
                       'max_cgroup_peak_MiB':max(g['cgroup_peak_bytes'] for g in games)/1024**2})
    color_table='\n'.join(f"| {r['color']} | {r['games']} | {r['wins']} | {r['own_errors']} | {r['max_prepare_s']:.6f} | {r['max_clock_s']:.6f} | {r['max_cgroup_peak_MiB']:.3f} |" for r in colors)
    before=json.loads((OUT/'preflight_01/estado_inicial.json').read_text())
    head=subprocess.check_output(['rtk','proxy','git','rev-parse','HEAD'],text=True).strip();assert head==before['HEAD']
    handoff=('8B.2 técnico de I0 completado, sin promoción ni cierre global8: Python/JIT extendidos pasan con TT20, dos fallos P originales reproducidos y cero fallos corregidos. '
             'Linux ARM64/imagen79983872, Python3.12.16/NumPy2.4.6/Numba0.67.0; paquete de respaldo sin caché propia pasa3fríos (máx2.717s/142.54MiB). '
             'Validador oficial con2humos separado y cuatro técnicas random(2por color,semilla20261221consumida),cero errores/timeouts/ilegales. '
             'Máximos tanda:preparación3.032s,reloj14.375s,pico cgroup143.36MiB. Fuentes principal/cachés/17fuentes5 intactos; no S13/test5. '
             'No acredita fuerza ni selecciona finalista. Ventana técnica TERMINADA; puede coordinarse después validación independiente deS13. '
             'Evidencia: resultados/medicion/fase8B2_runtime_20261029/CIERRE_8B2.md.')
    # Solo texto del traspaso; no se cambian nombres de rutas/archivos.
    for a,b in {'global8':'global 8','Python3.12':'Python 3.12','NumPy2.4':'NumPy 2.4','Numba0.67':'Numba 0.67',
                'pasa3fríos':'pasa 3 fríos','máx2.717':'máx. 2.717','con2humos':'con 2 humos','random(2por color,semilla20261221consumida),cero':'random (2 por color, semilla 20261221 consumida), cero',
                'tanda:preparación3.032s,reloj14.375s,pico':'tanda: preparación 3.032s, reloj 14.375s, pico',
                '/17fuentes5':'/17 fuentes de 5','deS13':'de S13'}.items():handoff=handoff.replace(a,b)
    text=f'''# Cierre técnico 8B.2 — I0 con respaldo

**Controles técnicos de I0 pasados en el alcance fijado. No es finalista, prueba de fuerza, promoción ni cierre global de fase8. La ventana técnica terminó.** S13 sigue fuera de este paquete y pendiente de validación independiente; no se abrió test5 ni se eligió modelo a partir de sus resultados.

## Identidades reales y preservación

Decisión explícita del director, conservada sin cambios: {link('DECISION_TECNICA.json',OUT/'DECISION_TECNICA.json')}, SHA `{frozen['decision_sha256']}`. Rama `{before['branch']}`, HEAD `{head}`. Principal e informe activo intactos; sin commit/merge/push/promoción. Las17 fuentes congeladas de5, runtime y30 archivos de caché activo coinciden antes/después. No se usó NPZ amplio rechazado, S13 o utilidad U. Activo P conserva sus fuentes originales; haber probado esta copia no lo integra.

I0 probado físicamente en `{participant}`, montado readonly como /bot. Búsqueda `cc888a3c7fc7ba896f5145c36c56c77120bc6f0ef0b0ba45f077023bbeb7a177`; estrategia `88ce9d4718d73e1fbb4ae5901b9ea08b79fc9d6f314a498d64ad7d0b25de4d43`; eval3 `e15319c06397fa01a35a5c335eb51e46170fe0c198dc7f52b0fcd922b444d4e6`; NPZ275 `37678d0bd700c27a29d0084558158db1a65b01409dbf54828e89706441c2250b`; libro `00fd209f4ad9973d422e6370e9799c1dd3e3b48992518dcaa5d834680dc5f4c0`. Ocho archivos idénticos a decisión, sin cambios de lógica/hojas/buffers/reloj/evaluación/pesos/libro.

Exportación con constructor8B.1 reutilizado: {link('participante.tar',exported/'participante.tar')}, SHA `{export_manifest['archive_sha256']}`,81920bytes,8archivos/74599bytes de contenido, idéntico al tar8B.1. Árbol exportado por SHA idéntico al staging realmente probado; receta/procedencia/manifiestos fuera del juego. No incluye cachés Mac/Linux, herramientas de entrenamiento, resultados o credenciales. Informe experimental de I0 conservado como parte de la decisión; informe final de entrega sigue pendiente.

## Ventana de CPU

Se rechazó pre_regresion_01 por Spotlight sostenido≈101% de un núcleo y lectura de66.58% idle; ninguna prueba de carga se ejecutó allí. pre_regresion_02 aceptada con76.30/76.82% idle; pre_linux_01 con81.43/85.84; pre_humo_01 con87.60/87.61; pre_tecnicas_01 con87.47/85.84. Puerta publicada ANTES de medir, sin bajar umbral después. Muestras/rawps/top y Kubernetes en ventanas/. No procesos5/entrenamiento/diagnóstico ajenos, sin jobs de laboratorio Running; control plane existente documentado y no detenido. No se pausó/reanudó GRIS, Spotlight u otra aplicación. Solo se gestionaron contenedores propios8b2-i0 con etiqueta específica; nunca limpieza global de CLI/imágenes/volúmenes.

## Regresiones y ejecución Linux

Suite original f5464d8f preservada byte a byte en copia de prueba. {link('Python extendido',OUT/'regresiones/python/stdout.log')} y {link('JIT extendido',OUT/'regresiones/jit/stdout.log')}: por pase10 ventanas originales,192 extendidas,15 raíces,TT1048576 entradas; dos controles negativos P ±210.20324999290867 frente a≈0, cero fallos corregidos frente al oráculo sin TT. Baseline P/eval/NPZ iguales y candidato realmente cargado cc888a3c, rutas absolutas en comandos y SHA emitidos por la suite. Cache de regresión temporal/aislado, no activo. No se reabre el panel cronometrado de pistas2.

Imagen oficial inmutable `{IMAGE}`, Linux/aarch64 (Docker Architecture arm64), Python `{abi['python'].split(' (')[0]}`, NumPy `{abi['numpy']}`, Numba `{abi['numba']}`. Motor real /torneo/sistema/motor/nucleo.py SHA `{abi['motor_sha256']}` coincide con host. Ruta de estrategia y co_filename /bot/strategy.py; búsqueda /bot/busqueda_eval3.py y dispatcher.py_func con misma ruta, gain={decision['ganancia_observada_8A']}. Identidades y compilación real registradas en linux/abi_01 y probes. Valida esta arquitectura/imagen; no certifica AMD64, otra ABI, otro hardware o un modelo posterior.

{link('FUNCIONAL_LINUX.json',OUT/'FUNCIONAL_LINUX.json')}:9 comprobaciones compiladas (prioridad segura/ambas ramas,forzada,reloj corto,raíz abortada y equivalencia TT desde Linux). {link('FALLBACK_ABORTO_LINUX.json',OUT/'FALLBACK_ABORTO_LINUX.json')}: elegir con tope1nodo en ambos colores produce respaldo legal,profundidad completa0,aborto1 y sin usar parcial; globals restaurados. No se cambió la política de búsqueda en fuentes ni en partidas. No repetir suites7A/8A: se verificó la composición seleccionada para esta autorización.

## Frío y caché: no fue necesaria una precomputada

Intento inicial SIN caché de participante: {initial_cold['total_prepare_s']:.6f}s y {int(initial_cold['cgroup']['/sys/fs/cgroup/memory.peak'])/1024**2:.3f}MiB de pico cgroup, pasó10s, conservado en linux/frio_sin_cache_01. Por ello no se construyó/incorporó caché propia: rama cache no ejecutada. Se mantiene el paquete sin cambios. Todos los probes finales registran cache_miss=1 en raíces/valor/evaluador, demostrando compilación JIT Linux real, no reutilización de caché Mac.

{link('FRIO_FINAL.json',OUT/'FRIO_FINAL.json')}: tres contenedores NUEVOS, tiempos {', '.join(f"{d['total_prepare_s']:.6f}" for d in cold['data'])}s; máximo **{cold['max_prepare_s']:.6f}s**, pico cgroup máximo **{cold['max_cgroup_peak_bytes']/1024**2:.3f}MiB**. Cuota cpu.max100000/100000=1CPU; memory.max1073741824,red none,root readonly,usuario65534,caps dropALL,pids64. memory.peak es contador real leído en cgroupv2, eventos OOM0; no inferido de --memory ni sustituido por RSS.

Frío significa contenedores/tmpfs de caché nuevos, con copia y calentamiento del motor oficial fuera del tiempo de importar/preparar al alumno, igual al runner. Volumen existente del motor solo lectura; no se vació caché de páginas del SO ni se alteró el sistema global. El intento inicial y tres fríos prefijados se distinguen; no se añadieron repeticiones para rescatar un fallo.

## Validador de entrega y tanda técnica separados

{link('Validador oficial con humo',OUT/'humo_oficial_01/VALIDADOR_OFICIAL.json')}: entrega=True,docker=True,jugar_humo=True,0errores/0avisos y humo realmente jugado. Dos partidas automáticas contra random (una por color),semilla2026 y reloj3s propio del validador,preparación10s; no se suman a las cuatro técnicas. Máximos I0 de humo: preparar **{smoke['max_prepare_s']:.6f}s**, reloj **{smoke['max_clock_s']:.6f}s**, pico **{smoke['max_cgroup_peak_bytes']/1024**2:.3f}MiB**,sin fallos propios.

Se usó API oficial completa con fábrica de contenedores fijada en memoria al digest/etiqueta propios, sin Entorno.preparar ni limpieza global. Fuentes oficiales intactas. Telemetría externa envuelve solo Canal.enviar al emitir adios FUERA de reloj para añadir cgroup y archivos realmente cargados; no altera decisiones, reglas o límites. Medida de preparar es el valor devuelto por _Arbitro._turno original. Inspecciones prueban imagen exacta,CPU1,memoria/swap1GiB,rednone y mounts readonly. Instrumentación/restauración y mensajes crudos preservados por partida.

{link('Auditoría de semilla previa',OUT/'AUDITORIA_SEMILLA_20261221.json')}:19 raíces,7456 textos de resultados y890 trazas conocidas,0usos/0referencias previas, con test256/S13/datasets sellados excluidos sin leer. Protocolo/orden/fuentes/imagen/rival/auditoría congelados en {link('ANTES_DE_PARTIDAS.json',OUT/'tecnicas_01/ANTES_DE_PARTIDAS.json')} antes de jugar. **20261221 ahora está CONSUMIDA por estas cuatro técnicas**; marcador separado en SEMILLA_20261221_CONSUMIDA.json. No se gastan otras reservas.

Cuatro técnicas I0 vs random,orden blancas/negras/blancas/negras,juegos0/0/1/1,pareja0,30s/10s/1CPU/1GiB. Resultado4–0 solo técnico: no es comparación contra P/A/U ni prueba de fuerza. Ninguna ampliación de muestra. {link('RESUMEN técnico',OUT/'tecnicas_01/RESUMEN.json')} y JSON oficiales/telemetrías por partida guardados.

| Color I0 | Partidas | Victorias técnicas | Errores propios | Preparar máx.s | Reloj máx.s | Pico cgroup MiB |
|---|---:|---:|---:|---:|---:|---:|
{color_table}

Máximos de la tanda: **{technical['max_prepare_s']:.6f}s preparación, {technical['max_clock_s']:.6f}s reloj, {technical['max_cgroup_peak_bytes']/1024**2:.3f}MiB**. Máximos medidos entre intento inicial/fríos/humo/tanda: **{measured_maxima['prepare_s']:.6f}s preparación, {measured_maxima['clock_s']:.6f}s reloj, {measured_maxima['cgroup_peak_MiB']:.3f}MiB**; el mayor pico pertenece al intento inicial, no se omite. Cero ilegales,timeouts,errores propios,partidas abortadas o fallas de infraestructura. Abortos internos de profundidad no se instrumentan durante las partidas; se comprueba su contrato/fallback con fixtures, no se inventa un contador cero de búsqueda. Métricas JSON/CSV por población/color en metricas.json/csv. Todo bajo30s/10s/1024MiB; contenido8archivos/74599bytes bajo100MiB/50MiB/2000archivos.

## Decisión, límites y fin de ventana

Pasados: identidad y composición elegida, regresiones Python/JIT, ABI real LinuxARM64, preparación/frío/cgroup, legalidad/respaldo ante reloj corto/aborto, validador de entrega con humo real y cuatro partidas técnicas. Fallos de ejecución observados: ninguno. Rechazo de entorno inicial conservado; caché de participante no requerida y no incorporada. No hay control técnico obligatorio pendiente de este encargo para ESTA versión/imagen/arquitectura. No extrapolar las pruebas finitas o cuatro random a rivales fuertes o cualquier carga/hardware.

Pendientes globales: selección deportiva e integración/entrega final, validación independiente deS13, decisión posterior de modelo/libro y examen/protocolo correspondiente. Un cambio de pesos, escala, fuentes, ABI o libro exige nuevo hash y repetir controles afectados. El informe activo y el paquete activo no se cambian; informe final de entrega todavía debe alinearse con la decisión final. No abrir test5/examen9,libro nuevo,entrenamiento o liga MC/tiers desde aquí.

**Ventana técnica terminada {now}. No quedan contenedores propios.** {link('VENTANA_TECNICA_TERMINADA.json',OUT/'VENTANA_TECNICA_TERMINADA.json')}. Puede coordinarse después la generación/validación independiente deS13; este encargo no la inicia ni espera indefinidamente.

## Evidencia y traspaso

{link('Protocolo previo',OUT/'PROTOCOLO.md')} · {link('Fuentes congeladas',OUT/'CONGELADO.json')} · {link('Comandos/recuperación',OUT/'LEEME.md')} · {link('Métricas CSV',OUT/'metricas.csv')} · {link('Paquete técnico',exported/'participante.tar')}. Manifiesto final con entorno/hashes/checks y catálogo SHA256SUMS. Nuevas herramientas solo bajo herramientas/,nota propia, sin Estado/Bitácora/Plan.

{handoff}
'''
    # Espaciado de frases concretas; nunca se alteran identificadores o rutas.
    replacements={'fase8.':'fase 8.','Las17 fuentes':'Las 17 fuentes','de5,':'de 5,','y30 archivos':'y 30 archivos','constructor8B.1':'constructor 8B.1',
        '`,81920bytes,8archivos/74599bytes':'`, 81920 bytes, 8 archivos/74599 bytes','tar8B.1':'tar 8B.1','sostenido≈':'sostenido ≈',
        'de66.58%':'de 66.58%','con76.30/76.82%':'con 76.30/76.82%','con81.43/85.84':'con 81.43/85.84',
        'con87.60/87.61':'con 87.60/87.61','con87.47/85.84':'con 87.47/85.84','procesos5/':'procesos de 5/',
        'propios8b2-i0':'propios 8b2-i0','por pase10':'por pase 10',',192 extendidas,15 raíces,TT1048576':', 192 extendidas, 15 raíces, TT1048576',
        'pistas2.':'pistas de fase 2.',':9 comprobaciones':': 9 comprobaciones',',forzada,reloj':', forzada, reloj',
        'tope1nodo':'tope 1 nodo',',profundidad completa0,aborto1':', profundidad completa 0, aborto 1','suites7A/8A':'suites 7A/8A',
        'pasó10s':'pasó 10s','cpu.max100000':'cpu.max 100000','memory.max1073741824,red none,root readonly,usuario65534,caps dropALL,pids64':'memory.max 1073741824, red none, root readonly, usuario 65534, caps dropALL, pids64',
        'eventos OOM0':'eventos OOM 0','jugar_humo=True,0errores/0avisos':'jugar_humo=True, 0 errores/0 avisos',
        'color),semilla2026 y reloj3s':'color), semilla 2026 y reloj 3s',',preparación10s':', preparación 10s',
        'MiB**,sin':'MiB**, sin','exacta,CPU1,memoria/swap1GiB,rednone':'exacta, CPU1, memoria/swap1GiB, red none',
        ':19 raíces,7456 textos':': 19 raíces, 7456 textos','y890 trazas':'y 890 trazas',',0usos/0referencias':', 0 usos/0 referencias',
        'test256/S13':'test 256/S13','random,orden':'random, orden',',juegos0/0/1/1,pareja0,30s/10s/1CPU/1GiB':', juegos 0/0/1/1, pareja 0, 30s/10s/1CPU/1GiB',
        'Resultado4–0':'Resultado 4–0','ilegales,timeouts,errores propios,partidas abortadas':'ilegales, timeouts, errores propios, partidas abortadas',
        'bajo30s/10s/1024MiB':'bajo 30s/10s/1024MiB','contenido8archivos/74599bytes bajo100MiB/50MiB/2000archivos':'contenido 8 archivos/74599 bytes bajo 100MiB/50MiB/2000 archivos',
        'deS13':'de S13','examen9,libro nuevo,entrenamiento':'examen9, libro nuevo, entrenamiento','herramientas/,nota':'herramientas/, nota'}
    for a,b in replacements.items():text=text.replace(a,b)
    (OUT/'CIERRE_8B2.md').write_text(text)
    note=f'''---
tipo: experimento
estado: 8B2-tecnico-I0-pasado-fase8-global-abierta
fecha: 2026-10-29
baseline_commit: {head}
---

# 8B.2 — controles técnicos reales de I0

I0 con respaldo seleccionado explícitamente solo para técnica. Búsqueda cc888a3c, estrategia88ce9d47, NPZ37678d0b, libro00fd209f; no S13/U/modelo amplio ni promoción. Principal/informe/caché y17 fuentes5 intactos.

Regresión original extendida Python/JIT pasa (10+192 ventanas,15raíces por pase, dos fallos P reproducidos y cero corregidos). Linux ARM64, imagen79983872…, Python3.12.16/NumPy2.4.6/Numba0.67.0. Tres fríos sin caché propia: máximo2.717s y142.54MiB cgroup; JIT real dentro de preparar. No hizo falta añadir caché Linux/Mac. Prioridad/forzada/legales/equivalencia TT y fallback por aborto real en ambos colores comprobados.

Validador oficial entrega con2humos separado,0errores/avisos. Cuatro técnicas random,2por color,semilla20261221 ahora consumida,4–0 sin fallos propios/timeout/ilegalidad. Máximos de tanda:preparar3.032s,reloj14.375s,pico143.36MiB. No prueba fuerza ni finalista; fase8global sigue abierta.

{link('Certificado y traspaso',OUT/'CIERRE_8B2.md')} · {link('Métricas',OUT/'metricas.csv')} · {link('Paquete técnico',exported/'participante.tar')} · {link('Ventana terminada',OUT/'VENTANA_TECNICA_TERMINADA.json')}.

Primera ventana rechazada por Spotlight, posteriores comprobadas sin carga ajena pesada; ningún servicio/GRIS/clúster detenido. Solo contenedores propios, sin limpieza global. Instrumentación cgroup al adios fuera del reloj; fuentes cargadas/SHA/co_filename reales. Ventana técnica terminada; puede coordinarse validación independiente deS13 después, sin iniciarla desde aquí. Pendientes selección deportiva/integración/informe final/otros modelos/arquitecturas. No notas globales ni examen9.
'''
    for a,b in {'estrategia88':'estrategia 88','NPZ376':'NPZ 376','libro00':'libro 00','y17 fuentes5':'y 17 fuentes de 5',
                'ventanas,15raíces':'ventanas, 15 raíces','imagen799':'imagen 799','Python3.12':'Python 3.12','NumPy2.4':'NumPy 2.4','Numba0.67':'Numba 0.67',
                'máximo2.717s y142.54':'máximo 2.717s y 142.54','con2humos separado,0errores':'con 2 humos separado, 0 errores',
                'random,2por color,semilla20261221 ahora consumida,4–0':'random, 2 por color, semilla 20261221 ahora consumida, 4–0',
                'tanda:preparar3.032s,reloj14.375s,pico143.36MiB':'tanda: preparar 3.032s, reloj 14.375s, pico 143.36MiB',
                'fase8global':'fase 8 global','deS13':'de S13'}.items():note=note.replace(a,b)
    NOTE.write_text(note)
    for file in [OUT/'CIERRE_8B2.md',NOTE]:
        assert all(Path(p).exists() for p in re.findall(r'\]\(<([^>]+)>\)',file.read_text()))
    tools=sorted((TEAM/'herramientas').glob('fase8b2_*.py'))
    manifest={'at':now,'decision_sha256':frozen['decision_sha256'],'branch':before['branch'],'HEAD':head,
              'git_status':subprocess.check_output(['rtk','proxy','git','status','--short','--branch'],text=True),
              'checks':checks,'active_report_preserved':True,'tested_runtime':runtime,'export_manifest':export_manifest,
              'ABI':abi,'image':IMAGE,'regressions':regressions,'cold':{'trials':3,'max_prepare_s':cold['max_prepare_s'],'max_peak_MiB':cold['max_cgroup_peak_bytes']/1024**2},
              'smoke':smoke,'technical':technical,'technical_by_color':colors,'measured_maxima':measured_maxima,'Linux_functional_cases':11,
              'cache_built_or_copied':False,'cache_not_needed_reason':'No-cache import+prepare and3cold trials under10s with real JIT cache misses',
              'seed20261221_consumed':True,'own_containers_remaining':0,'window_ended':True,
              'technical_scope_passed':True,'phase8_global_closed':False,'sports_strength_certified':False,'S13_used':False,
              'new_test5_read':False,'active_integration':False,'handoff':handoff,
              'tools':{str(p):sha(p,budget) for p in tools},'note':str(NOTE),'note_sha256':sha(NOTE),
              'host_environment':{'python':sys.version,'numpy':importlib.metadata.version('numpy'),'numba':importlib.metadata.version('numba'),'architecture':platform.machine()},
              'global_pending':['independent_S13_validation','sports_selection','final_integration_and_report','protocol_for_other_models_or_architectures']}
    save(OUT/'MANIFIESTO_FINAL.json',manifest)
    save(OUT/'progreso.json',{'status':'8B2_I0_technical_passed_window_ended','required_work_remaining':[],
                            'global_phase8_open':True,'own_heavy_processes_finished':True,'no_automatic_next_phase':True})
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(sha(p,budget)+'  '+str(p.relative_to(OUT))+'\n' for p in files))
    budget.wait()
    print(json.dumps({'technical_passed':True,'phase8_global_closed':False,'window_ended':True,'files_hashed':len(files),
                      'technical_games':4,'smoke_games':2,'max_technical_prepare_s':technical['max_prepare_s'],
                      'max_technical_clock_s':technical['max_clock_s'],'max_technical_cgroup_MiB':technical['max_cgroup_peak_bytes']/1024**2,
                      'phase5_sources_preserved':17},indent=2))


if __name__=='__main__':main()
