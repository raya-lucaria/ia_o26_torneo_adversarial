"""Resume el panel completo y declara explicitamente casos/profundidades pendientes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--resultados',type=Path,required=True)
    p.add_argument('--adicional',type=Path,action='append',default=[])
    args=p.parse_args()
    casos=[json.loads(x) for x in args.panel.read_text().splitlines()]
    encontrados={}
    for carpeta in [args.resultados]+args.adicional:
        for f in carpeta.glob('*_d*.json'):
            r=json.loads(f.read_text())
            clave=(r['id'],r['profundidad'])
            if clave in encontrados:
                raise ValueError(f'Diagnostico duplicado: {clave}')
            encontrados[clave]=r
    resumen=[]
    for caso in casos:
        rs=sorted([r for (ident,_),r in encontrados.items() if ident==caso['id']],key=lambda r:r['profundidad'])
        completos=[r for r in rs if r['completo'] and not r.get('aborto_buscador',True)]
        clase='pendiente' if not rs else 'evidencia_insuficiente'
        if any(r.get('certificado_decision') for r in rs):
            clase='error_decision_demostrado'
        elif any(r.get('coincide_buscador') is False for r in completos):
            clase='discrepancia_busqueda_requiere_revision'
        elif completos:
            clase='sin_discrepancia_mismo_horizonte'
            if any(not(set(a['jugadas_optimas'])&set(b['jugadas_optimas']))
                   for a,b in zip(completos,completos[1:])):
                clase='sensibilidad_al_horizonte'
        resumen.append({'id':caso['id'],'categoria':caso['categoria_panel'],'color':caso['color_propio'],
                        'clasificacion':clase,'profundidades_completas':[r['profundidad'] for r in completos],
                        'profundidades_truncadas':[r['profundidad'] for r in rs if not r['completo']],
                        'jugada_observada':caso['jugada_observada'],
                        'brechas_observadas_por_horizonte':{str(r['profundidad']):r['brecha_observada'] for r in completos},
                        'limite':'Solo arbol truncado. Diferir de jugada observada a menor horizonte no demuestra error del modelo'})
    salida={'casos':len(casos),'categorias':dict(Counter(x['clasificacion'] for x in resumen)),
            'carpetas_fuente':[str(x) for x in [args.resultados]+args.adicional],
            'casos_analizados':sum(bool(r['profundidades_completas']) for r in resumen),
            'casos_con_dos_horizontes':sum(len(r['profundidades_completas'])>=2 for r in resumen),
            'detalle':resumen,'errores_modelo_demostrados':None,'fuerza_medida':False}
    args.resultados.mkdir(parents=True,exist_ok=True)
    (args.resultados/'RESUMEN.json').write_text(json.dumps(salida,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in salida.items() if k!='detalle'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
