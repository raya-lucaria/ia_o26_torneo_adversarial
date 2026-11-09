"""Regenera datos de la entrega local. Ejecutar offline con motor instalado.

El corpus de identidades no contiene valores: fija qué estados resolver. Se
comprueba su cierre antes de propagar exactamente todos los resultados de azar.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

PAQUETE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAQUETE))
from compacto import identidad,reconstruir
from geometria import NOMBRES_V4,pesos_iniciales_v4


def referencias_historicas():
    """Constantes documentadas; no se cargan en la estrategia entregable."""
    return {
        'neutral.json': {'formato':1,'version':'neutral_sin_entrenar',
            'rasgos':['constante','puntos_diferencia','material_diferencia',
                      'avance_max_propio','avance_max_rival','avance_total_diferencia',
                      'a_un_paso_propio','a_un_paso_rival','apoyo_diferencia'],
            'pesos':[0,0,0,0,0,0,0,0,0],'nota':'Referencia neutral conservada.'},
        'referencia.json': {'formato':2,'version':'referencia_manual_v2',
            'rasgos':['constante','puntos','material','tres_avanzados','movilidad',
                      'apoyo','anotaciones_disponibles','exposicion_rival_menos_propia',
                      'amenaza_tercer_punto'],
            'pesos':[0,5,1,2,0.5,0.5,3,1,5],
            'nota':'Hipotesis manual previa a validacion. Salida acotada, sin calibracion demostrada.'}}


def logico(estados,valores,resueltos):
    return hashlib.sha256(estados.tobytes()+valores.tobytes()+resueltos.tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--salida',type=Path)
    p.add_argument('--verificar',action='store_true',help='Recalcula y compara, sin sobrescribir datos')
    a=p.parse_args()
    if not a.verificar and a.salida is None:p.error('Indica --verificar o --salida para evitar sobrescribir por accidente')
    modelo={'formato':4,'version':'geometria_carreras_manual_v1','rasgos':list(NOMBRES_V4),
            'pesos':pesos_iniciales_v4().tolist(),'nota':'Hipotesis manual; salida sin calibrar.'}
    texto=json.dumps(modelo,indent=2,ensure_ascii=False)+'\n'
    originales=json.loads((PAQUETE/'data/modelo.json').read_text())
    if originales!=modelo:raise RuntimeError('El modelo entregado no coincide con los pesos manuales definidos')
    auxiliares=referencias_historicas()
    for nombre,datos in auxiliares.items():
        if json.loads((PAQUETE/'data'/nombre).read_text())!=datos:
            raise RuntimeError('Referencia histórica distinta: '+nombre)
    estados=np.load(Path(__file__).parent/'estados_finales.npy',allow_pickle=False)
    if estados.dtype!=np.uint8 or estados.shape[1:]!=(22,):raise ValueError('Corpus inválido')
    claves=[k.tobytes() for k in estados];permitidas=set(claves);cache={}
    if len(permitidas)!=len(claves):raise ValueError('Identidades repetidas')
    def visitar(e):
        k=identidad(e)
        if k in cache:return cache[k]
        if k not in permitidas:raise RuntimeError('El corpus no es cerrado: falta un sucesor')
        g=e.ganador()
        v=float(g==e.turno) if g is not None else max(
            sum(p*(1-visitar(h)) for p,h in e.siguiente(j)) for j in e.jugadas())
        cache[k]=v;return v
    valores=np.asarray([visitar(reconstruir(k)) for k in claves],dtype=np.float64)
    resueltos=np.ones(len(claves),np.uint8)
    digest=logico(estados,valores,resueltos)
    esperado=json.loads((Path(__file__).parent/'origen_datos.json').read_text())
    if digest!=esperado['finales_sha256_logico']:raise RuntimeError('No se reprodujeron exactamente los valores')
    with np.load(PAQUETE/'data/finales.npz',allow_pickle=False) as f:
        if not np.array_equal(estados,f['estados']) or not np.array_equal(valores,f['valores']) or not np.array_equal(resueltos,f['resueltos']):
            raise RuntimeError('La tabla entregada difiere de la solución recalculada')
    if not a.verificar:
        a.salida.mkdir(parents=True,exist_ok=True)
        (a.salida/'modelo.json').write_text(texto)
        np.savez(a.salida/'finales.npz',estados=estados,valores=valores,resueltos=resueltos)
        for nombre,datos in auxiliares.items():
            (a.salida/nombre).write_text(json.dumps(datos,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'modelo_identico':True,'finales_recalculados':len(cache),
                     'valores_identicos_bit_a_bit':True,'referencias_historicas_identicas':True,
                     'sha256_logico':digest,'solo_verificacion':a.verificar}))


if __name__=='__main__':main()
