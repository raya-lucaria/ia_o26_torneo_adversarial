"""MCTS adversarial acotado. Sus hojas usan un evaluador congelado.

W/N guarda retornos aproximados desde el padre, por resultado de combate.
UCT selecciona acciones; el azar selecciona resultados según el motor. Q
pondera los promedios de ambos resultados con sus probabilidades conocidas.
"""
import math
import random
import time


class Accion:
    def __init__(self, estado, jugada, evaluar, previos):
        self.jugada = jugada
        self.ramas = []
        self.n = 0
        for k,(p,h) in enumerate(estado.siguiente(jugada)):
            ganador = h.ganador()
            previo = previos.get((str(jugada),k))
            n,w = (0,0.) if previo is None else previo
            base = 1.-evaluar(h) if ganador is None else float(ganador==estado.turno)
            self.ramas.append([p,h,n,w,base,ganador is not None])
            self.n += n

    def q(self):
        return sum(p*(base if terminal or not n else w/n) for p,h,n,w,base,terminal in self.ramas)


class Nodo:
    def __init__(self, estado, evaluar, previos):
        self.acciones = [Accion(estado,j,evaluar,previos) for j in estado.jugadas()]
        self.n = sum(a.n for a in self.acciones)


def explorar(estado, evaluar, iteraciones=128, semilla=1, previos=None,
             segundos=30., max_nodos=2048, max_profundidad=12, reloj=time.perf_counter):
    from compacto import identidad
    inicio = reloj();plazo = inicio+max(0.,segundos)
    rng = random.Random(semilla)
    arbol = {}
    metricas={'profundidad_visitada':0,'terminales_visitados':0,'hojas_modelo':0}
    if estado.ganador() is not None:
        return {'valor':float(estado.ganador()==estado.turno),'acciones':[],
                'iteraciones':0,'nodos':0,'cobertura':True,'resuelto':True,'segundos':0.}

    def visitar(e,prof):
        metricas['profundidad_visitada']=max(metricas['profundidad_visitada'],prof)
        if reloj() >= plazo:
            raise TimeoutError
        ganador = e.ganador()
        if ganador is not None:
            metricas['terminales_visitados']+=1
            return float(ganador==e.turno)
        if prof >= max_profundidad:
            metricas['hojas_modelo']+=1
            return evaluar(e)
        clave = identidad(e)
        nodo = arbol.get(clave)
        if nodo is None:
            if len(arbol) >= max_nodos:
                return evaluar(e)
            # Una expansión nueva por iteración; los sucesores se evalúan una vez.
            nodo = Nodo(e,evaluar,previos or {} if prof==0 else {})
            arbol[clave] = nodo
            metricas['hojas_modelo']+=1
            return max(a.q() for a in nodo.acciones)
        menores = [a for a in nodo.acciones if a.n<4]
        if menores:
            a = min(menores,key=lambda a:a.n)
        else:
            a = max(nodo.acciones,key=lambda a:a.q()+.4*math.sqrt(math.log(nodo.n+1)/(a.n+1)))
        u=rng.random();acum=0.;k=len(a.ramas)-1
        for i,rama in enumerate(a.ramas):
            acum += rama[0]
            if u<acum:
                k=i;break
        p,h,n,w,base,terminal = a.ramas[k]
        retorno = base if terminal else 1.-visitar(h,prof+1)
        a.ramas[k][2] += 1
        a.ramas[k][3] += retorno
        a.n += 1;nodo.n += 1
        return max(opcion.q() for opcion in nodo.acciones)

    hechas=0
    for _ in range(iteraciones):
        try:
            visitar(estado,0)
        except TimeoutError:
            break
        hechas += 1
    raiz = arbol.get(identidad(estado))
    acciones=[]
    if raiz:
        for a in raiz.acciones:
            acciones.append({'jugada':str(a.jugada),'q':a.q(),'visitas':a.n,
                'ramas':[{'prob':p,'estado':identidad(h),'visitas':n,'suma':w,
                          'previo':base,'terminal':terminal}
                         for p,h,n,w,base,terminal in a.ramas]})
    return {'valor':max((a['q'] for a in acciones),default=evaluar(estado)),
            'acciones':acciones,'iteraciones':hechas,'nodos':len(arbol),
            'cobertura':bool(acciones) and all(a['visitas']>=4 for a in acciones),
            'resuelto':False,'segundos':reloj()-inicio,**metricas}
