"""Rivales offline: UCT con azar explicito y dos estilos de una jugada.

UCT guarda retornos desde un objetivo fijo, MAX en su turno y MIN en el rival.
Los resultados de combate se muestrean por p; no se seleccionan como acciones.
"""
from __future__ import annotations
import math,random
from motor import simular_al_azar


class Arista:
    def __init__(self,jugada):
        self.jugada=jugada;self.n=0;self.suma=0.;self.hijos={}


class Nodo:
    def __init__(self,estado):
        self.estado=estado;self.n=0
        self.aristas=[Arista(j) for j in estado.jugadas()]


def uct(estado,semilla,iteraciones=512,filtrar=True):
    if iteraciones<1:raise ValueError('Iteraciones debe ser positiva')
    rng=random.Random(semilla);objetivo=estado.turno
    legales=estado.jugadas()
    if not legales:raise ValueError('Estado terminal')
    if filtrar:
        for j in legales:
            if all(h.ganador()==objetivo for _,h in estado.siguiente(j)):
                return j,{'victoria_inmediata':True,'iteraciones':0}
    raiz=Nodo(estado)
    for _ in range(iteraciones):
        nodo=raiz;camino=[];visitados=[nodo]
        while nodo.estado.ganador() is None:
            sin_visitar=[a for a in nodo.aristas if not a.n]
            if sin_visitar:
                a=rng.choice(sin_visitar)
            else:
                signo=1 if nodo.estado.turno==objetivo else -1
                a=max(nodo.aristas,key=lambda a:signo*a.suma/a.n+math.sqrt(2*math.log(nodo.n+1)/a.n))
            ramas=nodo.estado.siguiente(a.jugada)
            u=rng.random();acum=0.;indice=len(ramas)-1
            for k,(p,_) in enumerate(ramas):
                acum+=p
                if u<acum:indice=k;break
            camino.append(a)
            nuevo=indice not in a.hijos
            if nuevo:a.hijos[indice]=Nodo(ramas[indice][1])
            nodo=a.hijos[indice];visitados.append(nodo)
            if nuevo:break
        ganador=nodo.estado.ganador()
        retorno=float(ganador==objetivo) if ganador is not None else float(simular_al_azar(
            nodo.estado,1,rng.randrange(2**31),para=objetivo))
        for n in visitados:n.n+=1
        for a in camino:a.n+=1;a.suma+=retorno
    mejor=max(raiz.aristas,key=lambda a:a.n)
    return mejor.jugada,{'iteraciones':iteraciones,'aristas':[
        {'jugada':str(a.jugada),'visitas':a.n,'valor':a.suma/a.n if a.n else None} for a in raiz.aristas]}


def valor_estilo(estado,objetivo,estilo):
    ganador=estado.ganador()
    if ganador is not None:return 10000. if ganador==objetivo else -10000.
    rival='negras' if objetivo=='blancas' else 'blancas'
    def avances(color):
        return [(int(s[1])-1 if color=='blancas' else 9-int(s[1])) for s in estado.piezas(color)]
    a,b=avances(objetivo),avances(rival)
    puntos=100*(estado.puntos[objetivo]-estado.puntos[rival])
    if estilo=='agresivo':return puntos+5*(len(a)-len(b))+.1*(sum(a)-sum(b))
    return puntos+4*((max(a) if a else -1)-(max(b) if b else -1))+.2*(sum(a)-sum(b))


def estilo(estado,nombre,semilla):
    objetivo=estado.turno;rng=random.Random(semilla)
    opciones=[]
    for j in estado.jugadas():
        valor=sum(p*valor_estilo(h,objetivo,nombre) for p,h in estado.siguiente(j))
        opciones.append((valor,j))
    mejor=max(v for v,_ in opciones)
    return rng.choice([j for v,j in opciones if abs(v-mejor)<1e-12]),{'estilo':nombre,'valor':mejor}
