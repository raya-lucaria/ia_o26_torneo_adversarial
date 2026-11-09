"""Monte Carlo plano offline de presupuesto fijo; no es un benchmark de liga."""
import random
from motor import simular_al_azar

def montecarlo(estado, semilla, muestras=128, rollouts=20):
    if muestras < 1 or rollouts < 1:
        raise ValueError('Presupuestos positivos requeridos')
    objetivo=estado.turno; rng=random.Random(semilla); legales=estado.jugadas()
    if not legales: raise ValueError('Estado terminal')
    for j in legales:
        if all(h.ganador()==objetivo for _,h in estado.siguiente(j)):
            return j, {'victoria_inmediata':True}
    valores=[]
    for j in legales:
        ramas=estado.siguiente(j); total=0.
        for _ in range(muestras):
            u=rng.random(); acumulada=0.; hijo=ramas[-1][1]
            for prob, candidato in ramas:
                acumulada+=prob
                if u<acumulada:
                    hijo=candidato; break
            total+=simular_al_azar(hijo,rollouts,rng.randrange(2**31),para=objetivo)
        valores.append(total/muestras)
    mejor=max(range(len(legales)),key=lambda i:valores[i])
    return legales[mejor], {'muestras_por_jugada':muestras,'rollouts_por_muestra':rollouts,'eval':valores[mejor]}
