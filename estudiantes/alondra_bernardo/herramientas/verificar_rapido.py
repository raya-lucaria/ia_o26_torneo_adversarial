"""Compara la búsqueda numba (sin quietud) con expectiminimax sin poda en Python."""
import random, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from motor import Estado
import rapido as r

ctx = r.Contexto(r.pesos_iniciales()); ctx.control[4] = 0
f = np.zeros(r.RASGOS)

def ev(e):
    c = e._color
    return r.evaluar(e._tablero, e._puntos, c, ctx.w, f)

def ref(e, prof, ply):
    g = e.ganador()
    if g is not None:
        return -(1.0 - 1e-4 * ply)  # quien acaba de mover ganó
    if prof == 0:
        return ev(e)
    js = e.jugadas()
    if not js:
        return -(1.0 - 1e-4 * ply)
    return max(sum(p * -ref(h, prof - 1, ply + 1) for p, h in e.siguiente(j)) for j in js)

rng = random.Random(7); err = 0; casos = 0; t0 = time.time()
estados = []
for g in range(40):
    e = Estado.inicial(); n = 0
    while e.ganador() is None:
        if n in (14, 30, 46, 60): estados.append(e)
        j = rng.choice(e.jugadas()); e = rng.choice([h for _, h in e.siguiente(j)]); n += 1
for e in estados[:60]:
    for prof in (1, 2, 3):
        ctx.limpiar()
        v, jug = r.buscar(ctx, e._tablero, e._puntos, e._color, prof, 10**9)
        vr = ref(e, prof, 0)
        err = max(err, abs(v - vr)); casos += 1
        legal = {f"{j}" for j in e.jugadas()}
        assert r.texto(jug) in legal, (r.texto(jug), legal)
print(f"{casos} comparaciones, diferencia máxima {err:.2e}, {time.time()-t0:.0f}s")
