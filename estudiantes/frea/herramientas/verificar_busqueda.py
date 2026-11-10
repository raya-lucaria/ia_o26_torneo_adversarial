"""Compara _buscar sin selectividad (ctx[5] = 1) contra expectiminimax de fuerza bruta sin podas.

Incluye posiciones de carrera (2 a 2), donde un resultado de ataque vale ±1 exacto.
"""

import sys
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARPETA))
import mi_motor as m  # noqa: E402

B = m.crear_buffers(np.load(CARPETA / "data" / "pesos.npy"), bits_tt=20)
tab, pts, mo, md, ctx, f, mk, Z = B[0], B[1], B[2], B[3], B[9], B[10], B[11], B[13]


def bruto(t, prof, ply):
    n = m.generar(tab, t, mo, md, ply)
    if n == 0:
        return -1.0
    jugadas = [(int(mo[ply, i]), int(md[ply, i])) for i in range(n)]
    meta = 8 if t == 0 else 0
    mejor = -2.0
    if prof <= 0:
        mejor = float(m.evaluar(tab, pts, t, f, mk, B[12]))
        if prof <= -m.PROF_QUIESCENCIA:
            return mejor
    for o, d in jugadas:
        antes = int(tab[d])
        p = float(m.prob_ataque(tab, t, o, d)) if antes else 1.0
        if prof <= 0 and not (d // 9 == meta or (antes and p >= 0.64)):
            continue
        m.hacer(tab, pts, t, o, d, True, 0, Z)
        v1 = 1.0 if pts[t] >= 3 else -bruto(1 - t, prof - 1, ply + 1)
        m.deshacer(tab, pts, t, o, d, antes)
        val = v1
        if antes:
            m.hacer(tab, pts, t, o, d, False, 0, Z)
            val = p * v1 + (1 - p) * -bruto(1 - t, prof - 1, ply + 1)
            tab[o] = t + 1
        mejor = max(mejor, val)
    return mejor


rng = np.random.default_rng(9)
for nombre, carrera in (("general", False), ("carrera 2-2", True)):
    difs = []
    for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 150):
        tab[:] = 0
        tab[:18] = 1
        tab[-18:] = 2
        pts[:] = 0
        t, viva = 0, True
        for _ in range(int(rng.integers(20, 85))):
            n = m.generar(tab, t, mo, md, 0)
            if n == 0:
                viva = False
                break
            i = rng.integers(n)
            o, d = int(mo[0, i]), int(md[0, i])
            gana = rng.random() < m.prob_ataque(tab, t, o, d) if tab[d] else True
            m.hacer(tab, pts, t, o, d, gana, 0, Z)
            if pts[t] >= 3:
                viva = False
                break
            t = 1 - t
        if not viva or m.generar(tab, t, mo, md, 0) == 0:
            continue
        if carrera:
            pts[:] = 2
        for prof in (1, 2):
            esperado = bruto(t, prof, 0)
            B[5][:] = 0
            B[8][:] = 0
            ctx[:] = 0
            ctx[1] = 10**12
            ctx[5] = 1
            valor = float(m.buscar_raiz(B, t, int(m.clave_de(tab, pts, t, Z)), prof))
            difs.append(abs(valor - esperado))
    difs = np.array(difs)
    print(f"{nombre}: {len(difs)} casos, {int((difs > 1e-9).sum())} con diferencia, máx {difs.max():.4f}")
