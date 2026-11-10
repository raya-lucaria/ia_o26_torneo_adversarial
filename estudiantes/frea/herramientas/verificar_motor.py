"""Compara mi_motor con el motor oficial en partidas al azar: jugadas, p, resultado y clave."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mi_motor  # noqa: E402
from motor import Estado  # noqa: E402

rng = random.Random(1)
buf = mi_motor.crear_buffers(mi_motor.PESOS_INICIALES, bits_tt=4)
tab, pts, mo, md, Z = buf[0], buf[1], buf[2], buf[3], buf[13]


def cargar(e):
    tab[:] = 0
    for v, nombre in ((1, "blancas"), (2, "negras")):
        for c in e.piezas(nombre):
            tab[(int(c[1]) - 1) * 9 + ord(c[0]) - 97] = v
    pts[0], pts[1] = e.puntos["blancas"], e.puntos["negras"]


revisadas = 0
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
    e = Estado.inicial()
    while not e.terminado():
        cargar(e)
        turno = 0 if e.turno == "blancas" else 1
        n = mi_motor.generar(tab, turno, mo, md, 0)
        propias = sorted((int(mo[0, i]), int(md[0, i])) for i in range(n))
        assert propias == sorted((j.origen, j.destino) for j in e.jugadas())
        h = int(mi_motor.clave_de(tab, pts, turno, Z))
        for j in e.jugadas():
            antes_tab, antes_pts = tab.copy(), pts.copy()
            antes = tab[j.destino]
            for k, (p, hijo) in enumerate(e.siguiente(j)):
                gana = k == 0
                if antes != 0:
                    assert abs(mi_motor.prob_ataque(tab, turno, j.origen, j.destino) - e.prob_ataque(j)) < 1e-9
                h2 = mi_motor.hacer(tab, pts, turno, j.origen, j.destino, gana, h, Z)
                esperado = np.zeros(81, np.int8)
                for v, nombre in ((1, "blancas"), (2, "negras")):
                    for c in hijo.piezas(nombre):
                        esperado[(int(c[1]) - 1) * 9 + ord(c[0]) - 97] = v
                assert np.array_equal(tab, esperado), (str(e), str(j), gana)
                assert (pts[0], pts[1]) == (hijo.puntos["blancas"], hijo.puntos["negras"])
                assert h2 == mi_motor.clave_de(tab, pts, 1 - turno, Z)
                if gana or antes == 0:
                    mi_motor.deshacer(tab, pts, turno, j.origen, j.destino, antes)
                else:
                    tab[j.origen] = turno + 1
                assert np.array_equal(tab, antes_tab) and np.array_equal(pts, antes_pts), (str(e), str(j), gana)
            revisadas += 1
        j = rng.choice(e.jugadas())
        e = rng.choices([s for _, s in e.siguiente(j)], [p for p, _ in e.siguiente(j)])[0]
print(f"ok: {revisadas} jugadas revisadas")
