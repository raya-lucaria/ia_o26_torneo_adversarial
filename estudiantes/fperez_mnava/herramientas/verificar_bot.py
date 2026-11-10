"""Verifica que la representación y la búsqueda de strategy.py coinciden con el motor oficial.

Uso (desde la raíz del repo, con .venv activo):
    python estudiantes/<equipo>/herramientas/verificar_bot.py

Revisa, sobre posiciones de partidas al azar:
1. que el generador de jugadas y sus probabilidades coinciden con `Estado.jugadas()` / `prob_ataque`;
2. que hacer/deshacer restaura el tablero y coincide con `Estado.aplicar`;
3. que la búsqueda con poda alfa-beta da el mismo valor que expectimax sin poda.
Correrlo después de CUALQUIER cambio en strategy.py: un error aquí es una jugada ilegal en la liga.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import strategy as S  # noqa: E402
from motor import Estado, Jugada  # noqa: E402

random.seed(5)


def posiciones(n_partidas=6):
    """Posiciones de partidas jugadas al azar (con dados al azar)."""
    salida = []
    for _ in range(n_partidas):
        e = Estado.inicial()
        while e.ganador() is None:
            salida.append(e)
            j = random.choice(e.jugadas())
            if e.es_ataque(j):
                e = e.aplicar(j, gana_ataque=random.random() < e.prob_ataque(j))
            else:
                e = e.aplicar(j)
    return salida


def plano(b, lado, prof):
    """Expectimax sin ninguna poda (referencia lenta)."""
    if prof == 0:
        return S._evaluar(b.tab, b.pos, b.pts, lado)
    js = S._generar(b.tab, b.pos, lado)
    if not js:
        return 0.0
    mejor = -1.0
    for o, d, p in js:
        resultados = ((p, True), (1 - p, False)) if p < 1 else ((1.0, True),)
        mv = 0.0
        for prob, gana in resultados:
            u = S._hacer(b.tab, b.pos, b.pts, lado, o, d, gana)
            v = 1.0 if b.pts[lado] >= 3 else 1.0 - plano(b, 1 - lado, prof - 1)
            S._deshacer(b.tab, b.pos, b.pts, lado, u)
            mv += prob * v
        mejor = max(mejor, mv)
    return mejor


def main():
    lista = posiciones()
    print("posiciones de prueba:", len(lista))
    fallas = 0

    mal = 0
    for e in random.sample(lista, 200):
        tab, pos, pts = S._desde_estado(e)
        lado = 0 if e.turno == "blancas" else 1
        mias = {S._nombre(o) + "-" + S._nombre(d): p for o, d, p in S._generar(tab, pos, lado)}
        ref = {str(j): e.prob_ataque(j) for j in e.jugadas()}
        if mias.keys() != ref.keys() or any(abs(mias[k] - ref[k]) > 1e-12 for k in ref):
            mal += 1
    print("1. generador distinto del motor en", mal, "de 200 posiciones")
    fallas += mal

    mal = 0
    for e in random.sample(lista, 200):
        tab, pos, pts = S._desde_estado(e)
        lado = 0 if e.turno == "blancas" else 1
        for o, d, p in S._generar(tab, pos, lado):
            for gana in (True, False) if p < 1 else (True,):
                antes = (list(tab), (set(pos[0]), set(pos[1])), list(pts))
                u = S._hacer(tab, pos, pts, lado, o, d, gana)
                j = Jugada.desde_texto(S._nombre(o) + "-" + S._nombre(d))
                ref = e.aplicar(j, gana_ataque=gana if p < 1 else None)
                if S._desde_estado(ref) != (tab, pos, pts):
                    mal += 1
                S._deshacer(tab, pos, pts, lado, u)
                if (tab, pos, pts) != antes:
                    mal += 1
    print("2. hacer/deshacer inconsistentes:", mal)
    fallas += mal

    mal = 0
    muestra = random.sample(lista[len(lista) // 4 :], 25)
    for e in muestra:
        lado = 0 if e.turno == "blancas" else 1
        for prof in (1, 2, 3):
            v1 = S._Buscador(*S._desde_estado(e), lado).valor(lado, prof, -1.0, 2.0)
            v2 = plano(S._Buscador(*S._desde_estado(e), lado), lado, prof)
            if abs(v1 - v2) > 1e-9:
                mal += 1
    print("3. valores con poda distintos del plano:", mal, "de", 3 * len(muestra))
    fallas += mal

    tab, pos, pts = S._desde_estado(Estado.inicial())
    b = S._Buscador(tab, pos, pts, 0)
    t = time.perf_counter()
    b.valor(0, 3, -1.0, 2.0)
    dt = time.perf_counter() - t
    print(f"velocidad: {b.nodos} nodos en {dt:.2f} s ({b.nodos / dt:.0f} nodos/s, prof. 3 desde el inicio)")
    print("TODO BIEN" if fallas == 0 else f"HAY {fallas} FALLAS: no entregues así")


if __name__ == "__main__":
    main()
