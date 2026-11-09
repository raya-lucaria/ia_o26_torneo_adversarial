"""Clasifica las partidas perdidas que guardó `arena.py --guardar` (no corre en la liga).

Para cada partida reconstruye la evaluación de nuestro bot (A) en cada una de sus jugadas y
la «suerte de dados» de cada bando: suma de (resultado − p) en sus ataques, en ataques
ganados de más. Clasifica cada derrota en:
- mala suerte: nuestra suerte − la del rival ≤ −1.5 ataques;
- colapso: la evaluación llegó a ≥ +0.5 (íbamos claramente ganando) y se perdió;
- nunca bien: el resto.

Uso:
    docker run --rm -v "$PWD":/torneo -w /torneo/estudiantes/t_regex torneo-bot-base:local \
        python herramientas/analizar_perdidas.py /torneo/resultados/entrenamiento/perdidas_mc.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import busqueda as bq  # noqa: E402


def analizar(partida):
    yo = 0 if partida["a_es"] == "blancas" else 1
    tab = np.zeros(81, np.int8)
    tab[:18] = 1
    tab[-18:] = 2
    pts = np.zeros(2, np.int64)
    buf = bq.nuevos_buferes()
    suerte = [0.0, 0.0]
    ataques = [0, 0]
    evals = []
    color = 0
    for jugada, gana in partida["jugadas"]:
        o, d = jugada // 81, jugada % 81
        if color == yo:
            evals.append(bq.evaluar(tab, pts, yo, buf))
        if tab[d] != 0:
            p = bq.prob_ataque(tab, color, o, d)
            suerte[color] += (1.0 if gana else 0.0) - p
            ataques[color] += 1
        bq.hacer(tab, pts, color, o, d, gana, np.uint64(0))
        color = 1 - color
    neta = suerte[yo] - suerte[1 - yo]
    if neta <= -1.5:
        tipo = "mala suerte"
    elif max(evals) >= 0.5:
        tipo = "colapso"
    else:
        tipo = "nunca bien"
    return tipo, neta, max(evals), len(partida["jugadas"]), ataques[yo], ataques[1 - yo], partida


def main():
    partidas = json.loads(Path(sys.argv[1]).read_text())
    res = [analizar(p) for p in partidas]
    cuenta = Counter(r[0] for r in res)
    print(f"{len(res)} derrotas:")
    for tipo, n in cuenta.most_common():
        sub = [r for r in res if r[0] == tipo]
        print(f"  {tipo:12} {n:3}  suerte neta media {np.mean([r[1] for r in sub]):+.2f}"
              f"  eval máx media {np.mean([r[2] for r in sub]):+.2f}"
              f"  plies {np.mean([r[3] for r in sub]):.0f}"
              f"  ataques nuestros/rival {np.mean([r[4] for r in sub]):.1f}/"
              f"{np.mean([r[5] for r in sub]):.1f}")
    print(f"suerte neta media en todas las derrotas: {np.mean([r[1] for r in res]):+.2f}")


if __name__ == "__main__":
    main()
