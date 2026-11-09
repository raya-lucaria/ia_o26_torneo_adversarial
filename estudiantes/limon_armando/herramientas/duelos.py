"""Resumen de duelos contra un rival: python3 duelos.py <desde YYYY-mm-dd_HH-MM> [rival].

Lee resultados/<fecha>/*.json de `./torneo jugar` (cada partida se guarda al terminar) de las
corridas desde esa fecha, y da victorias, intervalo de Wilson 95 %, reloj y preparar máximos.
"""
import glob, json, math, os, sys
from collections import defaultdict

desde = sys.argv[1]
rival = sys.argv[2] if len(sys.argv) > 2 else "montecarlo"
g, n, reloj, prep, err = defaultdict(int), defaultdict(int), defaultdict(float), defaultdict(float), defaultdict(int)
for d in sorted(glob.glob("resultados/*/")):
    if os.path.basename(d.rstrip("/")) < desde:
        continue
    for f in glob.glob(d + "*.json"):
        r = json.load(open(f))
        m = r["meta"]
        if rival not in (m["blancas"], m["negras"]):
            continue
        bot = m["negras"] if m["blancas"] == rival else m["blancas"]
        color = "blancas" if m["blancas"] == bot else "negras"
        n[bot] += 1
        g[bot] += r.get("ganador") == color
        err[bot] += r.get("motivo") not in (None, "puntos")
        for j in r.get("jugadas", []):
            if j["color"] == color:
                reloj[bot] = max(reloj[bot], j.get("reloj", 0))
        prep[bot] = max(prep[bot], (r.get("preparar_usado") or {}).get(color) or 0)

def wilson(k, m, z=1.96):
    p = k / m; c = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m)
    mid = (p + z * z / (2 * m)) / (1 + z * z / m)
    return mid - c, mid + c

print(f"{'bot':>16} {'gana':>5} {'n':>4}  {'%':>5}  IC 95 %        reloj máx  prep máx  no-puntos")
for bot in sorted(n, key=lambda b: -g[b] / n[b]):
    lo, hi = wilson(g[bot], n[bot])
    print(f"{bot:>16} {g[bot]:>5} {n[bot]:>4}  {100*g[bot]/n[bot]:5.1f}  [{100*lo:4.1f}, {100*hi:4.1f}]   {reloj[bot]:6.1f}   {prep[bot]:6.1f}     {err[bot]}")
