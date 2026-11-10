"""Corrige los pesos con partidas reales contra un rival (offline, no corre en la liga).

1. Reconstruye cada posición de las partidas, calcula `strategy.rasgos` desde quien mueve y
   la etiqueta «quien mueve ganó».
2. Ajusta una regresión logística regularizada hacia los pesos de autojuego (campo
   "pesos_autojuego" de data/pesos.json; si no está, los de strategy.py):
       min  logloss(y, σ(w·x + b·s)) + λ/2·||w − w_autojuego||²
   donde s = +1 si mueve nuestro bot y −1 si mueve el rival; b (sin penalizar) estima la
   ventaja real sobre ese rival en log-momios (la usamos para fijar VENTAJA).
3. Compara con validación cruzada por partidas (5 pliegues): pesos de partida, más b, y
   corregidos. Con --escribir data/pesos.json guarda los corregidos en "pesos" (los que usa el bot).

Acepta los .json de `./torneo jugar` (carpeta resultados/<fecha>/) y los .jsonl de nuestro
arnés de pruebas (una partida por línea con "color_a", "ganador" y "jugadas").

Uso (desde estudiantes/<equipo>/):
    python herramientas/ajustar_vs_rival.py --equipo equipo_gg --lam 0.3 ../../resultados/*/*.json \
        --escribir data/pesos.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

CARPETA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARPETA))


def partidas(rutas, equipo):
    """Genera (color_nuestro, ganador, [(jugada_texto, gano_ataque_o_None), ...])."""
    for ruta in rutas:
        texto = Path(ruta).read_text()
        if ruta.endswith(".jsonl"):
            for linea in texto.splitlines():
                r = json.loads(linea)
                if r.get("motivo") == "puntos":
                    yield r["color_a"], r["ganador"], [(j[0], j[1][1] if j[1] else None) for j in r["jugadas"]]
        else:
            r = json.loads(texto)
            if "meta" not in r or r.get("motivo") != "puntos":
                continue
            nuestro = "blancas" if r["meta"]["blancas"] == equipo else "negras"
            if r["meta"][nuestro] != equipo:
                continue
            yield nuestro, r["ganador"], [(j["jugada"], j["combate"]["gano"] if j["combate"] else None)
                                          for j in r["jugadas"]]


def ajustar(X, y, s, w0, lam):
    nf = X.shape[1]

    def fobj(t):
        w, b = t[:nf], t[nf]
        p = 1 / (1 + np.exp(-(X @ w + b * s)))
        ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        g = np.concatenate([X.T @ (p - y) / len(y), [np.mean((p - y) * s)]])
        d = w - w0
        g[:nf] += lam * d
        return ll + 0.5 * lam * np.sum(d * d), g

    r = minimize(fobj, np.concatenate([w0, [0.0]]), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    return r.x[:nf], r.x[nf]


def logloss(X, y, s, w, b):
    p = 1 / (1 + np.exp(-(X @ w + b * s)))
    return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("--equipo", default="equipo_gg")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--escribir", default=None, help="ruta de data/pesos.json a actualizar")
    a = ap.parse_args()
    import strategy as bot
    from motor import Estado, Jugada

    f = np.zeros(bot.NFEAT)
    B = np.zeros(bot.B_TAM, np.int64)
    X, Y, S, G = [], [], [], []
    for gid, (nuestro, ganador, jugadas) in enumerate(partidas(a.archivos, a.equipo)):
        e = Estado.inicial()
        for texto, gano in jugadas:
            tab, pts, lado = bot._desde_estado(e)
            bot.rasgos(tab, pts, lado, f, B)
            X.append(f[: bot.NFEAT_USADOS].copy())
            Y.append(1.0 if e.turno == ganador else 0.0)
            S.append(1.0 if e.turno == nuestro else -1.0)
            G.append(gid)
            j = Jugada.desde_texto(texto)
            e = e.aplicar(j, gana_ataque=gano) if gano is not None else e.aplicar(j)
    X, Y, S, G = map(np.array, (X, Y, S, G))
    w0 = np.array(bot.PESOS_AJUSTADOS, dtype=np.float64)
    ruta = Path(a.escribir) if a.escribir else CARPETA / "data" / "pesos.json"
    previo = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    if previo.get("rasgos") == bot.NOMBRES_RASGOS and "pesos_autojuego" in previo:
        w0 = np.array(previo["pesos_autojuego"], dtype=np.float64)
        print(f"punto de partida: pesos_autojuego de {ruta}")
    juegos = np.unique(G)
    np.random.default_rng(1).shuffle(juegos)
    res = np.zeros(3)
    pliegues = [pl for pl in np.array_split(juegos, min(5, len(juegos))) if len(pl)]
    if len(pliegues) < 2:
        raise SystemExit("Hacen falta al menos 2 partidas.")
    for pl in pliegues:
        te = np.isin(G, pl)
        w, b = ajustar(X[~te], Y[~te], S[~te], w0, a.lam)
        _, b0 = ajustar(X[~te], Y[~te], S[~te], w0, 1e6)
        res += [logloss(X[te], Y[te], S[te], w0, 0.0), logloss(X[te], Y[te], S[te], w0, b0),
                logloss(X[te], Y[te], S[te], w, b)]
    res /= len(pliegues)
    print(f"{len(Y)} posiciones de {len(juegos)} partidas")
    print(f"log-loss (validación cruzada): actuales {res[0]:.4f} | actuales + ventaja {res[1]:.4f} | "
          f"corregidos {res[2]:.4f}")
    w, b = ajustar(X, Y, S, w0, a.lam)
    print(f"ventaja estimada: {b:+.3f} log-momios")
    for i in np.argsort(-np.abs(w - w0))[:12]:
        print(f"  {bot.NOMBRES_RASGOS[i]:26s} {w0[i]:+.4f} -> {w[i]:+.4f}")
    print("PESOS_AJUSTADOS = [" + ", ".join(f"{v:.5f}" for v in w) + "]")
    if a.escribir:
        previo["rasgos"] = bot.NOMBRES_RASGOS
        previo.setdefault("pesos_autojuego", [round(float(v), 6) for v in w0])
        previo["pesos"] = [round(float(v), 6) for v in w]
        previo.setdefault("ventaja", 1.0)
        previo["origen"] = (f"Corrección con {len(juegos)} partidas ({len(Y)} posiciones), lam = {a.lam}. "
                            f"Log-loss en validación cruzada: {res[0]:.4f} sin corregir, {res[1]:.4f} con "
                            f"ventaja, {res[2]:.4f} corregidos. Ventaja estimada: {b:+.2f}.")
        ruta.write_text(json.dumps(previo, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"guardado en {ruta} (pesos)")


if __name__ == "__main__":
    main()
