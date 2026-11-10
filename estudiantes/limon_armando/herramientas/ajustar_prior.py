"""Ajusta el prior del modelo del rival y lo guarda en data/prior_oponente.npz (offline).

Uso (desde la raíz del repo, con .venv activo):

    python estudiantes/limon_armando/herramientas/ajustar_prior.py resultados/<corrida> [...]

Lee los registros de partidas (un JSON por partida de `./torneo jugar`, o `parejas/N.json` de
`./torneo liga`), las reproduce con el motor y obtiene, para cada jugada, las características
de todas las jugadas legales y cuál se eligió. Solo usa partidas de nuestros bots y de los bots
del curso jugadas en nuestra máquina; ignora `random` (su modelo es theta = 0).

Para cada bot ajusta theta por máxima verosimilitud (con un poco de L2) en características
estandarizadas. El prior es:
    mu0    = promedio de los theta de los bots,
    Sigma0 = diag(TAU^2) en escala estandarizada (TAU fijado a mano).
También imprime una validación dejando fuera un bot: log-verosimilitud media por jugada contra
la del azar (-log|A|), acierto de la jugada más probable y el efecto de actualizar con Bayes.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from motor import Estado, Jugada  # noqa: E402

import modelo_oponente as mo  # noqa: E402

TAU = 0.1          # desviación del prior por característica (escala estandarizada);
                   # elegida comparando 0.5, 0.2, 0.1 y 0.05 en la validación de abajo
L2_AJUSTE = 1.0    # regularización al ajustar cada bot (evita pesos infinitos si una
                   # característica rara separa perfectamente, como «defiende»)
ESCALA_MINIMA = 0.1  # piso de la desviación usada para estandarizar
IGNORAR = {"random", "humano"}
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "prior_oponente.npz")


def registros(rutas):
    for ruta in rutas:
        for f in sorted(glob.glob(os.path.join(ruta, "**", "*.json"), recursive=True)):
            with open(f) as fh:
                d = json.load(fh)
            for r in d.get("registros", [d]):
                if "meta" in r and "jugadas" in r:
                    yield r


def observaciones(rutas):
    """{bot: [(F, indice_elegida, id_partida), ...]} reproduciendo cada partida."""
    datos = {}
    for n, r in enumerate(registros(rutas)):
        estado = Estado.inicial()
        for jug in r["jugadas"]:
            bot = r["meta"][jug["color"]]
            legales = estado.jugadas()
            j = Jugada.desde_texto(jug["jugada"])
            if bot not in IGNORAR:
                F = mo.matriz_caracteristicas(estado, legales)
                datos.setdefault(bot, []).append((F, legales.index(j), n))
            combate = jug.get("combate")
            estado = estado.aplicar(j, gana_ataque=combate["gano"] if combate else None)
    return datos


def ajustar(obs, escala, mu0, prec0, pasos=60):
    """MAP de theta en escala estandarizada (F / escala), con el mismo Newton del bot."""
    theta, _ = mo.map_laplace(mu0, prec0, [(F / escala, b) for F, b, _ in obs], mu0.copy(),
                              pasos=pasos, tol=1e-7)
    return theta


def evaluar(obs, theta_raw):
    """(log-verosimilitud media, log-verosimilitud del azar, acierto top-1, acierto del azar)."""
    ll = ll0 = top = top0 = 0.0
    for F, b, _ in obs:
        pi = mo.probabilidades(theta_raw, F)
        ll += np.log(pi[b])
        ll0 -= np.log(len(pi))
        top += float(np.argmax(pi) == b)
        top0 += 1.0 / len(pi)
    n = len(obs)
    return ll / n, ll0 / n, top / n, top0 / n


def evaluar_bayes(obs, mu0_raw, prec0_raw):
    """Log-verosimilitud media prediciendo cada jugada solo con las anteriores de su partida."""
    ll = 0.0
    partida, vistas, theta = None, [], mu0_raw
    for F, b, pid in obs:
        if pid != partida:
            partida, vistas, theta = pid, [], mu0_raw
        ll += np.log(mo.probabilidades(theta, F)[b])
        vistas.append((F, b))
        theta, _ = mo.map_laplace(mu0_raw, prec0_raw, vistas, theta)
    return ll / len(obs)


def main(rutas):
    datos = observaciones(rutas)
    todas = [F for obs in datos.values() for F, _, _ in obs]
    # Desviación de cada característica, con piso: una característica rara (std ~ 0) no
    # debe inflarse al estandarizar.
    escala = np.maximum(np.vstack(todas).std(axis=0), ESCALA_MINIMA)
    cero = np.zeros(mo.D)
    reg = L2_AJUSTE * np.eye(mo.D)
    print(f"Bots: { {b: len(o) for b, o in datos.items()} }  (jugadas observadas)")

    thetas = {b: ajustar(o, escala, cero, reg) for b, o in datos.items()}
    for b, th in thetas.items():
        print(f"theta {b:>14}: " + "  ".join(f"{n}={v:+.2f}" for n, v in zip(mo.NOMBRES, th)))

    prec0_raw = np.diag(escala**2 / TAU**2)  # Sigma0 = diag(TAU^2) en escala estandarizada
    print("\nValidación dejando fuera un bot (el prior se ajusta con los demás):")
    for fuera, obs in datos.items():
        otros = [x for b, o in datos.items() if b != fuera for x in o]
        if not otros:
            continue
        mu_raw = ajustar(otros, escala, cero, reg) / escala
        ll, ll0, top, top0 = evaluar(obs, mu_raw)
        llb = evaluar_bayes(obs[:4000], mu_raw, prec0_raw)
        llp = evaluar(obs[:4000], mu_raw)[0]
        print(f"  {fuera:>14}: logver/jugada {ll:+.3f} (azar {ll0:+.3f})  "
              f"top-1 {top:.1%} (azar {top0:.1%})  |  con Bayes {llb:+.3f} vs solo prior {llp:+.3f}")

    mu0_std = np.mean(list(thetas.values()), axis=0)
    np.savez(
        SALIDA,
        mu0=mu0_std / escala,      # en la escala original de las características
        prec0=prec0_raw,
        escala=escala,
        nombres=np.array(mo.NOMBRES),
        tau=TAU,
        bots=np.array(sorted(thetas)),
    )
    print(f"\nGuardado {os.path.normpath(SALIDA)}")
    print("mu0 (estandarizado): " + "  ".join(f"{n}={v:+.2f}" for n, v in zip(mo.NOMBRES, mu0_std)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
