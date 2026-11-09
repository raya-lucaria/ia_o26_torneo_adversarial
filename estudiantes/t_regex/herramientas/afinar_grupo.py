"""Afinado de pesos contra un GRUPO de rivales fijos (no corre en la liga).

`afinar.py` compara cada candidato contra la versión vigente, y eso premia cambios que solo
explotan a la vigente (nos pasó: una cadena de adopciones terminó empatando con la versión 1).
Aquí el candidato y la vigente juegan contra el mismo grupo de rivales distintos (versiones
viejas con otras evaluaciones y montecarlo), con los mismos dados, y se comparan sus puntajes:
- filtro: 50 juegos de dados por rival (300 partidas por bando); pasa si el candidato saca
  ≥ 3 puntos porcentuales más que la vigente;
- confirmación: otras 300 por bando con dados nuevos; se adopta si en las 600 mantiene
  ≥ +2.5 puntos.

Uso (desde la raíz del repo; corre durante horas, retoma desde estado.json):
    docker run --rm --cpus=8 -v "$PWD":/torneo -w /torneo/estudiantes/t_regex \
        torneo-bot-base:local python herramientas/afinar_grupo.py \
        --carpeta /torneo/resultados/afinado_grupo
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arena  # noqa: E402
import busqueda as bq  # noqa: E402
from afinar import ORDEN, PASO_MINIMO, PASO_RELATIVO, cambios  # noqa: E402

GRUPO = ["../t_regex_v1", "../t_regex_v3", "montecarlo:1000"]
PARES_POR_RIVAL = 50
FILTRO_VENTAJA = 0.03
CONFIRMA_VENTAJA = 0.025


def puntaje(procesos, pesos, semilla, nodos):
    """Fracción de partidas que ganan estos pesos contra todo el grupo (ambos colores)."""
    yo = (".", cambios(pesos))
    trabajos = []
    for r, rival in enumerate(GRUPO):
        for k in range(PARES_POR_RIVAL):
            s = semilla + 1000 * r + k
            trabajos.append((yo, (rival, ""), s, nodos))
            trabajos.append(((rival, ""), yo, s, nodos))
    victorias = 0
    with Pool(procesos) as pool:
        for i, (ganador, *_) in enumerate(pool.imap(arena.partida, trabajos)):
            victorias += ganador == i % 2  # índice par: yo con blancas (gano si ganador = 0)
    return victorias / len(trabajos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default="/torneo/resultados/afinado_grupo")
    ap.add_argument("--nodos", type=int, default=100_000)
    ap.add_argument("--procesos", type=int, default=8)
    args = ap.parse_args()
    carpeta = Path(args.carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / "estado.json"
    if archivo.exists():
        estado = json.loads(archivo.read_text())
    else:
        pesos = [float(w) for w in bq.PESOS]
        estado = {"pesos": pesos, "pasos": [max(abs(w) * PASO_RELATIVO, PASO_MINIMO) for w in pesos],
                  "posicion": 0, "version": 0, "adoptados_en_vuelta": 0, "semilla": 200_000,
                  "cache": {}}
        np.save(carpeta / "pesos_v0.npy", np.array(pesos))

    def anotar(texto):
        linea = f"{time.strftime('%Y-%m-%d %H:%M')}  {texto}"
        print(linea, flush=True)
        with open(carpeta / "historial.txt", "a") as f:
            f.write(linea + "\n")

    def puntaje_vigente(semilla):
        """La vigente juega cada juego de dados una sola vez: se guarda en el estado."""
        llave = f"{estado['version']}:{semilla}"
        if llave not in estado["cache"]:
            estado["cache"][llave] = puntaje(args.procesos, estado["pesos"], semilla, args.nodos)
        return estado["cache"][llave]

    anotar(f"inicio: versión {estado['version']}, posición {estado['posicion']}, grupo {GRUPO}")
    while True:
        i = ORDEN[estado["posicion"] // 2]
        signo = 1 if estado["posicion"] % 2 == 0 else -1
        candidato = list(estado["pesos"])
        candidato[i] += signo * estado["pasos"][i]
        nombre = f"{bq.NOMBRES_RASGOS[i]} {estado['pesos'][i]:+.3f} → {candidato[i]:+.3f}"

        # Los dados del filtro cambian con cada vuelta, no con cada candidato: así la vigente
        # juega una vez por vuelta y todos los candidatos se comparan con los mismos dados.
        s1 = estado["semilla"]
        v, c = puntaje_vigente(s1), puntaje(args.procesos, candidato, s1, args.nodos)
        adoptado = False
        texto = f"{nombre}: grupo {c:.1%} contra {v:.1%} de la vigente"
        if c - v >= FILTRO_VENTAJA:
            s2 = s1 + 500_000 + 10_000 * estado["posicion"]
            v2 = puntaje(args.procesos, estado["pesos"], s2, args.nodos)
            c2 = puntaje(args.procesos, candidato, s2, args.nodos)
            adoptado = (c + c2) / 2 - (v + v2) / 2 >= CONFIRMA_VENTAJA
            texto += f"; confirmación {c2:.1%} contra {v2:.1%}"
        anotar(texto + ("  ← ADOPTADO" if adoptado else ""))

        if adoptado:
            estado["pesos"] = candidato
            estado["version"] += 1
            estado["adoptados_en_vuelta"] += 1
            np.save(carpeta / f"pesos_v{estado['version']}.npy", np.array(candidato))
            np.save(carpeta / "pesos_mejores.npy", np.array(candidato))
            if signo == 1:
                estado["posicion"] += 1
        estado["posicion"] += 1
        if estado["posicion"] >= 2 * len(ORDEN):
            if estado["adoptados_en_vuelta"] == 0:
                estado["pasos"] = [max(p / 2, PASO_MINIMO / 2) for p in estado["pasos"]]
                anotar("vuelta sin mejoras: paso a la mitad")
            estado["posicion"] = 0
            estado["adoptados_en_vuelta"] = 0
            estado["semilla"] += 7_000
        archivo.write_text(json.dumps(estado, indent=1))


if __name__ == "__main__":
    main()
