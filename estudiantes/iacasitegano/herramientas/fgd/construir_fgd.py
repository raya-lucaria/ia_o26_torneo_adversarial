"""Reconstruye la evaluacion del bot (data/pesos_eval3_ligero.npz y las constantes CAL_B0 / CAL_STM
de busqueda_eval3.py) a partir de las posiciones etiquetadas con una busqueda profunda.

Que hace: ajusta por regresion ridge (lambda=100, en validacion cruzada por partida) el logit de
`eval3` a la etiqueta profunda de cada posicion (data/etiquetas_prof7.jsonl: tablero, puntos, turno y el
valor de una busqueda de profundidad 7 sin tope de nodos). Cada posicion aporta dos filas: vista desde
el que mueve (etiqueta y, indicador de turno +1) y vista desde el rival (etiqueta -y, indicador -1).
Columnas libres: las familias activas (puntos, patrones, filas, marcador) y el indicador de turno.

Uso (desde la carpeta del bot, con el motor del torneo en PYTHONPATH):
    PYTHONPATH=<repo>/sistema python herramientas/fgd/construir_fgd.py --salida /tmp/pesos_nuevos.npz
Imprime CAL_B0 y CAL_STM (los dos numeros que van en busqueda_eval3.py) y compara contra los pesos que
se entregan: el error maximo debe ser ~1e-9 o menos. No sobreescribe nada salvo que se lo pidas con --salida."""
import argparse, json, sys
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent.parent.parent   # carpeta del bot
sys.path.insert(0, str(AQUI))
import eval3 as e3

ap = argparse.ArgumentParser()
ap.add_argument('--etiquetas', default=str(AQUI / 'data/etiquetas_prof7.jsonl'))
ap.add_argument('--base', default=str(AQUI / 'data/pesos_eval3_ligero.npz'), help='aporta las entradas no activas y la clave "modelo"')
ap.add_argument('--lam', type=float, default=100.0)
ap.add_argument('--salida', default=None)
a = ap.parse_args()

R = [json.loads(l) for l in open(a.etiquetas) if l.strip()]
base = np.load(a.base); pes = base['pesos'].astype(np.float64)
F = []; Y = []; S = []
for r in R:
    tab = np.array([int(c) for c in r['tab']], np.int8); pts = np.array(r['pts'], np.int64); col = r['col']
    y = 2 * np.arctanh(float(np.clip(r['v'], -0.97, 0.97)))     # logit: v = tanh(logit/2)
    for persp, sg in ((col, 1), (1 - col, -1)):
        F.append(e3.features_anchas_nb(tab, pts, persp)); Y.append(sg * y); S.append(sg)
F = np.array(F, np.float64); Y = np.array(Y); S = np.array(S, float)
fam = e3.FAMILIAS
act = np.concatenate([np.arange(*fam[n]) for n in ['puntos', 'patrones', 'filas', 'marcador']])
X = np.c_[F[:, act], S]; mu = X.mean(0); sd = X.std(0) + 1e-9; Xs = (X - mu) / sd; ym = Y.mean()
w = np.linalg.solve(Xs.T @ Xs + a.lam * np.eye(Xs.shape[1]), Xs.T @ (Y - ym))
coef = w / sd; b0 = ym - coef @ mu; beta = coef[:-1]; stm = coef[-1]
nuevo = pes.copy(); nuevo[act] = beta
print('posiciones: %d   CAL_B0 = %r   CAL_STM = %r' % (len(R), float(b0), float(stm)))
print('error maximo contra los pesos entregados: %.3g' % np.abs(nuevo - pes).max())
if a.salida:
    np.savez(a.salida, pesos=nuevo, crudo=nuevo[:267], modelo=base['modelo']); print('escrito', a.salida)
