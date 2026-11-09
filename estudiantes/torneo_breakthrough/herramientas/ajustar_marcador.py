"""Ajusta los pesos que juega el bot (data/pesos.npy): marcador ordenado + tablas de patrones 3×3.

Regresión logística con etiquetas suaves t = (1 + v)/2, donde v es el valor de una búsqueda de 500k
nodos (las etiquetas que guarda etiquetas_busqueda.py con --guardar-en). Modelo (rasgos.py):
  z = Σ_k W[cb]_k·φ_k(Blancas) − Σ_k W[cn]_k·φ_k(Negras) + marcador y turno + tabla(Blancas) − tabla(Negras)
con cb = (puntos de Blancas, de Negras) y cn = (puntos de Negras, de Blancas): cada bando usa los
pesos de SU situación en el marcador (9 juegos de 46). Se ajusta en tres pasos, cada uno desde el
anterior:
  1. los 46 rasgos solos (desde data/pesos_46.npy);
  2. W[a][b] = w + Δ[a][b], con L2 pequeño en Δ (se queda el de menor error en validación);
  3. lo mismo más la tabla: un peso por (fila del ancla, orilla o centro, contenido de las 9 casillas),
     con el espejo izquierda-derecha atado (un patrón y su reflejo comparten peso);
  4. una tabla por grupo de marcador del dueño de la ventana (nadie en 2, él en 2, su rival en 2,
     2-2): T_g = T + Δ_g, con L2 en Δ_g (se queda el de menor error en validación).
Marcador y turno no salen de `_lado`: son los mismos en las 9 clases. Separa el 10 % de las partidas
para medir el error en valor, compara con los 46 rasgos y verifica que `rasgos.evaluar_lineal` con
los pesos cuantizados reproduce el ajuste.

Uso (desde la raíz del repo). Así salió el data/pesos.npy entregado (INFORME §5):
  python estudiantes/torneo_breakthrough/herramientas/ajustar_marcador.py pesos_profundas_etiquetas.npz \\
      pesos_lab_etiquetas.npz --guardar
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit
from scipy import sparse
from scipy.optimize import minimize

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))

import rasgos as rg  # noqa: E402

K = rg.K
FUERA = np.array([rg.I_MARCADOR, rg.I_MARCADOR + 1, rg.I_MARCADOR + 2, rg.I_TURNO])  # no salen de _lado
DENTRO = np.ones(K, bool)
DENTRO[FUERA] = False
VENTANAS = rg.ANCLAS * rg.ANCLAS  # ventanas 3×3 por bando


@njit(cache=False)
def matrices(bbs, ptss, turnos, X, FW, FB):
    """X: los 46 rasgos desde Blancas (marcador y turno incluidos); FW / FB: los de cada bando solo."""
    phi = np.zeros(K, np.float64)
    for i in range(bbs.shape[0]):
        rg.rasgos(bbs[i], ptss[i], turnos[i], phi)
        X[i, :] = phi
        rg.rasgos_lado(bbs[i], 0, ptss[i, 0], phi)
        FW[i, :] = phi
        rg.rasgos_lado(bbs[i], 1, ptss[i, 1], phi)
        FB[i, :] = phi


@njit(cache=False)
def espejo(idx):
    """Índice del mismo contenido de 3×3 reflejado izquierda-derecha."""
    salida = 0
    pot = 1
    for dr in range(3):
        for dc in range(3):
            salida += ((idx // 3 ** (dr * 3 + 2 - dc)) % 3) * pot
            pot *= 3
    return salida


@njit(cache=False)
def entradas_tabla(bbs, filas, cols, vals):
    """Entradas (posición, columna, ±1) de la matriz de la tabla, con el espejo atado."""
    idx = np.zeros((2, VENTANAS), np.int64)
    n = 0
    for i in range(bbs.shape[0]):
        rg.indices_patrones(bbs[i], idx)
        for yo in range(2):
            for v in range(VENTANAS):
                bloque, conf = divmod(idx[yo, v], rg.NCONF)
                filas[n] = i
                cols[n] = bloque * rg.NCONF + min(conf, espejo(conf))
                vals[n] = 1.0 if yo == 0 else -1.0
                n += 1


def matriz_tabla(bbs):
    """La matriz dispersa de la tabla, solo con las columnas que aparecen (y cuáles son)."""
    total = bbs.shape[0] * 2 * VENTANAS
    filas = np.zeros(total, np.int64)
    cols = np.zeros(total, np.int64)
    vals = np.zeros(total, np.float32)
    entradas_tabla(bbs, filas, cols, vals)
    m = sparse.coo_matrix((vals, (filas, cols)), shape=(bbs.shape[0], rg.TAM_TABLA)).tocsr()
    usadas = np.unique(m.indices)
    return m[:, usadas], usadas


@njit(cache=False)
def entradas_grupos(bbs, ptss, filas, cols, vals):
    """Como entradas_tabla, pero cada ventana va al bloque Δ del grupo de marcador de su dueño."""
    idx = np.zeros((2, VENTANAS), np.int64)
    n = 0
    for i in range(bbs.shape[0]):
        rg.indices_patrones(bbs[i], idx)
        for yo in range(2):
            g = rg.grupo_tabla(ptss[i, yo], ptss[i, 1 - yo])
            for v in range(VENTANAS):
                bloque, conf = divmod(idx[yo, v], rg.NCONF)
                filas[n] = i
                cols[n] = g * rg.TAM_TABLA + bloque * rg.NCONF + min(conf, espejo(conf))
                vals[n] = 1.0 if yo == 0 else -1.0
                n += 1


def matriz_grupos(bbs, ptss):
    total = bbs.shape[0] * 2 * VENTANAS
    filas = np.zeros(total, np.int64)
    cols = np.zeros(total, np.int64)
    vals = np.zeros(total, np.float32)
    entradas_grupos(bbs, ptss, filas, cols, vals)
    m = sparse.coo_matrix((vals, (filas, cols)),
                          shape=(bbs.shape[0], rg.GRUPOS_TABLA * rg.TAM_TABLA)).tocsr()
    usadas = np.unique(m.indices)
    return m[:, usadas], usadas


def tabla_completa(w_tabla, usadas):
    """La tabla que lee el bot: un peso por (bloque, contenido), con el espejo ya expandido."""
    por_columna = np.zeros(rg.TAM_TABLA)
    por_columna[usadas] = w_tabla
    canon = np.array([min(i, espejo(i)) for i in range(rg.NCONF)])
    return por_columna.reshape(-1, rg.NCONF)[:, canon].ravel()


def ajustar(X, t, w0, l2):
    """Regresión logística con etiquetas suaves; l2 es un vector (una fuerza por columna)."""

    def f(w):
        z = X @ w
        p = 0.5 * (1.0 + np.tanh(0.5 * z))
        perdida = np.mean(np.logaddexp(0.0, z) - t * z) + 0.5 * np.sum(l2 * w * w)
        return perdida, X.T @ (p - t) / len(t) + l2 * w

    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000}).x


def error_valor(z, v):
    return float(np.mean((np.tanh(0.5 * z) - v) ** 2))


@njit(cache=False)
def evaluar_todas(bbs, ptss, turnos, pesos, salida):
    """Valor de la evaluación del bot desde Blancas, para verificar que reproduce el ajuste."""
    for i in range(bbs.shape[0]):
        v = rg.evaluar_lineal(bbs[i], ptss[i], turnos[i], pesos)
        salida[i] = v if turnos[i] == 0 else -v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etiquetas", nargs="+", help=".npz de etiquetas_busqueda.py (bb, pts, turno, partida, valor)")
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 de los 46 pesos base")
    ap.add_argument("--l2-delta", type=float, nargs="+", default=[1e-5, 1e-4], help="L2 de los Δ por marcador")
    ap.add_argument("--l2-tabla", type=float, default=1e-4)
    ap.add_argument("--l2-grupo", type=float, nargs="+", default=[1e-4, 3e-5], help="L2 de los Δ por grupo")
    ap.add_argument("--inicial", default=str(BOT / "data" / "pesos_46.npy"), help="46 pesos para arrancar")
    ap.add_argument("--guardar", action="store_true", help="escribe data/pesos.npy y data/pesos.json")
    ap.add_argument("--guardar-en", default="", help="otra ruta .npy (para probar antes de adoptar)")
    a = ap.parse_args()
    partes = [np.load(f) for f in a.etiquetas]
    bbs = np.concatenate([p["bb"] for p in partes]).astype(np.int64)
    ptss = np.concatenate([p["pts"] for p in partes]).astype(np.int64)
    turnos = np.concatenate([p["turno"] for p in partes]).astype(np.int64)
    partida = np.concatenate([p["partida"] + 10_000_000 * i for i, p in enumerate(partes)])
    v = np.concatenate([p["valor"] for p in partes])
    t = 0.5 * (1.0 + np.clip(v, -0.999, 0.999))
    rng = np.random.default_rng(0)
    juegos = np.unique(partida)
    val = np.isin(partida, rng.choice(juegos, size=max(1, len(juegos) // 10), replace=False))
    ent = ~val
    n = len(t)
    print(f"{n} posiciones ({ent.sum()} entrenamiento, {val.sum()} validación)")

    def error(z):
        return error_valor(z[val], v[val])

    X = np.zeros((n, K))
    FW = np.zeros((n, K))
    FB = np.zeros((n, K))
    matrices(bbs, ptss, turnos, X, FW, FB)
    # 1. Los 46 rasgos solos.
    w46 = ajustar(X[ent], t[ent], np.load(a.inicial), np.full(K, a.l2))
    base = error(X @ w46)
    print(f"1. 46 rasgos:                     error en valor {base:.5f}")
    # 2. Marcador ordenado: columnas Δ[s] = φ(Blancas)·[cb = s] − φ(Negras)·[cn = s].
    cb = ptss[:, 0] * 3 + ptss[:, 1]
    cn = ptss[:, 1] * 3 + ptss[:, 0]
    bloques = []
    for s in range(9):
        B = FW * (cb == s)[:, None] - FB * (cn == s)[:, None]
        B[:, ~DENTRO] = 0.0
        bloques.append(sparse.csr_matrix(B))
    XO = sparse.hstack([sparse.csr_matrix(X)] + bloques).tocsr()
    mejor = None
    for l2d in a.l2_delta:
        l2 = np.concatenate([np.full(K, a.l2), np.full(9 * K, l2d)])
        w = ajustar(XO[ent], t[ent], np.concatenate([w46, np.zeros(9 * K)]), l2)
        e = error(XO @ w)
        print(f"2. + marcador ordenado (L2 Δ {l2d:g}): error en valor {e:.5f} ({100 * (e / base - 1):+.1f} %)")
        if mejor is None or e < mejor[0]:
            mejor = (e, w)
    # 3. Más la tabla 3×3.
    T, usadas = matriz_tabla(bbs)
    M = sparse.hstack([XO, T]).tocsr()
    l2 = np.concatenate([np.full(K, a.l2), np.full(9 * K, a.l2_delta[0]), np.full(T.shape[1], a.l2_tabla)])
    w = ajustar(M[ent], t[ent], np.concatenate([mejor[1], np.zeros(T.shape[1])]), l2)
    e = error(M @ w)
    print(f"3. + tabla 3×3 ({T.shape[1]} patrones vistos): error en valor {e:.5f} ({100 * (e / base - 1):+.1f} %)")
    # 4. Una tabla por grupo de marcador: T_g = T + Δ_g.
    G, usadas_g = matriz_grupos(bbs, ptss)
    MG = sparse.hstack([M, G]).tocsr()
    elegido = None
    for l2g in a.l2_grupo:
        l2 = np.concatenate([np.full(K, a.l2), np.full(9 * K, a.l2_delta[0]), np.full(T.shape[1], a.l2_tabla),
                             np.full(G.shape[1], l2g)])
        wg = ajustar(MG[ent], t[ent], np.concatenate([w, np.zeros(G.shape[1])]), l2)
        eg = error(MG @ wg)
        print(f"4. + tabla por grupo de marcador (L2 {l2g:g}): error en valor {eg:.5f} ({100 * (eg / base - 1):+.1f} %)")
        if elegido is None or eg < elegido[0]:
            elegido = (eg, wg)
    e, w = elegido
    z = MG @ w
    print(f"   elegido: error en valor {e:.5f} | entrenamiento {error_valor(z[ent], v[ent]):.5f}")
    W = w[:K][None, :] + w[K:10 * K].reshape(9, K)
    W[:, ~DENTRO] = w[:K][~DENTRO]  # marcador y turno: los mismos en las 9 clases
    nt = T.shape[1]
    tabla = tabla_completa(w[10 * K:10 * K + nt], usadas)
    delta = np.zeros(rg.GRUPOS_TABLA * rg.TAM_TABLA)
    delta[usadas_g] = w[10 * K + nt:]
    canon = np.array([min(i, espejo(i)) for i in range(rg.NCONF)])
    tablas = [tabla + delta[g * rg.TAM_TABLA:(g + 1) * rg.TAM_TABLA].reshape(-1, rg.NCONF)[:, canon].ravel()
              for g in range(rg.GRUPOS_TABLA)]
    pesos = np.concatenate([W.ravel()] + tablas)
    # Verificación: la evaluación del bot (int16) reproduce el ajuste.
    muestra = np.where(val)[0][:20000]
    vb = np.zeros(len(muestra))
    evaluar_todas(bbs[muestra], ptss[muestra], turnos[muestra], rg.cuantizar(pesos), vb)
    dif = np.abs(vb - np.tanh(0.5 * z[muestra])).max()
    print(f"verificación: |evaluación del bot − ajuste| máxima {dif:.4f} en {len(muestra)} posiciones")
    if dif > 0.01:
        raise SystemExit("la evaluación del bot no reproduce el ajuste: no se guarda nada")
    if a.guardar or a.guardar_en:
        ruta = Path(a.guardar_en) if a.guardar_en else BOT / "data" / "pesos.npy"
        np.save(ruta, pesos)
        clases = [f"{s // 3}-{s % 3}" for s in range(9)]
        ruta.with_suffix(".json").write_text(json.dumps(
            {"modelo": "marcador ordenado (9 × 46) + 4 tablas de patrones 3×3 según el marcador; ver rasgos.py",
             "nombres": rg.NOMBRES, "clases (mis puntos-los del rival)": clases,
             "pesos por clase": {c: [round(float(x), 6) for x in W[s]] for s, c in enumerate(clases)},
             "tablas": {"una por grupo de marcador del dueño de la ventana":
                            ["nadie en 2", "él en 2", "su rival en 2", "2-2"],
                        "pesos por tabla": int(rg.TAM_TABLA), "patrones vistos (con espejo)": int(T.shape[1]),
                        "dónde": "en pesos.npy, después de los 9 × 46, en el orden de los grupos"},
             "l2": a.l2, "l2_delta": a.l2_delta, "l2_tabla": a.l2_tabla, "l2_grupo": a.l2_grupo,
             "fuentes": [Path(f).name for f in a.etiquetas], "posiciones": int(ent.sum()),
             "error en valor (validación)": {"46 rasgos": round(base, 6), "final": round(e, 6)}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"guardado en {ruta} ({len(pesos)} pesos) y {ruta.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
