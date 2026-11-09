"""Contrafactuales offline por acción, con motor oficial y oráculo sin poda.

No usa dados futuros ni resultados finales. No mide fuerza ni rendimiento.
Escribe un checkpoint por caso/profundidad; una búsqueda truncada no es exacta.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ["NUMBA_DISABLE_JIT"] = "1"
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"
sys.dont_write_bytecode = True
EQUIPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EQUIPO), str(EQUIPO.parents[1] / "sistema")]

import numpy as np
import eval3
from motor import Estado
from motor.jugada import casilla_de_texto
from motor.nucleo import MAX_JUGADAS


class Limite(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cargar_busqueda(path):
    spec = importlib.util.spec_from_file_location("fase5_busqueda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.QUIESCENCIA_MAX != 0:
        raise ValueError("El oráculo truncado requiere quiescencia cero")
    return mod


def arreglos(estado):
    t = np.zeros(81, np.int8)
    for c, nombre in enumerate(("blancas", "negras")):
        for casilla in estado.piezas(nombre):
            t[casilla_de_texto(casilla)] = c + 1
    p = np.array([estado.puntos[c] for c in ("blancas", "negras")], np.int64)
    return t, p, int(estado.turno == "negras")


def clave_completa(estado):
    return (tuple(estado.piezas("blancas")), tuple(estado.piezas("negras")),
            estado.turno, estado.puntos["blancas"], estado.puntos["negras"])


def analizar(caso, profundidad, busqueda, pesos, max_nodos, segundos):
    estado = Estado.desde_dict(caso["estado"])
    objetivo = int(estado.turno == "negras")
    nombre_objetivo = estado.turno
    ganancia = float(busqueda.GANANCIA)
    inicio = time.monotonic()
    cache = {}
    nodos = 0

    def valor(e, d):
        nonlocal nodos
        nodos += 1
        if nodos > max_nodos or time.monotonic() - inicio > segundos:
            raise Limite()
        k = (clave_completa(e), d)
        if k in cache:
            return cache[k]
        ganador = e.ganador()
        if ganador is not None:
            exacto = ganancia if ganador == nombre_objetivo else -ganancia
            v = (exacto,exacto,exacto)
        elif d == 0:
            t, p, _ = arreglos(e)
            # Ruta Python independiente de evaluar_nb del buscador.
            z = float(eval3.features_anchas_py(t, p, objetivo) @ pesos[:eval3.N_FEATURES])
            # Las cotas describen el valor REAL del juego, independientemente del modelo.
            v = (z,-ganancia,ganancia)
        else:
            vs = []
            for j in e.jugadas():
                ramas = [(prob,valor(hijo,d-1)) for prob,hijo in e.siguiente(j)]
                vs.append(tuple(sum(prob*v[k] for prob,v in ramas) for k in range(3)))
            elegir = max if e.turno == nombre_objetivo else min
            v = tuple(elegir(x[k] for x in vs) for k in range(3))
        cache[k] = v
        return v

    salida = {"id": caso["id"], "profundidad": profundidad,
              "categoria": caso["categoria_panel"], "color": nombre_objetivo,
              "jugada_observada": caso["jugada_observada"], "alternativas": [],
              "completo": False, "clasificacion": "evidencia_insuficiente"}
    try:
        for j in estado.jugadas():
            ramas = []
            for prob,hijo in estado.siguiente(j):
                estimado,inferior,superior = valor(hijo,profundidad-1)
                ramas.append({'probabilidad':float(prob),'valor':float(estimado),
                              'inferior':float(inferior),'superior':float(superior)})
            salida["alternativas"].append({"jugada": str(j), "ramas": ramas,
                                            **{k:sum(x['probabilidad']*x[k] for x in ramas)
                                               for k in ('valor','inferior','superior')}})
        mejor = max(x["valor"] for x in salida["alternativas"])
        optimas = [x["jugada"] for x in salida["alternativas"] if abs(x["valor"] - mejor) < 1e-9]
        observado = next(x["valor"] for x in salida["alternativas"] if x["jugada"] == caso["jugada_observada"])
        # Contraste del árbol completo truncado con el buscador de producción.
        t, p, c = arreglos(estado)
        contador = np.zeros(2, np.int64)
        origen, destino, v, aborto, _ = busqueda._mejor_jugada_numba(
            t, p, c, profundidad, pesos, ganancia, -1, -1, contador, max_nodos)
        salida.update(completo=True, valor_oraculo=mejor, jugadas_optimas=optimas,
                      brecha_observada=mejor - observado, valor_buscador=float(v),
                      aborto_buscador=bool(aborto), nodos_buscador=int(contador[0]))
        if not aborto:
            from motor import Jugada
            elegida = str(Jugada(int(origen), int(destino)))
            salida["jugada_buscador"] = elegida
            coincide = abs(v - mejor) < 1e-8 and elegida in optimas
            salida["coincide_buscador"] = bool(coincide)
            salida["clasificacion"] = "sin_discrepancia_mismo_horizonte" if coincide else "discrepancia_busqueda_requiere_revision"
        salida["limite_certeza"] = "Optimo solo del arbol truncado con este evaluador; no valor optimo del juego"
        inferior = max(x['inferior'] for x in salida['alternativas'])
        superior = max(x['superior'] for x in salida['alternativas'])
        salida['cotas_probabilidad_victoria_optima'] = [.5*(1+inferior/ganancia),.5*(1+superior/ganancia)]
    except Limite:
        salida["limite_certeza"] = "Presupuesto agotado; alternativas parciales no identifican la mejor jugada"
    observado = next((x for x in salida['alternativas'] if x['jugada']==caso['jugada_observada']),None)
    if observado:
        testigo = next((x for x in salida['alternativas'] if x['inferior']>observado['superior']+1e-9),None)
        if testigo:
            salida['certificado_decision'] = {
                'jugada_observada':caso['jugada_observada'],'alternativa':testigo['jugada'],
                'probabilidad_observada_maxima':.5*(1+observado['superior']/ganancia),
                'probabilidad_alternativa_minima':.5*(1+testigo['inferior']/ganancia),
                'referencia':'Cotas minimax/esperanza con hojas desconocidas [-1,1]; independiente de eval3'}
    salida.update(nodos_oraculo=nodos, segundos_diagnostico=time.monotonic() - inicio)
    return salida


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--buscador", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--profundidades", default="1,2")
    ap.add_argument("--max-nodos", type=int, default=30000)
    ap.add_argument("--segundos-caso", type=float, default=5.0)
    ap.add_argument("--limite-casos", type=int, default=0)
    args = ap.parse_args()
    if args.max_nodos < 1 or args.segundos_caso <= 0:
        ap.error("Los presupuestos deben ser positivos")
    profundidades = [int(x) for x in args.profundidades.split(",")]
    if not profundidades or min(profundidades) < 1:
        ap.error("Profundidades deben ser >=1")
    buscador = args.buscador.resolve()
    if (buscador.parent / "eval3.py").read_bytes() != (EQUIPO / "eval3.py").read_bytes():
        ap.error("eval3 del buscador no coincide con la referencia local")
    with np.load(EQUIPO / "data/pesos_eval3_ligero.npz", allow_pickle=False) as z:
        pesos = z["pesos"].copy()
    # Evita computar familias desactivadas en la ruta de features anchas.
    for nombre, activo in zip(eval3.BANDERAS, pesos[eval3.N_FEATURES:]):
        if not activo:
            a, b = eval3.FAMILIAS[nombre]
            pesos[a:b] = 0
    mod = cargar_busqueda(buscador)
    mod.cargar_pesos(pesos)
    if abs(mod.GANANCIA - (mod.cota_eval3(pesos) + 100)) > 1e-10:
        ap.error('Terminales incompatibles con el oraculo de logits')
    casos = [json.loads(x) for x in args.panel.read_text().splitlines()]
    # Evita usar por accidente una referencia de logits contra una escala transformada.
    for caso in casos[:3]:
        t,p,c = arreglos(Estado.desde_dict(caso['estado']))
        esperado = float(eval3.features_anchas_py(t,p,c) @ pesos[:eval3.N_FEATURES])
        if abs(mod._evaluar_numba(t,p,c,pesos)-esperado)>1e-10:
            ap.error('El buscador no usa logits equivalentes; requiere otro oraculo de escala')
    manifiesto = {"panel_sha256": digest(args.panel), "buscador": str(buscador),
                  "buscador_sha256": digest(buscador), "eval3_sha256": digest(EQUIPO / "eval3.py"),
                  "pesos_sha256": digest(EQUIPO / "data/pesos_eval3_ligero.npz"),
                  "herramienta_sha256": digest(__file__), "profundidades": profundidades,
                  "max_nodos": args.max_nodos, "segundos_caso": args.segundos_caso,
                  "JIT": False, "terminal": mod.GANANCIA,
                  "limite_casos": args.limite_casos, "fuerza_medida": False}
    args.salida.mkdir(parents=True, exist_ok=True)
    archivo_m = args.salida / "manifest.json"
    texto = json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n"
    if archivo_m.exists() and archivo_m.read_text() != texto:
        raise ValueError("Manifiesto diferente: usar otra salida")
    archivo_m.write_text(texto)
    if args.limite_casos:
        casos = casos[:args.limite_casos]
    for caso in casos:
        ident = hashlib.sha256(caso["id"].encode()).hexdigest()[:16]
        for profundidad in profundidades:
            archivo = args.salida / f"{ident}_d{profundidad}.json"
            if archivo.exists():
                continue
            res = analizar(caso, profundidad, mod, pesos, args.max_nodos, args.segundos_caso)
            if digest(buscador) != manifiesto["buscador_sha256"]:
                raise RuntimeError("Buscador cambio durante el diagnostico")
            tmp = archivo.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
            tmp.replace(archivo)
            print(json.dumps({k: res[k] for k in ("id", "profundidad", "completo", "clasificacion")}), flush=True)


if __name__ == "__main__":
    main()
