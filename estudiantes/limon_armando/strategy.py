"""Bot limon_armando. Con ALGORITMO = "plano" juega el Monte Carlo plano mejorado de plano.py (el
bot entregado); con "uct", el MCTS/UCT de este archivo (nuestro bot anterior, ver INFORME.md)."""

import itertools
import json
import os
import random
import time
import math

import numpy as np

from motor import Estado, Jugada, reportar, simular_al_azar

import modelo_oponente as mo
import plano
import simulacion as sim

# "uct" = MCTS/UCT de este archivo; "plano" = Monte Carlo plano mejorado (plano.py)
ALGORITMO = "plano"

SIMULACIONES_POR_VUELTA = 8  # partidas al azar por cada muestra de una jugada
RESERVA = 0.3  # fracción del reloj que nunca gastamos pensando (la comunicación también cuesta)
EXPLORACION = 0.5 #Peso del bono de exploracion de UCT

# Modelo del rival (logit condicional con prior en data/prior_oponente.npz; ver modelo_oponente.py)
USAR_A = False  # A: en turnos del rival, revisar primero sus jugadas probables (sesgo en UCT)
USAR_B = False  # B: simulaciones donde cada bando juega según el modelo en vez de al azar
PASOS_GUIADOS = 10**6  # B: cuántas jugadas al inicio de cada simulación usan el modelo
PESO_SESGO = 1.0  # peso w del sesgo progresivo w·|A|·pi(a|s)/(1+N(s,a)) de la opción A
EPS_MODELO = 0.1  # mezcla con uniforme: (1 - eps)·pi + eps/|A|, por si el modelo se equivoca

# Reloj: "fijo" = 1/25 de lo que queda sobre la reserva del 30 %;
# "stop" = STOP + UNST (Baier y Winands, 2016); "critico" = más tiempo si la posición no es quieta.
RELOJ = "fijo"
RESERVA_S = 5.0     # stop/critico: segundos de reserva fijos
F_STOP = 1.5        # stop: el plan es 1.5 veces el turno normal (gasta lo que STOP ahorra)
P_STOP = 0.5        # stop: parar si la 2.ª jugada ya no alcanza a la 1.ª ni con p·(vueltas que quedan)
F_UNST = 0.75       # stop: si la más visitada no es la de mayor valor, buscar 0.75 del plan más
MAX_UNST = 2        # stop: cuántas veces se puede extender
FACTOR_CRITICA = 1.6  # critico: tiempo en posiciones con capturas, peones cerca de anotar o 2 puntos
FACTOR_QUIETA = 0.6   # critico: tiempo en posiciones quietas
USAR_TACTICA = False  # Q: simulaciones con jugadas decisivas y antidecisivas (simulacion.py)
REUSAR_ARBOL = False  # R: empezar cada turno desde el subárbol del turno anterior
# Evaluación de posiciones (logística, data/eval_posicion.npz; ver herramientas/ajustar_eval.py)
CORTE_SIMULACION = sim.SIN_CORTE  # tras tantas jugadas, la simulación se detiene en la primera
                                   # posición quieta y usa la evaluación (requiere simulacion.py)
ALFA_MINIMAX = 0.0    # implicit minimax: UCT usa (1 - α)·promedio + α·minimax de la evaluación
LIBRO = False  # usar el libro de aperturas data/libro_aperturas.json (herramientas/generar_libro.py)
TACTICA_ARBOL = False  # en todo nodo donde el rival de quien mueve tiene 2 puntos, conservar
                       # solo las jugadas de menor riesgo (el filtro de la raíz, en todo el árbol)
SIMULAR_TACTICO = USAR_TACTICA or CORTE_SIMULACION < sim.SIN_CORTE
NECESITA_TABLERO = SIMULAR_TACTICO or ALFA_MINIMAX > 0 or TACTICA_ARBOL

def preparar(config): #Configura al bot antes de comenzar a jugar
    if ALGORITMO == "plano":
        plano.preparar(config)
        return
    global color, rng, reserva  # se guardan aquí para usarlos en cada llamada a jugar
    global previo, mu0, prec0, mu, observaciones, simulador, mi_indice, arbol_previo
    global w_eval, f_eval, libro
    color = config.color  # evaluamos SIEMPRE desde nuestro color
    rng = random.Random(config.semilla) #Crea un generador de numeros aleatorio.
    reserva = RESERVA * config.reloj_s
    # Calentamos numba ahora: la primera simulación tarda segundos y aquí no gasta reloj.
    simular_al_azar(Estado.inicial(), 1, 0, para=color)
    mi_indice = 0 if color == "blancas" else 1
    arbol_previo = None  # (raíz, jugada elegida) del turno anterior, para REUSAR_ARBOL
    libro = {}
    if LIBRO:
        with open(os.path.join(config.carpeta_datos, "libro_aperturas.json")) as archivo:
            libro = json.load(archivo)
    f_eval = np.zeros(sim.D_EVAL)
    w_eval = np.zeros(sim.D_EVAL)
    if CORTE_SIMULACION < sim.SIN_CORTE or ALFA_MINIMAX > 0:
        w_eval = np.load(os.path.join(config.carpeta_datos, "eval_posicion.npz"))["w"]
    if NECESITA_TABLERO:
        # Compilar los kernels de numba aquí (fuera del reloj).
        inicial = Estado.inicial()
        tab, pts, turno = mo.tablero_de(inicial)
        sim.hijo_tablero(tab, pts, turno, inicial.jugadas()[0], True)
        if SIMULAR_TACTICO:
            sim.simular_tactico(tab, pts, turno, 1, 0, mi_indice, w_eval, CORTE_SIMULACION)
        if ALFA_MINIMAX > 0:
            sim.evaluar(tab, pts, turno, w_eval, f_eval)
        if TACTICA_ARBOL:
            sim.riesgo_jugada(tab, pts, turno, 4, 13)

    # Estado y jugada de nuestro turno anterior, para deducir qué jugó el rival.
    previo = (Estado.inicial(), None) if color == "negras" else None
    if USAR_A or USAR_B:
        prior = np.load(os.path.join(config.carpeta_datos, "prior_oponente.npz"))
        mu0, prec0 = prior["mu0"], prior["prec0"]
        mu = mu0.copy()  # media posterior: empieza en el prior y se actualiza en cada turno
        observaciones = []
        # Compilar los kernels de numba aquí (fuera del reloj).
        inicial = Estado.inicial()
        mo.matriz_caracteristicas(inicial, inicial.jugadas())
        if USAR_B:
            # Pesos del modelo precalculados en tablas: nosotros con el prior, el rival con
            # la posterior (se recalcula su tabla cada vez que cambia mu).
            simulador = mo.SimuladorRapido(EPS_MODELO, PASOS_GUIADOS)
            simulador.poner_pesos(color, mu0)
            simulador.poner_pesos(otro_color(), mu)
            simulador.fraccion(inicial, 1, 0, color)

def otro_color():
    return "negras" if color == "blancas" else "blancas"

def actualizar_modelo(estado):
    """Deduce la última jugada del rival y actualiza la media posterior mu (Bayes + Laplace)."""
    global mu
    if previo is None:
        return  # jugamos con Blancas y es nuestra primera jugada: el rival aún no jugó
    candidatos = mo.reconstruir_jugada_rival(previo[0], previo[1], estado)
    if len(candidatos) != 1:
        return  # varias jugadas dejan el mismo tablero (≈4 %): no se sabe cuál fue, se ignora
    estado_rival, jugada_rival = candidatos[0]
    jugadas = estado_rival.jugadas()
    F = mo.matriz_caracteristicas(estado_rival, jugadas)
    observaciones.append((F, jugadas.index(jugada_rival)))
    mu, _ = mo.map_laplace(mu0, prec0, observaciones, mu)
    if USAR_B:
        simulador.poner_pesos(otro_color(), mu)

def sesgo_rival(nodo):
    """|A|·pi(a|s) de cada jugada del rival en este nodo (se calcula una vez por nodo)."""
    if "sesgo" not in nodo:
        jugadas = list(nodo["acciones"])
        F = mo.matriz_caracteristicas(nodo["estado"], jugadas)
        pi = mo.probabilidades(mu, F, EPS_MODELO)
        nodo["sesgo"] = {j: len(jugadas) * p for j, p in zip(jugadas, pi)}
    return nodo["sesgo"]

def puntaje_uct(suma, visitas, visitas_padre, turno):
    """Equilibra el resultado observado con la exploración."""
    if visitas == 0:
        return float("inf")

    promedio = suma / visitas
    bono = EXPLORACION * math.sqrt(math.log(visitas_padre) / visitas)

    if turno == color:
        return promedio + bono
    return -promedio + bono

def crear_nodo(estado):
    """Crea un nodo y guarda su deteccion de victoria inmediata"""
    ganador = ganador_tactico(estado)
    jugadas = [] if ganador is not None else estado.jugadas()

    return {
        "estado": estado,
        "ganador": ganador,
        "visitas": 0,
        "suma": 0.0,
        "acciones": {
            j: {
                "visitas": 0,
                "suma": 0.0,
                "resultados": None,
                "hijos": {},
            }
            for j in jugadas
        },
    }

def valor_accion(datos):
    """Valor estimado de una jugada para UCT.

    En un ataque cuyos dos resultados (gana / pierde) ya tienen nodo, combina el
    valor de cada resultado con su probabilidad exacta p (depende de los apoyos),
    en vez de confiar en la proporción que salió al sortear el dado.
    En los demás casos regresa el promedio de siempre.
    """
    hijos = datos["hijos"]
    if datos["resultados"] is not None and len(datos["resultados"]) == 2 and len(hijos) == 2:
        return sum(
            p * hijos[i]["suma"] / hijos[i]["visitas"]
            for i, (p, _) in enumerate(datos["resultados"])
        )
    return datos["suma"] / datos["visitas"]

def valor_eval(nodo):
    """Valor del nodo para nosotros en [-1, 1] según la evaluación (±1 si ya hay ganador)."""
    if nodo["ganador"] is not None:
        return 1.0 if nodo["ganador"] == color else -1.0
    p_blancas = sim.evaluar(nodo["tab"], nodo["pts"], nodo["turno"], w_eval, f_eval)
    p = p_blancas if mi_indice == 0 else 1.0 - p_blancas
    return 2.0 * p - 1.0

def preparar_nodo(nodo):
    """Datos extra de un nodo nuevo que ya tiene su tablero numpy."""
    if ALFA_MINIMAX > 0:
        nodo["v"] = valor_eval(nodo)
    if TACTICA_ARBOL and nodo["ganador"] is None and nodo["pts"][1 - nodo["turno"]] == 2:
        # El rival de quien mueve tiene 2 puntos: solo las jugadas que menos le dejan anotar.
        riesgos = {
            j: sim.riesgo_jugada(nodo["tab"], nodo["pts"], nodo["turno"], j.origen, j.destino)
            for j in nodo["acciones"]
        }
        menor = min(riesgos.values())
        nodo["acciones"] = {
            j: datos for j, datos in nodo["acciones"].items() if riesgos[j] <= menor + 1e-12
        }

def respaldar_minimax(camino_nodos, camino_acciones):
    """Implicit minimax (Lanctot et al., 2014): sube el minimax de la evaluación por el camino.

    El valor de una jugada promedia sus resultados de dado ya visitados (pesados con p); el de
    un nodo es el máximo de sus jugadas si nos toca y el mínimo si le toca al rival.
    """
    for i in range(len(camino_acciones) - 1, -1, -1):
        accion = camino_acciones[i]
        total = suma = 0.0
        for k, hijo in accion["hijos"].items():
            p = accion["resultados"][k][0]
            total += p
            suma += p * hijo["v"]
        accion["v"] = suma / total
        nodo = camino_nodos[i]
        valores = [d["v"] for d in nodo["acciones"].values() if "v" in d]
        nodo["v"] = max(valores) if nodo["estado"].turno == color else min(valores)

def valor_mezclado(datos):
    """Valor de una jugada para UCT: el promedio, mezclado con el minimax si ALFA_MINIMAX > 0."""
    q = valor_accion(datos)
    if ALFA_MINIMAX > 0 and "v" in datos:
        return (1.0 - ALFA_MINIMAX) * q + ALFA_MINIMAX * datos["v"]
    return q

def seleccionar_uct(nodo, generador):
    """Selecciona una jugada para explorar desde un nodo no terminal."""
    acciones = nodo["acciones"]

    # Opción A: en turnos del rival, el modelo decide qué revisar primero.
    sesgo = sesgo_rival(nodo) if USAR_A and nodo["estado"].turno != color else None

    sin_visitar = [
        j for j, datos in acciones.items()
        if datos["visitas"] == 0
    ]
    if sin_visitar:
        if sesgo is not None:
            return max(sin_visitar, key=sesgo.get)  # la respuesta más probable primero
        return generador.choice(sin_visitar)

    puntajes = {
        j: puntaje_uct(
            valor_mezclado(datos) * datos["visitas"],
            datos["visitas"],
            nodo["visitas"],
            nodo["estado"].turno,
        ) + (PESO_SESGO * sesgo[j] / (1 + datos["visitas"]) if sesgo is not None else 0.0)
        for j, datos in acciones.items()
    }

    mayor = max(puntajes.values())
    mejores = [j for j, valor in puntajes.items() if valor == mayor]
    return generador.choice(mejores)

def avanzar_uct(nodo, jugada, generador):
    """Sortea un resultado de la jugada y obtiene su nodo hijo."""
    datos = nodo["acciones"][jugada]

    if datos["resultados"] is None:
        datos["resultados"] = nodo["estado"].siguiente(jugada)

    resultados = datos["resultados"]
    probabilidades = [p for p, _ in resultados]

    indice = generador.choices(
        range(len(resultados)),
        weights=probabilidades,
    )[0]

    nuevo = indice not in datos["hijos"]
    if nuevo:
        _, estado_hijo = resultados[indice]
        hijo = crear_nodo(estado_hijo)
        if NECESITA_TABLERO:
            # Tablero numpy del hijo a partir del de su padre (el resultado 0 es «gana»).
            hijo["tab"], hijo["pts"] = sim.hijo_tablero(
                nodo["tab"], nodo["pts"], nodo["turno"], jugada, indice == 0
            )
            hijo["turno"] = 1 - nodo["turno"]
            preparar_nodo(hijo)
        datos["hijos"][indice] = hijo

    return datos["hijos"][indice], nuevo

def ganador_tactico(estado):
    """Ganador actual, o jugador que puede anotar su tercer punto sin combate."""
    ganador = estado.ganador()
    if ganador is not None:
        return ganador

    turno = estado.turno

    # Esta detección busca una victoria por el tercer punto.
    if estado.puntos[turno] != 2:
        return None

    # Internamente las filas se numeran de 0 a 8.
    fila_meta = 8 if turno == "blancas" else 0

    for jugada in estado.jugadas():
        llega_a_meta = jugada.destino // 9 == fila_meta
        if llega_a_meta and not estado.es_ataque(jugada):
            return turno

    return None

def vuelta_uct(raiz, generador):
    """Ejecuta selección, expansión, simulación y retropropagación."""
    nodo = raiz
    camino_nodos = [raiz]
    camino_acciones = []

    # Selección y expansión.
    while nodo["ganador"] is None:
        jugada = seleccionar_uct(nodo, generador)
        camino_acciones.append(nodo["acciones"][jugada])

        hijo, nuevo = avanzar_uct(nodo, jugada, generador)
        camino_nodos.append(hijo)
        nodo = hijo

        if nuevo:
            break

    # Resultado conocido si terminó o hay una victoria inmediata segura.
    ganador = nodo["ganador"]
    if ganador is not None:
        resultado = 1.0 if ganador == color else -1.0
    elif USAR_B:
        # Opción B: nosotros jugamos según el prior y el rival según la posterior.
        fraccion = simulador.fraccion(
            nodo["estado"],
            SIMULACIONES_POR_VUELTA,
            generador.randrange(2**31),
            color,
        )
        resultado = 2 * fraccion - 1
    elif SIMULAR_TACTICO:
        fraccion = sim.simular_tactico(
            nodo["tab"],
            nodo["pts"],
            nodo["turno"],
            SIMULACIONES_POR_VUELTA,
            generador.randrange(2**31),
            mi_indice,
            w_eval,
            CORTE_SIMULACION,
        )
        resultado = 2 * fraccion - 1
    else:
        fraccion = simular_al_azar(
            nodo["estado"],
            SIMULACIONES_POR_VUELTA,
            generador.randrange(2**31),
            para=color,
        )
        resultado = 2 * fraccion - 1

    # Retropropagación: el mismo resultado actualiza todo el camino.
    for visitado in camino_nodos:
        visitado["visitas"] += 1
        visitado["suma"] += resultado

    for accion in camino_acciones:
        accion["visitas"] += 1
        accion["suma"] += resultado

    if ALFA_MINIMAX > 0:
        respaldar_minimax(camino_nodos, camino_acciones)

    return resultado

def riesgo_inmediato(estado, jugada, rival):
    """Probabilidad de que el rival pueda ganar en su siguiente turno."""
    riesgo = 0.0
    # Nuestra jugada puede tener uno o dos resultados si hay combate.
    for prob_nuestra, hijo in estado.siguiente(jugada):
        if hijo.ganador() == color:
            continue  # Ya ganamos: el rival no tendrá respuesta.
        mayor_riesgo_en_hijo = 0.0
        # El rival elegirá la respuesta que más le convenga.
        for respuesta in hijo.jugadas():
            prob_ganar = sum(
                prob_rival
                for prob_rival, nieto in hijo.siguiente(respuesta)
                if nieto.ganador() == rival
            )
            mayor_riesgo_en_hijo = max(mayor_riesgo_en_hijo, prob_ganar)
        riesgo += prob_nuestra * mayor_riesgo_en_hijo
    return riesgo

def jugar(estado, tiempo_restante):
    global previo
    if ALGORITMO == "plano":
        return plano.jugar(estado, tiempo_restante)
    if USAR_A or USAR_B:
        actualizar_modelo(estado)
    mejor = decidir(estado, tiempo_restante)
    previo = (estado, mejor)  # se guarda sin importar por cuál salida decidió `decidir`
    return mejor

def es_critica(estado):
    """True si la posición no es quieta: hay capturas, un peón a 2 filas o menos de anotar, o 2 puntos.

    Es la idea de «posición quieta» del curso (7.3.2): si nadie puede capturar ni está por
    anotar, pensar más casi no cambia la jugada.
    """
    if max(estado.puntos.values()) >= 2:
        return True
    tab, _, _ = mo.tablero_de(estado)
    for casilla in range(81):
        pieza = tab[casilla]
        if pieza == 0:
            continue
        k = pieza - 1
        fila = casilla // 9
        if (fila if k == 0 else 8 - fila) >= 6:
            return True  # a 2 filas o menos de anotar
        frente = casilla + (9 if k == 0 else -9)
        col = casilla % 9
        for dc in (-1, 1):
            if 0 <= col + dc < 9 and tab[frente + dc] == 2 - k:
                return True  # este peón puede capturar
    return False

def buscar_en_arbol(previo_arbol, estado):
    """Nodo del árbol del turno anterior que corresponde a `estado` (None si no está)."""
    raiz_prev, jugada_prev = previo_arbol
    datos = raiz_prev["acciones"].get(jugada_prev)
    if datos is None:
        return None
    clave = estado.clave
    # Dos niveles abajo: nuestra jugada (y su dado), luego la respuesta del rival (y su dado).
    for hijo in datos["hijos"].values():
        for accion in hijo["acciones"].values():
            for nieto in accion["hijos"].values():
                if nieto["estado"].clave == clave and nieto["ganador"] is None:
                    return nieto
    return None

def ya_decidido(raiz, nuevas, usado, queda):
    """STOP: la segunda jugada más visitada ya no alcanza a la primera con lo que queda."""
    visitas = sorted((d["visitas"] for d in raiz["acciones"].values()), reverse=True)
    if len(visitas) < 2:
        return True
    return visitas[0] - visitas[1] > nuevas * (queda / usado) * P_STOP

def inestable(raiz):
    """UNST: la jugada más visitada no es la de mayor valor (entre las que tienen ≥ 5 % de vueltas)."""
    minimo = 0.05 * raiz["visitas"]
    acciones = [(j, d) for j, d in raiz["acciones"].items() if d["visitas"] >= max(minimo, 1)]
    if len(acciones) < 2:
        return False
    mas_visitada = max(acciones, key=lambda x: x[1]["visitas"])[0]
    mejor_valor = max(acciones, key=lambda x: valor_accion(x[1]))[0]
    return mas_visitada != mejor_valor

def decidir(estado, tiempo_restante):
    global arbol_previo
    inicio = time.perf_counter()
    previo_arbol, arbol_previo = arbol_previo, None  # se vuelve a guardar solo si se busca

    # 0. Libro de aperturas: jugada ya calculada con mucho tiempo (solo si es legal aquí).
    if LIBRO and str(estado.clave) in libro:
        jugada = Jugada.desde_texto(libro[str(estado.clave)])
        if jugada in estado.jugadas():
            return jugada

    # 1. Presupuesto (ver RELOJ).
    disponible = tiempo_restante - (reserva if RELOJ == "fijo" else RESERVA_S)
    jugadas = estado.jugadas()

    # 2. Si una jugada gana la partida sin importar el dado, la tomamos.
    for j in jugadas:
        if all(hijo.ganador() == color for _, hijo in estado.siguiente(j)):
            return j
    if disponible <= 0:  # solo queda la reserva: no hay tiempo de pensar
        return rng.choice(jugadas)

    rival = "negras" if color == "blancas" else "blancas"
    # Si el rival ya tiene 2 puntos, podría ganar en su próxima jugada.
    if estado.puntos[rival] == 2:
        riesgos = {j: riesgo_inmediato(estado, j, rival) for j in jugadas}
        menor_riesgo = min(riesgos.values())
        # Conservar solo las jugadas que reducen más esa amenaza.
        jugadas = [
            j for j in jugadas
            if riesgos[j] <= menor_riesgo + 1e-12
        ]
    if RELOJ != "fijo" and len(jugadas) == 1:
        return jugadas[0]  # jugada forzada: no hay nada que pensar

    if RELOJ == "fijo":
        t_plan = disponible / 25
        tope = t_plan
    else:
        if RELOJ == "stop":
            factor = F_STOP
        else:
            factor = FACTOR_CRITICA if es_critica(estado) else FACTOR_QUIETA
        t_plan = disponible / 25 * factor
        tope = min(3 * t_plan, 0.25 * disponible)

    # Construimos el árbol (o seguimos el del turno anterior) con las jugadas que pasaron
    # el filtro de riesgo.
    raiz = None
    if REUSAR_ARBOL and previo_arbol is not None:
        raiz = buscar_en_arbol(previo_arbol, estado)
    reusadas = 0 if raiz is None else raiz["visitas"]
    if raiz is None:
        raiz = crear_nodo(estado)
        if NECESITA_TABLERO:
            raiz["tab"], raiz["pts"], raiz["turno"] = mo.tablero_de(estado)
            if ALFA_MINIMAX > 0:
                raiz["v"] = valor_eval(raiz)
    raiz["acciones"] = {
        j: raiz["acciones"][j]
        for j in jugadas
    }

    # Cada vuelta amplía el árbol y actualiza sus estadísticas.
    fin_plan = inicio + t_plan
    fin_tope = inicio + tope
    extensiones = 0
    nuevas = 0
    while True:
        ahora = time.perf_counter()
        if ahora >= fin_plan:
            if (RELOJ == "stop" and extensiones < MAX_UNST and fin_plan < fin_tope
                    and inestable(raiz)):
                fin_plan = min(fin_tope, fin_plan + F_UNST * t_plan)
                extensiones += 1
            else:
                break
        if (RELOJ == "stop" and nuevas > 0 and nuevas % 16 == 0
                and ya_decidido(raiz, nuevas, ahora - inicio, fin_plan - ahora)):
            break
        vuelta_uct(raiz, rng)
        nuevas += 1

    # Al jugar elegimos por visitas; UCT se usa durante la exploración.
    max_visitas = max(
        datos["visitas"]
        for datos in raiz["acciones"].values()
    )
    # Si no hubo tiempo de explorar (o el árbol reusado solo tenía visitas en jugadas que el
    # filtro de riesgo quitó), devolvemos una legal.
    if max_visitas == 0:
        return rng.choice(jugadas)
    candidatas = [
        j for j, datos in raiz["acciones"].items()
        if datos["visitas"] == max_visitas
    ]
    mejor = rng.choice(candidatas)

    datos = raiz["acciones"][mejor]
    promedio = datos["suma"] / datos["visitas"]

    reportar(
        vueltas=raiz["visitas"],
        vueltas_nuevas=nuevas,
        reusadas=reusadas,
        extensiones=extensiones,
        visitas_elegida=datos["visitas"],
        jugadas_exploradas=sum(
            d["visitas"] > 0 for d in raiz["acciones"].values()
        ),
        eval=(promedio + 1) / 2,
    )
    if REUSAR_ARBOL:
        arbol_previo = (raiz, mejor)
    return mejor
