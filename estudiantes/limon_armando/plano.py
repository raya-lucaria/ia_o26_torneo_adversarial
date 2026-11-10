"""Monte Carlo plano mejorado: el bot «montecarlo» del curso con estas mejoras.

1. Reloj por criticidad: más tiempo en posiciones no quietas (capturas, un peón a 2 filas o menos
   de anotar, o alguien con 2 puntos) y menos en las quietas; reserva fija de 5 s.
2. Sequential halving (Karnin et al. 2013): el tiempo se parte en rondas; en cada una se descarta
   la mitad de jugadas con peor promedio, así las mejores reciben más simulaciones.
3. Dado exacto: en un ataque se simulan los dos resultados y se pesan con su probabilidad.
4. Números aleatorios comunes: en cada pasada todas las jugadas usan la misma semilla.
5. Filtro de riesgo: si el rival tiene 2 puntos, solo las jugadas que menos le dejan anotar.
6. Desempate al azar: si las dos finalistas no se distinguen, elige entre ellas al azar.

`strategy.py` lo usa cuando ALGORITMO = "plano".
"""

import math
import random
import time

from motor import Estado, reportar, simular_al_azar

SIMULACIONES = 20  # partidas al azar por cada muestra de una jugada
RESERVA_S = 5.0  # segundos del reloj que nunca gastamos pensando
FACTOR_CRITICA = 1.6  # tiempo en posiciones no quietas
FACTOR_QUIETA = 0.6  # tiempo en posiciones quietas
FILTRO_RIESGO = True  # 5: descartar jugadas que le dejan el tercer punto al rival
# 6: si la diferencia de las 2 finalistas es menor que su error estándar, elegir al azar entre
# ellas (menos predecibles ante un rival con libro o con árbol reusado)
DESEMPATE_AL_AZAR = True


def preparar(config):
    global color, rival, rng
    color = config.color  # evaluamos SIEMPRE desde nuestro color
    rival = "negras" if color == "blancas" else "blancas"
    rng = random.Random(config.semilla)
    # Calentamos numba ahora: la primera simulación tarda segundos y aquí no gasta reloj.
    simular_al_azar(Estado.inicial(), 1, 0, para=color)


def jugar(estado, tiempo_restante):
    inicio = time.perf_counter()
    jugadas = estado.jugadas()

    # Si una jugada gana la partida sin importar el dado, la tomamos.
    for j in jugadas:
        if all(hijo.ganador() == color for _, hijo in estado.siguiente(j)):
            return j
    disponible = tiempo_restante - RESERVA_S
    if disponible <= 0:  # solo queda la reserva: no hay tiempo de pensar
        return rng.choice(jugadas)
    if FILTRO_RIESGO and estado.puntos[rival] == 2:
        riesgos = {j: riesgo_inmediato(estado, j) for j in jugadas}
        menor = min(riesgos.values())
        jugadas = [j for j in jugadas if riesgos[j] <= menor + 1e-12]
    if len(jugadas) == 1:
        return jugadas[0]  # jugada forzada: no hay nada que pensar

    factor = FACTOR_CRITICA if es_critica(estado) else FACTOR_QUIETA
    limite = inicio + min(disponible / 25 * factor, 0.25 * disponible)

    # Sequential halving: rondas de igual duración; tras cada una se descarta la mitad peor.
    total = {j: 0.0 for j in jugadas}
    cuadrados = {j: 0.0 for j in jugadas}
    muestras = {j: 0 for j in jugadas}
    rondas_vistas = []  # candidatas de cada ronda, ordenadas, antes de recortar
    resultados = {j: estado.siguiente(j) for j in jugadas}
    candidatas = list(jugadas)
    rondas = math.ceil(math.log2(len(jugadas)))
    inicio_busqueda = time.perf_counter()
    for r in range(rondas):
        fin_ronda = inicio_busqueda + (limite - inicio_busqueda) * (r + 1) / rondas
        while True:  # al menos una muestra por candidata en cada ronda
            semilla = rng.randrange(2**31)  # números comunes: la misma para todas en esta pasada
            for j in candidatas:
                v = valor_muestra(resultados[j], semilla)
                total[j] += v
                cuadrados[j] += v * v
                muestras[j] += 1
            if time.perf_counter() >= fin_ronda:
                break
        candidatas.sort(key=lambda j: total[j] / muestras[j], reverse=True)
        rondas_vistas.append(list(candidatas))
        candidatas = candidatas[: math.ceil(len(candidatas) / 2)]

    mejor = candidatas[0]
    if DESEMPATE_AL_AZAR:
        mejor = desempatar(mejor, rondas_vistas[-1], total, cuadrados, muestras)
    reportar(muestras=sum(muestras.values()), eval=total[mejor] / muestras[mejor])
    return mejor


def desempatar(mejor, finalistas, total, cuadrados, muestras):
    """Al azar entre las 2 finalistas si su diferencia es menor que su error estándar.

    El error estándar de la diferencia se calcula con la varianza de las muestras de cada una,
    como si fueran independientes (con números comunes es algo mayor que el real).
    """
    a, b = finalistas[0], finalistas[1]
    media = {j: total[j] / muestras[j] for j in (a, b)}
    var = sum(max(cuadrados[j] / muestras[j] - media[j] ** 2, 1e-9) / muestras[j] for j in (a, b))
    return rng.choice([a, b]) if media[a] - media[b] < math.sqrt(var) else mejor


def valor_muestra(resultados, semilla):
    """Fracción de partidas al azar ganadas tras una jugada, sin sortear el dado.

    Con ataque, simula los dos resultados (la mitad de partidas cada uno) y los pesa con su
    probabilidad exacta: misma media que sortear el dado, con menos varianza.
    """
    if len(resultados) == 1:
        return simular_al_azar(resultados[0][1], SIMULACIONES, semilla, para=color)
    return sum(
        p * simular_al_azar(hijo, SIMULACIONES // 2, semilla, para=color)
        for p, hijo in resultados
    )


def riesgo_inmediato(estado, jugada):
    """Probabilidad de que el rival gane en su siguiente turno si hacemos `jugada`."""
    riesgo = 0.0
    for prob_nuestra, hijo in estado.siguiente(jugada):
        if hijo.ganador() == color:
            continue  # ya ganamos: el rival no tendrá respuesta
        peor = 0.0  # el rival elige la respuesta que más le conviene
        for respuesta in hijo.jugadas():
            prob = sum(p for p, nieto in hijo.siguiente(respuesta) if nieto.ganador() == rival)
            peor = max(peor, prob)
        riesgo += prob_nuestra * peor
    return riesgo


def es_critica(estado):
    """True si la posición no es quieta: hay capturas, un peón a 2 filas o menos de anotar, o 2 puntos."""
    if max(estado.puntos.values()) >= 2:
        return True
    ocupadas = {}
    for k, c in enumerate(("blancas", "negras")):
        for nombre in estado.piezas(c):
            ocupadas[(int(nombre[1:]) - 1) * 9 + ord(nombre[0]) - ord("a")] = k
    for casilla, k in ocupadas.items():
        fila, col = divmod(casilla, 9)
        if (fila if k == 0 else 8 - fila) >= 6:
            return True  # a 2 filas o menos de anotar
        frente = casilla + (9 if k == 0 else -9)
        for dc in (-1, 1):
            if 0 <= col + dc < 9 and ocupadas.get(frente + dc) == 1 - k:
                return True  # este peón puede capturar
    return False

