"""Bot «montecarlo»: Monte Carlo plano. Tu punto de partida: léelo, cámbialo, mejóralo."""

import itertools
import random
import time

from motor import Estado, reportar, simular_al_azar

SIMULACIONES = 20  # partidas al azar por cada muestra de una jugada
RESERVA = 0.3  # fracción del reloj que nunca gastamos pensando (la comunicación también cuesta)


def preparar(config):
    global color, rng, reserva  # se guardan aquí para usarlos en cada llamada a jugar
    color = config.color  # evaluamos SIEMPRE desde nuestro color
    rng = random.Random(config.semilla)
    reserva = RESERVA * config.reloj_s
    # Calentamos numba ahora: la primera simulación tarda segundos y aquí no gasta reloj.
    simular_al_azar(Estado.inicial(), 1, 0, para=color)


def jugar(estado, tiempo_restante):
    # 1. Presupuesto: una fracción fija del reloj que queda (sin tocar la reserva).
    disponible = tiempo_restante - reserva
    # AQUÍ PUEDES MEJORAR: ¿todas las jugadas merecen el mismo tiempo?
    # Baja el 25 para pensar más; vigila «Reloj máx» (⚠ desde 27 s).
    limite = time.perf_counter() + disponible / 25
    jugadas = estado.jugadas()

    # 2. Si una jugada gana la partida sin importar el dado, la tomamos.
    for j in jugadas:
        if all(hijo.ganador() == color for _, hijo in estado.siguiente(j)):
            return j
    if disponible <= 0:  # solo queda la reserva: no hay tiempo de pensar
        return rng.choice(jugadas)

    # 3. Para cada jugada sumamos resultados (1 = gané, 0 = perdí) y contamos muestras.
    total = {j: 0.0 for j in jugadas}
    muestras = {j: 0 for j in jugadas}
    for hechas, j in enumerate(itertools.cycle(jugadas), 1):
        # 4. Si la jugada tiene un ataque, "tiramos el dado" para elegir un resultado.
        # AQUÍ PUEDES MEJORAR: ¿tirar el dado es la única manera de tratar el azar?
        probs, hijos = zip(*estado.siguiente(j), strict=True)
        hijo = rng.choices(hijos, weights=probs)[0]
        # 5. Medimos ese estado con partidas al azar, desde NUESTRO color.
        # AQUÍ PUEDES MEJORAR: ¿qué tan buenas son esas partidas al azar?
        total[j] += simular_al_azar(hijo, SIMULACIONES, rng.randrange(2**31), para=color)
        muestras[j] += 1
        # Paramos al acabar el tiempo, pero solo cuando toda jugada tiene una muestra.
        if hechas >= len(jugadas) and time.perf_counter() >= limite:
            break

    # 6. Elegimos la jugada con mejor promedio.
    # AQUÍ PUEDES MEJORAR: ¿es el promedio lo único que importa para elegir?
    mejor = max(jugadas, key=lambda j: total[j] / muestras[j])
    # Lo que reportes lo ves con -v y en el visor (eval: promedio de la jugada elegida).
    reportar(muestras=sum(muestras.values()), eval=total[mejor] / muestras[mejor])
    return mejor
