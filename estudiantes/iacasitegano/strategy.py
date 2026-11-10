"""Bot «iacasitegano»: expectiminimax exacto (usa la probabilidad real del dado,
no una tirada) + evaluación lineal a mano. Fase 1 del plan del equipo.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np

from motor import Estado, Jugada, reportar

import busqueda_eval3 as _busqueda

# T2 (reloj): la reserva de 9 s era ~6 veces el peor sobrecosto de comunicación medido en
# 96 partidas oficiales (máx. 1.51 s por partida, 0.12 s por jugada). Se baja a ~4 s y el
# reparto deja de ser "1/16 de lo que queda" (que dejaba ~6 s sin usar) para repartir lo
# disponible entre las jugadas propias que se esperan, con piso y tope por jugada.
RESERVA = 0.135  # fracción del reloj: 4.05 s con el reloj oficial de 30 s
JUGADAS_ESPERADAS = 40  # jugadas propias típicas por partida (mediana medida: 41)
MINIMO_RESTANTES = 10   # nunca repartir entre menos: lo que quede decae geométricamente
TOPE_JUGADA = 2.0       # ninguna jugada se lleva más de esto, pase lo que pase
# Cuánto más o menos tiempo (relativo al reparto base) le damos a una jugada según qué
# fracción de las jugadas legales son ataques: más ataques disponibles = posición más
# táctica = vale más la pena pensarla. En una posición tranquila (sin ataques) gastamos
# menos de lo normal; en una con TODO ataques, hasta el doble. El tiempo que no se gasta
# en jugadas tranquilas queda disponible (vía `tiempo_restante`) para las tácticas.
PESO_MINIMO = 0.5
PESO_MAXIMO = 2.0
FASE_APERTURA = 0.6
FASE_CONTACTO = 1.6
PESOS_EVAL3 = "pesos_eval3_ligero.npz"


def preparar(config):
    global color, rng, reserva, libro, propias  # se guardan aquí para usarlos en cada llamada a jugar
    propias = 0  # jugadas propias ya hechas en esta partida (para repartir el reloj)
    color = config.color  # evaluamos SIEMPRE desde nuestro color
    rng = random.Random(config.semilla)
    reserva = RESERVA * config.reloj_s
    # Los pesos eval3 son un NPZ empaquetado con familias de features activas.
    ruta_pesos = config.carpeta_datos / PESOS_EVAL3
    if ruta_pesos.exists():
        with np.load(ruta_pesos) as datos:
            _busqueda.cargar_pesos(datos["pesos"])
    # Libro de apertura (herramientas/libro_apertura.py): {clave del tablero: [origen, destino]},
    # solo para las primeras jugadas. Si no existe el archivo, `libro` queda vacío y `jugar`
    # siempre busca en vivo — el libro nunca es obligatorio.
    ruta_libro = config.carpeta_datos / "libro.json"
    libro = json.loads(ruta_libro.read_text()) if ruta_libro.exists() else {}
    # Calentamos numba: la búsqueda entera es numba ahora (ver busqueda_numba.py), y
    # compilarla tarda segundos — aquí no gasta reloj (10 s aparte para `preparar`).
    _busqueda.calentar()


def jugar(estado: Estado, tiempo_restante: float):
    global propias
    propias += 1
    jugadas = estado.jugadas()
    if len(jugadas) == 1:  # forzada: no hay nada que decidir, no gastamos reloj pensándola
        return jugadas[0]

    # Si una jugada gana la partida sin importar el dado, la tomamos sin pensar más.
    for j in jugadas:
        if all(hijo.ganador() == color for _, hijo in estado.siguiente(j)):
            return j

    entrada_libro = libro.get(str(estado.clave))
    if entrada_libro is not None:
        jugada_libro = Jugada(*entrada_libro)
        if jugada_libro in jugadas:  # por si acaso: nunca confiamos un libro a ciegas
            return jugada_libro

    disponible = tiempo_restante - reserva
    if disponible <= 0:  # solo queda la reserva: no hay tiempo de pensar
        return rng.choice(jugadas)

    n_ataques = sum(1 for j in jugadas if estado.es_ataque(j))
    peso = PESO_MINIMO + (PESO_MAXIMO - PESO_MINIMO) * (n_ataques / len(jugadas))
    # FMTF: reparto por fase. Medido en 411 partidas oficiales: las derrotas se deciden en
    # las jugadas propias ~12 a ~38 (el contacto), y ahi F gastaba MENOS (0.3-0.4 s) que en
    # la apertura (0.5 s), usando solo ~15 s de 30. Se mueve tiempo de apertura/final al contacto.
    if propias <= 8:
        fase = FASE_APERTURA
    elif propias <= 38:
        fase = FASE_CONTACTO
    else:
        fase = 1.0
    inicio = time.perf_counter()
    restantes = max(MINIMO_RESTANTES, JUGADAS_ESPERADAS - propias + 1)
    limite = inicio + min(TOPE_JUGADA * max(1.0, fase), fase * peso * disponible / restantes)
    jugada, valor, profundidad = _busqueda.elegir(estado, color, limite)
    reportar(prof=profundidad, eval=valor, tiempo=time.perf_counter() - inicio)
    return jugada
