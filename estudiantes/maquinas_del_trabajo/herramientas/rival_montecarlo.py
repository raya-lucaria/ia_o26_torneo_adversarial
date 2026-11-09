"""Rival de práctica: el bot de referencia «montecarlo» del curso, adaptado a una clase.

Es una copia de `estudiantes/montecarlo/strategy.py` (el bot público de referencia, que también
fue la plantilla de nuestra carpeta) con su estado dentro de un objeto, para jugar varias
partidas en un mismo proceso. Solo lo usa la arena; `herramientas/` no corre en la liga.
"""

import itertools
import random
import time

from motor import Estado, simular_al_azar

SIMULACIONES = 20  # partidas al azar por cada muestra de una jugada
RESERVA = 0.3  # fracción del reloj que nunca gasta pensando


class Montecarlo:
    def preparar(self, config):
        self.color = config.color
        self.rng = random.Random(config.semilla)
        self.reserva = RESERVA * config.reloj_s
        simular_al_azar(Estado.inicial(), 1, 0, para=self.color)

    def jugar(self, estado, tiempo_restante):
        disponible = tiempo_restante - self.reserva
        limite = time.perf_counter() + disponible / 25
        jugadas = estado.jugadas()
        for j in jugadas:  # si una jugada gana sin importar el dado, la toma
            if all(hijo.ganador() == self.color for _, hijo in estado.siguiente(j)):
                return j
        if disponible <= 0:
            return self.rng.choice(jugadas)
        total = {j: 0.0 for j in jugadas}
        muestras = {j: 0 for j in jugadas}
        for hechas, j in enumerate(itertools.cycle(jugadas), 1):
            probs, hijos = zip(*estado.siguiente(j), strict=True)
            hijo = self.rng.choices(hijos, weights=probs)[0]
            total[j] += simular_al_azar(
                hijo, SIMULACIONES, self.rng.randrange(2**31), para=self.color
            )
            muestras[j] += 1
            if hechas >= len(jugadas) and time.perf_counter() >= limite:
                break
        return max(jugadas, key=lambda j: total[j] / muestras[j])
