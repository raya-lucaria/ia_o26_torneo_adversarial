# Informe: alondra_bernardo

## 1. Integrantes

- Alondra Valdivia, clave única: 218713.
- Bernardo Murrieta, clave única: 220734.

## 2. Idea en un párrafo

Nuestro bot busca hacia adelante con **expectiminimax**: en cada turno imagina sus jugadas,
las respuestas del rival y así sucesivamente, y trata cada ataque como un nodo de azar cuyo
valor es p·V(gana) + (1 − p)·V(pierde), con la probabilidad de combate del motor. Para que
la búsqueda llegue a 7–9 jugadas de profundidad (~1 millón de posiciones por segundo) está
escrita en numba y usa poda alfa-beta con la poda Star1 en los nodos de azar, búsqueda de
quietud, tabla de transposición y reducción de jugadas tardías. Las posiciones del fondo se
valoran con una **función logística de 69 rasgos hechos a mano**, cuyos pesos entrenamos por
regresión logística sobre ~675 mil posiciones de partidas de autojuego, etiquetadas con el
valor de una búsqueda más profunda y con el resultado final de la partida. Al final afinamos 8
de esos pesos (los de combate) con SPSA, jugando contra los bots de otros dos equipos. El reloj
se reparte según las jugadas que probablemente faltan, apartando el costo fijo de cada jugada
en Docker.

## 3. Búsqueda (algoritmo y por qué)

Empezamos con Monte Carlo plano + UCB (sección 8), pero se estancó en ~55 % contra
`montecarlo`, porque el rival usa casi el mismo algoritmo y el Monte Carlo plano se satura:
triplicar el tiempo casi no ayudaba. Elegimos expectiminimax porque el juego tiene
información perfecta y el único azar está en los combates, con probabilidades conocidas: el
valor esperado es exacto y no hay que estimarlo con simulaciones.

La búsqueda (`rapido.py`) es un negamax: todo valor está en [−1, 1] desde el lado que mueve.

- **Nodos de azar con Star1** (Ballard, 1983). Primero se busca el resultado más probable del
  combate; como |V| ≤ 1, a veces se puede cortar sin buscar el otro resultado. La ventana
  alfa-beta de cada resultado se transforma con la probabilidad, así que la poda sigue siendo
  exacta. `herramientas/verificar_rapido.py` compara contra un expectiminimax sin poda en
  Python: diferencia 0 en 180 posiciones.
- **Quietud** de hasta 4 plies en las hojas, solo con jugadas que anotan o atacan con
  p ≥ 0.5, para no valorar una posición a media pelea.
- **Tabla de transposición** (2^20 entradas) con claves Zobrist, calculadas de forma
  incremental al aplicar cada jugada, y un caché de evaluaciones.
- **Orden de jugadas:** jugada de la tabla, anotar, ataques por probabilidad, dos jugadas
  asesinas (*killers*) por nivel e historial.
- **Reducción de jugadas tardías (LMR):** una jugada tranquila, lejos de la meta y mal
  ordenada se mira primero con un nivel menos y ventana nula. Si sorprende, se repite completa.
- **Profundización iterativa** con un tope de nodos (numba no puede leer el reloj). Si una
  profundidad se aborta, se juega la mejor jugada de la raíz que sí terminó a esa
  profundidad. Si ni la profundidad 1 alcanza, se busca a 1 ply sin quietud.

## 4. Función de evaluación

V = tanh(0.5 · w·f), equivalente a 2·σ(w·f) − 1, donde σ(w·f) es la probabilidad estimada de
ganar. f son 69 rasgos, siempre «propios menos rivales» desde el lado que mueve:

| Grupo | Rasgos |
|---|---|
| Posición | Tabla de avance × columna (8 × 5, doblando el tablero por el centro) |
| Estructura | Peones con 1 y con 2 apoyos, falange (lado a lado), bloqueados, movilidad |
| Combate | Peones amenazados con p ≥ 0.5 y con p ≥ 0.65, colgados (amenazados y sin apoyo), Σ(p − 0.5) de los ataques posibles |
| Carrera | Corredores (sin rival en el cono de delante) por avance 3–7, casi corredores (un solo rival), peones seguros en avance 5–7, quién llega antes y por cuánto |
| Marcador | Diferencia de puntos, «tengo 2 puntos», «el rival tiene 2», 2 puntos y un peón en avance ≥ 5 |
| Fase | Material × (1 − piezas / 36): el material pesa más al final |

Elegimos una función lineal sobre rasgos interpretables, sin redes neuronales, porque es
barata (~0.7 µs por posición) y se entrena con regresión logística. Su limitación es que no
ve interacciones entre rasgos que no hayamos escrito a mano.

## 5. Precomputación (qué y cómo regenerarla)

`data/pesos.json` contiene los 69 pesos de la versión **sc400**: los de r69 con 8 pesos de
combate afinados por SPSA. Es lo único que el bot lee de `data/`. En `preparar` el bot compila
numba y mide sus nodos por segundo, fuera del reloj.

Los pesos de r69 se generaron con el código de `herramientas/`:

1. `generar_posiciones.py`: partidas de autojuego con la búsqueda a profundidad fija,
   8 % de jugadas al azar para variar las posiciones. Guarda cada posición con el valor
   de la búsqueda y el resultado final. Los tres juegos de datos de r69 fueron:
   - 3000 partidas a profundidad 6 con LMR y los pesos `mezcla30.json`, semilla 7
     (264 mil posiciones);
   - 3000 partidas a profundidad 6 con los pesos `m8.json`, semilla 9 (276 mil);
   - 1500 partidas a profundidad 7 con los pesos `m8.json`, semilla 10 (135 mil).
2. `ajustar_pesos.py`: regresión logística (entropía cruzada, L-BFGS, L2 pequeña, 10 % de las
   partidas para validar). La etiqueta es 70 % el valor de la búsqueda y 30 % el resultado
   final (`--mezcla 0.3`).
3. `arena.py`: duelo entre dos versiones a nodos fijos, para decidir si los pesos nuevos
   se adoptan.

```bash
H=estudiantes/alondra_bernardo/herramientas
python $H/generar_posiciones.py --partidas 3000 --prof 6 --lmr --azar 0.08 --procesos 9 \
    --semilla 7 --pesos $H/pesos/mezcla30.json --salida v6.npz
python $H/generar_posiciones.py --partidas 3000 --prof 6 --lmr --azar 0.08 --procesos 9 \
    --semilla 9 --pesos $H/pesos/m8.json --salida v9.npz
python $H/generar_posiciones.py --partidas 1500 --prof 7 --lmr --azar 0.08 --procesos 9 \
    --semilla 10 --pesos $H/pesos/m8.json --salida v10.npz
python $H/ajustar_pesos.py v6.npz v9.npz v10.npz --mezcla 0.3 --salida pesos.json
```

Después, `spsa_combate.py` afinó los pesos de apoyos, amenazas, ataques posibles, colgados,
falange y bloqueados (8 de 69) por partidas ganadas (SPSA, Spall 1992). En cada vuelta mueve
esos pesos ±c al azar, juega las dos versiones contra cada rival con las mismas semillas y
también una contra otra, y empuja los pesos hacia la que ganó. Fueron 400 vueltas
(~9 600 partidas), a 60 mil nodos por jugada, contra `pamplinotas` y `torneo_breakthrough` con
6 s de reloj. Usamos sus bots con el acuerdo de los dos equipos y el visto bueno del
profesor, solo como rivales; no copiamos su código ni sus datos. Los rivales juegan por
reloj, así que la corrida no se repite exactamente, pero sí el método:

```bash
python $H/spsa_combate.py $H/pesos/r69.json --rivales RUTA/torneo_breakthrough RUTA/pamplinotas \
    --vueltas 400 --salida sc   # sc_400.json es data/pesos.json
```

`herramientas/pesos/` guarda los pesos de cada versión (v1 → r63 → r63b → mezcla30 → m7 → m8 →
r69 → sc400). La historia completa, con commits y resultados, está en `CAMBIOS.md`.

## 6. Manejo del reloj

Cada jugada recibe un objetivo de tiempo:

```
restantes  = max(10, 3 + 0.085·(piezas − 16)²)        # jugadas propias que faltan
disponible = tiempo_restante − reserva − 0.05·restantes
objetivo   = 1.1 · disponible / restantes
```

- La **reserva** es el 5 % del reloj (1.5 s), nunca menos de 1 s.
- Los **0.05 s por jugada futura** cubren el costo fijo que el árbitro cuenta en Docker aunque
  no pensemos (~40 ms medidos). Sin ellos, las partidas largas contra bots que buscan
  (110–136 jugadas) llegaban a 29.5 s.
- La fórmula de **jugadas restantes** está ajustada al percentil 75 de 96 partidas reales, no
  a la mediana: las partidas largas eran las que perdíamos con 0.04–0.17 s por jugada al final.

La profundización iterativa no empieza otra profundidad si el tiempo usado × 2 ya rebasa el
objetivo. Cada búsqueda lleva un tope de nodos igual a 1.5 × objetivo × nodos por segundo,
medidos en `preparar` y actualizados en cada jugada.

En Docker usamos en promedio 23–27 s de los 30. El máximo en las pruebas de sc400 fue 28.7 s,
y ninguna partida se perdió por tiempo.

## 7. Resultados vs random y montecarlo

Todo con `./torneo jugar` en Docker: reloj de 30 s por bot, `preparar` de 10 s, límites del
torneo y colores alternados. Para no elegir la versión que tuvo suerte con ciertas semillas,
las versiones se compararon con las mismas semillas y luego se confirmaron con semillas nuevas.
Con 96 partidas, ±7 victorias es ruido normal (2 desviaciones estándar).

**Versión entregada (sc400):**

| Rival | Semillas | Partidas | Resultado |
|---|---|---:|---:|
| montecarlo | 10001–10012 | 96 | 81–15 (84.4 %) |
| random | 9611–9612 | 16 | 16–0 |
| Nuestra versión anterior (m7) | 10001–10012 | 96 | 60–36 (62.5 %) |

**r69, de la que sale sc400 (mismo código, otros pesos):** contra montecarlo 201–39 (83.8 %) en
240 partidas, contra random 16–0 y contra m7 209–127 (62 %) en 336.

**Contra los bots de otros dos equipos** (con su acuerdo; arena local con reloj real de 30 s y
Docker; no se entregan), sumando todas las corridas:

| Rival | Partidas | r69 | sc400 |
|---|---:|---:|---:|
| pamplinotas | 912 | 597–315 (65.5 %) | 614–298 (67.3 %) |
| torneo_breakthrough | 912 | 474–438 (52.0 %) | 487–425 (53.4 %) |

sc400 queda +1.4 a +1.8 puntos arriba contra ambos sin bajar contra montecarlo ni m7, pero la
diferencia cabe en el ruido (±59 victorias en 1 824 partidas): es la mejor versión medida, no
una mejora demostrada. El código entregado incluye la clave Zobrist incremental y el caché de
evaluaciones (misma jugada, valor y nodos que sin ellos en 200 posiciones, ~5 % más rápido).

Evolución contra `montecarlo` en Docker: Monte Carlo + UCB 57 % → búsqueda con pesos a mano
22 % → pesos entrenados 66 % → error de la raíz corregido 76 % → LMR 84 % → m8, r69 y sc400
~84 %. r69 se adoptó porque le gana a m8 en la arena (391–329) y 62 % a m7 en Docker.

## 8. Qué intentamos y no funcionó

**El bot anterior: Monte Carlo + UCB.** Monte Carlo plano con los combates ponderados por su
probabilidad, UCB en la raíz y un detector de victorias inmediatas. Llegó a 34–26 contra
`montecarlo`. Ninguna variante se separó del ruido: reloj (55–45 y 43–57), Ridge reentrenado
(51–49), ronda cortable (22–18) y expectimax en Python (15–25). Lo dejamos por la búsqueda.

**Ideas probadas sobre el bot de búsqueda** (arena: 360 partidas a nodos fijos contra la
versión vigente, donde ±19 es ruido):

| Idea | Resultado | Lectura |
|---|---|---|
| Ventana nula (PVS) | 168–192 | Las repeticiones cuestan más de lo que ahorra |
| Jugada nula (R = 2) | 175–185 | Sus errores pesan más que lo que poda |
| Star2 (sondear los nodos de azar) | 347–373 en 720 | 33 % menos nodos, pero juega peor a nodos fijos |
| Quietud de 2 o de 6 plies | 168–192, 171–189 | 4 es el equilibrio |
| LMR más agresivo; tabla 4 veces mayor | 178–182; 43–45 en Docker | Sin ganancia |
| Pesos por fase (apertura y final) | 202–158 contra un control de 204–156 | La ventaja era del reentrenamiento |
| Más rasgos (r65, r73) | 35–13 contra el fuerte (antes 42–6); 173–187 | Los pesos nuevos quedaron en ~0 |
| Más vueltas de autojuego, maestro a profundidad 7 | 156–204 a 186–174 | La evaluación lineal ya está en su techo |
| 20 % de partidas contra Monte Carlo; ventaja asumida | 589–131 y 579–141 contra 582–138 | Igual |
| Doble de tiempo en jugadas críticas | 169–23 contra 165–27 en Docker | Ruido |

**Última tanda: buscar 75 % contra los dos rivales.** Contra pamplinotas y torneo_breakthrough
(720 partidas por rival, reloj de 30 s) y contra r69 (720 partidas, reloj de 10 s, o 360 a
300 mil nodos), comparado con r69 en las mismas semillas:

| Idea | Pamplinotas | torneo_breakthrough | Lectura |
|---|---:|---:|---|
| r69 (referencia) | 477–243 | 376–344 | — |
| Pesos con partidas contra los rivales (r74, r75) | 476–244 (r75) | 184–176 en 360 | Igual |
| Más autojuego (r69b) | 474–246 | 389–331 | Igual |
| Repartir el reloj hacia el final (2 variantes) | 216–144 y 222–138 en 360 | — | Igual |
| Quietud con avances, tabla que siempre reemplaza, LMR, poda en la frontera | — | — | 336–384 a 368–352 contra r69: igual o peor |
| Evaluación 25 % más rápida (mismas decisiones) | 457–263 | 369–351 | Igual |
| SPSA de todos los pesos contra sí mismo (sb500) | 467–253 | 349–371 | Le gana a r69, pero a los rivales no |
| SPSA de 29 pesos contra los rivales (sd800) | 489–231 | 380–340 | Igual que sc400 |
| *Comparison training* (ordenar jugadas como una búsqueda profunda) | 225–135 en 360 | 168–192 en 360 | Peor; con pesos más libres, 99–261 contra r69 |

Dos mediciones explican el techo. Con 4 veces más reloj, r69 sacó 65.6 % y 53.8 %. Y al
repasar 192 partidas con una búsqueda fija, los dados movieron ~4 veces más valor que las malas
elecciones de cada lado, con casi ningún error grande (≥ 0.15) que castigar. Una tabla de
patrones 2×3 y un modelo flexible (solo como medición fuera del bot) tampoco predijeron mejor
que r69.

**Aprendizajes:** el error que más costó fue de código: si la tabla cortaba en la raíz, el
bot jugaba la primera jugada legal (~20 % de los turnos); corregirlo subió una serie de 27–21 a
41–7. Contra nosotros mismos duplicar los nodos casi no ayuda (191–169), contra Monte Carlo sí
(293–67 contra 272–88). Y 28 de 31 derrotas contra `montecarlo` se voltean justo después de
perder un combate en una posición pareja.

## 9. Fuentes y uso de IA

- Ballard, B. (1983). «The *-minimax search procedure for trees containing chance nodes».
  *Artificial Intelligence* 21. Es la poda Star1.
- Hauk, T., Buro, M. y Schaeffer, J. (2006). «\*-Minimax Performance in Backgammon». Lo
  usamos para Star1 y Star2.
- Tesauro, G. (1995), TD-Gammon, y el método Texel de ajuste de pesos (Chess Programming
  Wiki): son la idea de entrenar la evaluación con resultados de partidas propias.
- Spall, J. C. (1992). «Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation», *IEEE Trans. on Automatic Control* 37. Es SPSA.
- Chess Programming Wiki: Zobrist hashing, killers, history heuristic, LMR, quiescence search.
- Lorentz, R. y Horey, T. (2013). «Programming Breakthrough», *Computers and Games*. Lo
  consultamos para ideas de rasgos de carrera y de defensa de la casa.
- Notas del curso (rayalucaria.org/ia_o26/juegos): expectiminimax y tablas de transposición.
- Repositorio oficial del torneo: el motor (`motor.nucleo`) y el bot `montecarlo`, base de
  nuestro bot anterior.
- Leímos los `INFORME.md` de otros dos equipos (solo los informes, no su código ni sus datos).
  De ahí salió la idea general de cambiar a expectiminimax con una evaluación entrenada.
- Jugamos contra los bots de los equipos `pamplinotas` y `torneo_breakthrough`, con el acuerdo
  de ambos y el visto bueno del profesor, para medir y para afinar los 8 pesos de combate con
  SPSA. Los usamos solo como rivales: no copiamos su código, sus rasgos ni sus datos, y no se
  entregan. El código, los rasgos, los datos y los pesos son nuestros.

**Uso de IA.** En el bot anterior usamos ChatGPT para explicar código, proponer cambios y
redactar. En el bot de búsqueda usamos Claude Code (Anthropic) como asistente de
programación. Nosotros decidimos la estrategia, qué probar y qué conservar. Claude escribió
la mayor parte del código de `rapido.py`, `strategy.py` y `herramientas/` siguiendo esas
decisiones, corrió las arenas y las pruebas en Docker, y ayudó a redactar `CAMBIOS.md` y este
informe. Los integrantes revisamos los resultados y somos responsables de entender y verificar
el código y cada afirmación de este informe. El bot no usa redes neuronales: su evaluación es
una suma ponderada de rasgos.
