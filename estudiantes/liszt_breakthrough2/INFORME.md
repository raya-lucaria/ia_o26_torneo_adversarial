# Informe: liszt_breakthrough2

## 1. Integrantes
- Emilio Martínez (clave única 211675)
- Santiago Quiñones (clave única 214196)

## 2. Idea en un párrafo
El bot hace una búsqueda **expectiminimax con poda alfa-beta** (variante Star1) escrita en numba:
piensa varias jugadas adelante suponiendo que el rival también elige su mejor respuesta, y cada
ataque es un nodo de azar cuyo valor es exacto, `p·V(gana) + (1−p)·V(pierde)`. Al llegar al límite
de profundidad sigue solo con ataques y jugadas que anotan (quiescencia) y califica la posición con
una **evaluación logística**: 20 características del tablero combinadas con pesos que **aprendimos
offline** con regresión logística sobre 1 000 partidas del propio bot (*Texel tuning*). La
evaluación estima la probabilidad de ganar del bando que mueve, así que el promedio en los nodos de
azar es un promedio de probabilidades. La profundización iterativa reparte el reloj de 30 s.

## 3. Búsqueda (algoritmo y por qué)
**Algoritmo.** Negamax con poda alfa-beta y nodos de azar Star1 (`alfabeta.py`, función `negamax`):

- **Nodos de azar exactos.** En un ataque se buscan las dos ramas (gana / pierde) con su
  probabilidad real. Star1 poda también aquí: con las cotas de la evaluación ([−5000, 5000]) y la
  ventana [α, β] calcula una ventana para la rama «gana» y, si no basta para decidir, otra para la
  rama «pierde». Comprobamos que la poda no cambia el resultado: en 40 posiciones al azar, a
  profundidad 1 y 2, el valor de la raíz coincide exactamente con un expectiminimax sin poda.
- **Profundización iterativa** (profundidad 1, 2, 3…) con límite de nodos según el tiempo: siempre
  hay una jugada lista y la tabla de transposición ordena la siguiente iteración.
- **Tabla de transposición** de 2²⁰ entradas con el hash Zobrist del motor (`nucleo.clave`):
  guarda valor, profundidad, tipo de cota y mejor jugada.
- **Orden de jugadas:** primero la jugada de la tabla, luego las que anotan, luego los ataques por
  probabilidad de ganar (y por qué tan cerca de anotar estaba la víctima), luego dos jugadas
  *killer* por profundidad y al final las demás por avance.
- **Quiescencia** de hasta 6 medias jugadas: solo ataques y jugadas que anotan, con la evaluación
  estática como opción de quedarse quieto. Evita cortar la búsqueda a mitad de un intercambio.

**Por qué este y no otro.** Empezamos con MCTS (ver sección 8). Nuestro MCTS (sección 8) y este expectiminimax resultaron equivalentes en fuerza contra `montecarlo` y empataron entre sí (20–20 en 40 partidas). Elegimos el expectiminimax porque su evaluación se puede explicar y medir (log-loss de validación) y porque rindió más contra `montecarlo` en nuestras pruebas (sección 7). Además, el árbol de este juego es manejable (unas 25 jugadas por posición) y el azar solo
aparece en los ataques, con dos resultados, así que un expectiminimax con buena poda llega a
profundidad 4–6 en unas décimas de segundo (≈ 280 000 posiciones por segundo en nuestra máquina).

## 4. Función de evaluación
`alfabeta.caracteristicas` calcula 20 características enteras desde el bando que mueve
(«mío − suyo» salvo que se diga otra cosa) y `alfabeta.evaluar` las combina:

`P(gana el que mueve) = 1 / (1 + exp(−Σ W[i]·F[i] / 1000))`, y la búsqueda usa `10000·P − 5000`.

| # | Característica | Peso ajustado |
|---|---|---|
| 0 | peones | 254 |
| 1–7 | peones con avance 1…7 (7 = a un paso de anotar) | −29, 45, 82, 287, 485, 795, 1019 |
| 8 | apoyos (peones propios en diagonal de atrás) | −19 |
| 9, 10 | peones libres (cono de avance sin rivales): cuántos y avance total | 196, −7 |
| 11 | puntos anotados | 1892 |
| 12 | carrera: ¿quién completa 3 puntos primero solo con peones libres? (+1/−1/0) | 2090 |
| 13 | tengo un peón que anota en esta jugada | 212 |
| 14 | peones del rival a un paso de anotar | −109 |
| 15, 16 | mejor probabilidad de ataque disponible (×100): mía, del rival | 6, −4 |
| 17, 18 | ataques con p ≥ 0.65: míos, del rival | 125, −126 |
| 19 | constante (tener el turno) | 61 |

Los pesos están en milésimas de logit (`data/pesos.json`). Lo que aprendió el ajuste tiene
sentido de juego: el valor del avance crece mucho cerca de la meta, los apoyos casi no valen por sí
solos (su efecto ya está en los ataques buenos de cada bando, ±125) y la carrera de peones libres
pesa más que un punto anotado. La evaluación es simétrica: verificamos en 200 posiciones al azar que
intercambiar los colores da exactamente las mismas características.

## 5. Precomputación (qué y cómo regenerarla)
`data/pesos.json` tiene los 20 pesos de la evaluación (y de dónde salieron). `strategy.py` los lee
en `preparar`; si faltara el archivo, usa `alfabeta.PESOS`. Se regenera con las dos herramientas de
`herramientas/`, desde la raíz del repo y con `.venv` activado (unos 15 min con 7 núcleos):

```bash
export NUMBA_CACHE_DIR=/tmp/numba_liszt
C=estudiantes/liszt_s            # carpeta del equipo
H=$C/herramientas
python $H/generar_datos.py --partidas 400 --seg 0.05 --salida ronda1.npz
python $H/ajustar_pesos.py ronda1.npz                 # escribe $C/data/pesos.json
cp $C/data/pesos.json pesos_r1.json
python $H/generar_datos.py --partidas 600 --seg 0.05 --pesos pesos_r1.json --semilla 2 --salida ronda2.npz
python $H/ajustar_pesos.py ronda1.npz ronda2.npz      # pesos finales
```

- `generar_datos.py` juega el bot contra sí mismo (0.05 s por jugada; las primeras 2 a 10 medias
  jugadas y un 5 % de las demás son al azar, para que las partidas no se repitan) y guarda, de cada
  posición, sus características y si el bando que movía terminó ganando.
- `ajustar_pesos.py` ajusta una regresión logística con regularización L2 (método de Newton con
  búsqueda de paso, sobre características normalizadas) y separa un 20 % de las **partidas** para
  validar. Ronda 1: 400 partidas, 29 217 posiciones; ronda 2: 600 partidas con los pesos de la
  ronda 1. Pesos finales con las dos rondas: 74 245 posiciones; log-loss de validación 0.538
  (pesos a mano: 0.640; predecir siempre 0.5: 0.693).

No hay datos que sean código: solo una lista de números enteros.

## 6. Manejo del reloj
- **Compilación fuera del reloj.** `strategy.py` llama `alfabeta.calentar()` al importarse, así que
  numba compila en `preparar`. En nuestra máquina nuestro código tarda ≈ 3.7 s en compilar
  (evaluación 1.0 s, orden de jugadas 0.3 s, búsqueda 2.2 s), lejos de los 20 s.
- **Presupuesto por jugada:** `disponible = restante − 2.5 s de reserva − sobrecosto × jugadas que
  faltan`, y la jugada usa `factor × disponible / jugadas que faltan`, con 40 jugadas propias
  supuestas (nunca menos de 8), `factor = 1.4` si hay ataques posibles y `0.7` si no, y nunca más del
  25 % de lo disponible ni de 2.5 s.
- **Sobrecosto medido en la partida:** cada turno compara lo que bajó el reloj con lo que midió
  pensando; la diferencia (comunicación con el árbitro) se promedia y se descuenta por adelantado.
- **Profundización iterativa con límite de nodos** calculado con los nodos por segundo medidos, para
  no pasarse; si no hay tiempo para nada, juega profundidad 1, que es instantánea.
- Medido con Docker con la versión final (36 partidas contra `montecarlo`): el máximo usado fue **27.2 s de 30**; la versión previa llegó a 23.3 s. Dimos esa holgura a cambio de más profundidad: el piso de 8 jugadas restantes y el sobrecosto medido evitan que el reloj se agote.

## 7. Resultados vs random y montecarlo
Todo con `./torneo jugar` (reloj de 30 s por partida para cada bot, 40 partidas por corrida salvo
indicación). Cada corrida alterna colores. «Sin Docker» cuenta reloj y `preparar` pero sin límite de
CPU ni de memoria; «Docker» usa los límites del torneo (1 núcleo, 1 GB).

**Versión entregada (`liszt_breakthrough2`)**, medida por nosotros:

| Rival | Condición | Semilla | Partidas | Ganadas | Reloj máx. |
|---|---|---|---|---|---|
| `montecarlo` | sin Docker | 5001 | 40 | 29 (73 %) | 27.4 s |
| `montecarlo` | sin Docker | 8001 | 40 | 34 (85 %) | < 27 s |
| `montecarlo` | Docker | 9001 | 12 | 6 (50 %) | 27.2 s |
| `montecarlo` | Docker | 9101 | 24 | 17 (71 %) | 27.0 s |

En total: **63 de 80 (79 %) sin Docker** y **23 de 36 (64 %) con Docker**. Con Docker el bot
busca menos profundo (1 núcleo) y el porcentaje baja; esa es la cifra más cercana a las condiciones
de la liga. Con 12 a 40 partidas por corrida el margen de error es de ±8 a ±15 puntos porcentuales,
así que los porcentajes son orientativos. Ninguna partida pasó de 27.4 s de los 30 s.

**Cambio respecto a la versión previa (`liszt_s`).** Solo cambian tres constantes del reparto del
reloj (sección 6): `JUGADAS_ESPERADAS` 60→40, `TOPE_FRACCION` 0.15→0.25 y `TOPE_SEG` 1.5→2.5. El
motivo es que la versión previa terminaba las partidas con 9 a 19 s sin usar (en una partida de 63
jugadas usó 10.9 s). Contra `montecarlo` con las mismas semillas: 29–11 y 34–6 (versión entregada)
frente a 26–14 y 26–14 (versión previa); con Docker, 17–7 frente a 14–10 en la semilla 9101. Entre
las dos versiones empataron 20–20 (semilla 6001): la mejora se ve contra `montecarlo` y no contra
la versión previa.

**Comparación con nuestro MCTS (versión vigente, sin Docker).** Le gana a `montecarlo` 51 de 80
(64 %) y empata 20–20 con `liszt_s` en 40 partidas. Es equivalente a la versión previa del expectiminimax; la versión entregada rindió más contra
`montecarlo` (79 % contra 64 % sin Docker, con semillas distintas, así que la diferencia es
orientativa) y por eso la elegimos.

**Referencia: mediciones de la versión previa, hechas por Santiago con Docker.** No son de la versión
entregada y se conservan solo como contexto:

| Rival | Partidas | Ganadas | Reloj máx. |
|---|---|---|---|
| `random` | 4 | 4 (100 %) | 12.7 s |
| `montecarlo` (8 nov) | 20 | 15 (75 %) | < 20 s |
| `montecarlo` (7 nov) | 6 | 4 | 23.3 s |

Con 20 partidas el 75 % tiene un margen de ±19 puntos, y nuestras 80 partidas de esa versión dan
65 %. En el duelo con nuestro MCTS de esa época, Negras ganó 15 de 20 partidas: entre dos bots que
castigan los errores, salir primero parece una desventaja.

## 8. Qué intentamos y no funcionó
- **MCTS con UCB1, Progressive Bias y esperanza exacta en los combates** (nuestra primera
  versión, de Emilio): **8 de 20 contra `montecarlo`** (Docker, 30 s) y 11–9 contra el bot final.
- **MCTS en numba con playouts informados y corte con evaluación** (anotar si se puede, capturar al
  peón que va a anotar, preferir ataques con apoyo; evaluar tras 16 medias jugadas): 2–2 contra
  `montecarlo` en 4 partidas, llegando a 27.6 s de 30; después de corregir el reloj, 10–10 contra
  esa misma versión y **6–14 contra `montecarlo`** (20 partidas, Docker, 30 s). Las partidas al
  azar, aunque informadas, estiman mal las posiciones.
- **Expectiminimax con pesos a mano**: 10/20 contra `montecarlo`. Medimos que esos pesos predecían
  el resultado peor que una moneda (log-loss 0.716 contra 0.693): la búsqueda optimizaba la cosa
  equivocada. Ajustar los pesos con partidas lo llevó a 18/20 (sección 7).
- **Cinco características más** (centralidad, falange, peones avanzados sin apoyo, peón más
  adelantado, peones avanzados en las orillas) y usar el resultado parcial de la última iteración:
  la log-loss de validación no mejoró (0.5428 contra 0.5417) y quedó 16/30 contra la versión actual.
  Lo descartamos por no aportar.
- **Búsqueda más rápida y más reloj** (PVS, historial de jugadas, hash Zobrist incremental,
  conservar la mejor jugada de una iteración interrumpida y repartir el reloj para partidas de 45
  jugadas): con el mismo medio segundo llegaba a profundidad 4.33 en promedio contra 4.17, pero
  perdió 8–12 contra la versión actual (20 partidas, Docker) y usaba hasta 26 s del reloj. Lo
  descartamos: una décima de profundidad no se notó y gastaba más tiempo.
- **Ajuste final del reloj (sí funcionó).** A diferencia del intento anterior, que cambiaba el algoritmo de búsqueda, este solo modifica tres constantes de reparto del tiempo (sección 6) y se midió contra `montecarlo` en dos semillas sin Docker y dos con Docker (sección 7). Contra la versión previa empata, así que lo conservamos por la mejora contra `montecarlo` y porque el reloj máximo quedó en 27.4 s de 30.
- **Medir con pocas partidas engaña.** Con los mismos dados y colores cambiados, muchas veces el
  mismo color gana las dos partidas; entre versiones parecidas, 20–30 partidas apenas distinguen
  algo. Por eso comparamos siempre por pares de dados.

## 9. Fuentes y uso de IA
- Russell y Norvig, *Artificial Intelligence: A Modern Approach*, capítulo de búsqueda adversaria
  (minimax, alfa-beta, expectiminimax).
- B. W. Ballard (1983), «The *-minimax search procedure for trees containing chance nodes»,
  *Artificial Intelligence* 21: algoritmo Star1 para podar en nodos de azar.
- T. Hauk, M. Buro y J. Schaeffer (2004), «Rediscovering *-Minimax Search», *Computers and Games*.
- Chess Programming Wiki: *Texel's Tuning Method* (ajuste de la evaluación con regresión logística
  sobre resultados de partidas), *Killer Heuristic*, *Transposition Table*, *Quiescence Search*.
- Reglas y motor del torneo (`raya-lucaria/ia_o26_torneo_adversarial`); usamos `motor.nucleo` para
  generar jugadas, aplicar jugadas, calcular `p` y el hash Zobrist. No copiamos código de internet.
- **Uso de IA.** La primera versión (MCTS) se escribió con ayuda de Google Antigravity. La versión
  actual se escribió con Claude (Anthropic): nosotros decidimos el enfoque (pensar varias jugadas
  adelante poniéndose en el lugar del rival, usar enteros, aprender los pesos offline) y Claude
  escribió el código, las pruebas de exactitud de la poda y los scripts de entrenamiento y medición.
  El ajuste final del reparto del reloj, la elección de qué versión entregar y la actualización de este informe se hicieron con apoyo de Claude (Anthropic) para analizar los resultados y escribir los parches; las pruebas las corrimos nosotros. Todas las decisiones de qué conservar las tomamos con partidas medidas, no con la opinión de la IA.
