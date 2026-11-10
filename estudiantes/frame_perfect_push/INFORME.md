# Informe: frame_perfect_push

## 1. Integrantes

Bernardo Pérez de la Torre (clave única 000212575). Equipo de un integrante, con autorización del
profesor (ver `equipo.toml`).

## 2. Idea en un párrafo

Monte Carlo en la raíz, pero gastando mejor las simulaciones que el `montecarlo` de referencia.
Cada jugada legal es un brazo de un bandido UCB1: primero recibe una tirada (40 partidas al azar
del motor) y después el bandido manda más tiradas a las jugadas prometedoras. El azar del combate
se trata de forma exacta: los dos desenlaces de un ataque salen de `estado.siguiente(j)` y se
visitan en proporción exacta a su probabilidad (muestreo estratificado, sin tirar un dado propio),
así el promedio de la jugada converge a p·V(gana) + (1 − p)·V(pierde). Antes de simular hay dos
reglas tácticas exactas: si una jugada gana segura (todo desenlace gana) se juega de inmediato,
un desenlace que deja al rival anotar su tercer punto sin combate vale 0, y si el rival tiene 2
puntos y puede llevar un peón a la fila anterior a su meta, calculamos exactamente con qué
probabilidad sobrevivimos y usamos eso como tope del valor. Las primeras jugadas salen de un libro
de aperturas calculado offline con la misma búsqueda y mucho más tiempo.

## 3. Búsqueda (algoritmo y por qué)

- **Bandido UCB1 en la raíz** (`C = 0.5`, valores en [0, 1]). Ronda inicial: una tirada por
  jugada. Luego, mientras haya presupuesto, la jugada con mayor `media + C·sqrt(ln N / n)` recibe
  4 tiradas seguidas. Se juega la jugada más simulada (desempate: mejor media).
- **Nodos de azar exactos.** Un ataque tiene dos resultados con probabilidades p y 1 − p del
  motor. Cada tirada valora el resultado más atrasado respecto a su probabilidad
  (`p·(n+1) − visitas`), así que tras n tiradas cada resultado lleva ⌊p·n⌉ visitas. No se usa ni
  se intenta predecir el dado del árbitro.
- **Por qué no un árbol.** Probamos (sección 8) expectiminimax con heurística, MCTS con nodos de
  azar y una segunda fase con la respuesta del rival; todos perdieron contra la versión plana. Con
  ~50 000 partidas al azar por segundo y ~1 s por jugada, el promedio de 40 partidas por tirada en
  la raíz es mucho más estable que árboles profundos con pocas simulaciones por hoja.

## 4. Función de evaluación

No hay heurística: el valor de un estado es la fracción de victorias de nuestro color en partidas
al azar de ambos lados (`simular_al_azar`, compilado con numba en el motor), salvo en estados cuyo
valor se conoce exacto:
- partida terminada → 1 o 0;
- quien está en turno tiene 2 puntos y puede anotar el tercero sin combate → 1 si somos nosotros,
  0 si es el rival (la partida al azar casi nunca vería esa jugada);
- **amenaza en dos jugadas:** si el rival está en turno con 2 puntos, para cada avance suyo sin
  combate a la fila anterior a su meta buscamos nuestra mejor respuesta (con sus desenlaces
  exactos) y medimos la probabilidad de que después ya no pueda anotar seguro (lo capturamos o
  ganamos antes). El rival escoge el avance que más nos daña. Esa probabilidad es una cota
  superior exacta del valor: el valor simulado se recorta a `min(simulado, cota)`, y si la cota es 0
  el desenlace vale 0 sin simular.

Por qué: al analizar las 42 derrotas de la validación de v8 (sección 7), en 17 la caída decisiva de
nuestra evaluación ocurrió sin combate: el rival con 2 puntos empujaba un peón que ya no podíamos
capturar, mientras nuestra evaluación seguía en 0.5–0.6 porque en las partidas al azar el rival casi
nunca lo empuja.

Probamos 10, 20, 40 y 80 partidas por tirada (sección 7): 40 fue lo mejor.

## 5. Precomputación (qué y cómo regenerarla)

**Libro de aperturas** en `data/libro.json` (1 318 posiciones, 43 KB): clave de posición del motor
(`estado.clave`) → jugada. Cubre TODAS las respuestas del rival en: nuestras jugadas 1–3 con
Blancas y 1–2 con Negras. Cada posición se resolvió con la misma búsqueda del bot
(`strategy.analizar`) con 60 s (primera jugada de Blancas), 30 s, 15 s o 10 s según la ronda, es
decir 8–50 veces el tiempo de una jugada en partida. En partida, si la posición está en el libro y
la jugada es legal, se juega al instante (ahorra ~2.5 s de reloj por partida); si el archivo falta
o está dañado, el bot juega sin libro.

Regenerar (desde la raíz, con `.venv`; ~1.7 h con 3 procesos):
`python estudiantes/frame_perfect_push/herramientas/generar_libro.py --procesos 3`

En `preparar` se carga el libro y se llama una vez a `simular_al_azar` para calentar numba fuera
del reloj.

## 6. Manejo del reloj

- Presupuesto por jugada: `min(1.5 s, (reloj restante − 3 s) / 22)`. Es una fracción del reloj
  que queda, así que la suma nunca alcanza la reserva de 3 s aunque la partida sea larga.
- La ronda inicial (una tirada por jugada) cuesta tiempo aunque el presupuesto sea mínimo. Si el
  presupuesto no alcanza para esa ronda (estimando 20 000 partidas/s, una máquina lenta), las
  tiradas iniciales usan 4 partidas en vez de 40.
- El chequeo de «el rival gana seguro» solo genera las jugadas una vez (antes recorría todos los
  desenlaces y costaba decenas de ms por jugada con 2 puntos; causó una derrota por reloj en una
  prueba con reloj de 4 s).
- Se reporta `tiradas` y `eval` con `reportar` (sin `print`).
- Las jugadas del libro son instantáneas (~2.5 s de reloj ahorrados por partida). La cota de
  amenaza en dos jugadas solo se calcula cuando el rival tiene 2 puntos y cuesta pocos ms.
- Si todos los desenlaces de todas las jugadas tienen valor exacto, no se simula.
- Medido con el reloj real de 30 s: reloj máximo usado en una partida 25.4 s, decisión más lenta
  1.24 s, 0 derrotas por reloj ni jugadas ilegales en 470 partidas oficiales de estas versiones
  (v4r, v5, v8, v12). Con Docker: `preparar` (incluida la carga del libro) 0.0 s. Con reloj reducido a 4 s (prueba de
  estrés, 20 partidas): 0 derrotas por reloj.

## 7. Resultados vs random y montecarlo

Partidas con `./torneo --sin-docker jugar`, reloj real de 30 s, alternando colores, 3 partidas en
paralelo cada una fijada a un núcleo (`herramientas/torneo_paralelo.py`). Sin Docker no se limita
memoria/CPU: la validación oficial es la de GitHub.

| Versión | Rival | Partidas | Resultado | Blancas | Negras |
|---|---|---:|---|---|---|
| v2 (expectiminimax + heurística) | montecarlo | 40 | 5–35 (12.5 %) | — | — |
| v5 (UCB1, 20 por tirada) | montecarlo | 60 | 38–22 (63 %) | 17–13 | 21–9 |
| v5 con 40 por tirada | montecarlo | 40 | 23–17 (57.5 %) | 8–12 | 15–5 |
| v5 con 40 por tirada | v5 | 40 | 22–18 (55 %) | 9–11 | 13–7 |
| v8 | montecarlo | 100 | 58–42 (58 %) | 27–23 | 31–19 |
| v12 sin libro | v8 | 40 | 22–18 (55 %) | 10–10 | 12–8 |
| v12 con libro parcial | v8 | 40 | 23–17 (57.5 %) | 10–10 | 13–7 |
| **v12 (final)** | montecarlo | 100 | **64–36 (64 %)** | 34–16 | 30–20 |
| **v12 (final), otros dados** | montecarlo | 100 | **64–36 (64 %)** | 35–15 | 29–21 |
| **v12 (final)** | random | 10 | 10–0 | 5–0 | 5–0 |

Intervalo de confianza del 95 % (Wilson) de v12 contra montecarlo: 54 %–73 % en la primera serie;
juntando las dos series, 128–72 en 200 partidas (64 %, IC 95 %: 57 %–70 %). Con Docker, 2
partidas contra montecarlo: 1–1, `preparar` 0.0 s, reloj máximo 23.4 s.

**Análisis de derrotas (v8, 42 derrotas de la validación)**: hay pocos combates (~4 ataques por
bando y partida; al final cada lado conserva ~12 peones), así que el juego es sobre todo una carrera
de peones. Nuestra evaluación cae bajo 0.35 a mitad de partida (mediana: media jugada 59 de ~85),
no en la apertura. En 19 derrotas la mayor caída vino de perder un ataque propio, en 6 de un ataque
ganado por el rival y en 17 sin combate: el rival con 2 puntos empujaba un peón imparable. Eso
motivó la cota de amenaza en dos jugadas de v12.

Cribado rápido (script local de duelo en un solo proceso con reloj escalado a 6 s, 30
partidas; no se entrega porque carga los bots dinámicamente):
40 por tirada ganó 22/30 a 20 por tirada; 10 por tirada ganó 10/30; 80 contra 40, 14/30;
`C = 0.3` contra `C = 0.5`, 16/30.

## 8. Qué intentamos y no funcionó

| Intento | Resultado | Por qué creemos que falló |
|---|---|---|
| v2: expectiminimax (MAX/MIN/CHANCE exactos) con heurística (avance, apoyo, riesgo, amenazas) y profundización iterativa | 5–35 vs montecarlo | Python puro: ~4 000 nodos/s, profundidad 2–3; la heurística evalúa peor que partidas al azar. |
| Tabla de transposición sobre v2 | sin medir; descartada | No arregla el problema de fondo (`versiones/v2_tt/fase4_tt.patch`). |
| MCTS (UCT) con nodos de azar estratificados y reutilización del árbol | 2–7 vs montecarlo | 2 simulaciones por hoja: el árbol sigue al ruido. |
| Partidas de simulación «informadas» en numba (anotar si se puede, capturar al peón que va a anotar) | 1–5 vs montecarlo; 7/30 vs la misma versión con simulaciones del motor | ~2.5 veces más lentas y política sesgada; además compilaban en ~4 s dentro de `preparar`. |
| v6: segunda fase con la mejor respuesta del rival para las 4 mejores jugadas | 3/20 vs v5 (y derrotas por reloj) | El mínimo de promedios ruidosos sesga a la baja y gasta el presupuesto. |
| v7: suponer que el rival anota un punto si puede sin combate | 16/30 vs base (sin diferencia) | Se descartó por no mejorar. |
| v9: más reloj (reserva 2 s, tope 1.8 s) y pensar 1.6 veces más cuando las dos mejores jugadas empatan | 14–26 vs v8 (oficial) | Gasta demasiado en la apertura (empates frecuentes); llegó a 27.2 s de reloj. |
| v11: menos reloj en la apertura (tope 1.0 s) | 10–14 vs v8 (cortado) | Sin mejora visible. |
| v10d: simulaciones propias en numba con jugada decisiva (anotar si se puede sin combate) | 41/60 vs v8 en duelos con reloj 6 s, pero 13/24 con 15 s, 21–19 oficial con 30 s y 22–18 vs montecarlo | La ventaja se diluye con más reloj y cuesta ~2–3 s de compilación en `preparar`. |
| v10: v10d + atacar al peón rival que va a anotar | 15/30 vs v8 (duelo 6 s) | Sesga las simulaciones. |
| Exploración UCB `C = 1.0` | 15/30 vs `C = 0.5` | Sin diferencia. |
| v13: evaluación exacta de carreras de peones imparables (cono de avance libre; verificada en 14 825 casos sin fallas) | 41–39 vs v12 (80 oficiales) | Decide solo ~4 medias jugadas antes del final, cuando ya está claro. |
| v14: v13 + simulaciones con jugada decisiva | 33–47 vs v12 | Igual que v10d: con reloj real empeora. |
| v15: v13 + más reloj (reserva 2 s, divisor 20) | 36–44 vs v12 | Como v9: gastar más reloj no ayuda. |
| v16: v13 + cota exacta tras cada respuesta del rival (8 mejores jugadas) | 34–46 vs v12 | Ver abajo: las derrotas demostradas valían 0. |
| v17 / v12s: derrotas demostradas «suaves» (0.02 × simulado en vez de 0) | 38–42 y 42–38 vs v12 | Sin diferencia medible. |
| Control: v12 contra una copia de v12 | 40–40 | El sistema de pruebas no tiene sesgo. |
| v18: corrección aprendida (regresión logística sobre el valor simulado y 14 rasgos: puntos, material, peones pasados, huecos en nuestras dos últimas filas; 725 partidas propias, 49 027 posiciones) | 110–90 vs v12 en 200 (55 %); 153–127 en 280 | Predice mejor el resultado (pérdida log 0.529 vs 0.549 en partidas de prueba), pero no alcanzó el umbral que fijamos antes de medir (57 %); no se entrega. |
| Cribado con duelos de reloj reducido | v10d 41/60 y v12 41/60 en duelos, pero 21–19 y 22–18 oficiales | El duelo rápido exagera las mejoras; solo cuenta lo medido con reloj real. |

## 9. Fuentes y uso de IA

- Browne et al. (2012), «A Survey of Monte Carlo Tree Search Methods», IEEE TCIAIG (UCB1, UCT,
  Monte Carlo plano).
- Auer, Cesa-Bianchi y Fischer (2002), «Finite-time Analysis of the Multiarmed Bandit Problem»
  (UCB1).
- El `montecarlo` del curso como punto de partida; el motor oficial para reglas y simulaciones.
- Uso de IA: el código, las pruebas y este informe se escribieron con ayuda de asistentes de IA
  (un agente en VS Code para las fases 1–3 y Claude para Monte Carlo con bandido, el análisis de
  derrotas, la cota de amenaza, el libro de aperturas y los intentos v13–v17),
  siguiendo las decisiones y la revisión del equipo. Todos los resultados de las tablas salen de
  partidas ejecutadas; sus registros quedan en `resultados/` de nuestra copia local (ignorada por
git, no se entrega).
