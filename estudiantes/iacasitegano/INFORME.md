## 1. Integrantes

José Malfavaun Gorostizaga (000213398) y Daphne Ivette Vela Villegas
(000211927).

## 2. Idea en un párrafo

El bot busca con expectiminimax exacto (usa la probabilidad real de combate del motor, no una tirada simulada), con poda alfa-beta, poda Star1 en los nodos de azar, profundización iterativa *anytime* con tope de nodos y tabla de transposición. Evalúa las hojas con un modelo logístico pequeño (`eval3`, familias activas: puntos, filas, patrones y marcador) y convierte el logit a una utilidad esperada de victoria `tanh(logit/2)` en `[-1, 1]` antes de combinarlo con la probabilidad del dado; el terminal vale ±1. La versión entregada (llamada FGD en nuestros registros) se distingue de las anteriores en dos cosas: (a) los **pesos de la evaluación se reajustaron por regresión ridge para imitar una búsqueda mucho más profunda** (profundidad 7, sin tope de nodos) sobre 11,504 posiciones reales de partidas previas, y se añadió un término de "a quién le toca"; (b) el reloj se reparte por fase de la partida (más tiempo al contacto, que es donde se deciden las derrotas). Antes de buscar revisa si alguna jugada gana la partida sin importar el dado, y solo después consulta un libro de apertura de 8 entradas.

## 3. Búsqueda (algoritmo y por qué)

`busqueda_eval3.py`: expectiminimax exacto compilado con Numba sobre el
tablero crudo, no sobre objetos `Estado`, porque Python puro a profundidad
3–4 es demasiado lento para competir con `montecarlo`. En los nodos de
azar (hasta dos ramas: éxito/fracaso del ataque, con su probabilidad real)
la poda Star1 acota el mejor/peor caso con la utilidad terminal
(`GANANCIA=1.0`) y corta la segunda rama si ese caso ya cae fuera de la
ventana alfa-beta.

Profundización iterativa **anytime** (`elegir`, en `busqueda_eval3.py`):
antes de intentar una profundidad nueva solo revisa si queda reloj
(`limite_tiempo`, fijado por `strategy.jugar`, ver sección 6); si queda,
calcula un **tope de nodos** para esa profundidad a partir de los
nodos/segundo REALES medidos en la última profundidad que sí terminó
completa y el tiempo real que falta (`tope = nodos_por_segundo ×
tiempo_restante × FRACCION_TOPE`, con `FRACCION_TOPE=0.8` y un piso de
`TOPE_MINIMO=20 000` nodos). La primera profundidad no tiene medición
todavía, así que corre con un tope fijo y generoso
(`TOPE_PRIMERA_PROFUNDIDAD=2 000 000`) que en la práctica nunca se agota —
eso garantiza siempre una jugada de respaldo. El buscador cuenta nodos
visitados y, si pasa ese tope a MEDIA profundidad, aborta: si ya había al
menos otras dos jugadas de la raíz completamente evaluadas a esa
profundidad, toma la mejor de esas pocas como resultado parcial; si no,
descarta el intento y conserva la última profundidad **completa**. En
ningún caso se reporta como alcanzada una profundidad que no terminó.
Dos constantes de una versión anterior, `FACTOR_PESIMISTA` y
`MARGEN_SEGURIDAD` (que decidían con un factor fijo si la siguiente
profundidad completa iba a caber, antes de intentarla), ya no se usan:
quedan en el código solo como comentario histórico, no las lee ninguna
función.

Tabla de transposición: arreglo fijo de NumPy (hash Zobrist XOR
profundidad, reemplaza siempre en colisión), que solo guarda un valor como
exacto si el ciclo terminó sin corte propio **y** el resultado cae dentro
de la ventana alfa-beta de entrada del nodo — corrección sobre una versión
anterior que guardaba exacto con solo terminar el ciclo, reproducida con
±210.20 contra ≈0 esperado en un caso de prueba.

Ordenamiento de jugadas (barato, calculado en Numba): heurístico y lexicográfico (victoria inmediata,
ataque que anota, probabilidad del ataque, bloqueo de amenaza rival),
nunca cambia el valor minimax. Quiescencia apagada (`QUIESCENCIA_MAX=0`):
medida contra alfa-beta real, no compensó su costo.

## 4. Función de evaluación

`eval3.py`: lineal por familias de features acumuladas por *lookup* en tablas pequeñas. En los pesos activos (`data/pesos_eval3_ligero.npz`, 275 valores) solo **puntos, filas, patrones y marcador** están encendidos; el resto de familias tiene bandera propia pero está apagada.

Los pesos activos de esta entrega **no** vienen de regresión logística sobre el resultado de la partida (como en versiones anteriores), sino de una **regresión ridge (λ=100, variables estandarizadas)** contra la etiqueta de una búsqueda de profundidad 7 sin tope de nodos: `data/etiquetas_prof7.jsonl` guarda 11,504 posiciones (tablero, puntos, turno, valor buscado). Cada posición aporta dos filas (vista del que mueve y vista del rival, con la etiqueta en espejo) y una columna adicional de turno. De ese ajuste salen los pesos de las familias activas y dos constantes en `busqueda_eval3.py`: `CAL_B0 = 0.0` (el sesgo es 0 por la simetría de las dos vistas) y `CAL_STM = 0.2691`, que se suma al logit si le toca mover al jugador evaluado y se resta si no. Dato honesto: las posiciones provienen de partidas anteriores nuestras (12 por partida, mezcladas) y las etiquetas se generaron con una evaluación del propio proyecto, así que hay algo de circularidad; lo que sí se midió es el resultado en partidas oficiales (sección 7). La utilidad `tanh(logit/2)` se aplica solo en la búsqueda, no en el ajuste.

## 5. Precomputación (qué y cómo regenerarla)

`data/` contiene `pesos_eval3_ligero.npz` (275 valores: 267 pesos + 8 banderas de familia), `libro.json` (8 posiciones de apertura, `{clave_tablero: [origen, destino]}`), `etiquetas_prof7.jsonl` (las 11,504 etiquetas del ajuste) y `pesos_eval3_FGE_etiquetador.npz` (los pesos de la versión anterior, usados para generar las etiquetas; solo documentación). Nada de código compilado: no hay caché de Numba en `data/` (el arranque frío mide ~2.7 s, muy por debajo de los 10 s de `preparar()`).

Regeneración de los pesos entregados: `PYTHONPATH=<repo>/sistema python herramientas/fgd/construir_fgd.py` (desde esta carpeta) reajusta la ridge desde `etiquetas_prof7.jsonl` e imprime `CAL_B0`, `CAL_STM` y el error máximo contra `pesos_eval3_ligero.npz`; lo verificamos: error 0. `herramientas/fgd/etiquetar_prof7.py` documenta cómo se produjeron las etiquetas (profundidad 7 sin tope de nodos, 12 posiciones por partida); se corrió en nuestro laboratorio con una variante previa del buscador (mismo valor alfa-beta, distinto orden de jugadas) y necesita partidas oficiales que no se entregan, por eso es documentación y no se puede correr tal cual desde esta carpeta. El libro viene de `herramientas/libro_apertura.py` (self-play con el evaluador previo a `eval3`, por eso `busqueda_numba.py` y `eval.py` se conservan aunque `strategy.py` ya no los use). El resto de `herramientas/` es el historial de experimentos (ver `herramientas/README_REGENERAR_DATA.md`).

## 6. Manejo del reloj

De los 30 s por partida, `strategy.py` reserva `RESERVA=0.135` (≈4 s) que nunca gasta pensando; esa reserva se fijó tras medir el peor sobrecosto de comunicación en 96 partidas oficiales (máx. 1.5 s por partida). Lo demás (`disponible`) se reparte entre las jugadas propias que se esperan (`JUGADAS_ESPERADAS=40`, con piso `MINIMO_RESTANTES=10`). El límite de cada jugada es `min(TOPE_JUGADA·max(1,fase), fase · peso · disponible / restantes)`, con `TOPE_JUGADA=2 s`, un `peso` entre 0.5 y 2.0 según qué fracción de las jugadas legales son ataques (posición más táctica, más tiempo) y un factor de **fase**: 0.6 en las primeras 8 jugadas propias, 1.6 de la 9 a la 38 (contacto) y 1.0 después. La fase sale de medir 411 partidas oficiales: las derrotas se decidían en las jugadas propias ~12 a ~38 y ahí el bot gastaba menos que en la apertura. Dentro del límite, el reloj solo decide si se **empieza** una profundidad nueva; una vez iniciada la corta el tope de nodos (sección 3), calculado con la velocidad real medida. Jugadas forzadas y victorias seguras se devuelven sin gastar presupuesto. Si solo queda la reserva se elige una jugada legal al azar. En las campañas oficiales de la sección 7 el reloj promedio usado fue ≈20.6 s y el máximo observado ≈25.2 s de 30 s.

## 7. Resultados vs random y montecarlo

**Contra `montecarlo`, bajo el árbitro oficial (Docker, colores alternados, semillas distintas por campaña):** tres campañas independientes de 80 partidas dieron **60/80, 58/80 y 55/80, en total 173/240 (72.1 %)**. Medimos con la misma metodología el bot anterior (F, sin el reajuste por búsqueda profunda): 134/240 (55.8 %). Una covariable de "suerte" de combate (balance neto de combates ganados por el dado) explica parte de la variación entre campañas, por eso comparamos por pares (misma semilla y color) y con intervalos de Wilson antes de decidir. Probamos además variantes (más reloj en finales de 2 puntos, desempate por riesgo, otras calibraciones) y ninguna superó a esta.

**Contra `random`:** esta versión exacta no tiene una campaña propia de partidas oficiales contra `random`; el bot anterior ganó 100/100 con el mismo motor de búsqueda y evaluación del mismo tipo, y el validador oficial (`./torneo validar iacasitegano --entrega`) juega 2 partidas de prueba contra `random` con esta carpeta y pasa. No presentamos el 100/100 como resultado de esta versión.

## 8. Qué intentamos y no funcionó

- **Pistas de la TT para ordenar nodos internos:** ahorro de tiempo de solo 1.7 % (se exigía <5 %); no se incorporó.
- **Poda Star2** en nodos de azar: correcta (2,000 posiciones sin diferencias contra expectiminimax exhaustivo) pero solo +0.032 niveles de profundidad media; se mantiene Star1.
- **Híbrido con rollouts Monte Carlo:** mejoró en desarrollo (55 vs 49 victorias de 100) pero no en validación reservada (24 vs 29 de 40).
- **Finales exactos y detector de carrera pura:** 0 activaciones en las poblaciones oficiales; no se integró.
- **Reentrenamiento amplio, lote de reajuste de 256 episodios, entrenamiento con exploración épsilon-greedy:** rechazados por peor desempeño o peor retención histórica (el reentrenamiento amplio ganó 6/80 contra `montecarlo`; la exploración, 1/40 y 1/40).
- **Caché de Numba precompilada en `data/`:** retirada; el arranque frío es rápido y el curso la prohíbe como código nativo propio.
- **Pesos S13 (13 coeficientes libres):** 41/80 (F) contra 44/80 (S13) pero una semilla en contra y sin llegar al margen exigido; rechazado.
- **Extra de reloj en finales de 2 puntos (variante FGX, ×1.8 en posiciones decisivas):** no mejoró de forma sostenida en la campaña oficial; descartado.
- **Desempate por riesgo en la raíz (variante FGAZ2):** tampoco mejoró; descartado.
- **Calibración ajustada con partidas oficiales:** una versión previa ajustaba sesgo y turno con partidas contra `montecarlo`; la entregada deriva ambos de las etiquetas profundas (sesgo 0, turno 0.2691).

Chequeos de robustez del bot anterior (misma arquitectura) contra un rival de expectiminimax simple (156/160) y contra un rival UCT/MCTS propio (143/162) no mostraron un punto ciego.

## 9. Fuentes y uso de IA

Russell y Norvig, *Artificial Intelligence: A Modern Approach*, capítulo de juegos (expectiminimax, poda alfa-beta). Ballard, B. W. (1983), «The *-minimax search procedure for trees containing chance nodes», Artificial Intelligence 21(3), base de la poda Star1. Sutton y Barto, *Reinforcement Learning: An Introduction*, para el enfoque de ajustar una función de evaluación con datos en vez de pesos fijos. Hastie, Tibshirani y Friedman, *The Elements of Statistical Learning*, para la regresión ridge.

Sin redes neuronales ni dependencias más allá de `numpy`/`scipy`/`numba`. OpenSpiel se consultó solo como referencia conceptual para verificar un MCTS propio usado como rival de desarrollo; no es dependencia del bot ni se copió su Breakthrough.

Usamos varios agentes de IA durante todo el desarrollo, no solo para escribir código: **Codex** y **Claude Code** (incluidas sesiones en la nube) ejecutaron de forma **autónoma** entrenamientos, generación de partidas, etiquetado de posiciones con búsqueda profunda y tandas de evaluación completas, incluidas corridas nocturnas sin supervisión en vivo. Eso ocurrió dentro de protocolos que el equipo fijó antes de cada corrida (hipótesis, semillas, muestra y criterio de aceptación o rechazo), y la decisión de qué entra a `estudiantes/iacasitegano/`, incluida la elección de la versión final entre los candidatos, la tomó el equipo con la evidencia a la vista. **Claude** (chat) llevó la revisión independiente, las reglas del torneo y la memoria del proyecto; **Antigravity** se usó para investigación puntual. Todo el código de `estudiantes/iacasitegano/` fue escrito o revisado por el equipo; ningún fragmento se copió de otro equipo o repositorio del curso.
