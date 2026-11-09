# Informe: maquinas_del_trabajo

## 1. Integrantes
- Luis Alberto Islas Muro (000212541)
- Julia Rojas Pereyra (000214639)

Código medido: commit `ceb9794` (rama `codigo`, 1 de noviembre de 2026). La versión entregada solo cambia comentarios, documentación y este informe: juega idéntico.

## 2. Idea en un párrafo
Nuestro bot hace expectiminimax: mira varias jugadas hacia adelante y, en cada ataque, promedia los dos resultados posibles del dado según su probabilidad real. La búsqueda es un negamax con poda alfa-beta y poda Star1 en los nodos de azar, sobre un motor en numba que es una copia validada del oficial. En las hojas, una evaluación entrenada offline estima la probabilidad de ganar: una regresión logística con 25 rasgos y una tabla de patrones 3×3, corregida por dos reglas exactas para el final de la partida. El reloj se reparte según las jugadas que estimamos que faltan, y siempre dejamos una reserva de 3.5 s sin tocar.

## 3. Búsqueda (algoritmo y por qué)
Como el combate se decide con un dado, minimax no basta: hay que hacer expectimax. Cada ataque tiene solo dos resultados con probabilidad conocida (en la práctica, entre 0.2 y 0.8), así que usamos Star1 (Ballard, 1983): buscamos primero el resultado más probable y, si con ese ya sabemos que el nodo queda fuera de la ventana (α, β), no buscamos el otro. Los valores van de −1 a 1 y son 2·P(ganar) − 1 desde el bando que mueve. Una victoria o derrota segura vale ±(1 − 0.0005·ply): preferimos ganar rápido y perder lento, y queda por encima de cualquier evaluación (tope de 0.95).

El resto es lo estándar de un buscador de juegos, y casi todo lo medimos antes de dejarlo:

- **Profundización iterativa** con ventana de aspiración de ±0.1. La profundidad 1 se busca sin límite, así que siempre hay jugada.
- **Tabla de transposición** con claves Zobrist (las del motor oficial), 4 millones de lugares (~88 MB) en cubetas de 2: un lugar guarda lo más profundo de la búsqueda actual y el otro lo más reciente. En la raíz nunca cortamos por la tabla, porque ahí se necesita la jugada.
- **Orden de jugadas:** anotar, jugada de la tabla, ataques por ganancia esperada, *killers* e historia. Los ataques con ganancia esperada negativa van al final: ~15 % menos nodos y +66 Elo en SPRT a 500k nodos.
- **Caché de evaluaciones** en las hojas (4 MB): la misma búsqueda, 18–23 % más nodos por segundo.
- **Regla del último punto.** Los peones no retroceden, así que nadie puede tapar la meta de su rival: contra un peón a una jugada de anotar, la única defensa es capturarlo. Si el rival tiene 2 puntos y una amenaza que no podemos atacar, o dos amenazas (solo capturamos una), la hoja vale derrota segura; si nosotros tenemos 2 puntos y un peón que anota sin combate, vale victoria. Coincide con una búsqueda 1–2 plies más profunda en 3,000 posiciones y aporta +21 Elo (95 %: +2 a +39).

El motor numba pasó un fuzz diferencial contra el oficial (jugadas, orden, probabilidades, posiciones y claves Zobrist) con cero discrepancias en 200,000 posiciones. Con la configuración de la liga, el bot hace ~2.18 M nodos/s en una Mac M2 Pro y ~1.05 M en GitHub, con profundidad mediana de 7 plies a media partida. No tiene PVS ni LMR (los probamos y los quitamos, sección 8), búsqueda de quietud, *null move* ni extensiones.

## 4. Función de evaluación
Es una regresión logística, sin capas ocultas: x = Σ (peso · rasgo) + Σ (±tabla[patrón de cada peón]) y v = tanh(x/2) = 2·σ(x) − 1, recortado a ±0.95. Todo se calcula desde el bando que mueve (sus peones suman, los del rival restan), así que un solo conjunto de pesos sirve para los dos colores.

**25 rasgos:** material, puntos, avance por filas, carrera hacia la meta (con todos los peones y solo con los libres), peones libres o interceptados a 1–4 jugadas de la meta, apoyo, turno y ataques disponibles. Los que más pesan, ya entrenados:

| Rasgo | Qué mide | Peso |
|---|---|---|
| `PUNTO_2` / `PUNTO_1` | tener 2 / 1 puntos (propio − rival) | 1.827 / 0.741 |
| `MEJOR_ATAQUE` | mejor ataque del que mueve, max(2p − 1) si es positivo | 0.696 |
| `ATAQUES_RIVAL` | suma de los ataques positivos del rival (solo podemos responder a uno) | −0.484 |
| `LIBRE_1` / `LIBRE_2` | peón a 1 / 2 jugadas de la meta sin rivales en su cono de amenaza | 0.367 / 0.275 |
| `CARRERA_LIBRE` | la carrera contando solo peones libres | 0.225 |
| `MATERIAL` | peones propios − rivales | 0.193 |
| `CARRERA` | jugadas que le faltan al rival menos las nuestras para anotar lo que queda | 0.156 |

El «cono de amenaza» de un peón son las casillas desde donde un rival podría atacarlo o cerrarle el paso antes de que anote.

Algunos pesos salen negativos porque los rasgos se traslapan y la regresión reparte el efecto entre ellos. El apoyo ya lo ve la tabla de patrones (las dos diagonales de atrás están en el vecindario 3×3), así que `APOYO` solo hace una corrección pequeña (−0.034). Un peón libre a 3–4 jugadas de la meta también suma por `CARRERA`, `CARRERA_LIBRE` y el avance, y `LIBRE_3`/`LIBRE_4` (−0.185 y −0.246) compensan ese doble conteo. Leídos juntos, avanzar un peón libre sigue valiendo positivo.

**Tabla de patrones 3×3.** Por cada peón miramos sus 8 vecinas (vacía, propia o rival), vistas desde el dueño y reflejadas para que ambos lados y ambos colores compartan la tabla. Con 8 filas de avance × 5 columnas × 3^8 patrones salen 262,440 pesos (2 MB). Es «buscar y sumar»: no hay red neuronal.

**Carrera decidida** (exacta, sin pesos). Un corredor es seguro si en su cono, ampliado una columna más, no hay ningún peón: nadie puede tocarlo y anota en tantas jugadas como su distancia. Si con corredores seguros anotamos lo que nos falta en n jugadas y el rival no puede hacerlo en menos de n ni con todos sus peones sin bloqueos, la posición vale +(0.965 − 0.0001·n); en el caso simétrico, lo contrario. Junto con la regla del último punto (que vive en la búsqueda), cubre el final de la partida.

La evaluación cuesta ~670 ns por posición: recorremos las 81 casillas una sola vez, sin crear arreglos, y las tablas de vecinos se precalculan al importar (−25 % de tiempo, cero diferencias en 1.2 millones de posiciones).

## 5. Precomputación (qué y cómo regenerarla)
Lo único precalculado es `data/pesos.npz`: un arreglo de 262,465 `float64` (25 rasgos más 262,440 pesos de la tabla), 229 KB. `strategy.cargar_pesos` acepta 25 o 262,465 valores finitos; si el archivo falta o viene mal, el bot juega con pesos a mano. No guardamos nada compilado en `data/`: numba compila en `preparar`.

**Datos.** 13,500 partidas y 1,166,684 posiciones (1,049,988 de autojuego y 116,696 contra montecarlo). Cada partida empieza con 2 a 8 jugadas al azar. Nuestro bot juega con 20,000 nodos por jugada y montecarlo con 4 s de reloj. Cada posición guarda sus rasgos, el resultado y el valor de la búsqueda (las que mueve montecarlo solo llevan el resultado).

**Ajuste.** Etiqueta = λ·(v + 1)/2 + (1 − λ)·ganó, con λ = 0.5, porque el resultado solo es muy ruidoso por los dados. Minimizamos entropía cruzada con L2 (1e-4 en rasgos, 1e-3 en la tabla) con L-BFGS-B de scipy, partiendo de los pesos a mano con la tabla en cero. Elegimos λ por SPRT a 500k nodos contra la versión anterior: 0.5 dio +83 Elo, 0.7 dio +58 y 0.0 dio +27. En el 20 % de las **partidas** apartadas para validar, la pérdida bajó de 0.5578 a 0.5304 y el acierto subió de 70.3 % a 70.9 %.

**Regenerarlo** (desde `estudiantes/maquinas_del_trabajo/`). Los datos (~35 MB) van a `datos_entrenamiento/`, que el `.gitignore` de nuestra carpeta deja fuera de git. Las partidas se jugaron con los pesos **anteriores** (commit `354df4b`), que se recuperan del historial:

```bash
mkdir -p datos_entrenamiento
git show 354df4b:estudiantes/maquinas_del_trabajo/data/pesos.npz > datos_entrenamiento/pesos_354df4b.npz
../../.venv/bin/python -m herramientas.datos --rival nuestro --partidas 12000 --nodos 20000 \
    --semilla 2001 --pesos datos_entrenamiento/pesos_354df4b.npz --salida datos_entrenamiento/r2_autojuego.npz
../../.venv/bin/python -m herramientas.datos --rival montecarlo --partidas 1500 --nodos 20000 \
    --reloj-rival 4 --semilla 2002 --pesos datos_entrenamiento/pesos_354df4b.npz \
    --salida datos_entrenamiento/r2_montecarlo.npz
../../.venv/bin/python -m herramientas.entrenar datos_entrenamiento/r2_autojuego.npz \
    datos_entrenamiento/r2_montecarlo.npz --lambda 0.5 --l2-patrones 1e-3 \
    --salida datos_entrenamiento/r2_pesos_l0.5.npz
cp datos_entrenamiento/r2_pesos_l0.5.npz data/pesos.npz
```

Hay que usar `--lambda 0.5` y `--l2-patrones 1e-3`: no son los valores por defecto y sin ellos no se reproducen los pesos. Verificamos (1 de noviembre) que el entrenamiento con nuestros datos da `data/pesos.npz` idéntico bit a bit. Las partidas de autojuego deberían repetirse igual con la misma semilla, porque nuestro bot juega por nodos; las de montecarlo no del todo, porque montecarlo juega por reloj.

## 6. Manejo del reloj
Estimamos las jugadas que faltan con n = ⌊1.5·peones propios⌋ + 6 − 2·(puntos totales), acotado entre 8 y 70. El presupuesto de cada jugada es:

```text
útil   = tiempo_restante − 3.5 − 0.015·n
blando = útil / n
duro   = min(3·blando, tiempo_restante − 3.5)
```

El tiempo **blando** es lo que queremos pensar: no empezamos una iteración si, según lo que creció la anterior, no alcanzaría a terminar (factor 4.0; en la arena rindió 75 % contra 58 % con 1.5). El **duro** es un corte que nunca se pasa, revisado dentro de numba cada 4,096 nodos. Con una sola jugada legal respondemos al instante.

- **Reserva de 3.5 s** que nunca tocamos: ni una partida larga pasa de ~27 s, el 90 % del reloj, donde el torneo marca ⚠.
- **15 ms por jugada** por el costo del protocolo con el árbitro aunque contestemos al instante (en Docker: 7.1 ms promedio, 14.7 ms en el percentil 90).

**Medido con el reloj real de 30 s:** en 32 partidas de autojuego usamos 21.7 s en promedio (13.4–25.3 s); en las partidas de la arena contra montecarlo, 25.6 s como máximo; en Docker, 25.0 s; en GitHub, 23.5 s. Ninguna partida se perdió por reloj ni marcó ⚠.

**`preparar`** (límite 20 s, incluye importar `strategy.py`): carga los pesos, crea toda la memoria y corre una búsqueda de profundidad 3 desde la posición inicial, que compila la búsqueda de numba antes de que empiece el reloj. Para compilar menos usamos `NUMBA_OPT=1` (sin perder velocidad), funciones internas sin envoltura de Python y copia del tablero con bucles. Tarda 2.16 s en la Mac, 2.6–2.7 s en Docker local y 6.5–7.5 s en GitHub (el validador avisa arriba de 10 s).

## 7. Resultados vs random y montecarlo
Todas con reloj de 30 s por partida y la versión final (`af61aae`), salvo donde se indica:

| Rival | Dónde | Partidas | Resultado |
|---|---|---|---|
| montecarlo | arena propia (semillas 9100 y 9300) | 400 | **335–65 (84 %)**: 164–36 y 171–29 |
| montecarlo | Docker con límites del torneo (semilla 9200) | 20 | **17–3 (85 %)** |
| montecarlo | pruebas de GitHub (`48a9b11`, `21af7dd`, `af61aae`) | 30 | 21–9 (7–3 en cada una) |
| random | pruebas de GitHub (las mismas tres) | 30 | **30–0** |

En la arena, 84 % equivale a unos +285 Elo sobre montecarlo, con reloj máximo de 25.6 s y cero derrotas por tiempo. Las pruebas de GitHub son de 10 partidas (±17 % de ruido); que las tres dieran 7–3 es casualidad. Contra random solo medimos en GitHub, y no tenemos mediciones contra los bots Tier del curso.

Para saber por qué perdemos, revisamos las 29 derrotas de la segunda corrida con un juez que busca 4 M nodos por jugada (unas 4 veces lo que busca el bot en partida). Coincidimos con su jugada en el 71 % de los casos; nuestros errores explican solo el 13 % de la caída de cada derrota, y el resto ocurre en combates decididos por el dado. Contra cuatro rivales propios de estilo (expectimax de 2–3 plies: agresivo, corredor, defensivo y equilibrado) ganamos 96–99 % en 100 partidas cada uno (`48a9b11`).

## 8. Qué intentamos y no funcionó
Aceptamos un cambio solo si mejoraba en partidas o si dejaba la búsqueda idéntica y más rápida. Estas ideas no pasaron ese filtro:

| Intento | Qué resultó | Decisión |
|---|---|---|
| **PVS** (ventana nula tras la primera jugada) | +4.9 % de nodos | Quitado |
| **LMR** (reducir jugadas tardías, 3 variantes) | Dos «no es mejor» (+6); la de profundidad 4 pasó un SPRT (+31), pero al confirmarla dio −4 | Quitado |
| **Reetiquetar con búsqueda profunda** (180,000 posiciones a 500k nodos) | Validación casi igual (0.5304 → 0.5297); en partidas −17 (−53 a +18) | No adoptado |
| **Pesos por fase** (2–2, «yo con 2», «rival con 2») | Bajó 12.5 % el error contra la búsqueda profunda, pero en partidas ≈ 0 | No adoptado |
| **Escala de reloj 1.5** | −7 Elo (−40 a +26) y +2.3 s por partida | Se queda en 1.0 |
| **`NUMBA_OPT=3`** | 1.35 contra 1.38 M nodos/s | Se queda en 1 |
| **Tabla de patrones cuantizada** (`float32`, `int16`, `int8`) | Misma velocidad (1.92–1.94 M nodos/s) y cambiaba algunas jugadas | No adoptada |
| **Conteo por distancia** en vez de `_insertar` | +29 % de tiempo (crea un arreglo por evaluación) | Descartado |
| **Evaluación incremental** | Techo de 1.27× aunque actualizar fuera gratis | Descartada sin programar |

Otras las descartamos con datos antes de programarlas: rasgos extra (movilidad, peones atacados, peones en casa, corredores lejanos), que explican solo 1–3 % del error cada uno; pensar en `preparar` (≤ +5–10 Elo); resolver como quietud el combate pendiente en la meta (esas posiciones son el 0.45 % del total), y un libro de aperturas (+2–5 Elo estimados).

En dos ocasiones (reetiquetado y pesos por fase), una evaluación que mejoraba en validación no se notó en las partidas: con 6–7 plies de búsqueda, el árbol corrige casi todo lo que la evaluación ve mal, y nuestro ruido (±30 Elo en 400 partidas) no distingue diferencias de +10. Lo que sí se tradujo en fuerza fue la búsqueda y la velocidad: pensar el doble (500k contra 250k nodos) vale +38 Elo.

## 9. Fuentes y uso de IA
**Fuentes**
- Ballard, B. W. (1983). «The *-minimax search procedure for trees containing chance nodes». *Artificial Intelligence* 21(3). Base de Star1.
- Hauk, T., Buro, M. y Schaeffer, J. (2004). «Rediscovering *-Minimax Search». *Computers and Games*.
- Zobrist, A. L. (1970). «A New Hashing Method with Application for Game Playing».
- Knuth, D. y Moore, R. (1975). «An Analysis of Alpha-Beta Pruning».
- Russell, S. y Norvig, P. *Artificial Intelligence: A Modern Approach*, capítulo de juegos (expectiminimax).
- Chess Programming Wiki: tabla de transposición, *killer* e *history heuristic*, *aspiration windows*, *Texel's tuning method* y SPRT.
- Fishtest (Stockfish): el SPRT por pares con aproximación normal que adaptamos en `herramientas/sprt.py`.
- Documentación del curso: `docs/reglas.md`, `docs/avanzado.md` y `docs/probar.md`.

Como código base usamos el motor oficial (`sistema/motor/nucleo.py`), copiado a `motor_rapido.py` como permite `docs/avanzado.md`, y la plantilla `estudiantes/montecarlo`. `herramientas/rival_montecarlo.py` es una copia del `strategy.py` de montecarlo, adaptada a una clase para usarla de rival en la arena. No usamos código de internet.

**Uso de IA**
- **Claude (Claude Code):** escribió la mayor parte del código bajo nuestra dirección (motor numba, búsqueda, evaluación, herramientas y pruebas), corrió las mediciones (arena, SPRT, perfiles, fuzz) y propuso y descartó ideas con datos. Cada cambio se aceptó solo con pruebas y mediciones.
- **Decisiones del equipo:** qué probar, los criterios de aceptación, cuándo congelar y el reetiquetado con búsqueda profunda.
- **Este informe:** se redactó a partir de nuestra documentación interna con ayuda de IA y lo revisamos nosotros.
