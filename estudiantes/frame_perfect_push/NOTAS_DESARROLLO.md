# Notas de desarrollo: Frame Perfect Push

Auditoria inicial, 2026-11-07. Esta fase no modifica la estrategia.

## Resumen

- `versiones/v1/strategy.py` conserva la copia inicial de Monte Carlo plano. La estrategia actual `strategy.py` es expectiminimax v2; sus resultados contra `montecarlo` siguen por debajo del filtro competitivo.
- `equipo.toml` registra un integrante; el equipo reporta tener autorizacion del profesor. La validacion normal del alumno aun marca el numero de integrantes como error; la excepcion requiere aceptacion manual.
- `data/` solo contiene `.gitkeep`; `requirements.txt` no declara dependencias adicionales.
- Trabajar y guardar cambios unicamente en esta carpeta. No subir ni modificar carpetas de otros equipos o archivos del curso.

## Protocolo del bot

- `strategy.py` debe definir `jugar(estado, tiempo_restante)`. `preparar(config)` es opcional; el runner lo llama una vez si existe.
- Cada partida inicia un proceso nuevo. El runner importa `strategy.py` y llama `preparar` antes de jugar; el tiempo de importacion se incluye en el limite de preparar (20 s). El objetivo del equipo es menos de 8 s.
- `preparar` recibe `config.color` (`"blancas"` o `"negras"`), `config.semilla` (azar del bot), `config.carpeta_datos` (ruta a `data/`) y `config.reloj_s` (30 s).
- Cada turno, `jugar` recibe un `Estado` y el reloj restante, como `float`. No recibe historial ni lista de jugadas legales ya calculada. El estado contiene tablero, turno y puntos; `estado.jugadas()` calcula las legales bajo demanda.
- La respuesta debe ser un `Jugada` legal de `estado.jugadas()` o su texto, por ejemplo `"e2-e3"`. El runner convierte la respuesta a texto; el arbitro la analiza y verifica. Tuplas y otros tipos pierden por ilegal.
- No se llama al bot cuando la partida ya termino o el jugador no tiene jugadas.

## Estado, tablero y jugadas

- El tablero es 9 x 9, columnas `a`–`i`, filas `1`–`9`. Blancas avanzan hacia filas mayores; Negras, hacia filas menores. Blancas empiezan.
- `Estado` es inmutable; cada `aplicar` o resultado de `siguiente` es un estado nuevo. `Estado.desde_dict` recibe listas de casillas por color, `turno` y opcionalmente `puntos`.
- El motor guarda casillas como `fila * 9 + columna`, con `a1 == 0`, `b1 == 1`, etc. El tablero inicial tiene 18 peones por color en las dos filas propias.
- `Estado.jugadas()` usa `nucleo.generar`: por peon genera diagonales hacia adelante si no hay peon propio, y el avance recto solo si esta vacio. No genera ataques rectos, movimientos laterales ni hacia atras.
- `Jugada` es una tupla nombrada `(origen, destino)` de indices enteros; `str(jugada)` da `"e2-e3"`. `Jugada.desde_texto(...)` convierte ese formato.
- Apoyo: peones propios en las dos diagonales de atras. El atacante se cuenta desde su origen antes de mover; el defensor desde su casilla. `p = 0.5 + 0.15 * (apoyo_atacante - apoyo_defensor)`, limitado a [0.2, 0.9]. Ejemplo: Blancas e4 con d3/f3 atacan Negras f5 con g6: 2 contra 1, `p=0.65`. En el ejemplo de Negras d6 apoyado por c7 contra Blancas e5 apoyado por d4/f4: 1 contra 2, `p=0.35`.
- `Estado.siguiente(j)` da `(probabilidad, estado)` para cada desenlace; `aplicar(j, gana_ataque=...)` aplica un combate. Si el atacante pierde, su peon sale; si llega a la fila meta, anota y sale del tablero.
- El arbitro resuelve combate con un numero `u` independiente por color: gana el atacante si `u < p`. Los dados usan PCG64 sembrado a partir de semilla maestra, pareja, juego y color. La semilla que recibe el bot es distinta y nunca revela la de los dados.
- Gana quien llega a 3 puntos o deja al rival sin jugadas legales. No hay empate.

## API reutilizable y restricciones

- API publica del motor permitida y prevista: `Estado`, `Jugada`, `prob_combate`, `simular_al_azar`, `reportar` y `tomar_reporte`, importados desde `motor`. Tambien son utiles `estado.piezas`, `puntos`, `turno`, `jugadas`, `siguiente`, `aplicar`, `es_ataque`, `prob_ataque`, `ganador`, `terminado` y `clave`.
- `simular_al_azar(estado, n, semilla, para)` devuelve la fraccion de victorias del color indicado; ambos bandos eligen jugadas al azar.
- No importar el sistema del torneo (`arbitro`, `liga`, `validador`, `cli`, `visor`), otros equipos ni redes neuronales. El validador marca carga/ejecucion dinamica para revision manual. No usar codigo ofuscado, binarios propios, red ni manipular dados.
- `data/` se puede leer durante la partida y es de solo lectura; lo generado offline va ahi. `requirements.txt` opcional esta en esta carpeta: una dependencia por linea y version exacta; no cambiar numpy, scipy o numba; maximo 300 MB instalados.
- `INFORME.md` de entrega: 2–4 paginas y 9 secciones numeradas: Integrantes, Idea, Busqueda, Evaluacion, Precomputacion, Reloj, Resultados, Intentos fallidos, Fuentes/IA. Debe describir el bot real y no copiar la plantilla.
- `equipo.toml`: nombre igual a la carpeta, integrante(s) con nombre y `clave_unica` real entre comillas (6–12 digitos, distintas). El validador verifica formato, no consulta un padron de alumnos. Un solo integrante exige permiso del profesor y aun puede requerir aceptacion manual en el PR.

## Reglas clave (resumen, 10 lineas)

1. Tablero 9 x 9; Blancas avanzan hacia fila 9, Negras hacia fila 1; empiezan Blancas.
2. Se mueve un peon una casilla adelante: recto solo vacio; diagonal vacio o ataque; nunca atras ni de lado.
3. No hay ataque recto. El apoyo usa diagonales traseras; atacante desde origen y defensor desde su casilla.
4. Ataque: `p = 0.5 + 0.15 * diferencia_de_apoyo`; gana si el dado oculto `u < p`.
5. Llegar a la ultima fila anota 1 punto y quita ese peon del tablero.
6. Gana quien llega a 3 puntos o deja al rival sin jugadas; no hay empates.
7. El bot pierde por jugada ilegal, excepcion, reloj agotado o `preparar` (incluida importacion) sobre 20 s; reloj objetivo 30 s por partida.
8. Limites oficiales: 1 CPU, 1024 MB, carpeta 100 MB total/50 MB por archivo, `print` 64 KB y dependencias 300 MB.
9. Carpeta de solo lectura durante el juego, sin red ni persistencia entre partidas; se permite leer `data/`.
10. Prohibidos: redes neuronales, codigo ofuscado/nativo propio, salir del aislamiento, adivinar/manipular dados o copiar/compartir codigo/datos de otro equipo.

## Comandos de prueba

- Una partida: `./torneo jugar frame_perfect_push random --partidas 1 --semilla 7`.
- Comparacion por colores: `./torneo jugar frame_perfect_push montecarlo --partidas 20 --semilla 7`; `jugar` alterna colores.
- Mini liga: `./torneo mini frame_perfect_push`. Liga indicada: `./torneo liga frame_perfect_push random montecarlo --juegos-dados 2 --semilla 7`.
- Validacion: `./torneo validar frame_perfect_push`; entrega: `./torneo validar frame_perfect_push --entrega`.
- `jugar` y `liga` aceptan semilla; `jugar` acepta cantidad de partidas. No existe bandera de paralelismo para `jugar`. `liga` paraleliza parejas cuando hay suficientes nucleos: requiere al menos 6, asigna 2 por partida concurrente y deja 2 libres. Esta maquina tiene 4 nucleos logicos (`nproc`), por lo que `liga` corre una pareja a la vez.
- Hay localmente `random` y `montecarlo`; Tier 1–5 no estan en el repositorio y aparecen en la liga del curso.
- Salida: el comando Docker `./torneo` fija resultados en la raiz del repo; `TORNEO_RESULTADOS` no se propaga por la lista de variables de `docker-compose.yml`. Con `--sin-docker`, esa variable si puede dirigir resultados a esta carpeta, pero no se limitan CPU ni memoria. No correr otra prueba Docker hasta resolver cómo respetar la restriccion de salida sin escribir fuera de esta carpeta.

## Medicion de referencia

Comando ejecutado: `./torneo jugar random montecarlo --partidas 2 --semilla 20261107` (Docker, dos colores, una pareja de dados repetida por color). Wall time del comando: 48.66 s para dos partidas, incluyendo arranque. Resultado registrado: `montecarlo` gano 2–0. No hubo avisos de reloj ni fallas.

| Blancas | Negras | Ganador | Jugadas | Puntos B–N | Reloj MC | Preparar MC | Max decision MC | Suma tiempos de jugada |
|---|---|---|---:|---|---:|---:|---:|---:|
| random | montecarlo | montecarlo | 66 | 0–3 | 15.962 s | 0.034 s | 0.876 s | 17.128 s |
| montecarlo | random | montecarlo | 83 | 3–0 | 17.714 s | 0.043 s | 0.871 s | 19.002 s |

Estimacion lineal desde el wall time observado: 24.33 s/partida. Para 20, 60 y 100 partidas en serie: ~8.1, 24.3 y 40.6 min. Paralelismo ideal de 3 trabajadores (4 nucleos menos 1): ~2.7, 8.1 y 13.5 min; es una proyeccion, no medida. El CLI oficial en esta maquina usa una pareja concurrente como maximo por su umbral de 6 nucleos, asi que la estimacion oficial aqui es la serie.

Registros y resumen de esta medicion: `resultados/random_montecarlo_2026-11-07_15-01-38/`, `resultados/medicion_random_montecarlo_salida.txt` y `resultados/medicion_random_montecarlo_tiempo.txt`. Se midio usando Docker; `TORNEO_RESULTADOS` no fue propagada por `./torneo`, por lo que el directorio de esta corrida se reubico despues dentro de la carpeta del equipo.

## Pendientes y decisiones

- La medicion de dos partidas de Fase 1 es solo una referencia corta; no prueba ventaja estadistica. Las estimaciones de 20/60/100 son lineales y la paralela es teorica.
- Para pruebas con Docker, el wrapper oficial no pasa `TORNEO_RESULTADOS`; sus registros caen en la raiz. Las herramientas de esta carpeta usan `--sin-docker` y colocan resultados/cache aqui, pero no limitan memoria ni CPU cgroup.

## Fase 2: base v1 congelada (commit 2806c22)

- `strategy.py` usa expectiminimax de 1–2 decisiones y el API permitido del motor (`estado.jugadas()` y `estado.siguiente()`); no implementa generacion de jugadas ni simulacion propias. Se conserva el motor como unica fuente de reglas y el combate se valora como suma de probabilidades por valor de cada desenlace.
- Evaluacion modular con `PESOS` centralizados: puntos, material, avance cubico hacia meta, apoyo, oportunidades de ataque, peones propios expuestos, movilidad y amenaza de anotar en el siguiente movimiento. Victoria terminal gana prioridad; se comprueban victorias garantizadas antes de buscar y profundidad 2 detecta respuestas inmediatas del rival.
- Seguridad: `jugar` calcula primero la lista legal y fija su primer movimiento como respaldo. La busqueda completa esta protegida; cualquier excepcion devuelve ese respaldo. Recurso limitado a 2400 nodos, revision temporal cada 16 nodos, profundidad maxima 2, presupuesto calculado con reloj restante, maximo 250 ms y reserva de 1.5 s. Solo se conserva una iteracion completa. `preparar` no hace trabajo pesado.
- Sin generador/simulador propio, no aplica la prueba diferencial de miles de posiciones. Pruebas unitarias en `test_strategy.py` cubren combate ganado/perdido, ataque recto ilegal, anotacion y salida del peon, tercer punto, ausencia de jugadas, apoyos de ambos ejemplos, victoria inmediata, defensa contra anotacion inmediata, respaldo con reloj agotado y respaldo ante excepcion.
- Herramientas: `herramientas/torneo_paralelo.py` ejecuta parejas de dos colores con semillas distintas, `taskset` y hasta nucleos disponibles menos uno; fuerza `--sin-docker`, y solo acepta salidas dentro de la carpeta. `herramientas/resumir_resultados.py` resume por color W-L, puntos, jugadas, relojes maximos e incidencias.
- Congelado de estrategia: `versiones/v1/strategy.py`; SHA-256 coincide con `strategy.py` al congelar: `ea137a184dcb10bbeacb3b7caf440e7ac0eaada09e3068edcde2748365309f0c`.

### Pruebas de v1

| Prueba | Resultado | Observaciones |
|---|---|---|
| `python -m unittest discover -s estudiantes/frame_perfect_push -p 'test_*.py' -v` | 13/13 pasan | Reglas, apoyos, tactica y fallback |
| `frame_perfect_push` vs `random`, 10 partidas | 10 victorias, 0 derrotas | 5 por color; 0 ilegales y 0 otras fallas |
| Reloj en humo vs `random` | Maximo de partida 15.370 s (blancas), 12.099 s (negras) | Maximo por decision: 0.447 s / 0.259 s |
| v1 vs `montecarlo`, 20 partidas | 0–20 para v1 | 10 por color; 0 ilegales/otras fallas; reloj v1 max. 9.647 s blancas y 8.844 s negras |

Comando del humo: `herramientas/torneo_paralelo.py random --bot frame_perfect_push --partidas 10 --semilla 20261108 --paralelo 3`. Se uso `--sin-docker`, fijacion a CPU y reloj oficial, pero no hubo limite de memoria. Registros: `resultados/paralelo_frame_perfect_push_vs_random_20261108/`.

Comando de referencia: `herramientas/torneo_paralelo.py montecarlo --bot frame_perfect_push --partidas 20 --semilla 20261109 --paralelo 3`. Registros en `resultados/paralelo_frame_perfect_push_vs_montecarlo_20261109/`.

El motor se eligio por ser la API permitida y fuente de verdad, con generacion Numba y desenlaces exactos; la implementacion evita duplicar reglas. El tiempo maximo por decision de v1 en el humo fue inferior a 0.5 s.

## Fase 3: expectiminimax v2

- Nodos MAX/MIN alternan por `estado.turno`; cada ataque es un nodo CHANCE que evalua completamente ambos resultados con las probabilidades exactas de `estado.siguiente(j)`. No se simulan dados. Alfa-beta se propaga solo por jugadas deterministas MAX/MIN; las llamadas bajo CHANCE reciben ventana completa `(-inf, +inf)`, por lo que ningun resultado de azar se poda.
- Valores terminales: `+1_000_000 - distancia` al ganar y `-1_000_000 + distancia` al perder; se prefiere victoria temprana y derrota tardia. Iterative deepening 1–4 conserva la jugada de la ultima iteracion completa.
- Recursos: maximo 2400 nodos, deadline adaptativo segun reloj y material, tope 300 ms, reserva 1.5 s, chequeo cada 4 nodos y tambien durante el prepaso de victorias garantizadas. Respaldo legal fijado antes de buscar; toda excepcion devuelve ese respaldo. `preparar` no compila ni carga trabajo pesado.
- Orden de jugadas: anotacion inmediata y ataques con mayor probabilidad primero, centro como desempate; en iterative deepening se busca primero la mejor jugada completa anterior. `PESOS` final: puntos 650, material 16, avance cuadratico 300, apoyo 8, ataque 16, riesgo 44, movilidad 1.2 y amenaza 300.
- El motor proporciona legalidad, transiciones y azar exacto; no se implementaron generador ni simulador propios. No aplica la prueba diferencial en miles de posiciones porque no se duplican reglas.
- Unit tests finales: 18/18 pasan. Cubren resultado ganado/perdido de ataques, ataque recto ilegal, anotar y retirar peon, tercer punto, sin jugadas, apoyos de ambos ejemplos, respaldo con excepcion/reloj agotado, prioridad de victoria, defensa de anotacion inmediata, orden central, monotonicidad de avance, distancias terminales, CHANCE exacto y metricas.
- Humo final v2 contra `random`: 10–0 (5 por color), 0 ilegales/otras fallas. Maximo de reloj en una partida: 12.967 s; maximo por decision: 0.438 s; profundidad maxima observada: 2; NPS maximo: 3,871.
- Filtros contra `montecarlo`: todas las iteraciones se reportan, no se ocultan las fallidas. El v2 inicial hizo 1–39 en 40; ajuste de avance/material, 5–35 en 40; avance central/movilidad, 4–36 en 40; presupuesto ampliado, 2–38 en 40; ordenamiento tactico, 5–35 en 40. Todas tuvieron 0 fallas tecnicas. El protocolo manda detener/corregir en esos resultados (≤24 victorias); no se consiguio una version que supere el filtro.
- En los 40 juegos del ordenamiento tactico, `frame_perfect_push` anoto 63 puntos (1.58 por partida) y `montecarlo` 109 (2.73 por partida); explica parte del 5–35, pero no se atribuye causalidad a un solo rasgo.
- Comparacion final actual v2 vs snapshot v1: 40–0 para v2, 20 victorias por color, 0 fallas. Reloj maximo v2: 12.725 s; maximo por decision 0.433 s; profundidad maxima 3; NPS maximo 5,282. Registros: `resultados/v2_vs_v1_20261360/`.
- Filtro contra `montecarlo` de la ultima reponderacion final (material 16, avance 300) no se ejecuto. El dato de v2-v1 no demuestra que esa version le gane a `montecarlo`.
- Snapshot v2: `versiones/v2/strategy.py`; SHA-256 `a6a0840f156a18f82e5e300d5356461b27497b51a2de68f72ed078f633c08c81`.
- El lanzador se corrigio para enviar todos los lotes al executor antes de esperar; verificamos 3 CLI concurrentes fijados a CPU 0,1,2 y CPU 3 libre. Las partidas v2–v1 se ejecutaron con `--sin-docker`, reloj contado pero sin cgroup de memoria/CPU.

## Preguntas para el curso

- Confirmar el procedimiento para que la excepcion autorizada de un integrante sea aceptada por el validador/PR; `./torneo validar` normal sigue marcandola como error.
- Confirmar una ruta admitida para guardar resultados de pruebas Docker dentro de la carpeta del equipo, sin modificar el wrapper ni archivos fuera de ella.

## Siguiente paso

La Fase 3 se cierra con v2 congelado y commit local. Antes de afirmar ventaja competitiva o ajustar la estrategia otra vez, ejecutar un nuevo filtro reproducible contra `montecarlo` con esta exacta reponderacion y respetando el criterio de 40 partidas. Las pruebas actuales fueron `--sin-docker`, no miden memoria/CPU cgroup ni reemplazan la validacion oficial.

## Fase 4 (replanteada): de expectiminimax a Monte Carlo con bandido (v5)

Diagnóstico: v2 (heurística + expectiminimax en Python) no pasaba de 5/40 contra `montecarlo`; la
tabla de transposición en curso (guardada en `versiones/v2_tt/fase4_tt.patch`) no cambia eso: la
búsqueda llega a profundidad 2-3 con ~4k nodos/s y la heurística evalúa peor que las partidas al
azar. Se cambió de enfoque y se midió cada intento (20 partidas oficiales contra `montecarlo`
salvo que se diga otra cosa; `--sin-docker`, reloj real, 3 procesos fijados a CPU):

| Versión | Idea | Resultado |
|---|---|---|
| v3a | MCTS (UCT) con nodos de azar, 2 simulaciones del motor por hoja | 2-7 (cortado) |
| v3b | v3a + simulación informada propia (numba: anotar / capturar al peón que va a anotar) | 0-3 (cortado) |
| v4 | Monte Carlo plano + UCB1 en la raíz + azar exacto + simulación informada | 1-5 (cortado); duelo rápido vs v4r 7/20 |
| v4r | v4 con `simular_al_azar` del motor (20 por tirada) | **14-6** (B 4-6, N 10-0) |
| v5 | v4r limpio + desenlace que deja al rival ganar seguro vale 0 | **12-8** (B 5-5, N 7-3); duelo rápido 13/20 |
| v6 | v5 + segunda fase: mejor respuesta del rival para las 4 mejores jugadas | duelo rápido vs v5 3/20 (y 2 derrotas por reloj) |

Acumulado v4r + v5 (misma idea): 26-14 = 65%. Sin jugadas ilegales ni excepciones. Reloj máximo
25.0 s de 30; decisión máxima 1.23 s.

Conclusiones:
- La simulación informada empeoró: era ~2.5 veces más lenta y la política sesgada.
- MCTS profundo con pocas simulaciones por hoja sigue al ruido; el promedio de 20 partidas por
  tirada en la raíz es más estable.
- La respuesta explícita del rival (v6) con estimaciones ruidosas sesga a la baja (el mínimo de
  promedios ruidosos) y gasta el presupuesto.

`strategy.py` = v5 (commit 7c94a83). Pruebas: `python -m unittest test_strategy` → 17/17.
Herramienta nueva: `herramientas/duelo.py` (duelo en un proceso, reloj escalado, solo para cribar).

## Fase 4 (cont.): v8 y validación final

- v8 = v5 con 40 partidas por tirada, ronda inicial barata cuando el presupuesto no alcanza
  (4 por tirada) y `_gana_seguro` barato (una generación de jugadas). Antes, con 2 puntos, el chequeo
  costaba decenas de ms por jugada y causó una derrota por reloj con reloj de 4 s.
- Cribado (duelo, reloj 6 s, 30 partidas): 40/tirada vs 20/tirada 22/30; 10 vs 20: 10/30;
  80 vs 40: 14/30; C=0.3 vs 0.5: 16/30; v7 (rival anota si puede) vs base 16/30.
- Oficiales (reloj 30 s): 40/tirada vs montecarlo 23-17; 40/tirada vs v5 22-18.
- **Validación final v8**: 100 vs montecarlo **58-42** (B 27-23, N 31-19; IC95 48-67 %), 10 vs
  random 10-0. Reloj máx. 25.5 s, decisión máx. 1.24 s, 0 ilegales, 0 fallas.
- `./torneo validar`: único error, 1 integrante (aceptación manual del profesor). `resultados/`
  está ignorado por git; localmente hace que el validador cuente >2000 archivos.
- `duelo.py` se sacó de la carpeta (el validador marca su carga dinámica para revisión manual);
  copia local fuera del repo.
- Commit: ver `git log` (v8 = bb36b2a, más informe).

## Ronda de mejoras (rama trabajo/mejoras-v9), 2026-11-08

`strategy.py` sigue siendo v8: ninguna variante la superó con reloj real.

| Variante | Prueba | Resultado |
|---|---|---|
| v9 (más reloj + extensión en empates) | oficial vs v8, 40 | 14-26; reloj máx 27.2 s |
| v11 (tope 1.0 s) | oficial vs v8 | 10-14 (cortado) |
| C_UCB 1.0 | duelo 6 s vs v8, 30 | 15/30 |
| v10 (simulación decisiva + antidecisiva, numba) | duelo 6 s vs v8, 30 | 15/30 |
| v10d (solo decisiva) | duelos 6 s vs v8 | 20/30 y 21/30 |
| v10d | duelo 15 s vs v8, 24 | 13/24 |
| v10d | oficial vs v8, 40 | 21-19 |
| v10d | oficial vs montecarlo, 40 | 22-18 (v8: 58-42 en 100) |

Lección: el duelo con reloj reducido exagera las mejoras que dependen de tener pocas
simulaciones; confirmar siempre con reloj real. v10d además suma ~2-3 s de compilación numba
en `preparar` (~4.5-5 s en total en esta máquina).

## v12: cota de amenaza en dos jugadas + libro de aperturas

- Análisis de las 42 derrotas de la validación de v8: ~4 ataques por bando; caída decisiva a mitad
  de partida; 17/42 sin combate (rival con 2 puntos y peón imparable). Ver INFORME §7.
- `_supervivencia`: con el rival en turno y 2 puntos, cota exacta de sobrevivir a su avance a la
  fila previa a su meta; recorta el valor simulado. Duelos vs v8: 21/30 (10 s), 20/30 (15 s);
  oficial vs v8: 22-18.
- Libro `data/libro.json`: 1318 posiciones (B 1-3, N 1-2), 60/30/15/10 s por posición, ~1.6 h con
  3 procesos (`herramientas/generar_libro.py`). Se usa en 2-3 jugadas por partida (también en Docker).
- v12 + libro parcial vs v8: 23-17. **Validación final v12: 64-36 vs montecarlo (IC95 54-73 %)**,
  10-0 vs random, reloj máx. 25.3 s, 0 fallas. Docker: preparar 0.0 s. Validador: solo el error de 1
  integrante.

## Cierre (2026-11-08): la entrega sigue siendo v12

Intentos posteriores, todos oficiales con reloj real (80 partidas contra v12 con libro):
v13 carreras exactas 41-39; v14 + simulación decisiva 33-47; v15 más reloj 36-44; v16 cota de
una jugada más 34-46; v17 derrotas suaves 38-42; v12s (v12 + derrotas suaves) 42-38; control v12
vs v12 40-40. Ninguno supera a v12 de forma medible (σ ≈ 5.6 puntos con 80 partidas). El código de
v13-v17 está en las ramas `trabajo/imparables` y `trabajo/libre`; aquí solo se documenta.

## v18: corrección aprendida (rama trabajo/correccion), 2026-11-08

- Datos: 725 partidas de v12 contra sí misma con reloj real (3 procesos, 3 h), 49 027 posiciones
  (rasgos de la posición tras nuestra jugada sin combate, valor simulado, resultado).
- Regresión logística L2=30: pérdida log en partidas de prueba 0.529 vs 0.549 (solo valor simulado).
  Pesos principales: material (+0.67 propio, −0.81 rival), puntos (±0.5), huecos en nuestras dos
  últimas filas (−0.09).
- Uso: elige entre jugadas con ≥10 % de las tiradas de la más simulada por valor corregido.
- Oficial vs v12: 110-90 en 200 (55 %); con las pruebas previas, 153-127 en 280. Regla fijada antes de
  medir: ≥57 % en 200 → no se cumple. **Entrega: v12** (`entrega/final`).
