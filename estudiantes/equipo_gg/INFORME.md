# Informe: equipo_gg

## 1. Integrantes

- Genevieve Martinez Teran — clave única 202728
- Ismael Cabrera Arroyo — clave única 217632

## 2. Idea en un párrafo

Nuestro bot busca con **expectiminimax**: alfa-beta para nuestras jugadas y las del rival y un
**nodo de azar** en cada ataque, que promedia sus dos resultados con la probabilidad real
`p = 0.5 + 0.15·Δapoyo`. Las hojas se valoran con una **regresión logística** de 55 rasgos
(puntos, avance, apoyos, ataques disponibles, peones imparables, interceptores, carreras y
amenazas inmediatas). La evaluación regresa `2·P(ganar) − 1`, así que promediar en los nodos de
azar es promediar probabilidad de ganar. Los pesos salen de partidas de autojuego y luego se
corrigen con 848 partidas reales contra `montecarlo` (`data/pesos.json`). Encima sumamos una
**ventaja asumida** de 1 log-momio a nuestro favor: la curva queda cóncava y el bot evita tiradas
de dado que no necesita. Todo el árbol corre en numba con profundización iterativa (5 a 9 capas
por jugada); las primeras 2–3 jugadas salen de un **libro de aperturas** (`data/aperturas.json`)
buscado offline a 8–11 capas contra todas las respuestas posibles del rival.

## 3. Búsqueda (algoritmo y por qué)

- **Por qué no Monte Carlo.** Con nuestra versión anterior (Monte Carlo con filtros) perdimos
  69 % de 800 partidas contra `montecarlo`. Las partidas al azar suponen que el rival responde al
  azar: no ven que un peón suelto se pierde ni que un hueco en la fila de casa deja pasar al
  rival. El juego tiene ~25 jugadas por posición y pocos ataques (~5 por bando por partida), así
  que una búsqueda en árbol con buena evaluación llega hondo.
- **Expectiminimax con alfa-beta (negamax).** Una jugada sin ataque es un nodo MAX/MIN normal; un
  ataque es un nodo de azar con dos hijos (gana con `p`, pierde con `1 − p`). Para podar debajo de
  los nodos de azar usamos **Star1** (Ballard, 1983): como todo valor está en `[−1, 1]`, de la
  ventana `(α, β)` del padre salen ventanas para cada hijo y a veces basta un hijo para cortar.
- **Profundización iterativa** con **tabla de transposición** (2^20 entradas, hash Zobrist propio),
  la mejor jugada de la capa anterior primero, **killers** e **historia** para ordenar.
- **PVS**: tras la primera jugada, ventana nula; si mejora α se repite completa.
- **LMR**: jugadas tranquilas tardías de peones lejos de la meta se buscan con 1–2 capas menos;
  si mejoran α se repiten completas.
- **Quietud** (hasta 3 capas): solo una anotación sin dado (son equivalentes), las anotaciones
  atacando y los ataques a peones rivales a uno o dos pasos de anotar; los ataques también con
  ventanas Star1. Una versión anterior exploraba todas con ventana completa y en posiciones con
  varios peones cerca de casa llegaba a 340 mil nodos a profundidad 3 (la profundidad real caía
  a 1–2 justo en los finales); la nueva reduce el peor caso a ~137 mil y el autojuego 7×.
- Reglas propias en numba sobre `int8[81]` (como `sistema/motor/nucleo.py`), verificadas contra el
  motor oficial en 200 partidas al azar: jugadas legales, `p`, aplicar/deshacer y hash coinciden en
  todas las posiciones. Velocidad: ~1 millón de nodos por segundo.
- Del motor oficial solo usamos su API pública y siempre regresamos una jugada de
  `estado.jugadas()`; si algo fallara hay un respaldo instantáneo (anotar, mejor ataque o avance).

## 4. Función de evaluación

`z = w · rasgos(posición)` y valor `= tanh(z/2) = 2·P(ganar) − 1`, siempre desde quien mueve
(«yo») contra su rival («él»). Los 55 rasgos (`NOMBRES_RASGOS` en `strategy.py`):

- **Puntos** de cada lado y si alguien tiene 2.
- **Avance**: peones por fila de progreso (diferencia en filas 0–4, por separado en 5–7).
- **Apoyos** por zona, peones avanzados sin apoyo, **ataques disponibles** con `p ≥ 0.65` y
  `p ≥ 0.5` de cada lado y el mejor ataque contra un intruso avanzado.
- **Corredores**: un peón es imparable si ningún rival está en su cono de intercepción (a `d`
  filas, distancia de columna `≤ d//2` si mueve su dueño y `≤ (d+1)//2` si mueve el rival; lo
  dedujimos de que el corredor avanza una fila por turno y el interceptor también). Con ellos, una
  **carrera**: cada jugada mueve un solo peón, así que el tiempo para ganar es la suma de las
  `3 − puntos` distancias más cortas.
- **Interceptores**: cuántos rivales hay en el cono de cada peón y una **carrera efectiva** con
  distancia + 3 por interceptor, para medir quién rompe antes.
- **Amenazas inmediatas**: peones a 1 o 2 pasos de anotar, si tienen casilla libre y si el rival
  puede atacarlos; **puntos virtuales** del rival (los suyos más las anotaciones que ya no puedo
  impedir) y si con eso llega a 3; **huecos en la fila de casa**.

**Cómo se obtuvieron los pesos** (sección 5): primero una regresión logística L2 con 248 mil
posiciones de autojuego (etiqueta: ganó quien mueve). Después, con partidas reales contra
`montecarlo`, una corrección regularizada hacia esos pesos con un término `b·s` (`s = +1` si
mueve nuestro bot, `−1` si mueve el rival). La primera corrección (228 partidas) bajó el log-loss
en validación cruzada de 0.4628 a 0.4308; la vigente usa 848 partidas (65,916 posiciones, 500 del
árbitro oficial) y lo baja de 0.4441 (autojuego) a 0.4223 (con `b`) y a 0.4033 (corregidos).
La corrección dice lo mismo que vimos en las derrotas: contra `montecarlo` importan más los huecos
en la fila de casa (−0.06 por hueco en autojuego puro, −0.13 en los pesos vigentes), los apoyos
cerca de casa (+0.01 → +0.05) y avanzar en bloque.
Ejemplos de pesos: «el rival llegará a 3 puntos» −3.4, «tengo 2 puntos y anoto ya» +3.9.

**Ventaja asumida.** Al valor le sumamos `VENTAJA = 1.0` log-momio a favor de nuestro color (en
contra cuando mueve el rival; la evaluación sigue siendo antisimétrica). Motivo medido: en las
derrotas contra `montecarlo` la suerte neta de dados era de −2 ataques en promedio y el bot tomaba
muchos ataques de 50 %. Contra un rival más débil una tirada pareja nos baja la probabilidad de
ganar; con la ventaja, `tanh` queda cóncava alrededor de las posiciones normales y la búsqueda
prefiere posiciones sin tiradas (las nuestras y las que le dejaríamos al rival). Sus ataques de
50 % bajaron de 69 a 47 en 40 partidas y los de 35 % desaparecieron. El ajuste con 848 partidas
estima `b = +1.04`, consistente con el 1.0 elegido.

## 5. Precomputación (qué y cómo regenerarla)

`data/` tiene dos archivos JSON que `preparar` lee de `config.carpeta_datos`. Si alguno falta o no
cuadra (por ejemplo, otros nombres de rasgos), el bot usa los pesos escritos en `strategy.py`
(los de la versión anterior) y juega sin libro.

- **`data/pesos.json`** (3 KB): los 55 pesos que usa el bot (`pesos`), la ventaja asumida, los
  nombres de los rasgos (el bot verifica que coincidan) y los pesos de autojuego de los que parte
  la corrección (`pesos_autojuego`).
- **`data/aperturas.json`** (≈110 KB): 1,111 posiciones → jugada. Con blancas cubre nuestras
  jugadas 1–3 y con negras la 1–2 contra **todas** las respuestas posibles del rival; cada posición
  se buscó con el mismo motor durante 6 s (nivel 1), 2 s (nivel 2) o 1.5 s (nivel 3): profundidad
  media 9.2 y mínima 8, contra ~8 en partida. No lo hicimos más largo porque no se usaría: en 500
  partidas contra `montecarlo`, desde nuestra 4.ª jugada ninguna posición se repitió. La clave es
  el tablero (81 casillas), quién mueve y los puntos.

Se regeneran con `herramientas/` (todo offline, desde la carpeta del equipo):

1. `python herramientas/autojuego.py datos --partidas 3000 --procs 2 --prof 3` — partidas del bot
   contra sí mismo (profundidad 3, de 0 a 10 jugadas al azar al inicio y 3 % de jugadas al azar)
   que guardan los rasgos de cada posición y si ganó quien mueve.
2. `python herramientas/ajustar.py datos_w*.npz --escribir data/pesos.json` — regresión
   logística (L-BFGS, validación separando partidas); guarda `pesos_autojuego`.
3. `python herramientas/ajustar_vs_rival.py --equipo equipo_gg --lam 0.3 <partidas> --escribir
   data/pesos.json` — corrección con partidas reales (`.json` de `./torneo jugar` o `.jsonl` de
   nuestro arnés); guarda `pesos` e imprime la validación cruzada y la ventaja estimada.
4. `python herramientas/generar_aperturas.py --procs 2` — el libro (~15 min con 2 núcleos).

Lo que hicimos: ronda 0 de autojuego con pesos a mano y 40 rasgos (4000 partidas, 352 mil
posiciones); ronda 1 con 55 rasgos (3000 partidas, 248 mil posiciones; validación: log-loss 0.531,
exactitud 0.70); una ronda 2 no mejoró (49.5 % en 200 partidas contra la ronda 1) y la descartamos.
Luego triplicamos a mano el peso de los huecos de casa (ver sección 7) y corregimos con partidas
reales: primero 228, al final 848.

## 6. Manejo del reloj

- Reserva fija de 2.5 s. Cada jugada tiene un **objetivo** = (reloj − reserva) / (42 − jugadas
  hechas, mínimo 10), la mitad en las primeras 8 jugadas (la apertura es tranquila y se llega a
  7–9 capas rápido), con tope de 1.5 s, y un **tope duro** = mín(2·objetivo, 12 % de lo que queda).
- Profundización iterativa: no empezamos otra capa si ya pasó 45 % del objetivo; dentro de numba
  el tope duro es un límite de nodos (con los nodos/s medidos en la misma partida) y la capa
  incompleta se descarta salvo que otra jugada ya haya superado a la anterior.
- Con menos de 0.4 s de reloj, búsqueda mínima. Ya en la reserva cada jugada cuesta ~0.05 s,
  así que la reserva alcanza para unas 40 jugadas más aunque la comunicación costara el doble.
- **preparar**: compilar numba toma ~6 s en una máquina y ~11 s en otra (500 partidas con Docker:
  mediana 11.2 s, máximo 12.7 s). Para no perder nunca por el límite de 20 s, compilamos en un
  hilo y esperamos hasta 14 s; si no terminara, `preparar` regresa igual y las primeras jugadas
  esperan hasta 4 s cada una con nuestro reloj. Si la posición está en el libro, juega la jugada
  del libro aunque la compilación no haya terminado; si no, el respaldo instantáneo. Con la espera
  anterior de 12 s, 10 de las 500 partidas usaron el respaldo al inicio (ganamos 9).
- Las jugadas del libro son instantáneas: el tiempo que ahorran se reparte en el resto.
- En 160 partidas contra `montecarlo` (reserva de 2 s): reloj medio 20.5 s, máximo 27.9 s
  (una partida de 135 jugadas). Por eso la versión final reserva 2.5 s.

## 7. Resultados vs random y montecarlo

Reloj real de 30 s por bando, colores alternados, dados del árbitro (misma clase `Dados`).

**500 partidas con `./torneo jugar` (Docker) contra `montecarlo`**, versión sin `data/` (pesos
corregidos con 228 partidas): **404/500 (80.8 %)**, 204/250 con blancas y 200/250 con negras;
intervalo de 95 %: 77–84 %. Marcadores al ganar: 186 veces 3–0, 120 veces 3–1, 98 veces 3–2.
Reloj medio 22.3 s, máximo 28.1 s. En esa máquina `preparar` tardó 11.2 s de mediana, así que en
151 partidas la primera jugada esperó a la compilación (hasta 4 s) y en 10 hubo jugadas de
respaldo; por eso esta versión espera 14 s en `preparar` y juega del libro mientras tanto.

**En nuestra máquina** (arnés propio que replica al árbitro sin Docker; cada semilla son dados
nuevos para ambos colores):

| Versión | Rival | Semilla | Partidas | Victorias | Con blancas | Con negras |
|---|---|---:|---:|---:|---:|---:|
| sin `data/` | `montecarlo` | 303 | 40 | 36 (90 %) | 17/20 | 19/20 |
| sin `data/` | `montecarlo` | 404 | 80 | 70 (87.5 %) | 36/40 | 34/40 |
| sin `data/` | `montecarlo` | 505 | 40 | 33 (82.5 %) | 15/20 | 18/20 |
| **con `data/` (final)** | `montecarlo` | 707 | 80 | **69 (86.2 %)** | 36/40 | 33/40 |
| sin `data/` | `random` | 606 | 20 | 20 (100 %) | 10/10 | 10/10 |

Con `data/`, el libro dio 2.5 jugadas por partida (3 con blancas, 2 con negras) y el reloj medio
fue 21.9 s (máximo 27.0 s). La diferencia entre 86 % y 87 % es ruido; lo que esperamos de
`data/` es una mejora chica en la apertura y no perder tiempo de reloj por la compilación.

Evolución contra `montecarlo` (las de 40 partidas con la misma semilla de dados, 303):

| Versión | Partidas | Victorias |
|---|---:|---:|
| Monte Carlo con filtros (versión anterior, registros) | 800 | 31 % |
| Expectiminimax + pesos a mano | 8 | 75 % |
| + regresión logística (ronda 0) | 20 | 75 % |
| + interceptores, carrera efectiva, puntos virtuales, PVS/LMR (ronda 1) | 40 | 75 % |
| + ventaja asumida 1.0 y reloj parejo | 40 | 82.5 % |
| con ventaja 1.6 en vez de 1.0 | 40 | 82.5 % |
| + quietud corregida | 40 | 80 % |
| + peso de huecos en casa ×3 | 40 | 87.5 % |
| + corrección con partidas contra montecarlo | 40 | 90 % |

Con 40 partidas el error estándar es de ±6 puntos, así que los pasos chicos son ruido; lo que sí
se repite es que cada cambio defensivo subió el porcentaje. En 160 de nuestras partidas, la
suerte neta de dados (ataques ganados de más, nuestros menos los del rival) fue de −2.1 en
promedio en las 21 derrotas y de +0.35 en las 139 victorias; solo 1 derrota tuvo suerte neta
positiva. En las 500 oficiales, perdiendo atacamos 6.5 veces por partida y ganando 2.3: las
derrotas son partidas que se volvieron de dados.

## 8. Qué intentamos y no funcionó

- **Monte Carlo plano con UCB** (versiones anteriores): 31 % contra `montecarlo` en 800 partidas.
- **Reloj cargado al inicio**: gastaba 0.7–0.8 s en las primeras 20 jugadas y < 0.1 s después de
  la 40, justo cuando se decide la carrera; perdimos partidas «ganadas» buscando a profundidad 2.
- **Pesos solo de autojuego**: con 2 puntos contra 1 la evaluación decía 99 % aunque el rival
  tuviera dos peones entrando por huecos de nuestra fila de casa. En autojuego los huecos casi no
  importan (el rival juega igual que nosotros); contra `montecarlo`, sí.
- **Más rondas de autojuego**: la ronda 2 cambió mucho los pesos (rasgos correlacionados) pero
  empató 99–101 contra la ronda 1.
- **Más ventaja asumida** (1.6): igual que 1.0.
- **Quietud con todas las anotaciones y ataques a ventana completa**: explotaba (ver sección 3).
- **Libro de aperturas más largo**: no se usaría; desde nuestra 4.ª jugada las posiciones ya no se
  repiten (el rival tiene ~25 respuestas por jugada y `montecarlo` las usa casi todas).

- **Bot anterior del equipo (versión 4)**: expectimax con Star1 y una evaluación de 6 rasgos con
  pesos a mano (peones, puntos, avance, apoyo, peones pasados, anotar sin dado). Le ganaba 14–6 a
  `montecarlo`, pero perdió 2–4 contra este bot en las partidas que alcanzamos a jugar; nos
  quedamos con este.
- **Agregar los peones pasados de la versión 4** a esta evaluación (sumados a los log-momios con
  peso 0.10, a mano, sin reajustar): 7–13 contra este bot sin el cambio en 20 partidas (semillas
  81 y 82). Esta evaluación ya mide algo parecido (carreras y distancias con interceptores) y un
  peso puesto a mano descompone el ajuste conjunto. No lo dejamos.

## 9. Fuentes y uso de IA

- Reglas, motor y árbitro del curso; el código de reglas en numba sigue `sistema/motor/nucleo.py`.
- B. W. Ballard (1983), «The *-minimax search procedure for trees containing chance nodes»,
  *Artificial Intelligence* 21: poda Star1 en nodos de azar.
- Russell y Norvig, *AIMA*, capítulo de juegos (expectiminimax, alfa-beta, profundización
  iterativa, tablas de transposición).
- Lorentz y Horey (2013), «Programming Breakthrough»: importancia de la defensa y de los peones
  imparables en Breakthrough.
- Usamos un agente de IA (Claude) para analizar los registros de partidas, escribir y probar el
  código, y correr los torneos de prueba; revisamos las decisiones y los números del informe.
