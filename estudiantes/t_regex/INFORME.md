# Informe: t_regex

## 1. Integrantes
- Ariadna Quijano (219656)
- Ben Zimbrón (212907)

## 2. Idea en un párrafo
Nuestro bot es un **expectiminimax con poda alfa-beta** que trata cada ataque como un nodo de
azar con sus dos resultados exactos (gana con probabilidad `p`, pierde con `1 − p`). La búsqueda
está compilada con numba (~1 millón de nodos por segundo en un núcleo) y profundiza de forma
iterativa mientras alcanza el reloj. En las hojas usa una **evaluación lineal** de rasgos del
tablero (puntos, peones por fila de avance, apoyo, peones libres y «amenazas seguras»)
convertida a una probabilidad de ganar. Antes de pensar consulta un **libro precalculado** de
1.44 millones de posiciones (aperturas y posiciones que salieron en miles de partidas de
práctica), cada una resuelta offline con una búsqueda mucho más profunda que la que cabe en el
reloj.

## 3. Búsqueda (algoritmo y por qué)
**Expectiminimax en forma negamax**, en `busqueda.py`:
- **Nodos de azar exactos.** Un ataque tiene solo dos resultados con `p` conocida, así que su valor
  es `p·v_gana + (1−p)·v_pierde`, sin muestrear. Elegimos esto sobre MCTS porque el azar se
  calcula sin varianza, el presupuesto es pequeño (30 s por partida) y el juego tiene trampas
  tácticas cortas que un mínimo detecta en cuanto llega a esa profundidad, mientras que un
  promedio de simulaciones las corrige poco a poco.
- **Poda alfa-beta con Star1** en los nodos de azar: como todo valor está en [−1, 1], a veces
  el primer resultado basta para saber que el promedio cae fuera de la ventana y no se busca el
  segundo. Verificamos que da exactamente el mismo valor que expectiminimax sin poda en 60
  posiciones al azar (`herramientas/probar_busqueda.py`).
- **Tabla de transposición** de 2^22 entradas (~84 MB; con 2^20 se llenaba y buscaba 11 % más
  nodos) con hash Zobrist (las mismas tablas que el motor, así que nuestro hash coincide con
  `estado.clave`; el libro usa la misma llave).
- **PVS:** tras la primera jugada de cada nodo preguntamos con ventana mínima si alguna la
  supera y solo entonces buscamos completo; es exacto y ahorra 17 % de nodos a profundidad 9.
- **Orden de jugadas:** la de la tabla, anotar, ataques por probabilidad, *killer moves* y
  heurística de historia. En la raíz ordenamos por avance (ver sección 8: esto importó mucho).
- **Profundización iterativa** con límite de nodos: si una iteración se aborta, se usa su mejor
  jugada si alcanzó a terminar la primera (la mejor de la iteración anterior).
- **Quedarse sin jugadas es perder, también en las hojas.** Antes, una posición sin jugadas
  legales al llegar a la profundidad límite se evaluaba como cualquier otra y podía parecer
  buena estando ya perdida. Ahora, antes de evaluar una hoja, revisamos si hay jugadas; si no
  hay, vale −1. Lo encontró y corrigió Ariadna (`herramientas/test_regresion_terminal.py`).
  Con el árbitro, contra la versión sin la corrección: 45 contra 42 de 64 partidas ante
  montecarlo con los mismos dados y 14–14 en el duelo directo. No juega peor y arregla un error
  real.
- Hacer/deshacer en sitio y una sola función recursiva (la raíz y los nodos de azar están dentro
  de `negamax`), lo que bajó la compilación de numba de 6.7 s a ~3 s.

Profundidad típica en partida: 6 en promedio (de 5 a 8). El código tiene una búsqueda de
quietud (`QS_PROF`, `QS_PMIN`), pero está **apagada** (`QS_PROF = 0`): ver sección 8.

## 4. Función de evaluación
Para quien está en turno calculamos 17 rasgos (los simétricos como «lo mío − lo del rival»):
puntos; peones según cuántas filas han avanzado (0 a 7); peones con apoyo; peones libres (ningún
rival en el cono por delante) y su avance; ataques favorables (`p ≥ 0.65`) propios y del rival;
amenazas seguras propias y del rival; y un sesgo. El valor es `0.85·tanh(z/2)` con
`z = pesos · rasgos`, que se lee como `2·P(ganar) − 1`. Ganar de verdad vale ~1.

**Amenaza segura:** un peón a un paso de anotar sin rivales en las dos casillas desde donde lo
podrían atacar. Esas casillas están en la fila inicial del rival, a la que ya no puede entrar
ningún peón, así que es un punto garantizado. Si las amenazas seguras de un bando le alcanzan
para llegar a 3 antes que el otro, la posición vale como carrera decidida (±0.89). Esto salió de
analizar partidas perdidas contra montecarlo: el bot creía ir ganando (0.77) con 5 peones de
ventaja mientras el rival tenía dos peones imparables.

**Cada peón cuenta una sola vez y su valor crece hasta el punto.** Un peón que es amenaza
segura cuenta solo como tal (1.3), no además como avanzado o libre; los pesos de «libre» son
chicos (0.3 + 0.05 por fila), así que un peón libre en la fila 6 de avance vale 1.25 < 1.3 < 1.5
(un punto anotado). Antes, un peón a un paso de anotar sumaba 3.6 (avanzado + libre + amenaza)
y anotar «costaba» 2.1: el bot no anotaba. En una partida tuvo dos puntos garantizados durante
24 jugadas sin cobrarlos mientras el rival anotaba 2. Además, en la raíz las jugadas que anotan
van primero, así que en un empate cobra ya en vez de posponerlo. Con esto las derrotas por
«colapso» (íbamos ganando según la evaluación y perdimos) bajaron de 35 a 24 por cada 200
partidas contra montecarlo.

**Pesos** (`PESOS_A_MANO` en `busqueda.py`): puntos 1.5; peón por avance 0.8 (fila de casa),
0.5, 0.5125, 0.525, 0.55, 0.5875, 0.65, 0.75; apoyo 0.05; libre 0.3 + 0.05 por fila; amenaza
segura ±1.3. Los pusimos a mano y los cambiamos uno por uno midiendo partidas. El cambio más útil
fue valorar más el peón de la fila de casa (0.8 en vez de 0.5): es el único que puede atacar a
un rival que llega a un paso de anotar (63 % contra la versión anterior en 400 partidas). Los
pesos afinados automáticamente no se usan (sección 8).

## 5. Precomputación (qué y cómo regenerarla)
`data/libro.npz` (13.0 MB): **1 435 427 posiciones** en dos arreglos, `claves` (hash Zobrist de
tablero + turno + puntos) y `jugadas` (`origen·81 + destino`). Al importar se carga a un
diccionario; en partida, si la posición está y la jugada es legal, se juega sin buscar y sin
gastar reloj. Cada jugada salió de una búsqueda de 2 a 5 millones de nodos (profundidad media ~8,
contra ~6 en vivo). Importar más `preparar` tarda ~4–4.7 s y el proceso ocupa ~420 MB
(medido en Docker con 1 núcleo y 1024 MB).

El libro se construye en capas; cada herramienta agrega al mismo archivo, guarda cada 15 min y
se puede interrumpir sin perder nada:
1. `herramientas/generar_libro.py --salida data/libro.npz` (~80 min con 8 núcleos): árbol fijo
   de aperturas hasta el ply 8 para ambos colores (todas las primeras jugadas del rival, luego
   la mejor según una búsqueda profunda y 3 más según una corta), 5 millones de nodos por
   posición.
2. `herramientas/ampliar_libro.py --rondas N --plies 100 --temprano 20`: juega rondas de
   partidas de nuestro bot contra 13 rivales (nuestras versiones v1…v8, montecarlo con 100,
   500 y 1700 muestras, random y él mismo) y agrega las posiciones donde nos tocaba mover que
   no estaban, se repitieron o son tempranas. Corrió varias noches; es la capa que más se usa
   (58 % de nuestras jugadas antes del ply 100 caen en el libro en partidas de prueba). Una
   última corrida solo contra la réplica de montecarlo (`--rivales montecarlo:1700`, 4
   instancias de 1 núcleo, ~3 h) agregó 10 415, pero ahí el acierto se quedó en ~9 %:
   montecarlo juega con mucho azar y casi ninguna posición se repite.
3. `herramientas/libro_de_partidas.py --partidas <carpetas de resultados>` (y `--posiciones`
   con lo que guarda `validar_todos.py`): lo mismo, a partir de partidas ya jugadas con el
   árbitro real o la arena.
4. `herramientas/explorar_arbol.py --rondas N`: recorridos al azar hasta el final de la
   partida, sin rival (jugadas legales y dados al azar para los dos bandos), guardando las
   posiciones tempranas y las 8 últimas antes de que alguien gane, donde hay una jugada que gana
   o una que evita perder. Las dos últimas corridas (~4 h con 8 núcleos y ~3.5 h con 4 instancias
   independientes, cada una en su archivo y luego juntadas) agregaron 109 683 y 41 843.

Para regenerarlo: correr 1 y después 2 (y opcionalmente 3 y 4) durante el tiempo que se quiera.
Las versiones viejas que usa el paso 2 como rivales son copias de nuestra carpeta en commits
anteriores. Hay que regenerarlo si cambia la evaluación. No hay `data/pesos.npy`: si existiera,
`busqueda.py` lo usaría en lugar de los pesos a mano.

Herramientas de medición (no corren en la liga): `arena.py` enfrenta dos versiones de la
búsqueda con presupuesto fijo de nodos, los mismos dados y ambos colores, y también una réplica
de `montecarlo`; usa `importlib` solo para cargar dos versiones de `busqueda.py` a la vez.
`validar_todos.py` compara un candidato con el bot actual en partidas idénticas contra todas
las versiones y montecarlo, intercalando partidas contra random. Todo se corre dentro de la
imagen de Docker del torneo; los comandos están en cada archivo.

## 6. Manejo del reloj
Reservamos 2 s que nunca gastamos. Del resto, cada jugada recibe lo disponible entre las jugadas
que creemos que faltan (`max(10, 42 − jugadas_hechas)`; medimos partidas de 39 a 49 jugadas
propias). La profundización se detiene si la siguiente iteración no cabe (estimamos su costo con
el crecimiento de la anterior) y se aborta por límite de nodos al doble del presupuesto o al
30 % de lo disponible. Las jugadas del libro no gastan reloj. En 100 partidas contra montecarlo
con el árbitro usamos 18.4 s de 30 en promedio (máximo 26.4, nunca perdimos por tiempo). La
compilación de numba (~3 s) ocurre en `preparar`, fuera del reloj.

## 7. Resultados vs random y montecarlo
Con el árbitro del torneo (`./torneo jugar`, reloj de 30 s, Docker, 1 núcleo por bot):

| Versión | Rival | Partidas | Resultado |
|---|---|---|---|
| v1 (expectiminimax básico) | random | 8 | 8–0 |
| v3 | montecarlo | 10 | 6–4 |
| v4 (amenazas seguras) | montecarlo | 10 | 6–4 |
| v7 (evaluación consistente, libro chico, PVS) | montecarlo | 20 | 16–4 |
| **final (v7 + libro de ~1 millón)** | montecarlo | 130 | **97–33 (74.6 %)** |
| **final** | random | 30 | **30–0** |
| **final** | v1 | 30 | **22–8** |
| final, antes de la corrección de terminales | montecarlo | 604 | 446–158 (73.8 % ± 3.5) |

En la arena (100 000 nodos por jugada, mismos dados, ambos colores), el bot final contra
cada versión anterior y montecarlo (200 partidas cada uno): v1 77 %, v2 82.5 %, v3 73 %, v4 63 %,
v5 55 %, v6 58.5 %, v7 49.5 %, v8 51 %, montecarlo con 1700 muestras 62 %. Contra random,
450 de 450.

Entre versiones propias (arena, mismos nodos): v3 sobre v2 59 %, v5 sobre v4 63 %, v6 (libro)
sobre v5 56 %. Las Blancas ganan entre 47 % y 75 % según la versión: mover primero pesa.

## 8. Qué intentamos y no funcionó
- **Búsqueda de quietud** (en las hojas seguir solo los ataques con `p ≥ 0.5` y las jugadas que
  anotan, hasta 4 plies más): en la arena, con el mismo número de nodos, ganaba +4.3 ± 2.1 puntos
  y +3.4 con dados nuevos. Con el reloj real cada nodo cuesta más (15–19 % menos nodos por
  segundo, media jugada menos de profundidad) y contra montecarlo quedó en 89–41 (68.5 %)
  contra 97–33 (74.6 %) sin ella, en 130 partidas cada una. Está apagada. Lección: medir con
  nodos fijos no cobra lo que cuesta cada nodo.
- **Pesos afinados automáticamente** (un peso a la vez, ±25 %, 200 partidas por candidato).
  Con la evaluación de v6, 17 candidatos y ninguno pasó de 52.5 %. Con la de v7 se adoptaron 3
  cambios (v8: 62.5 % contra v7 y 73 % contra la réplica de montecarlo), pero v8 solo empataba
  con v1 mientras v7 le ganaba 67.8 %: se había ajustado a jugar contra sí mismo. Volvimos a los
  pesos de v7. Una segunda vuelta adoptó 4 cambios que le ganaban cada uno a su predecesor
  (53–57 %) y juntos perdieron 41 % contra v8. Por eso ahora todo candidato se mide contra
  todas las versiones y montecarlo, con random intercalado (`validar_todos.py`).
- **Pesos de ataques favorables** (±0.2): +2.5 puntos en la arena, pero nada sobre la quietud
  (+0.9 ± 2.0). No se usa.
- **¿Ayuda el libro grande?** Lo medimos con y sin libro: 48 % ± 3.5 en la arena entre ellos;
  contra montecarlo con el árbitro 21–9 con libro y 23–7 sin él (30 partidas). La ganancia es
  pequeña, salvo contra v1 en la arena (78 % contra 70 %). Lo dejamos porque no cuesta reloj.
- **Explorar el árbol al azar** para el libro: agrega muchas posiciones, pero de 9 559 jugadas
  nuestras en partidas reales solo 1 cayó en una posición de esta capa (0.01 %), contra 58 %
  de la capa de partidas contra rivales. Las partidas al azar llegan a posiciones que un rival
  razonable casi nunca produce.
- **Killers e historia en la raíz** (40.5 % contra la versión anterior, 600 partidas): la
  evaluación tiene muchos empates y la búsqueda elige la primera jugada que ve. Con historia, el
  desempate dejó de favorecer el avance. Usarlas solo dentro del árbol dio 59 %.
- **Pesos por regresión logística** sobre 265 mil posiciones de autojuego: predicen mejor el
  ganador (log-loss 0.588 → 0.552) pero juegan peor (45 %): aprenden qué rasgos acompañan a las
  posiciones ganadoras, no qué jugadas llevan a ganar.
- **LMR (reducir jugadas tardías):** reducir 1 jugada perdió 36 % aun buscando más profundo; la
  rama reducida termina con el otro bando en turno y su evaluación no es comparable. Reducir 2
  quedó en 51 %. Está apagado.
- **Carrera generalizada** (peones libres a *n* pasos): 62.0 % contra 62.5 % sin ella, y subía
  la compilación; la quitamos.
- **Más reloj** (reserva 1.5 s y suponer 36 jugadas): 20–20 contra la versión anterior en 40
  partidas con el árbitro y se acercó a 27 s de 30; volvimos al reparto anterior.
- **Cobertura de la meta**, **peones avanzados con más peso**, **escalar la confianza de la
  evaluación** y **contar el punto inmediato del bando en turno**: entre 41 % y 52 %, neutrales
  o peores.

## 9. Fuentes y uso de IA
- Russell y Norvig, *Artificial Intelligence: A Modern Approach*, capítulo de juegos
  (expectiminimax, alfa-beta).
- Ballard (1983), «The *-minimax search procedure for trees containing chance nodes»
  (poda Star1).
- Material del curso: «Radiografía: un árbol, todos los algoritmos» y «Elegir el algoritmo».
- Programamos con **Claude Code** (agente de IA): nosotros decidimos la estrategia y el plan por
  etapas (contra sí mismo, random, montecarlo, todas las versiones), y el agente escribió el
  código, las pruebas y las herramientas de medición y del libro, y analizó las partidas
  perdidas. Cada cambio se aceptó solo si ganaba partidas medibles.
