# Informe: frea

## 1. Integrantes

- Luis Fernando Rodríguez Retama
- Emiliano Aguilar Castillo

## 2. Idea en un párrafo

Nuestro bot busca con expectiminimax: en sus jugadas y en las del rival elige como minimax, y en cada
ataque promedia los dos resultados del dado con su probabilidad real. La búsqueda corre en numba sobre
una copia propia de las reglas y llega a 6.7 plies de media. Las hojas se valoran con una evaluación
lineal de 30 rasgos, cuyos pesos ajustamos offline con regresión logística sobre posiciones de autojuego.
El valor de una posición se interpreta como `2 * P(ganar) - 1`, de modo que promediar en los nodos de azar
equivale a maximizar la probabilidad de ganar. Buena parte de la evaluación mide la carrera al tercer
punto: si un corredor puede llegar y qué tan fácil es interceptarlo.

## 3. Búsqueda (algoritmo y por qué)

Usamos negamax con poda alfa-beta pruning y nodos de azar en los ataques (`_buscar` en `mi_motor.py`). Lo
elegimos porque los nodos de azar son binarios y solo aparecen en ataques, y porque un prototipo de
MCTS perdió claramente contra él (sección 8).

- **Nodos de azar con Star1 (Ballard, 1983).** Como los valores están acotados entre -1 y 1, al buscar
  un resultado del dado suponemos el extremo para el otro y así estrechamos la ventana. Si el primero ya
  decide el corte, no buscamos el segundo. Los cortes se comprueban contra las cotas sin recortar a
  [-1, 1]; la sección 8 explica por qué importa.
- **Profundización iterativa** desde `strategy.py`: siempre hay una jugada lista al acabarse el tiempo.
  Si una iteración se interrumpe, usamos su mejor jugada solo si terminó al menos una jugada de la raíz.
- **Tabla de transposición** de 2^21 entradas, con hash Zobrist incremental propio. Se conserva durante
  toda la partida.
- **Orden de jugadas:** jugada de la tabla, anotar, ataques con p >= 0.65, ataques con p = 0.5, historia,
  y al final los ataques con p < 0.5.
- **Reducción de jugadas tardías (LMR).** Desde profundidad 3 y a partir de la cuarta jugada, las jugadas
  tranquilas se buscan 1 ply menos (2 si la profundidad es >= 5 y la jugada es la 13.ª o posterior) con
  ventana nula. Si superan alfa, se vuelven a buscar completas. No reducimos cuando alguien tiene 2 puntos
  ni en jugadas hacia las filas relativas 6 en adelante.
- **Extensión táctica.** Si quien mueve tiene 2 puntos y llega a la fila relativa 6 o más, esa jugada no
  gasta profundidad, para no perder de vista el tercer punto.
- **Quiescencia** de hasta 4 plies: en la frontera el bot puede plantarse con la evaluación o seguir
  solo por jugadas que anotan y ataques con p >= 0.65, para reducir el efecto horizonte.

`mi_motor.py` copia las reglas de `sistema/motor/nucleo.py` y agrega hacer/deshacer en sitio. Es para
pensar rápido: la jugada final se comprueba contra `estado.jugadas()`. `herramientas/verificar_motor.py`
lo compara con el motor oficial en partidas al azar (1,192,256 jugadas sin diferencias en la última corrida).
`herramientas/verificar_busqueda.py` compara `_buscar`, sin LMR ni extensión, con una búsqueda por fuerza
bruta a 1 y 2 plies más quiescencia: 0 diferencias en 600 casos, 300 de ellos de carreras 2–2.

## 4. Función de evaluación

Calculamos un logit `s = w · f`, con los rasgos `f` como blancas menos negras, y el valor es
`0.98 * tanh(s / 2)` desde quien mueve. Solo las posiciones terminales valen ±1. Los 30 rasgos son:

| Grupo | Rasgos |
|---|---|
| Material y avance | peones por fila relativa (8 rasgos) |
| Marcador | tener al menos 1 punto; tener al menos 2 puntos; con 2 puntos, peones en las filas relativas 6 y 7 |
| Apoyo | peones con apoyo 1; peones con apoyo 2 |
| Combate | suma de diferencias de apoyo en contactos diagonales; peones amenazados por un rival con más apoyo; ventaja del mejor ataque de quien mueve |
| Corredores | peones sin rivales en su cono de avance, en tres rangos de fila; quien mueve puede anotar ya; peones infiltrados (fila relativa >= 4, con salida y sin rival en contacto) |
| Carrera | ruta y peligro, cada uno también en versión «solo si el dueño tiene 2 puntos» (4 rasgos, abajo) |
| Forma | peones en el borde; peones bloqueados; columnas sin peones propios |
| Tempo | jugadas a casillas que el rival no puede atacar; a quién le toca mover |

Los rasgos de combate salen de una propiedad del juego: en un contacto diagonal, la probabilidad de
ganar de un bando es 1 menos la del otro. Quien tiene más apoyo lleva ventaja sin importar quién ataque.

Los rasgos de carrera toman el mejor corredor de cada bando:

- **Ruta**, para peones a 3 avances o menos de la meta: `_llega` recorre sus caminos y calcula la
  probabilidad del mejor, contando bloqueos, ataques necesarios y la exposición donde se detiene. Se divide
  entre los avances que faltan. Así distingue una ruta viable de un peón cercano pero bloqueado.
- **Peligro**, para peones en fila relativa >= 4: el producto de `1 - p_defensa` sobre los defensores de su
  cono de avance, dividido entre los avances que faltan. Mide lo difícil que será interceptarlo, antes de
  que el corredor esté al alcance del horizonte de búsqueda.

En los pesos entregados, el avance crece con la fila (de 0.19 en casa a 0.93 a un paso de anotar), los
puntos pesan 1.74 y 2.05, y los rasgos más fuertes son los de carrera con 2 puntos: peligro 2.86 y ruta
1.76. Los pesos no deben leerse uno por uno: varios rasgos están correlacionados y algunos cambian bastante
entre rondas de ajuste (peligro con 2 puntos varió de 2.86 a 4.64).

## 5. Precomputación (qué y cómo regenerarla)

`data/pesos.npy` contiene los 30 pesos. Se generan con ajuste tipo Texel en `herramientas/ajustar.py`:
cada ronda juega partidas de autojuego con los pesos actuales, etiqueta cada posición con quién ganó y
reajusta por regresión logística sin intercepto (L-BFGS, con regularización L2). Es un modelo lineal, no
una red neuronal. Para variar las partidas, los primeros 10 plies son al azar y después el 5 % de las
jugadas también.

Los pesos entregados son los de la última de 5 rondas consecutivas, de unas 170,000 posiciones cada una.
No elegimos la mejor ronda:

```bash
# 5 veces, unas 5 horas en total:
python estudiantes/frea/herramientas/ajustar.py --partidas 2000 --nodos 50000 --rondas 1 --procesos 10
```

El punto de partida fueron los 27 pesos de la versión anterior, ajustados con 4,000 nodos por jugada, con 3
pesos nuevos en 0.5. El rasgo de cercanía al tercer punto fue sustituido por «ruta con 2 puntos», y su peso
3.0 se fijó a mano. `ajustar.py` parte de `data/pesos.npy`, o de `PESOS_INICIALES` en `mi_motor.py` si no
existe o no tiene el tamaño correcto. Por eso, y porque las semillas se toman del sistema, regenerar desde
cero da pesos parecidos, no idénticos.

## 6. Manejo del reloj

Una reserva del 12 % del reloj (3.6 s de 30) nunca se gasta pensando. Del resto, cada jugada recibe
`disponible / max(10, 42 - jugadas_hechas)`, y no se empieza otra iteración si ya se gastó la mitad de ese
presupuesto.

numba no puede leer el reloj, así que limitamos la búsqueda por nodos: el tope es el tiempo permitido
por los nodos por segundo medidos, valor que se recalibra en cada jugada para ajustarse a la máquina. El
tope por jugada es 1.6 veces el presupuesto, sin pasar del 40 % de lo disponible. Si solo queda la
reserva, hace una búsqueda mínima de 200 nodos. Si la búsqueda falla, juega una jugada legal de respaldo.

`preparar` compila numba y mide la velocidad. Con Docker tardó de 4.9 a 5.0 s de 20 s. La liga puede ser hasta
2.5 veces más lenta al compilar, lo que daría unos 12.5 s.

Con reloj de 30 s, el mayor uso fue 26.37 s en 400 partidas sin Docker y 23.9 s en 4 partidas con Docker. Ninguna
se perdió por reloj. El reloj también cobra la pausa y reanudación del contenedor en cada jugada: unos 32 ms,
medidos con `random`. La reserva de 3.6 s alcanza para unas 110 jugadas solo con ese costo, y en nuestras
partidas cada bando hace unas 48. Con el reloj de prueba de 3 s la cuenta no alcanza, y `./torneo validar`
puede avisar derrota por reloj; es un aviso de ese reloj reducido, no del reloj de la liga.

## 7. Resultados vs random y montecarlo

Medimos con `herramientas/arena.py`: partidas pareadas (mismos dados, colores invertidos), con los
márgenes al 95 %. Corre sin Docker, así que no incluye la pausa de contenedores. Todas las filas son con la
versión entregada.

| Rival | Reloj | Partidas | Resultado | % de victorias |
|---|---|---|---|---|
| `random` | 30 s | 24 | 24–0 | 100 |
| `montecarlo` | 30 s | 48 | 41–7 | 85.4 ± 10.0 |
| `montecarlo`, con Docker y límites oficiales | 30 s | 4 | 4–0 | muestra mínima |
| Nuestra versión anterior (commit `77ddc88`, 27 rasgos) | 30 s | 400 | 232–168 | 58.0 ± 4.8 |


## 8. Qué intentamos y no funcionó

- **Nuestro primer Star1 tenía un error.** Comprobaba los cortes contra cotas recortadas a [-1, 1], así que
  un resultado de ±1 podía saltarse la otra rama del dado sin justificación. Las pruebas originales no lo
  vieron; al comparar con fuerza bruta en carreras 2–2, 79 de 300 casos daban un valor distinto, con
  diferencia de hasta 1.6. Tras corregirlo son 0 de 300. Cada versión anterior, sus pesos y sus resultados
  se produjeron con ese error. La lección fue probar la búsqueda justo en las posiciones donde más importa.
- **Pesos a mano que castigaban anotar.** En la primera versión, un peón a un paso de la meta valía más
  que el punto que ganaba al anotar, y el bot prefería no anotar. Perdía contra `montecarlo`: 25–35
  (41.7 %, con 3 s). El primer ajuste Texel dio 56–4 (93.3 %). La lección fue ajustar pesos con datos.
- **MCTS con nodos de azar**, con playouts al azar de 6 jugadas cortados con nuestra evaluación. Perdió
  contra el expectiminimax 107–293 (26.8 ± 4.3 %) con 3 s y 11–37 (22.9 ± 11.9 %) con 30 s. No lo
  incluimos en la entrega. No probamos playouts informados, así que esto descarta esta versión, no MCTS
  en general.
- **Medir solo con reloj corto.** Dos rasgos tras revisar partidas perdidas ganaron 424–376 (53.0 ± 3.5 %)
  con 3 s y estuvimos a punto de descartarlos. Con 30 s, y con 4 rondas más de ajuste, ganaron 250–150
  (62.5 ± 4.7 %). La lección fue medir las decisiones finales con el reloj real.
- **Bajar el nivel de optimización de numba** casi no ayudó a compilar más rápido. Lo que sirvió fue pasar
  los arreglos como parámetros explícitos en vez de una tupla: de 8 s a 3.5 s en la función de búsqueda.

Una limitación: LMR, extensión y rasgos de carrera entraron juntos con la corrección de Star1, sin medirse
por separado, y sus parámetros se fijaron a criterio. Por eso el 58 % del cuadro no se puede repartir entre
ellos.

## 9. Fuentes y uso de IA

**Fuentes**

- Ballard (1983), «The *-minimax search procedure for trees containing chance nodes», *Artificial
  Intelligence*: poda Star1.
- Russell y Norvig, *Artificial Intelligence: A Modern Approach*: minimax, alfa-beta y expectiminimax.
- Ajuste Texel, descrito en Chessprogramming Wiki: idea de ajustar pesos con regresión logística.
- Browne et al. (2012), «A Survey of Monte Carlo Tree Search Methods»: base del prototipo descartado.
- `sistema/motor/nucleo.py` del torneo: reglas copiadas a `mi_motor.py`, como permiten las reglas.

No usamos código de internet.

**Uso de IA**

- **Claude Code** propuso escribió el motor, la búsqueda, la evaluación, el manejo del
  reloj y las herramientas de medición y ajuste.
- **Codex Astra** escribió el prototipo de MCTS, que no se entrega, y analizó cinco partidas perdidas para proponer dos rasgos, adaptados y medidos antes de adoptarlos. Asímismo, hizo experimentos en copias separadas, de ahí salieron las ideas de defensa anticipada, rutas, reducción selectiva y extensión, y el
  aviso de un error en la poda, que confirmamos con fuerza bruta.
