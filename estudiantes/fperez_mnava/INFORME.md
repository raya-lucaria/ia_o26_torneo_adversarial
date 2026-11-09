# Informe: fperez_mnava

## 1. Integrantes
- Francisco Pérez Trofimoff — clave única 000214552
- Mateo Nava Baños — clave única 000214578

## 2. Idea en un párrafo
El juego es de suma cero, con información perfecta y azar solo en los ataques. Por eso el valor de
un estado es la probabilidad de ganar con juego óptimo, y se calcula con expectimax: máximo en
nuestros turnos, mínimo en los del rival y valor esperado en cada ataque. Como no se puede buscar
hasta el final, cortamos a cierta profundidad y estimamos la probabilidad de ganar con una regresión
logística: una suma lineal de rasgos del tablero pasada por una sigmoide. No usamos redes
neuronales. Se gana con 3 puntos, así que la evaluación premia la carrera hacia la meta más que el
material.

## 3. Búsqueda (algoritmo y por qué)
Negamax con poda alfa-beta y nodos de azar con poda tipo Star1 (como los valores están acotados en
[0, 1], si el primer resultado de un ataque ya decide, se omite el segundo), con profundización
iterativa y tiempo máximo por jugada. Las jugadas se ordenan: primero anotar, luego los ataques de
mayor probabilidad y luego avanzar; en la raíz, primero la mejor jugada de la capa anterior. Usamos
una representación propia (lista de 81 casillas con hacer/deshacer en sitio) porque el objeto
`Estado` es muy lento para buscar; la jugada final siempre se revisa contra `estado.jugadas()`.
Comprobamos con `herramientas/verificar_bot.py` que el generador y la búsqueda coinciden con el
motor oficial y con un expectimax sin poda: 0 diferencias en 200 posiciones del generador, 0
inconsistencias de hacer/deshacer y 0 diferencias de valor en 75 comparaciones con/sin poda.
Medido en la posición inicial con 30 s de reloj, sin numba y sin Docker: profundidad 5 con unos
34 000 nodos en 0.52 s (≈ 66 000 nodos/s; esa jugada recibe unos 0.6 s del reloj). En posiciones
con menos jugadas legales la búsqueda llega más hondo; no medimos la profundidad media de una partida.

## 4. Función de evaluación
P(ganar) = sigmoide(z), donde z es una suma ponderada, desde el punto de vista de quien mueve, de:
diferencia de puntos; carrera (jugadas que les faltan a los peones más avanzados para anotar lo que
falta); ventaja de mover; corredores (peones sin ningún rival adelante dentro de su cono, más
valiosos cuanto más cerca de la meta); carriles (corredores en zonas distintas del tablero); apoyos;
ataques disponibles para cada bando ponderados por p - 0.5; y material (peso bajo).

| Rasgo | Peso |
|---|---|
| puntos | 3.0 |
| carrera | 0.30 |
| mueve | 0.20 |
| corredor | 0.12 |
| carriles | 0.25 |
| apoyo | 0.04 |
| amenaza_mia | 1.2 |
| amenaza_rival | 0.6 |
| material | 0.06 |

Los pesos son una estimación hecha a mano; **no los ajustamos con partidas** (ni con búsqueda local
ni con otro método). Los pesos están en el diccionario `PESOS` de `strategy.py`.

## 5. Precomputación (qué y cómo regenerarla)
No hay precomputación: no existe la carpeta `data/` y el bot no carga archivos. Todo está en
`strategy.py`. La carpeta `herramientas/` solo tiene código de prueba que no corre en la liga:
`arena.py` (partidas entre bots; carga los bots con `importlib`, que es su función y por eso el
validador lo marca como aviso), `verificar_bot.py` y tres rivales de prueba en `rivales/`.

## 6. Manejo del reloj
Reserva fija del 12 % del reloj (3.6 s de 30). Del resto, cada jugada recibe lo disponible entre las
jugadas propias que esperamos que falten (se asume una partida de unas 45 jugadas propias, con
mínimo 8). La primera capa de búsqueda siempre se completa; si se acaba el tiempo en una capa más
profunda, se usa el resultado de la capa anterior. Si algo falla, se juega la primera jugada legal en
vez de perder. Reloj máximo usado en 40 partidas con reloj de 30 s (arena, sin Docker): 23.20 s, es
decir, un margen de unos 6.8 s. Las partidas duraron 58–61 jugadas propias en promedio, más que las
45 supuestas, y aun así no hubo derrotas por reloj. No lo hemos medido con los límites de CPU de Docker.

## 7. Resultados vs random y montecarlo
Con `herramientas/arena.py`, reloj de 30 s, semilla 2026, 10 juegos de dados con los dos colores
(20 partidas por rival), sin Docker y sin numba:

| Rival | Victorias | IC 95 % | Reloj máx. usado |
|---|---|---|---|
| random | 20 de 20 (10/10 con cada color) | 84 % a 100 % | 23.20 s |
| montecarlo | 20 de 20 (10/10 con cada color) | 84 % a 100 % | 23.19 s |

Con 20 partidas solo podemos decir que la tasa de victoria es alta; no la estimamos con precisión.
Además, el validador oficial en modo prueba jugó sus 2 partidas de humo contra `random` sin errores.
En una prueba anterior, con reloj de 10 s, el bot ganó 12/12 contra un rival codicioso y 7/8 contra
un rival de material (ambos propios, en `herramientas/rivales/`).

## 8. Qué intentamos y no funcionó
Experimento de las dos mitades (`PODA_ZONAS`, desactivado en la entrega): restringir nuestras jugadas
a las columnas a-e o e-i, con el rival respondiendo con todas las suyas y la evaluación viendo todo el
tablero. Resultado: 4 de 8 partidas contra la versión completa, sin diferencia detectable; con tan
pocas partidas no podemos concluir nada, y no lo repetimos con más. Tampoco probamos tabla de
transposición, búsqueda de quietud ni ajuste de pesos.

## 9. Fuentes y uso de IA
Russell y Norvig, *Artificial Intelligence: A Modern Approach*, capítulo de juegos (expectimax y
poda alfa-beta). Browne et al. (2012), «A Survey of Monte Carlo Tree Search Methods».
Usamos un asistente de IA (Claude, de Anthropic).

**Qué hizo la IA:** escribió la estructura inicial y el código de `strategy.py` (búsqueda con
poda y función de evaluación), el arnés de pruebas (`arena.py`, `verificar_bot.py`, los rivales de
prueba) y el borrador de este informe. En la entrega ayudó a ejecutar el validador, a medir contra
random y montecarlo y a llenar las cifras de este documento.

**Qué hicimos nosotros:** la parte teórica y las decisiones de diseño. Planteamos y pusimos a prueba
la idea de estimar la probabilidad de ganar con una sigmoide (regresión logística sobre rasgos del
tablero) en lugar de una diferencia de material. Propusimos e implementamos con la IA el rasgo de
los carriles (corredores en zonas distintas del tablero). Decidimos priorizar el análisis desde la
teoría de juegos (el juego como suma cero con azar solo en los ataques, donde el valor de un estado
es P(ganar)) en vez de optimizar numéricamente los parámetros; por eso los pesos son una estimación
a mano y no están ajustados con partidas. También hicimos nosotros, a mano, el experimento del tablero
dividido en dos mitades (sección 8), con el resultado ahí descrito.

**Conversacion con CLAUDE en torno a especificaciones de proyecto**: 

**1**
“Los pesos lineales o logísticos están permitidos explícitamente en las reglas. No se permiten redes neuronales.” Como se puede usar pesos lineales podemos utilizar una función como la sigmoide que pone todo acotado entre 0 y 1.

**2**
Si existe algun peón desprotegido es mejor atacar que reenforzarlo para que nos ataquen. La estrategia va a ser en hacer que el rival se equivoque antes de que nosotros, entonces cualquier peón que no esté reforzando una posición es inútil y posiblemente el movimiento de este sea leído por el oponente para que ellos “aprovechen” del movimiento erróneo pero lo que queremos es la esperanza de que este sea atacado no en el siguiente movimiento pero si en el segundo o nosotros atacar después de su primer movimiento. Necesitamos pensar más así y por la cantidad de peones, poder aprovechar que de 18 peones vamos a ganar 3 veces con mínimo .2 de probabilidad de ganar

**3**
En este caso, cortas a cierta profundidad y pones una aproximación. Esa aproximación es la regresión logística: tomas rasgos del tablero, los combinas con pesos, y la sigmoide los convierte en probabilidad. Cada peso dice cuánto sube o baja tu probabilidad de ganar por cada unidad de ese rasgo. E igual para proteger y no obsesionar el modelo, que mida una ventaja enorme ya vale casi uno, así que el bot no se obsesiona con ganar por mucho cuando basta con ganar.

**4**
Entonces como hacemos que el otro jugador nos ayude a limpiar uno o dos carriles de los bandos? Son más accesibles para atacar no? Luego los peones ya asegurados, no vale la pena moverlos ya que no tienen riesgo de ser comidos por los otros (cuando estén en frente del otro). Realmente cuál es el mejor método de outsmart un expectimax asumiendo que la mayoría serán así?

**5**
Hay que partir el tablero en dos, con la quinta columna siendo parte de los, dos tableros para optimizar los carriles 1,5 y 9. Y tener dos funciones expectimax que escoge solamente del mejor tablero, las dos leen las probabilidades de todo el tablero pero buscan el mejor solamente para su lado y los carriles, que piensas?

