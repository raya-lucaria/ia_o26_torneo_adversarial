# Informe: torneo_ia_na
## 1. Integrantes




**Náyade Velasco.**

El proyecto fue desarrollado individualmente. 

## 2. Idea en un párrafo

Desarrollé un bot para *Breakthrough de combate* basado en Monte Carlo probabilístico, con una política de simulación diferente según el color. El bot evalúa las jugadas legales estimando sus probabilidades de victoria a partir de partidas simuladas. Cuando un movimiento implica un ataque, considera explícitamente los posibles resultados del dado y sus probabilidades. Con blancas utiliza las simulaciones aleatorias del motor oficial; con negras utiliza una política propia que favorece avanzar, anotar puntos y realizar ataques con mayor probabilidad de éxito. También identifica victorias inmediatas garantizadas y limita el tiempo dedicado a cada decisión.

## 3. Búsqueda: algoritmo y justificación

La estrategia final utiliza **Monte Carlo plano con evaluación probabilística de los estados sucesores**.

En cada turno, el algoritmo:

1. Obtiene todas las jugadas legales y sus posibles estados sucesores.
2. Comprueba si existe un movimiento que garantice una victoria inmediata.
3. Evalúa las demás jugadas mediante simulaciones completas.
4. Calcula el valor esperado de cada acción, ponderando los resultados del dado por sus probabilidades.
5. Selecciona la jugada con mayor proporción estimada de victorias.

Para reducir cálculos repetidos, los estados sucesores se guardan durante la evaluación de cada turno.

La principal modificación respecto al bot de referencia es el uso de dos políticas:

- **Blancas:** Monte Carlo aleatorio del motor oficial, con 20 simulaciones por evaluación de un estado.
- **Negras:** Monte Carlo guiado, con 10 simulaciones por evaluación, utilizando Numba para acelerar las partidas.

Esta combinación surgió de las pruebas experimentales: la política guiada presentó buenos resultados con negras, pero resultados desfavorables con blancas. Por ello, decidí combinarla con la versión probabilística del simulador oficial.

También exploré UCB, MCTS-UCT y búsqueda adversarial, pero las versiones implementadas no consiguieron superar consistentemente al bot de referencia.

## 4. Función de evaluación

La evaluación principal es la probabilidad estimada de ganar desde el color del bot.

Para una jugada `a`, su valor se calcula como:

`V(a) = Σ P(s' | a) · W(s')`

Donde:

- `P(s' | a)` es la probabilidad de alcanzar el estado sucesor `s'` después del movimiento.
- `W(s')` es la proporción de victorias obtenidas mediante simulaciones desde ese estado.

La política guiada asigna a cada movimiento un peso inicial según el avance del peón:

`w = 1 + 0.25d + 8(d/8)^4`

Donde `d` es el avance del peón, entre 0 y 8.

A ese peso se le añaden:

- **22 unidades** si el movimiento permite anotar un punto.
- **45 unidades** si permite anotar cuando el jugador ya tiene dos puntos.

Cuando el movimiento es un ataque, su peso se multiplica por:

`0.4 + 1.1 × p_ataque`

Así, las simulaciones priorizan movimientos relacionados con el objetivo del juego, sin dejar de incorporar aleatoriedad. Las jugadas se seleccionan mediante muestreo proporcional a sus pesos.

Estos parámetros son heurísticos: se definieron manualmente y no mediante un proceso de aprendizaje.

## 5. Precomputación: qué se utiliza y cómo regenerarla

El bot no requiere archivos precomputados, modelos entrenados ni información externa.

Durante `preparar(config)` se inicializa el generador aleatorio con la semilla proporcionada y se ejecutan simulaciones de prueba para preparar las funciones compiladas con Numba.

Para las simulaciones guiadas, el estado del tablero se convierte a arreglos de NumPy que representan la ubicación de las piezas, el turno y los puntos acumulados.

Toda esta información se genera directamente durante la partida, por lo que no es necesario ejecutar un procedimiento adicional de precomputación.

## 6. Manejo del reloj

El bot dispone de 30 segundos acumulados de decisión por partida y hasta 20 segundos para la preparación.

Para reducir el riesgo de agotar el tiempo, reserva el 30 % del reloj inicial, equivalente a 9 segundos.

En cada turno, el presupuesto aproximado de búsqueda es:

`t_busqueda = max(0, tiempo_restante - reserva) / 25`

Las simulaciones se distribuyen circularmente entre las jugadas legales. El límite temporal se comprueba después de cada evaluación, procurando que todas las jugadas tengan al menos una muestra.

Cuando el tiempo disponible llega a la reserva, el bot selecciona una jugada legal sin continuar simulando.

En los minitorneos locales se utilizó el reloj de 30 segundos. No se observaron derrotas por tiempo. En la evaluación final de las semillas 1701–1705, el reloj máximo reportado por minitorneo no superó los 18.3 segundos.

Las pruebas se realizaron con `--sin-docker`, por lo que sus resultados temporales no sustituyen la ejecución en el entorno oficial.

## 7. Resultados contra random y montecarlo

Realicé diferentes series de minitorneos locales, alternando colores y utilizando distintas semillas. Cada semilla produce dos enfrentamientos contra cada rival.

Los resultados más relevantes fueron:

| Estrategia | Victorias vs. random | Victorias vs. montecarlo |
|---|---:|---:|
| Monte Carlo probabilístico | 30/30 (100 %) | 22/50 (44 %) |
| MCTS ajustado | 10/10 (100 %) | 5/10 (50 %) |
| Monte Carlo guiado, ambos colores | 12/12 (100 %) | 6/12 (50 %) |
| Híbrido inicial, antes de corregir un error | 6/6 (100 %) | 4/6 (66.7 %) |
| **Híbrido corregido (versión final evaluada)** | **10/10 (100 %)** | **5/10 (50 %)** |

### Resultados de la versión híbrida corregida

Las pruebas finales se realizaron con las semillas 1701–1705.

| Semilla | Vs. montecarlo | Vs. random |
|---|---:|---:|
| 1701 | 1/2 | 2/2 |
| 1702 | 1/2 | 2/2 |
| 1703 | 1/2 | 2/2 |
| 1704 | 1/2 | 2/2 |
| 1705 | 1/2 | 2/2 |
| **Total** | **5/10** | **10/10** |

Contra Montecarlo, el bot ganó **2 de 5 partidas con blancas** y **3 de 5 con negras**.

Un resultado importante durante el desarrollo fue que la estrategia completamente guiada ganó las seis partidas contra Montecarlo jugando negras, pero perdió las seis jugando blancas. Esta diferencia motivó el diseño híbrido.

La primera implementación híbrida consiguió cuatro victorias en seis partidas. Sin embargo, posteriormente detecté un error en la asociación entre las probabilidades de los ataques y sus estados sucesores. Después de corregirlo, repetí la evaluación con semillas nuevas.

**Interpretación:** la versión final superó consistentemente a `random` en estas pruebas y obtuvo resultados equilibrados contra `montecarlo`. No existe evidencia suficiente para afirmar que sea superior al bot de referencia, especialmente considerando el número de partidas y la variabilidad de los dados.

## 8. Qué intentamos y no funcionó

Durante el desarrollo probé diferentes alternativas antes de seleccionar la estrategia final.

**UCB en la raíz.** Busqué concentrar las simulaciones en movimientos prometedores, pero las pruebas no mostraron una mejora consistente.

**Semillas comunes en las simulaciones.** Intenté reducir la variabilidad de las comparaciones entre jugadas. El rendimiento siguió siendo similar al del algoritmo probabilístico original.

**Filtros tácticos y penalización por riesgo.** Probé mecanismos para evitar movimientos que permitieran derrotas inmediatas. Algunas versiones fueron demasiado conservadoras y perdieron efectividad.

**Selección progresiva de candidatos.** Evalué todas las jugadas inicialmente y después concentré las simulaciones en las mejores. No obtuve una ventaja clara.

**MCTS-UCT.** Implementé un árbol de búsqueda que consideraba decisiones de ambos jugadores y resultados probabilísticos de los ataques. También probé cambios en los parámetros de exploración, el número de simulaciones por hoja y la reutilización de nodos entre turnos. Estas modificaciones no demostraron una superioridad consistente frente a Montecarlo.

**Búsqueda adversarial de dos niveles.** Analicé posibles respuestas del rival mediante una evaluación heurística. Su implementación tenía limitaciones de tiempo que podían producir evaluaciones incompletas y no mejoró los resultados.

**Simulaciones guiadas.** Diseñé una política que favorecía el avance, las anotaciones y los ataques favorables. Aunque no superó globalmente a Montecarlo, mostró diferencias importantes según el color y sirvió como base para la estrategia híbrida.

**Corrección de errores y optimización.** Durante las pruebas también identifiqué una asociación incorrecta entre resultados del dado y estados sucesores. Corregí ese problema y eliminé cálculos repetidos que no eran necesarios para la evaluación de ciertos estados.

La principal conclusión del proceso es que **un algoritmo más complejo no necesariamente produce mejores decisiones cuando el tiempo de búsqueda es limitado**. Las pruebas con distintas semillas fueron esenciales para evitar seleccionar una estrategia únicamente por una racha favorable.

## 9. Fuentes y uso de inteligencia artificial

### Fuentes

- Repositorio oficial del torneo: `raya-lucaria/ia_o26_torneo_adversarial`. Motor, reglas, interfaz del bot y estrategia de referencia `montecarlo`.
- Material del curso de Inteligencia Artificial sobre búsqueda adversarial, minimax, expectiminimax y Monte Carlo Tree Search.
- Browne, C. B., et al. (2012). *A Survey of Monte Carlo Tree Search Methods*. IEEE Transactions on Computational Intelligence and AI in Games, 4(1), 1–43.
- Russell, S. y Norvig, P. *Artificial Intelligence: A Modern Approach*. Capítulos sobre juegos adversariales y búsqueda.

### Uso de IA

Utilicé ChatGPT como herramienta de apoyo para comprender el funcionamiento del motor, proponer alternativas de búsqueda, generar y modificar código, identificar errores, interpretar los resultados de las pruebas y organizar la documentación.

Durante el desarrollo se ejecutaron minitorneos locales para evaluar las propuestas y comparar su rendimiento. Varias alternativas generadas con apoyo de IA fueron descartadas porque no mostraron mejoras consistentes.

La estrategia presentada corresponde a la versión híbrida corregida y evaluada durante el desarrollo. No se utilizaron redes neuronales ni modelos entrenados.
