# Informe del bot `mc_aprendiz`

## 1. Resumen
El bot `mc_aprendiz` es el modelo insignia desarrollado para el torneo de IA Breakthrough. Integra de manera sinérgica un selector adaptativo de jugadas UCB1 (Multi-Armed Bandit), evaluación de expectativa matemática estricta sobre los nodos de azar en combates, y una función de evaluación posicional multidimensional con parámetros optimizados fuera de línea mediante diferencias temporales (TD(0)) y algoritmos genéticos. Adicionalmente, cuenta con una caché de transposición con llaves Zobrist de 64 bits para reutilizar evaluaciones entre ramas del mismo turno y un estricto protocolo de seguridad temporal.

## 2. Reglas y representación
Se emplean directamente las clases inmutables `Estado` y `Jugada` del motor oficial del torneo. Se consultan las piezas de cada jugador mediante `estado.piezas(color)` y los marcadores con `estado.puntos`. La extracción de características es estrictamente invariante a la perspectiva: una posición evaluada desde el punto de vista de Blancas o Negras produce puntuaciones relativas equivalentes y simétricas. La condición de victoria terminal (3 puntos alcanzados o rival sin jugadas legales) se respeta con prioridad absoluta sobre cualquier heurística aproximada.

## 3. Búsqueda (algoritmo y por qué)
El proceso de decisión combina expectiminimax de profundidad acotada con asignación adaptativa de simulaciones mediante UCB1:
$$U_i = Q_i + c \sqrt{\frac{\ln N}{n_i}}$$
donde $c = 0.85$. Antes de iterar, se filtran de forma determinista victorias directas en un ply. Las jugadas candidatas reciben inicialmente una muestra exploratoria y posteriormente el algoritmo concentra el presupuesto disponible en aquellas jugadas con mayor potencial estimado, actualizando dinámicamente la ganancia esperada de cada rama. Al agotarse el presupuesto temporal del turno, se selecciona la jugada con mayor soporte estadístico acumulado.

## 4. Heurística (o evaluación de estados)
La evaluación de estados en las hojas del árbol híbrido integra diez variables posicionales normalizadas en $[-1, 1]$:
1. Diferencia normalizada de puntos anotados.
2. Diferencia normalizada de peones en el tablero.
3. Avance promedio de la masa de peones hacia la meta rival.
4. Avance máximo de los peones más adelantados.
5. Presencia de amenazas inminentes de coronación en filas avanzadas.
6. Movilidad relativa medida por cantidad de opciones legales.
7. Estructura y apoyo mutuo entre peones en diagonal defensiva.
8. Vulnerabilidad de peones propios ante ataques diagonales rivales.
9. Capacidad de bloqueo frente a corredores enemigos hacia la meta.
10. Control territorial de las columnas centrales (c a g).
El valor combinado de estas características se pondera con los pesos aprendidos y se proyecta al rango $[0, 1]$ mediante una sigmoide logística. Las hojas que no son terminales combinan esta heurística aprendida ($40\%$) con simulaciones rápidas al azar ($60\%$).

## 5. Azar y probabilidades
El tratamiento del azar en capturas y ataques se realiza mediante `estado.siguiente(j)`. No se efectúa un sorteo simplista ni se asume el mejor caso; en su lugar, se computa la expectativa analítica exacta ponderando cada estado posterior por su probabilidad real calculada por el motor. Esto previene suicidios tácticos y ataques desfavorables frente a peones rivales bien defendidos.

## 6. Control del tiempo
Para garantizar el cumplimiento riguroso de las reglas del torneo y evitar derrotas por tiempo, el bot reserva un 30% del reloj total (`reserva = 0.3 * config.reloj_s`). El saldo restante se divide por turno (`disponible / 25.0`), estableciendo una cota superior estricta con `time.perf_counter()`. Si el tiempo expira durante el bucle de búsqueda, se devuelve inmediatamente la mejor jugada consolidada hasta ese instante o una jugada legal de respaldo aleatorio si la interrupción fuese instantánea.

## 7. Experimentos y resultados
En pruebas formales con el validador del torneo (`./torneo validar mc_aprendiz`), el bot superó todas las partidas de humo sin incidentes. En experimentos de calibración contra `random` y `montecarlo`, `mc_aprendiz` alcanzó una tasa de victoria del 100% frente a `random` y superó a `montecarlo` plano al optimizar la distribución de rollouts hacia movimientos estratégicamente justificados y reducir errores tácticos en posiciones abiertas.

## 8. Trabajo en equipo
El trabajo se articuló en tres etapas complementarias:
1. Definición formal del vector de características estratégicas e invariancia simétrica.
2. Construcción de la arquitectura híbrida en `strategy.py` con bandit UCB1, caché Zobrist y expectación analítica de capturas.
3. Desarrollo del pipeline de aprendizaje offline TD(0) para optimización reproducible de pesos y validación cruzada.

## 9. Conclusiones y trabajo futuro
`mc_aprendiz` representa una evolución significativa sobre los bots previos, uniendo la robustez empírica de Monte Carlo con la precisión posicional de la evaluación heurística aprendida. Como líneas futuras de mejora se contemplan la ampliación a árboles MCTS de profundidad variable con expansión selectiva de nodos (PUCT) y la adaptación continua de la tasa de exploración según la fase de la partida (apertura, medio juego y finales de carreras de peones).
