# Informe: limon_armando

## 1. Integrantes
- Ricardo Emilio Limon Santamaria (000219636)
- Carlos Armando Ibarrarán Arnaldo (000201598)

## 2. Idea en un párrafo

Nuestro bot (`plano.py`, que `strategy.py` llama con `ALGORITMO = "plano"`) es el `montecarlo`
del curso con seis mejoras. Igual que él, para cada jugada legal juega partidas al azar desde la
posición que resulta y elige la que más gana. Las mejoras cambian cuánto piensa (**reloj por
criticidad**), cómo reparte ese tiempo (**sequential halving**), con cuánto ruido compara las
jugadas (**dado exacto** y **números aleatorios comunes**) y evitan regalar el tercer punto al
rival (**filtro de riesgo**). Si nuestra medición no distingue cuál de las dos mejores jugadas es
mejor, elige entre ellas al azar (**desempate al azar**).

## 3. Búsqueda (algoritmo y por qué)

**Por qué Monte Carlo plano, sin árbol.** Nuestro bot anterior, MCTS con UCT, quedó 112–128
contra `montecarlo` (sección 8). Medimos que hacía la mitad de simulaciones: ~7,600 partidas al
azar por turno contra ~14,600, porque mover el árbol en Python costaba tanto como simular.
Repartidas en un árbol, cada respuesta del rival se exploraba solo 2 o 3 veces por turno.
Volvimos a un solo nivel, y el bot nuevo le gana 79 % al de árbol (sección 7).

**Cómo decide, en orden:**
1. Si una jugada gana sin importar el dado, la juega.
2. Si ya solo queda la reserva de tiempo (sección 6), juega una jugada legal al azar.
3. Si el rival tiene 2 puntos, el filtro de riesgo deja solo las jugadas más seguras.
4. Si queda una sola jugada, la juega.
5. Si no, piensa: prueba las jugadas que quedan y juega la mejor.
   - **Probar una jugada una vez** (una *muestra*): hacerla y, desde la posición que resulta, jugar
     20 partidas completas al azar y contar cuántas gana. Por ejemplo, 13 de 20 vale 0.65.
   - **Repartir el tiempo:** un torneo de eliminación por rondas (*sequential halving*, abajo)
     hasta que queda una jugada, o un desempate al azar entre las dos últimas.

**Sequential halving** (Karnin, Koren y Somekh, 2013; a diferencia del original, reparte tiempo
en vez de muestras y acumula los promedios de todas las rondas).
- **Qué hace:**
  - Parte el tiempo del turno en rondas iguales; son ⌈log₂ n⌉ rondas, 5 si hay 25 jugadas.
  - En cada ronda hace *pasadas*: una muestra a cada jugada viva, una tras otra.
  - Al cerrar la ronda conserva la mitad con mejor promedio (redondeando hacia arriba):
    25 → 13 → 7 → 4 → 2 → 1.
- **Por qué:** `montecarlo` da a cada una de 25 jugadas el 4 % del turno, aunque muchas se vean
  malas pronto. Aquí cada una de las dos finalistas recibe ~20 %, sumando todas las rondas: en principio, cinco veces más
  simulaciones donde se decide la jugada.

**Dado exacto** (muestreo estratificado).
- **Qué hace:** en un ataque, la muestra no sortea el dado. Juega 10 partidas desde «gané el
  combate» y 10 desde «lo perdí», y combina
  `valor = p · valor_si_gano + (1 − p) · valor_si_pierdo`. Aquí p es la probabilidad de ganar el
  combate que da el motor: 0.20, 0.35, 0.50, 0.65 u 0.80 según los apoyos.
- **Por qué:** en `montecarlo` las 20 partidas de una muestra dependen de un solo dado. Pesar los
  dos resultados conserva el promedio y quita ese ruido. Solo aplica a nuestra jugada; dentro de
  las partidas al azar el dado se sortea.

**Números aleatorios comunes** (Veness, Lanctot y Bowling, 2011).
- **Qué hace:** la *semilla* fija la secuencia de números al azar que usan las simulaciones, tanto
  para las jugadas como para los dados internos. En cada pasada usamos una semilla nueva, pero la
  misma para todas las jugadas.
- **Por qué:** con la misma secuencia, parte de la suerte es común y se cancela al comparar
  (parcialmente: desde posiciones distintas las partidas se separan pronto).
- Estas semillas salen de `config.semilla` (el azar propio del bot) y solo afectan nuestras
  simulaciones. Los dados reales los tira el árbitro con otra semilla que no vemos.

**Filtro de riesgo.**
- **Qué hace:** si el rival tiene 2 puntos, calcula para cada jugada nuestra la probabilidad de que
  él gane en su siguiente turno.
  Toma la mejor respuesta del rival (1 si llega a la meta sin combate, su p si tiene que ganar un
  combate) y, si nuestra jugada es un ataque, combina sus dos resultados con p. Se conservan las
  jugadas de menor riesgo, aunque sea 1 para todas.
- **Ejemplo:** un peón rival está a una casilla de la meta con el camino libre. Las jugadas que lo
  ignoran tienen riesgo 1. Atacarlo con p = 0.65 deja riesgo 0.35, y solo esa sobrevive.
- **Por qué:** en las partidas al azar el rival rara vez elige justo su jugada ganadora (es 1 de
  ~25), así que las simulaciones subestiman esa amenaza.

**Desempate al azar.**
- **Qué hace:** si la diferencia entre los promedios de las dos finalistas es menor que el error
  estándar de esa diferencia (calculado con la varianza de las muestras de cada una), elige entre
  ellas al azar con el generador propio del bot (`config.semilla`, distinta en cada partida).
- **Por qué:** un libro de aperturas o un árbol reusado solo le sirven al rival mientras la partida
  siga las líneas que preparó. Variar entre jugadas que nuestra medición no distingue nos saca
  antes de esas líneas sin renunciar a una jugada claramente mejor.

## 4. Función de evaluación

El valor de una jugada es el promedio de sus muestras. Cada muestra es la fracción de 20 partidas
al azar que gana nuestro color (en un ataque, combinada con p como en la sección 3). Las juega el
motor con `simular_al_azar`, con ambos bandos al azar. Mide «cuánto gano si después todos juegan
al azar»; como el rival real no juega así, existe el filtro de riesgo. El `eval` del visor es el
valor de la jugada elegida (solo cuando hay búsqueda).

## 5. Precomputación (qué y cómo regenerarla)

**No usamos nada precalculado: `data/` está vacía.** En `preparar` creamos el generador al azar
con `config.semilla` y corremos una simulación para que numba compile fuera del reloj (como máximo
0.2 s). Probamos tres cosas precalculadas y ninguna ayudó (sección 8).

`herramientas/` se corre offline. `duelo.sh` y `duelos.py` juegan series con `./torneo jugar` y
resumen las victorias; `hacer_variante.sh` copia el bot (fuera de esta carpeta) con una constante
cambiada. `ajustar_prior.py`, `ajustar_eval.py` y `generar_libro.py` generan los archivos de esas
tres pruebas (`prior_oponente.npz`, `eval_posicion.npz`, `libro_aperturas.json`) a partir de
partidas entre nuestros bots y `montecarlo`; hoy solo los leería el bot UCT anterior, apagado en
`strategy.py`, `modelo_oponente.py` y `simulacion.py`.

## 6. Manejo del reloj

1. **Reserva de 5 s** para la comunicación con el árbitro. `montecarlo` reserva el 30 % (9 s); con
   esa reserva nuestro bot usaba a lo más ~23.5 s de los 30.
2. **Tiempo base del turno** = (tiempo restante − reserva) / 25, el mismo divisor de `montecarlo`.
3. **Criticidad.**
   Las notas del curso (7.3.2) llaman *quieta* a una posición donde el jugador de turno no tiene
   capturas, para decidir dónde detener una búsqueda. Nosotros la usamos para el reloj y la
   extendemos: la posición no es quieta si alguien puede capturar, si un peón está a 2 filas o
   menos de anotar o si alguien tiene 2 puntos. Entonces, tiempo base × 1.6; si es quieta, × 0.6
   (factores elegidos a mano).

**Por qué:** `montecarlo` piensa lo mismo en una apertura tranquila que en un intercambio de
capturas. Tiempo medio por jugada propia, en las 240 partidas contra `montecarlo` (sección 7):

| Nuestras jugadas | 1–10 | 11–20 | 21–30 | Últimas 10 |
|---|---|---|---|---|
| Nuestro bot | 0.65 s | 0.80 s | 0.53 s | 0.30 s |
| `montecarlo` | 0.75 s | 0.50 s | 0.33 s | 0.23 s |

El reloj máximo usado fue 25.8 s de 30.

## 7. Resultados vs random y montecarlo

**Cómo medimos.** Con Docker (cada bot con los límites del torneo), reloj de 30 s y colores
alternados. Usamos *duelos*: series de 30 a 40 partidas contra un solo rival, varias a la vez,
cada una con su propia semilla de dados del árbitro. Intervalos de confianza del 95 % de Wilson.

| Rival | Partidas | Victorias | % | Intervalo 95 % |
|---|---:|---:|---:|---|
| `random` | 40 | 40 (todas 3–0) | 100 | [91, 100] |
| `montecarlo` | 240 | 177 | 73.8 | [67.8, 78.9] |
| Nuestro bot UCT anterior | 180 | 143 | 79.4 | [73.0, 84.7] |

**Control.** Una copia exacta de `montecarlo` le ganó 38 de 80 (47.5 %, [36.9, 58.3]) a
`montecarlo`: no detectamos sesgo en el método. Ninguna partida se perdió por tiempo ni por
error.

**Qué aportó cada paso**, contra `montecarlo`, en duelos anteriores a la confirmación (otras
partidas):

| Versión | Victorias |
|---|---|
| `montecarlo` + reserva de 5 s + criticidad | 53/80 (66 %) |
| + sequential halving | 50/80 (62.5 %) |
| + dado exacto + números aleatorios comunes | 152/200 (76 %) |
| + filtro de riesgo (bot final sin desempate) | 140/200 (70 %) |

Con márgenes de ±10 puntos (80 partidas) y ±6 (200) no atribuimos la ganancia a una sola pieza:
la reserva y la criticidad entraron juntas, y sequential halving sola no mostró ganancia. El
filtro le ganó 97–83 frente a frente a la versión sin él (180 partidas), dentro del margen; lo
conservamos porque solo actúa cuando el rival está a un punto de ganar.

**El bot entregado incluye el desempate al azar; las cifras de arriba son sin él.** Lo medimos
después, con y sin desempate en la misma tanda (120 partidas por rival):

| Rival | Sin desempate | Con desempate |
|---|---|---|
| `montecarlo` | 78/120 (65.0 %) | 78/120 (65.0 %) |
| Bot plano con libro de aperturas | 53/120 (44.2 %) | 60/120 (50.0 %) |
| Bot UCT con libro y árbol reusado | 71/120 (59.2 %) | 77/120 (64.2 %) |

Contra `montecarlo` no detectamos costo (el 65 % de esta tanda y el 73.8 % de la confirmación
difieren en 8.8 puntos porcentuales, ~1.7 errores estándar: no significativo). Contra los rivales preparados, versiones
nuestras con libro, gana 137/240 contra 124/240: ~1.2 errores estándar, no significativo. Contra
`random` solo jugó las 2 partidas de `./torneo validar` (las ganó).

## 8. Qué intentamos y no funcionó

Sobre el bot UCT, frente a frente contra él salvo donde se indica:
- **MCTS con UCT** (8 partidas al azar por iteración, detección del tercer punto, exploración
  0.5): 112–128 contra `montecarlo` en tres ligas (`./torneo liga`). Encima de él no cambiaron
  nada la p exacta al elegir ramas (40–40), reusar el árbol (39–41) ni simulaciones que anotan o
  defienden el tercer punto (40–40).
- **Modelar al rival** con una regresión logística (modelo lineal que da probabilidades; no es una
  red neuronal): que las partidas al azar lo imitaran las sesgó (4–76 contra `montecarlo`); como
  sesgo inicial en UCT empató (42–38).
- **Reloj STOP + UNST** (Baier y Winands, 2016; para cuando la mejor jugada ya no puede cambiar):
  gastaba 1.5 veces el tiempo base al principio y perdió 28–52.

Sobre el bot plano, contra `montecarlo` (referencia: bot final sin desempate, 73.8 %):
- **Criticidad aprendida** (¿pensar 4 veces más cambia la jugada?): daba más tiempo a la apertura
  y menos al final: 46/90 (51 %).
- **Libro de aperturas:** hecho con el UCT, 50/80 (62.5 %, contra 76 % de esa versión sin libro);
  rehecho con el bot plano y 40 partidas por muestra, empató frente a frente con el bot final
  sin desempate (105–105).
- **Cortar las partidas al azar con una evaluación logística:** 61/90 (68 %), y compilarla
  tardaba hasta 11 s de los 20 de `preparar`.
- **Constantes**, ninguna fuera del margen: reserva de 3 s, 89/120 (74 %) pero el reloj llegó a
  27.1 s; 10 o 40 partidas por muestra, 41/60 (68 %) y 47/60 (78 %).
- **Variar más que el desempate:** elegir al azar entre jugadas a menos de 0.02 de valor de la mejor
  costó contra `montecarlo` en toda posición (32/60, 53 %) y no ayudó solo en posiciones quietas.

## 9. Fuentes y uso de IA

- Repositorio del curso (motor, árbitro y el bot `montecarlo`, nuestro punto de partida):
  https://github.com/raya-lucaria/ia_o26_torneo_adversarial
- Notas del curso, 7.3.2: posición quieta.
- Karnin, Koren y Somekh (2013), *Almost optimal exploration in multi-armed bandits*, ICML.
- Veness, Lanctot y Bowling (2011), *Variance reduction in Monte-Carlo tree search*, NIPS.
- Baier y Winands (2016), *Time management for Monte Carlo tree search*, IEEE TCIAIG.

**Uso de IA.** Usamos ChatGPT para revisar código, encontrar errores, diseñar pruebas y analizar
repeticiones del bot UCT. Usamos Claude (Anthropic, con Claude Code) para implementar y medir las
variantes, correr ligas y duelos, buscar literatura y redactar este informe. Nosotros decidimos
qué probar y qué conservar, y revisamos el código y los números.
