# Informe: montecarlo

(Ejemplo de cómo llenar el informe. El tuyo debe ser más específico: números, tablas, decisiones.)

## 1. Integrantes
Equipo del curso (participante público de referencia).

## 2. Idea en un párrafo
Para cada jugada legal, jugamos muchas partidas rápidas al azar desde la posición que resulta y
medimos qué fracción gana nuestro color. Elegimos la jugada con mejor promedio. Si una jugada
gana la partida sin importar el dado, la tomamos de inmediato.

## 3. Búsqueda (algoritmo y por qué)
Monte Carlo plano: sin árbol, solo un nivel. Es lo más simple que funciona y deja claro dónde
mejorar. Cuando la jugada tiene un ataque, tiramos el dado con la probabilidad real de cada
resultado y medimos el estado que salió. Repartimos las muestras por turnos entre todas las
jugadas, sin importar si parecen buenas o malas (es un desperdicio conocido).

## 4. Función de evaluación
Es el promedio de victorias en partidas al azar de ambos lados. Es ruidosa y optimista: el azar
no se defiende. Siempre se mide desde nuestro color (`para=config.color`).

## 5. Precomputación (qué y cómo regenerarla)
Nada en `data/`. En `preparar` solo calentamos numba con una simulación de prueba.

## 6. Manejo del reloj
Guardamos una reserva del 30 % del reloj inicial que nunca gastamos pensando, porque cada
jugada también cuesta tiempo de comunicación con el árbitro (unos 7 ms por jugada en nuestra
máquina). Del resto, cada jugada usa `1/25` de lo disponible. Si solo queda la reserva, juega al
azar al instante. El reloj se revisa después de cada muestra, pero solo cuando todas las jugadas
tienen al menos una. Medido con el reloj de pruebas (3 s por partida): el mayor uso fue 2.55 s
de 3 s en 50 partidas, y ninguna se perdió por reloj. Con el reloj real de 30 s no lo hemos medido.

## 7. Resultados vs random y montecarlo
Con el reloj de pruebas (3 s), alternando colores: `montecarlo` ganó 36 de 40 partidas a
`random` (90 %). En 10 partidas de `montecarlo` contra `montecarlo` ganaron las blancas 3 y las
negras 7; son pocas partidas para decir si hay ventaja de color. No medimos resultados con el reloj de 30 s.

## 8. Qué intentamos y no funcionó
No intentamos otra cosa: este bot es solo el punto de partida. Ideas que se nos ocurrieron y no
probamos están marcadas con `AQUÍ PUEDES MEJORAR` en `strategy.py`.

## 9. Fuentes y uso de IA
Browne et al. (2012), «A Survey of Monte Carlo Tree Search Methods», IEEE Trans. on
Computational Intelligence and AI in Games: describe el Monte Carlo plano como base de MCTS.
Russell y Norvig, *Artificial Intelligence: A Modern Approach*, capítulo de juegos. Escrito por el equipo del curso con
ayuda de IA.
