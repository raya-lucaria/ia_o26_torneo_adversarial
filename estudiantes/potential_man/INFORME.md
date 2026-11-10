# Informe: potential_man

## 1. Integrantes

- Patricio Tanus Moctezuma, clave única 000195552.

El equipo es de **un integrante**. El validador de entrega exige dos; se solicitó autorización al
profesor para aceptarlo a mano (el mensaje del validador lo contempla). No se duplicó la clave ni
se inventó un segundo nombre.

## 2. Idea en un párrafo

Breakthrough de combate es una carrera a 3 puntos con dados en cada ataque. `potential_man`
combina cuatro piezas: un filtro táctico que toma mates, anotaciones quietas y paradas de peones
que ya pueden anotar; UCT solo en la raíz, con un prior heurístico para no explorar a ciegas;
`simular_al_azar` del motor, mezclado con una evaluación estática; y un expectimax corto de 2 plies
sobre las dos mejores candidatas si queda tiempo. La evaluación premia la carrera hacia 3 puntos y
castiga a un peón rival que avanza con el camino libre y sin forma inmediata de capturarlo o
bloquearlo. El resolvedor exacto de finales existe como atajo; contra `montecarlo` casi no se
activa, porque las partidas terminan con demasiados peones.

## 3. Búsqueda (algoritmo y por qué)

No se baja un árbol profundo de jugadas del rival. Con 20–40 legales y 30 s de reloj, minimax o
MCTS profundo gastan el tiempo en nodos lentos. Lo rápido del motor es `simular_al_azar`. Por eso
la búsqueda es **UCT plano (un nivel) + expectimax del dado + expectimax heurístico corto**.

Antes de simular, `_candidatos` recorta el conjunto:

1. **mate:** todas las ramas del dado ganan → se juega ya.
2. **para:** hay peón rival en la fila de anotar. El conjunto se reduce a capturas de ese peón
   (cualquier `p`), anotaciones quietas, bloqueos del paso de peones con avance ≥ 6 e
   intercambios con `p ≥ 0.65`. No se cierra el set solo por un peón a dos pasos: ese filtro
   bajaba el marcador.
3. **anota:** avance quieto que suma un punto, si no hay amenaza inminente.
4. **uct:** todas las legales.

Sobre las candidatas: una visita a cada una y, mientras alcance el tope del turno, UCB1
(`C = 0.85`) elige a quién volver a medir. Cada ataque se valora con expectimax
(`p·gana + (1−p)·pierde`), no con una sola tirada. El valor del hijo es
`0.55·winrate + 0.45·heurística`, más un bonus táctico. Si sobran unos 40 ms, las dos mejores
reciben un expectimax heurístico de 2 plies (peso 0.12). En los primeros 10 turnos hay un sesgo
leve de apertura (+0.10) hacia un ala elegida al azar en `preparar`. Si la posición cabe en el
resolvedor exacto (pocos peones o tempos bajos) y hay tiempo, se intenta alfa-beta/expectiminimax;
si aborta, se sigue con UCT.

## 4. Función de evaluación

`evaluar` mira el tablero desde nuestro color y devuelve un valor en `[-1, 1]`, cacheado por
`estado.clave` durante toda la partida (tope 20 000 entradas).

- Terminal: `+1` / `−1`.
- Puntos: diferencia × `W_PTS = 3.4`, más ±0.6 si un bando tiene 2.
- Avance no lineal: `PESOS_AVANCE = (0, 1, 2, 3.5, 6, 11, 20, 34)` por filas de avance 0–7,
  escalado por `W_AVANCE = 0.028`.
- Apoyo de combate (diagonales traseras) solo en peones con avance ≥ 4 (`W_APOYO = 0.07`).
- Material: `W_MAT = 0.30`.
- Carrera: diferencia de tempos a meta (`_tempos_a_meta`) × `W_CARRERA = 0.25`, recortada a
  `[-4, 4]` y comprimida con `tanh`.
- Táctica de 1 ply: peones que ya pueden anotar; penaliza fuerte si le toca al rival.
- **Camino libre:** si un peón rival tiene avance ≥ 6, el recto de delante está vacío y en un
  paso no se puede capturar ni bloquear, resta 0.08 (avance 6) o 0.12 (penúltima), con tope
  0.20. Si todavía se puede tocar, no resta.

La estructura pasa por `tanh`; la táctica se suma aparte. `simular_al_azar` es ruidoso: el azar
uniforme casi no anota ni para. La heurística y el filtro existen para corregir eso.

## 5. Precomputación

No hay `data/` en la entrega. En `preparar` se guarda color, rival, semilla, reserva (22 % del
reloj) y un ala de apertura; se limpia la caché de eval y de simulaciones; se llama
`simular_al_azar` una vez para calentar numba fuera del reloj de 30 s.

El código acepta, si existieran, `opening.json` y `tablas.pkl` en `carpeta_datos`: mapas de clave
de posición (o número de turno) a una jugada en texto. Si el archivo falta o está mal formado, se
ignora. Para regenerarlos habría que correr un script propio en `herramientas/` que explore las
primeras plies y escriba esos archivos; hoy no hay ninguno.

## 6. Manejo del reloj

Reserva del 22 % del reloj inicial (comunicación con el árbitro). Del resto, cada turno usa
`disponible / divisor`, con `divisor = max(12, 26 − 0.35·min(n_candidatos, 24))`, suelo 0.05 s y
techo 1.35 s. Si solo queda la reserva, se elige por heurística sin rollouts. El bucle UCT corta
con `perf_counter`. En las tandas vs `montecarlo` a 30 s, el reloj máximo de `potential_man`
anduvo entre ~19 s y ~23 s; ninguna partida se perdió por tiempo.

## 7. Resultados vs random y montecarlo

Todo con reloj oficial de **30 s**, `--sin-docker`. Los colores se alternan.

| Rival | Partidas | Semilla | Resultado | Reloj máx |
|---|---|---|---|---|
| `random` | 4 | 7 | **4–0** | ~20.6 s |
| `montecarlo` | 10 | 99 | **7–3** (versión entregada) | 22.7 s |
| `montecarlo` | 10 | 42 | **8–2** (misma versión) | 21.9 s |

La referencia de entrega es el **7–3** en semilla 99: misma arquitectura UCT + eval con camino
libre + filtro mate/para/anota. Con semilla 42 el mismo código hizo 8–2. Se probaron otras
variantes (sección 8) que no mejoraron ese marcador o lo bajaron. En la liga oficial cada pareja
juega 4 partidas; estos números locales no sustituyen esa liga.

## 8. Qué intentamos y no funcionó

- **Resolvedor exacto alfa-beta / expectiminimax de finales:** la infraestructura quedó (umbral
  de peones/tempos, 80 000 nodos, límites de tiempo). En las 10 partidas de semilla 99 no eligió
  ninguna jugada: la carrera termina con ~19–25 peones y la puerta está en ≤ 10.
- **Filtros agresivos de peones con avance ≥ 6:** cerrar el set de candidatas en cuanto aparecía
  un peón a dos pasos bajó el marcador de 7–3 a 5–5. Priorizar sin cerrar quedó en 6–4. Se
  revirtió el cierre; el set solo se restringe cuando el peón ya puede anotar.
- **Más simulaciones globales (40/52) y expectimax selectivo de 4 jugadas:** el reloj máximo
  subió a ~23.6 s, pero cada sonda se encareció y la apertura cambió en partidas que antes se
  ganaban. En semilla 99 el resultado bajó a 5–5.
- **Penalización más fuerte en la EVAL** (0.12 / 0.18 + 0.04 si el peón va solo): en semilla 99
  bajó a 6–4; se perdió una victoria que la versión suave (0.08 / 0.12) ganaba. Se volvió a
  0.08 / 0.12 con tope 0.20.
- **Experimento `corte` (solo capturas de peones avanzados):** en tandas anteriores dejó de lado
  la carrera y perdió. No se reintrodujo.

## 9. Fuentes y uso de IA

- Browne et al. (2012), «A Survey of Monte Carlo Tree Search Methods»: Monte Carlo plano y UCB
  en la raíz (flat UCB / UCT de un nivel).
- Russell y Norvig, *AIMA*: expectimax para nodos de azar (aquí el combate y un expectimax corto
  de 2 plies).
- Documentación del torneo (`docs/reglas.md`, `docs/avanzado.md`, `docs/entregar.md`,
  `AGENTS.md`): motor, reloj, `reportar`, rúbrica del informe.
- Cursor (agente Auto / Composer): implementación de `strategy.py`, tandas vs `montecarlo` y
  `random`, análisis de JSON de derrotas y redacción de este informe. El código se basa en el
  motor del curso (`Estado`, `simular_al_azar`, `reportar`); no se copió la carpeta de otro
  equipo. Las decisiones de revertir filtros agresivos y el paquete de más cálculo salieron de
  esas tandas.
