# Informe: pamplinotas

## 1. Integrantes

- Nadia Paola Medina García, clave única 220896.
- Alexis Cuevas Heras, clave única 219050.

## 2. Idea en un párrafo

Seguimos el mapa de decisión de la unidad: el árbol no cabe, hay dados y sí podemos escribir una
EVAL, así que el bot hace **expectiminimax con corte**: minimax con corte donde cada ataque es un
nodo de AZAR que promedia sus dos resultados con `p` y `1 − p`. Encima van las herramientas de la
clase 3: alfa-beta (y su extensión a nodos de azar, *-minimax), quietud, profundización iterativa
para el reloj y tabla de transposición, más reducción de jugadas tardías. La EVAL es un modelo logístico sobre 68 rasgos del tablero,
cada uno con un peso de apertura y otro de final, ajustados offline por regresión logística sobre miles de partidas (lo permite «pesos
lineales o logísticos»). Para llegar hondo reescribimos las reglas en numba (`buscador.py`), como
propone `docs/avanzado.md`.

## 3. Búsqueda (algoritmo y por qué)

**Por qué corte y no Monte Carlo.** Monte Carlo plano (`montecarlo`) estima cada jugada con partidas
al azar y no ve tácticas concretas. Con una EVAL razonable la teoría recomienda minimax con corte,
y con dados, nodos de azar: `p·V(gano) + (1−p)·V(pierdo)`.

**Cómo está escrito** (`buscador.negamax` y `valor_jugada`, forma negamax: todo desde quien mueve):

1. *Finales antes de profundidad 0.* Si el rival ya tiene 3 puntos o quien mueve no tiene jugadas,
   se regresa `−(M − ply)` con `M = 10⁶` sin evaluar; restar el ply prefiere la victoria cercana.
2. *Alfa-beta en MAX/MIN y \*-minimax (Star1, Ballard 1983) en AZAR.* Si todo valor del
   subárbol cumple `|v| ≤ U`, el primer resultado del dado (peso `p₁`) se busca con la ventana
   estirada `((α − p₂U)/p₁, (β + p₂U)/p₁)` y si cae fuera se corta sin mirar el segundo. Con
   `U = M` la ventana es casi infinita y no poda, así que `cota_valor` da una cota ajustada: `U = M`
   sólo si en los plies que quedan alguien puede llegar a 3 puntos (sus `3 − puntos` peones más
   cercanos suman a lo más los turnos que caben) o quedarse sin peones (sin jugadas ⇔ sin peones);
   si no, `U = 1000`, la cota de la EVAL (y una entrada de la tabla más profunda que el horizonte
   sólo se reutiliza si es una EVAL). Así Star1 recorta 10–14 % de los nodos con valores idénticos. Se mira primero el resultado más probable del dado.
   Variación principal: la primera jugada con ventana completa, las demás con ventana nula.
3. *Quietud* en las hojas, con *stand pat*: sólo sigue jugadas ruidosas (anotar, ataques que anotan
   y ataques con `p ≥ 0.5`), hasta 4 plies, contra el efecto horizonte.
4. *Profundización iterativa*: profundidad 1, 2, 3… hasta agotar el presupuesto; se juega la última
   búsqueda completa y se para si aparece un final forzado. Si ni la profundidad 1 termina, se juega
   la primera jugada legal: una búsqueda a medias no sirve.
5. *Tabla de transposición* de 2²⁰ entradas con clave Zobrist (la del motor oficial), valor
   exacto/piso/techo y mejor jugada; se reutiliza sólo si la profundidad guardada cubre la que falta.
   Orden de jugadas: jugada de la tabla, anotar, ataques por `p`, *killers*, historial.
6. *Reducción de jugadas tardías (LMR)*: desde la cuarta jugada del orden, una jugada tranquila (a
   casilla vacía, a más de tres filas de anotar) con 3 o más niveles por delante se busca primero
   con un nivel menos y ventana nula; sólo si supera α se busca completa. Apuesta a que el orden es
   bueno: no es exacta, así que se verifica sin ella y se mide con partidas (§7).

`herramientas/verificar_busqueda.py` comprueba que la búsqueda podada (TT viva, sin quietud ni LMR
porque la referencia no las tiene) da exactamente lo mismo que un expectiminimax sin poda en Python puro
(20 posiciones de medio juego a profundidades 1–4 y 60 finales con puntos a 1–6: 440 comparaciones,
0 diferencias) y que jugadas, probabilidades y claves coinciden con el motor oficial. `rasgos`, el 80 % del costo de un nodo, recorre el tablero sólo
dos veces (1.7× más rápida que la primera versión, con valores idénticos). En Docker la profundidad media es 6.8 (5.6 sin LMR).

## 4. Función de evaluación

68 rasgos `fᵢ` calculados desde quien mueve (propio − rival):

| Rasgos | Qué miden |
|---|---|
| Tabla peón-casilla (35: 7 filas de avance × 5 columnas simétricas), material (1), segunda fila de casa (1) | cuánto vale un peón por existir y según dónde está |
| Apoyos (2), amenazas (2), colgado (1) | peones con 1 y 2 apoyos; mayor `p` con que el rival puede atacar cada peón, y si es ≥ 0.65; peón en la mitad rival sin apoyo y atacable |
| Corredores (7) y casi corredores (3), por fila de avance | ningún rival puede frenarlo antes de anotar / sólo uno puede |
| Carrera (3) | mi corredor más cercano corona antes (o igual, porque muevo yo), el suyo antes, margen en filas |
| Mitad rival (1) y seguros (3, por fila) | peones en la mitad rival y, de ellos, los que nadie puede atacar ahora |
| Pares (1), cadenas (1) | peones lado a lado; apoyos que a su vez tienen apoyo |
| Puntos (3), cierre (1) | diferencia, tener 2, que el rival tenga 2; tener 2 y un peón a tres filas o menos de anotar |
| Movilidad (1), ataques (1), intercepto (1) | jugadas legales; Σ(p − 0.5) de mis ataques posibles; el sesgo del modelo |

Los 52 primeros salieron del curso y de mirar partidas; los otros 16, de medirnos contra el bot de
otro equipo (§8). **Fase:** cada rasgo entra dos veces, tal cual y multiplicado por `φ = 1 − peones/36`
(0 al inicio, cerca de 1 con el tablero vacío), así que tiene un peso de apertura y otro de final:
136 pesos. Una carrera de corredores casi no importa con el tablero lleno y lo decide todo al final.

**Cómo se ajustan.** Regresión logística `P = σ(k·Σ wᵢfᵢ)` sobre 1 532 007 posiciones de 16 800
partidas offline: 8 000 de autojuego a profundidad 4 con 8 % de jugadas al azar y 8 800 contra el
bot de `alondra_bernardo` (§9), estas contadas doble. La etiqueta mezcla dos: `y = 0.8·YB + 0.2·YR`.
`YB` es la probabilidad de ganar que estima una búsqueda a profundidad 4 desde esa posición (como
TD-Gammon: poco ruido, pero la EVAL hereda los puntos ciegos de nuestra búsqueda); `YR` es quién
ganó (sin ese sesgo, pero ruidosa por los dados). Sólo `YR` es mucho peor (70–330); la mezcla gana
a sólo `YB` (§7, §8). Se hizo en vueltas: generar, reajustar con todo, medir con 400 partidas y
adoptar sólo si gana. La EVAL que ve la búsqueda es `1000·(2P − 1) = 1000·tanh(k·Σ wᵢfᵢ / 2)`: los
nodos de AZAR promedian probabilidades de ganar y `|EVAL| < 1000 ≪ M`. Los pesos tienen sentido (apertura → final): un peón en la
mitad rival +118 → +621; a una fila de anotar ~250 → ~1500; ganar la carrera −102 → +472; un punto
~1055 → ~2870. Pesos chicos o de signo raro (apoyos, pares) vienen de la
correlación entre rasgos; lo que importa es la suma.

## 5. Precomputación (qué y cómo regenerarla)

`data/pesos.json` = `{"pesos": [136 valores], "logit_por_punto": k}`. Desde la raíz del repo con el
`.venv` (`H = estudiantes/pamplinotas/herramientas`), una vuelta nueva de autojuego y el ajuste:

```bash
PYTHONPATH=sistema:estudiantes/pamplinotas:$H python $H/generar_posiciones.py --partidas 2400 \
  --pesos estudiantes/pamplinotas/data/pesos.json --salida $H/posiciones/5_autojuego.npz
PYTHONPATH=sistema:estudiantes/pamplinotas python $H/ajustar_pesos.py $H/posiciones/*.npz \
  --mezcla 0.8 --salida estudiantes/pamplinotas/data/pesos.json
```

`generar_posiciones.py` guarda de cada posición el tablero y las dos etiquetas, no los rasgos, para
poder cambiar rasgos sin volver a jugar; `ajustar_pesos.py` calcula los rasgos y ajusta (L-BFGS de
scipy). Las partidas contra `alondra_bernardo` (`*_vs_alondra*.npz`) se jugaron con una variante que
carga el `strategy.py` de su carpeta; no va en el PR porque carga código de otro equipo. Jugar tarda horas y los rivales usan reloj, así que una regeneración no es bit
a bit igual; por eso entregamos `herramientas/posiciones/` (8 archivos, 10 MB, cuatro vueltas): con
el segundo comando solo, `ajustar_pesos.py` reproduce exactamente `data/pesos.json`. `arena.py` mide versiones entre sí y contra `random`/`montecarlo`;
`verificar_busqueda.py` es la prueba de corrección.

## 6. Manejo del reloj

`preparar` carga los pesos (falla si no cuadran con los rasgos) y compila numba con una búsqueda a
profundidad 3: de 3 a 7 s en Docker según la carga de la máquina (mediana 3.5; el límite es 10),
fuera del reloj; la primera jugada reporta ese
tiempo como `carga_s`. En cada jugada: disponible = restante − reserva, con reserva = 7 % del reloj
(3.5 s máximo), repartido entre las jugadas propias que faltan, `max(10, 42 − hechas)`.
Salen de medir: una partida dura 41 jugadas propias en promedio, y el reparto anterior
(`max(8, 34 − hechas)`, reserva 12 %) dejaba caer la profundidad al final; el nuevo le gana 218–182
a 6 s y 217–183 a 30 s. Dos frenos:
no se empieza una profundidad si lo transcurrido × 3 rebasa el objetivo, y un límite duro de 1.35 ×
objetivo en el que la búsqueda aborta (revisa el reloj cada 4096 nodos). Con una sola jugada legal
se responde al instante.

## 7. Resultados vs random y montecarlo

Partidas oficiales con `./torneo jugar` (Docker, reloj de 30 s, colores alternados) con la versión
final, también contra `alondra_bernardo`, el rival más fuerte que tuvimos:

| Rival | Semillas | Resultados por tanda | Total |
|---|---|---|---|
| `montecarlo` | 201, 202 | 6–2, 8–0 | **14 – 2** |
| `random` | 203 | 4–0 | **4 – 0** |
| `alondra_bernardo` | 204, 205, 206 | 4–4, 5–3, 4–4 | **13 – 11** |

(Versiones anteriores: 17–7, 12–4, 13–3 y 15–1 contra `montecarlo`; 6–10, 7–9, 11–13 y 11–13
contra `alondra_bernardo`; LMR le ganó 7–1 en Docker a la versión sin LMR. Tandas de 8 con el mismo bot van de 7–1 a 3–5: los dados pesan, y 24 partidas no
distinguen 50 % de 45 %.) Por eso cada decisión se midió en la arena con cientos de partidas
(colores alternados, 6 s salvo donde se indica):

| Enfrentamiento | Partidas | Resultado |
|---|---|---|
| versión final vs `alondra_bernardo` (al empezar: 64–136 a 30 s) | 400 a 30 s | **222 – 178** |
| la misma sin fase, mismas semillas / en 800 | 400 / 800 a 30 s | 206 – 194 / 409 – 391 |
| EVAL por fase vs sin fase | 800 / 200 a 30 s | **450 – 350 / 123 – 77** |
| LMR vs sin LMR | 400 / 200 a 30 s | **215 – 185 / 110 – 90** |
| pesos finales vs los del 6 de noviembre (sólo etiqueta de búsqueda) | 200 a 30 s | **124 – 76** |
| versión final vs `montecarlo` | 200 | 186 – 14 |
| etiqueta mezclada vs sólo búsqueda, mismos datos | 400 / 200 a 30 s | 259 – 141 / 127 – 73 |
| 9 rasgos estratégicos vs 52 rasgos | 400 / 200 a 30 s | 224 – 176 / 114 – 86 |
| época de 52 rasgos: 3× datos; búsqueda vs resultado | 400 c/u | 246 – 154; 224 – 176 |

Usa unos 23 de los 30 s por partida (28 como máximo), por diseño: la reserva de 2.1 s no se toca.

## 8. Qué intentamos y no funcionó

- **Pesos a mano.** Perdían con `montecarlo` 5–19 (dejaban pasar corredores); la primera regresión
  logística lo volteó a 20–4.
- **Medir con pocas partidas.** Con 60 u 80 partidas «descartamos» más datos (24–36); con 400
  ganaban (246–154). Desde entonces nada se decide con menos de 400. Más regularización empeoró
  (25–35, 16–44); la EVAL lineal no difiere de la logística (38–42, 39–41); con 52 rasgos y
  midiendo sólo contra nosotros mismos, mezclar etiquetas no aportó (195–205).
- **Medir sólo contra `montecarlo` tiene techo.** Con 95 % de victorias ya no distingue mejoras.
  Contra `alondra_bernardo` (mismo diseño) perdíamos 64–136 a 30 s. Descartado con 200–400
  partidas: los dados (suerte pareja), la velocidad (triple de reloj, 80–120), LMR con la EVAL
  vieja (189–211: con mal orden reduce las jugadas buenas; con la nueva gana, §7), quitar la
  quietud (187–213), prohibir ataques con p < 0.5 (75–125) y recalibrar los 52 pesos (179–221).
  Profundidad 6 contra 4 sólo dio 110–90: lo que faltaba no era táctico. Nueve rasgos nuevos
  ayudaron contra nosotros mismos (224–176) pero no contra ellos (71–129).
- **La pista que sí funcionó: comparar evaluaciones a la misma profundidad.** Con su búsqueda
  llamada a profundidad fija perdíamos 88–112 a 5 y 82–118 a 4: la diferencia estaba en la EVAL.
  Sobre posiciones de nuestras partidas contra ellos, nuestros 68 rasgos ajustados al resultado
  real predicen igual que los suyos (log-pérdida 0.498 contra 0.497), y ajustados a nuestra
  etiqueta de búsqueda, peor (0.550): no faltaban rasgos, la etiqueta enseñaba nuestros propios
  puntos ciegos. Mezclarla con el resultado (§4) volteó el marcador. Etiquetas a profundidad 5 no
  superaron a las de 4 (211–189); sólo el resultado fue mucho peor (70–330).
- **Al final ya no ganaron** (400 partidas o 200 a 30 s cada una): otro reparto del reloj (suponer
  50 jugadas: 75–125 contra ellos); una caché de evaluaciones (no más rápida: la tabla de
  transposición ya atrapa las repeticiones); la etiqueta TD(λ) (215–185 a 6 s, 96–104 a 30 s); el
  doble de tiempo en jugadas críticas (202–198, y 379–421 contra ellos); la poda de jugada nula
  (179–221: aquí ceder un turno sí cambia los ataques), y otros ajustes de LMR y de la quietud.
- **Ventanas de aspiración**: no redujeron nodos. **Star1 con `U = M`** no podaba nunca, y la
  primera cota ajustada olvidaba que con 1 punto se puede anotar dos veces en el horizonte: lo
  encontró una revisión con un contraejemplo; la prueba de finales con puntos lo vigila.
- **Más reloj ayuda poco** (20 % más: 193–207). **Seis rasgos de la primera época**: 180–220
  contra nosotros mismos, donde ambos lados comparten el punto ciego.

## 9. Fuentes y uso de IA

- Notas de la unidad «Juegos» del curso (clase 3 y mapa de decisión).
- Ballard, B. W. (1983). *The \*-minimax search procedure for trees containing chance nodes*.
  Artificial Intelligence 21.
- Ajuste de pesos estilo Texel y reducción de jugadas tardías (Chess Programming Wiki, «Texel's
  Tuning Method» y «Late Move Reductions»; LMR perdía con la EVAL vieja y gana con la nueva).
- Jugamos contra el bot del equipo `alondra_bernardo`, con el acuerdo de ambos equipos y el visto
  bueno del profesor, para medirnos y para generar partidas de entrenamiento. Lo usamos sólo como
  rival: no copiamos su código, sus pesos ni sus datos, y no se entrega. Revisamos qué mide su
  evaluación para decidir qué rasgos probar; cada uno se escribió, se verificó y se midió aquí (§8).
- Usamos Claude Code como agente de programación para escribir el motor en numba, las herramientas
  de medición y este informe, a partir del diseño que fijamos nosotros con la teoría del curso; cada
  componente se verificó contra el motor oficial y con partidas, y un segundo agente revisó la
  entrega como «profesor estricto».
