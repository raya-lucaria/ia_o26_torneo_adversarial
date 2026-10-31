# Liga y calificación

## La liga

- **Participantes:** todos los equipos que pasen la validación, más `random`, `montecarlo` y
  Tier 1 a Tier 5. Nuestros bots son participantes normales: mismo árbitro, mismas reglas, mismos
  límites.
- **Todos contra todos.** Contra cada participante juegas **4 partidas**: 2 juegos de dados × 2
  colores (ver [docs/reglas.md](reglas.md#los-dados)).
- **Victoria = 1 punto**, derrota = 0. No hay empates.
- La liga oficial corre en GitHub después de cerrar las entregas (hora límite: lunes 9 de
  noviembre de 2026, 10:00, hora de la Ciudad de México, UTC−6; ver
  [docs/entregar.md](entregar.md)). Los dados se generan con una
  semilla nueva al lanzarla, y la semilla se publica al terminar: cualquiera puede repetir cada
  partida.
- Los resultados se publican: tabla, calificación por equipo (con el nombre del equipo, sin
  claves únicas) y los visores de las partidas.

## Desempates

Solo sirven para ordenar la tabla. Si dos o más participantes tienen los mismos puntos, se ordenan
por, en este orden:

1. **Puntos entre los empatados:** las partidas que jugaron entre ellos.
2. **Calidad de victorias (Sonneborn-Berger):** por cada partida que ganaste, sumas los puntos
   finales de liga del rival al que le ganaste. Ganarle a alguien fuerte vale más.
3. **Peones anotados** en todas tus partidas.
4. **Sorteo** con la semilla de la liga.

### Ejemplo

Una liga con 4 participantes: `alfa`, `beta`, `montecarlo` y `random`. Cada pareja juega 4
partidas. Las victorias de cada fila contra cada columna:

| | alfa | beta | montecarlo | random | **Puntos** |
|---|---|---|---|---|---|
| **alfa** | — | 2 | 1 | 4 | **7** |
| **beta** | 2 | — | 2 | 3 | **7** |
| **montecarlo** | 3 | 2 | — | 4 | **9** |
| **random** | 0 | 1 | 0 | — | **1** |

`alfa` y `beta` empatan con 7 puntos.

1. **Entre ellos:** 2 a 2. Siguen empatados.
2. **Sonneborn-Berger:**
   - `alfa` ganó 2 a `beta` (7 puntos), 1 a `montecarlo` (9) y 4 a `random` (1):
     2 × 7 + 1 × 9 + 4 × 1 = **27**.
   - `beta` ganó 2 a `alfa` (7), 2 a `montecarlo` (9) y 3 a `random` (1):
     2 × 7 + 2 × 9 + 3 × 1 = **35**.

`beta` queda arriba: le ganó más veces al rival más fuerte. Tabla final: 1. `montecarlo`,
2. `beta`, 3. `alfa`, 4. `random`.

## Calificación

**«Ganarle» a un bot de referencia (`random`, `montecarlo` o un tier) = terminar la liga con
estrictamente más puntos que él.**
Empatar en puntos no es ganarle, aunque el desempate te ponga arriba.

- **Lugares 1, 2 y 3 entre equipos de alumnos** (después de desempates): **10**.
- Los demás: la calificación del **mejor bot de referencia al que le ganas**. Son escalones
  fijos, sin promedios ni puntos intermedios.

<!-- calificacion:inicio -->
| Mejor bot de referencia al que le ganas (`random`, `montecarlo` o un tier) | Calificación |
|---|---|
| ninguno | 0 |
| `random` (juega al azar) | 1 |
| `montecarlo` (Monte Carlo plano) | 3 |
| Tier 1 | 5 |
| Tier 2 | 6 |
| Tier 3 | 7 |
| Tier 4 | 8.5 |
| Tier 5 | 10 |
| **Lugares 1, 2 y 3 entre equipos de alumnos** | **10** |
<!-- calificacion:fin -->

- **Sin `INFORME.md` completo o sin `equipo.toml` válido, tu PR no se acepta y no tienes calificación.**
- Los tiers aparecen el día de la liga. Su código no se publica.

### Ejemplo

Final de una liga (solo algunas filas):

| Lugar | Participante | Puntos | Calificación |
|---|---|---|---|
| 1 | Tier 2 | 70 | — |
| 2 | `top_1` | 69 | 10 (1.º entre alumnos) |
| 3 | `top_2` | 66 | 10 (2.º entre alumnos) |
| 4 | `top_3` | 64 | 10 (3.º entre alumnos) |
| 5 | `los_alfa_beta` | 59 | **5**: tiene más puntos que Tier 1 (58) |
| 6 | Tier 1 | 58 | — |
| 7 | `ana_luis` | 58 | **3**: empata con Tier 1, no le gana; sí le gana a `montecarlo` (40) |
| 8 | `montecarlo` | 40 | — |
| 9 | `random` | 5 | — |

## Qué es trampa

**Trampa o intento de trampa = 0 en el torneo para todo el equipo.** Intentarlo cuenta igual que
lograrlo.

1. **Redes neuronales** de cualquier tipo en la liga, aunque estén escritas a mano con numpy, y
   librerías de *deep learning* (`torch`, `tensorflow`, `keras`, `jax`, `onnxruntime`…).
2. **Código ofuscado o ilegible:** cifrado, texto codificado que se ejecuta, datos que en realidad
   son código.
3. **Código nativo propio:** C, Cython, `.so`, binarios. (numba sí se vale.)
4. **Salir del aislamiento:** leer fuera de tu carpeta, usar la red, guardar información entre
   partidas, atacar al árbitro, al protocolo o a los registros.
5. **Adivinar o manipular los dados.**
6. **Copiar código o datos de otro equipo**, o compartir los tuyos antes de la entrega.
7. **Ponerse de acuerdo con otro equipo** (por ejemplo, perder a propósito): 0 para **los dos**
   equipos.

**Cómo se revisa:** el aislamiento de Docker detiene lo técnico; el validador revisa interfaz,
tamaños, `requirements.txt` y binarios; un agente de IA revisa redes neuronales, ofuscación,
trampas y que informe, código y `data/` cuadren; el profesor decide cada caso marcado antes de poner
el 0.

## Zona gris

¿No sabes si algo se vale? **Pregunta antes de entregar** (en un *issue* del repo del torneo o por
correo al profesor). La respuesta se publica para todos. Lo que preguntes a tiempo no te cuesta
nada; lo que no preguntes, se juzga con estas reglas.

Lo que sí se vale está en [docs/reglas.md](reglas.md#qué-se-vale-y-qué-no). En resumen: cualquier
búsqueda, evaluación o modelo que no sea red neuronal, todo lo precomputado en `data/` y, offline,
absolutamente todo.
