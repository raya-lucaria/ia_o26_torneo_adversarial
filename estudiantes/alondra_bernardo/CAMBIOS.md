# Bitácora: del Monte Carlo a la búsqueda en numba

Registro de los cambios y de las pruebas desde el 30 de octubre de 2026, cuando dejamos el
bot de Monte Carlo + UCB y empezamos un bot nuevo desde cero, basado en búsqueda. Todo
está en la rama `experimento-busqueda-numba`; la versión entregada está en `main`.

## Versiones

Los pesos de cada versión están en `herramientas/pesos/`; el que usa el bot es
`data/pesos.json`. Cada versión adoptada tiene una etiqueta git local con su nombre.

| Versión | Commit | Pesos | Arena (contra la anterior) | Docker contra montecarlo / fuerte |
|---|---|---|---|---|
| v1, v2 | `5f1928b` | `v1.json`, `v2.json` | v2 a v1: 167–103 | 21–11 |
| r63 | `3b6c683` | `r63.json` | r63 a v2: 187–173 | — |
| r63b | `8481589` | `r63b.json` | — | 73–23 |
| r63b + LMR | `bf8cd0d` | `r63b.json` | 197–163 | 81–15 |
| mezcla 30 % | `c1f19b7` | `mezcla30.json` | 220–140 | 77–11 / 39–9 |
| m7 | `e354866` | `m7.json` | 210–150 | 80–16 / 42–6 |
| m8 + reloj nuevo | `b6a7183` | `m8.json` | 204–156 contra m7; contra mc:0.4, 284–76 (m7: 272–88) | 83–13, y 80–16 con 96 semillas nuevas (7001–7012) / — |
| r69 | ver Historia | `r69.json` | 391–329 contra m8 (720 partidas, con 12 % menos nodos) | 78–18 (8001–8012) y 82–14 (7001–7012); contra ab_actual 119–73 y 59–37 con semillas nuevas (4601–4612) / 121–23 en 144 |
| **sc400** (actual) | ver Historia | `sc400.json` | 190–170 contra r69 (360 a 300k nodos); contra pamplinotas 614–298 y torneo_breakthrough 487–425 en 912 (r69: 597–315 y 474–438) | 81–15 (10001–10012); contra ab_actual 60–36 / — |

## De dónde salió la idea

El bot de Monte Carlo + UCB se había estancado en ~52 % contra `montecarlo`, porque el
rival oficial usa prácticamente el mismo algoritmo. Lo medimos así:

- Darle 3 veces más tiempo ayudaba mucho a 0.05 s por jugada (46–14), pero casi nada a
  0.6 s (32–28): el Monte Carlo plano se satura.
- Ninguna variante nuestra (reloj, Ridge reentrenado, rondas cortables, expectimax en
  Python) se separó con claridad:

| Variante anterior | Docker contra montecarlo |
|---|---|
| reloj | 25–15, 55–45, 43–57 |
| Ridge reentrenado | 21–19, 51–49 |
| expectimax en Python + Ridge | 15–25 |
| ronda cortable | 22–18 |

Leímos el `INFORME.md` de otros dos equipos (solo los informes; **no** abrimos su código ni
usamos sus datos). De ahí tomamos ideas generales, todas publicadas en la literatura de juegos:

- expectiminimax con poda (Star1, Ballard 1983);
- una evaluación logística sobre rasgos;
- pesos entrenados con etiquetas de la propia búsqueda (estilo Texel / TD-Gammon).

El código, los rasgos y los pesos son nuestros: escritos desde cero sobre el motor
oficial (`motor.nucleo`) y entrenados con nuestras propias partidas.

## Cómo funciona el bot nuevo

**`rapido.py`** es la búsqueda, compilada con numba:

- negamax con nodos de azar: el valor de un ataque es p·V(gana) + (1−p)·V(pierde);
- alfa-beta con poda Star1 en los nodos de azar;
- quietud de hasta 4 plies con jugadas que anotan o atacan con p ≥ 0.5;
- tabla de transposición con las claves Zobrist del motor;
- ordenamiento de jugadas: jugada de la tabla, anotar, atacar, asesinas, historial;
- reducción de jugadas tardías (LMR);
- un tope de nodos para abortar, porque numba no puede leer el reloj.

**La evaluación** es `tanh(0.5·w·f)` sobre 63 rasgos hechos a mano:

- tabla de avance × columna;
- apoyos y amenazas;
- corredores y peones seguros;
- puntos;
- ataques posibles;
- peones colgados, falange, bloqueos y movilidad;
- carrera a la meta.

**`strategy.py`** organiza el tiempo de cada jugada:

- `preparar` compila numba y mide nodos por segundo, fuera del reloj;
- `jugar` usa profundización iterativa con un presupuesto por jugada, según las piezas
  que quedan;
- deja una reserva del 10 % del reloj (mínimo 1 s).

**`data/pesos.json`** guarda los pesos entrenados que se usan hoy (m7).

**`herramientas/`**:

| Archivo | Para qué sirve |
|---|---|
| `generar_posiciones.py` | Autojuego (o contra Monte Carlo con `--contra-mc`); guarda posiciones con la etiqueta de la búsqueda |
| `ajustar_pesos.py` | Ajusta los pesos (entropía cruzada logística, L-BFGS); `--mezcla` combina la etiqueta con el resultado de la partida |
| `arena.py` | Duelos rápidos entre dos versiones a nodos fijos, sin Docker |
| `verificar_rapido.py` | Compara la búsqueda con un expectiminimax sin poda en Python |
| `posiciones_archivo.py` | Convierte partidas archivadas a posiciones |

`verificar_rapido.py` da diferencia 0 en 180 posiciones.

## Historia de cambios

| Commit | Cambio | Resultado |
|---|---|---|
| `80eb698` | Búsqueda nueva desde cero con pesos a mano | Docker 7–25 contra montecarlo |
| `5f1928b` | Dos vueltas de entrenamiento (v1, v2) | Arena: v1 a mano 147–33, v2 a v1 167–103. Docker 21–11 |
| `3b6c683` | Nueve rasgos nuevos (63 en total) | Arena: r63 a v2 187–173 |
| `8481589` | **Error corregido:** si la tabla cortaba en la raíz, no había jugada y se jugaba la primera legal (~20 % de los turnos). Además, más reloj y pesos r63b (etiquetas a profundidad 5) | Docker de 27–21 a **41–7** con las mismas semillas. Confirmación: 73–23 en 96 partidas nuevas, 8–0 contra random. Total 114–30 (79 %) |
| `bf8cd0d` | Reducción de jugadas tardías (LMR) | Arena 197–163. Docker **81–15 (84 %)** con las mismas 96 semillas que dieron 73–23 |
| `8d302d9` | Opciones nuevas en herramientas (`--lmr`, `+lmr`, `+qN`) | — |
| `c1f19b7` | Etiquetas mezcladas: 70 % valor de la búsqueda + 30 % resultado final de la partida | Arena 220–140 contra r63b. Docker **77–11** contra montecarlo (mismas semillas; una tanda se cayó por Docker) y **39–9** contra el montecarlo fuerte (antes 34–14) |
| `e354866` | Vuelta 7 (m7): autojuego con el bot nuevo + LMR a profundidad 6, mezcla 30 % | Arena 210–150; Docker 26–22 en duelo directo; contra montecarlo 80–16, contra el fuerte 42–6 |
| m8 | Pesos m8: reentrenados desde cero solo con las posiciones de la vuelta 7 (mezcla 30 %). Reloj: reserva del 5 % (antes 10 %), jugadas restantes al percentil 75 en vez de la mediana y objetivo × 1.3 (`USO`). Salida de emergencia a 1 ply en vez de la primera jugada legal | Arena 204–156 contra m7; contra Monte Carlo (mc:0.4, 600 mil nodos) 284–76, m7 272–88. Docker contra montecarlo con las mismas 96 semillas: **83–13 (86.5 %)**, m7 80–16. Solo los pesos con el reloj anterior: 72–16 en 88. Reloj medio 23.2 s, máximo 28.3 s |
| `42e4c65` | Profundidad abortada: si se acaba el tiempo a media profundidad, juega la mejor de las jugadas de la raíz que sí terminaron (la mejor anterior va primero). Freno 3 → 2 y `USO` 1.3 → 1.1 | Docker: 85–11 con las semillas de siempre, pero 79–17 con las nuevas (m8: 83–13 y 80–16). En total 164/192 (85.4 %) contra 163/192 (84.9 %): igual. Se queda porque es correcta y no empeora nada. Reloj medio 24 s, máximo 28.5 s |
| reloj con sobrecosto | Cada jugada cuesta ~40 ms fijos en Docker aunque no pensemos, y contra bots que buscan las partidas duran 110–136 jugadas: con la reserva del 5 % llegamos a 29.5 s y jugábamos el final con ~10 ms por jugada. Ahora se apartan 50 ms por cada jugada que falta y el mínimo de jugadas restantes sube de 6 a 10 | Docker contra ab_actual (m7), 192 partidas con semillas 4301–4312 y 4401–4412: **101–91**, reloj máx 28.4 s (ninguna ≥ 29 s). Con reserva 8 % en las mismas semillas: 99–93, máx 28.8 s. Con el reloj anterior y semillas 4001–4012: 56–40, máx 29.5 s; contra rival_ataque 55–41 y rival_defensa 51–45. Nivel real contra m7: ~53 % |
| r69 | 6 rasgos nuevos (69 en total): casi corredor (un solo rival en el cono, avance 4–6), margen de la carrera, material × fase y «dos puntos y un peón en avance ≥ 5». Pesos entrenados con v6_m3 + v9_m8 + v10_p7 (650 k posiciones, mezcla 0.3). La evaluación cuesta ~9 % de nodos/s | Arena contra m8 con 0.88× nodos: 196–164 y 195–165. Con solo v6_m3: 178–182 (los rasgos nuevos necesitan más datos). Docker con las semillas 4301–4412 contra ab_actual: **119–73 (62 %)**, m8 101–91; contra montecarlo 78–18 (m8 81–15). Reloj máx 29.0 s. Confirmación con semillas que no se usaron para elegir: ab_actual **59–37**, montecarlo **82–14**, reloj máx 28.6 s. Contra rival_ataque 57–39 (m8 55–41) y rival_defensa 60–36 (m8 51–45), mismas semillas. Contra el montecarlo fuerte, pareado con m8 en 144 partidas (3001–3018): 121–23 contra 124–20, igual |

Notas sobre las pruebas:

- **Arena:** 360 partidas a 300 mil nodos por jugada, colores alternados.
- **Docker:** `./torneo jugar alondra_bernardo montecarlo` en tandas de 8 partidas, con
  semillas fijas para poder comparar versiones.
- **Ruido:** con 360 partidas, ±19 victorias es ruido normal (2 desviaciones estándar).
- **Docker en paralelo:** con 6 tandas a la vez, a veces un bot tarda más de 10 s en
  `preparar` (numba compila en cada partida porque el torneo ignora `__pycache__`; solo
  tarda ~3.7 s). `tandas.sh` corre 4 a la vez y repite las tandas con ese fallo.

| sc400 | 8 pesos de combate (apoyo1, apoyo2, amenaza, amenaza grave, ataques posibles, colgados, falange, bloqueados) afinados con SPSA (`herramientas/spsa_combate.py`): 400 vueltas desde r69, cada una con 8 partidas contra pamplinotas, 8 contra torneo_breakthrough (reloj de 6 s, con el acuerdo de esos equipos y el visto bueno del profesor) y 8 entre las dos versiones, a 60k nodos. Cambios: apoyo1 +0.016 → −0.003, apoyo2 −0.040 → −0.031, amenaza +0.088 → +0.094, amenaza grave +0.049 → +0.041, ataques +0.473 → +0.474, colgados +0.014 → +0.033, falange −0.015 → −0.020, bloqueados 0.000 → −0.023 | Arena con reloj real de 30 s (`herramientas/reloj_arena.py`), dos semillas de 360 por rival: pamplinotas **486–234** (r69 477–243), torneo_breakthrough **388–332** (r69 376–344). Contra r69 a 300k nodos: 190–170. Docker con semillas nuevas: pamplinotas 66–30 y torneo_breakthrough 47–49 (10001–10012); pareado con r69 en 11001–11012: pamplinotas 62–34 (r69 63–33), torneo_breakthrough 52–44 (r69 44–52). Contra montecarlo 81–15 y ab_actual 60–36. Total contra los dos rivales: 1101 contra 1071 de r69 en 1 824 partidas, +30, dentro del ruido (±59): mejor versión medida, no mejora demostrada. Reloj máx 28.7 s |

## Ideas que probamos y descartamos

| Idea | Prueba | Resultado |
|---|---|---|
| Vuelta 5 de pesos (r63c, profundidad 5) | Arena contra r63b | 165–195 |
| LMR más agresivo (reduce 2 niveles) | Arena contra LMR | 178–182 |
| Quietud de 6 plies | Arena contra 4 | 171–189 |
| Quietud de 2 plies | Arena contra 4 | 168–192 |
| Vuelta 6 de pesos (r63e, profundidad 6 + LMR) | Arena 189–171; Docker | 77–19, peor que 81–15 |
| Ventana nula (PVS) | Arena contra LMR | 168–192 |
| Mezcla 15 % y 45 % (en vez de 30 %) | Arena contra mezcla 30 % | 167–193 y 180–180 |
| Mezcla 60 % | Arena contra r63b | 210–150, peor que el 30 % |
| Rasgos de carrera garantizada (65 rasgos, r65) | Docker contra montecarlo / fuerte | 83–13 / **35–13**, peor que m7 (80–16 / 42–6) |
| Pesos con 600 partidas contra Monte Carlo (mc8) | Arena contra m7 | 166–194 |
| Tabla de transposición de 2^22 (en vez de 2^20) | Docker en duelo directo | 43–45, y usa más reloj (media 26.5 s, máx 28.0) |
| Pesos por fase (apertura y final mezclados según peones y puntos; se quitó del código) | Arena contra m7, con un control reentrenado sin fases con los mismos datos | 202–158, pero el control da 204–156: las fases no aportan; la ventaja es del reentrenamiento |
| Pesimismo con nuestros ataques (usar p − 0.05 en la búsqueda; se quitó del código) | Arena contra mc:0.4 | 271–89 contra 272–88 sin pesimismo |
| Pesos con autojuego + posiciones contra Monte Carlo | Arena contra mc:0.4 | 282–78, igual que m8 |
| mc8 contra Monte Carlo (antes solo se probó contra m7) | Arena contra mc:0.4; Docker con el reloj de m8 y las mismas 96 semillas | Arena 288–72, pero en Docker **79–17**, peor que m8 (83–13); además pierde 166–194 contra m7 |
| Corredores en O(1) con la fila del rival más adelantado por columna | Mismos rasgos en 8 520 posiciones, pero más lento | 507 ns por evaluación contra 399: crear los arreglos cuesta más de lo que ahorra |
| Vuelta 9 de autojuego con m8 (3000 partidas, prof 6, 276 k posiciones) | m9 = v6_m3 + v9, m9_solo = solo v9, ambos con mezcla 0.3; arena contra m8 a 300k nodos | m9 **156–204**, m9_solo 179–181: más vueltas de autojuego ya no mejoran los pesos |
| Promedio de pesos m8 y m9_solo | Arena contra m8 (720 partidas); Docker con el reloj nuevo contra ab_actual (semillas 4301–4412) y montecarlo (8001–8012) | Arena 387–333; contra ab_actual 106–86 (m8: 101–91), pero contra montecarlo **72–24** (m8: 81–15): gana un poco en el duelo y pierde contra Monte Carlo |
| Jugadas restantes ajustadas a partidas contra bots que buscan (lineal: 2 + 1.9·(piezas − 16)), y el promedio de esa y la actual | Docker contra ab_actual (semillas 4301–4412) y montecarlo (8001–8012), comparado con el reloj con sobrecosto (101–91 y 81–15) | Lineal: 108–84 y 75–21 (contra montecarlo deja ~9 s sin usar). Promedio: 97–95 y 77–19. Todo cabe en el ruido (±14 en 192 partidas): el reparto del reloj ya no se puede medir con Docker |
| Jugada nula (null-move pruning, R = 2, solo si la evaluación estática ≥ beta; se quitó del código) | Arena contra m8 a 300k nodos | 175–185: no ahorra lo suficiente para compensar sus errores |
| Autojuego a profundidad 7 (1500 partidas, 135 k posiciones) como maestro más fuerte | p7 = solo esos datos, p7_v6 = con v6_m3, mezcla 0.3; arena contra m8 a 300k nodos | p7 186–174, p7_v6 164–196: un maestro más profundo tampoco mejora los pesos |
| Vuelta 11 de autojuego con r69 (3000 partidas, prof 6, 272 k posiciones) | r69b = v6_m3 + v9 + v10 + v11, r69c = v9 + v10 + v11, mezcla 0.3; arena contra r69 a 300k nodos | r69b 184–176, r69c 181–179: con los 69 rasgos, más datos tampoco mejoran los pesos |
| 4 rasgos más (columnas sin peones, peones en casa cuando el rival está cerca, mejor ataque propio y mejor ataque rival, por separado); ~2 % más lento; se quitaron del código | r73 con los 4 juegos de datos, mezcla 0.3; arena contra r69 a 300k nodos | **173–187**; la validación casi no cambia (0.1150 → 0.1149) y dos de los cuatro pesos quedaron en ~0 |
| Star2: antes de buscar un ataque que no es la primera jugada, sondear cada resultado con solo la primera respuesta del rival; si la cota ya no supera alfa, se descarta (se quitó del código) | Comparación exacta contra expectiminimax sin poda hasta profundidad 4 (240 casos, diferencia 0); nodos hasta profundidad 7 en 40 posiciones; arena contra r69 a 300k nodos, dos semillas | 33 % menos nodos para la misma profundidad, pero **173–187 y 174–186** (347–373 en 720): a nodos fijos juega un poco peor, quizá porque los sondeos ensucian las asesinas, el historial y la tabla |
| Doble de tiempo en jugadas críticas (peón rival en avance ≥ 6 o alguien con 2 puntos), commit `a8e871a`, revertido | Docker contra r69 con las mismas semillas: montecarlo 9101–9212 (192), ab_actual 9301–9312 (96). Antes, diagnóstico de las 32 derrotas de r69 contra montecarlo: 28 de 31 se voltean justo después de perder un combate, en posiciones parejas (valor ~+0.1, jugada ~61), con ~11 s de reloj sin usar | montecarlo **169–23 contra 165–27**, ab_actual **54–42 contra 61–35**: nada claro a favor y quizá peor contra bots que buscan. Ojo: en el control muchas tandas fallaron en `preparar` y se repitieron; se cuentan solo las válidas |
| Ventaja asumida (*contempt*): sumar un sesgo al logit de la evaluación a favor del lado de la raíz, para preferir posiciones tranquilas a volados contra un rival más débil (se quitó del código) | Comparación exacta con sesgo 0 y 1 (diferencia 0); arena contra Monte Carlo plano (mc:0.4) a 600k nodos, 720 partidas | Sesgo 0: **582–138**; sesgo 0.7: **579–141**. Igual; no se probó 1.4 |
| Meta de 75 % contra pamplinotas y torneo_breakthrough (r69: 65.5 % y 52.0 %) | Muchas pruebas, abajo. Arena con reloj real de 30 s salvo donde se indica; contra cada rival, dos semillas de 360 | Ninguna llega; la mejor es sc400 (67.3 % y 53.4 %) |
| Pesos con 1500 partidas contra pamplinotas (r75, 137 k posiciones) | Contra pamplinotas, dos semillas; contra torneo_breakthrough y r69 | 476–244 (r69 477–243); 184–176 (r69 185–175); 361–359 contra r69. Igual |
| Más autojuego (r69b) contra los rivales | Dos semillas por rival | 474–246 y 389–331. Igual |
| Repartir el reloj hacia el final (más jugadas restantes estimadas) | Contra pamplinotas, 360 | 216–144 y 222–138 (r69 223–137). Igual |
| Más reloj para nosotros (2× y 4×) | r69 con 60 s y 120 s contra 30 s del rival | Pamplinotas 64.2 % y 65.6 %; torneo_breakthrough 53.8 % y 53.8 %. El tiempo casi no cuenta |
| Quietud que sigue avances a 1 o 2 pasos de la meta; tabla que siempre reemplaza; LMR desde la 3.ª o la 6.ª jugada; poda en la frontera | Contra r69, 720 partidas con reloj de 10 s | 368–352, 356–364, 358–362, 352–368, 336–384, 354–366. Igual o peor |
| Evaluación con máscaras de bits (mismos rasgos en 2.9 M posiciones, misma búsqueda en 200; 1.47 M contra 1.12 M nodos/s) | Contra r69 (720) y dos semillas por rival | 364–356; pamplinotas 457–263, torneo_breakthrough 369–351. Más rápida sin ganar más; no se integró |
| Paquete: sc400 + máscaras de bits + quietud con avances | Dos semillas por rival | 475–245 y 376–344. No aporta |
| SPSA de los 69 pesos contra sí mismo (sb500) | Contra r69 y dos semillas por rival | 390–330 contra r69 a 300k, pero pamplinotas 467–253 y torneo_breakthrough 349–371: se especializó contra sí mismo |
| SPSA de 29 pesos contra los rivales, 800 vueltas más desde sc400 (sd800) | Dos semillas por rival; contra r69 | 489–231 y 380–340; 187–173. Igual que sc400 |
| *Comparison training*: pesos para que, a 1 ply, las jugadas queden ordenadas como con una búsqueda a profundidad 5 (40 k posiciones de autojuego) | Coincidencia de la mejor jugada en posiciones nuevas; partidas | Coincidencia 32.5 % → 35.0 %, pero contra r69 174–186 (λ 0.1) y 99–261 (λ 0.01); contra los rivales 225–135 y 168–192 en 360. Peor: cambia la escala que usan los nodos de azar |
| Tabla de patrones 2×3 (40 k pesos) y un modelo flexible sobre el tablero, solo como medición fuera del bot | Error contra el resultado en 150 k posiciones nuevas | 0.5371 con r69, 0.5373–0.5388 con patrones, peor con el modelo flexible: la evaluación ya no deja información aprovechable |
| Auditoría de 192 partidas contra los dos rivales (búsqueda fija a profundidad 5) | Valor perdido por elegir mal contra valor movido por los dados | Nosotros 0.48–0.69 por partida, rivales 0.67–0.83, dados 2.6–2.8; ~0.1 errores grandes por lado. Contra pamplinotas atacamos con p media 0.61 (rival 0.53); contra torneo_breakthrough 0.57 (rival 0.55) |
| Pesos con 20 % de posiciones contra Monte Carlo (r74): 3000 partidas de r69 contra mc:0.4 (240 k posiciones, etiquetas a profundidad 6) más los 4 juegos de autojuego, mezcla 0.3 | Arena a 600k nodos contra mc:0.4 (720), contra r69 a 300k (360) y contra mc:0.1 a 120k (360); control r69b = mismos datos sin las partidas contra Monte Carlo | r74: **589–131**, 172–188 y 313–47; r69: 582–138 y 320–40; r69b: **595–125**. Todo en el ruido; los datos contra Monte Carlo no aportan |

Las vueltas de autojuego dejaron de mejorar la evaluación después de r63b.

## Reloj

En las 96 partidas de Docker con LMR:

- usa en promedio 23.7 s de 30;
- el máximo fue 27.4 s;
- no perdió ninguna partida por tiempo.

En una jugada, la búsqueda llega a profundidad 7–9, a ~1.3–1.45 millones de nodos por
segundo.

## Cómo regenerar los pesos

```bash
source .venv/bin/activate
H=estudiantes/alondra_bernardo/herramientas
python $H/generar_posiciones.py --partidas 3000 --prof 5 --azar 0.08 --procesos 9 \
    --semilla 5 --pesos estudiantes/alondra_bernardo/data/pesos.json --salida v.npz
python $H/ajustar_pesos.py v.npz --salida pesos_nuevos.json
python $H/arena.py pesos_nuevos.json+lmr estudiantes/alondra_bernardo/data/pesos.json+lmr \
    --partidas 360 --nodos 300000
```

## Cuánto vale el tiempo

Duelo a 600 mil contra 300 mil nodos por jugada: **191–169**. Contra nosotros mismos, duplicar
los nodos casi no ayuda.

**Contra Monte Carlo sí ayuda:** m7 a 1.2 millones de nodos le gana 293–67 a mc:0.4, contra
272–88 a 600 mil. Por eso m8 usa más del reloj (ver «Historia»).

## Por qué perdemos contra montecarlo

Análisis de las 16 derrotas de m7 en Docker (de 96):

- 11 fueron con Negras.
- En las derrotas atacamos más (7.9 ataques por partida, p media 0.59) que en las victorias
  (4.7, p 0.64), y perdimos unos 2 combates netos más de lo esperado.
- La caída decisiva de la evaluación casi siempre es un ataque desesperado desde nuestra
  última fila contra un peón que se coló, y que luego anota.
- Hasta la jugada 50 no hay diferencia en peones en casa, intrusos ni material entre partidas
  ganadas y perdidas: las derrotas se deciden tarde y por dados.
- Con el reloj anterior las jugadas 40 en adelante recibían 0.04–0.17 s.

**Ojo con elegir con las mismas semillas:** con m8 y con la profundidad parcial, las 96
semillas de siempre dieron 83–13 y 85–11, y las 96 nuevas (7001–7012) 80–16 y 79–17. Con 96
partidas, ±7 victorias es ruido, y elegir la versión que mejor salió en unas semillas favorece a
la que tuvo suerte con esos dados. El nivel real de ambas es ~85 %.

## Rivales de otros estilos

Son carpetas locales `estudiantes/rival_*` que **no se suben**. Resultados en Docker con los
pesos r63b (16 partidas cada uno, salvo el fuerte):

| Rival | Estilo | Resultado |
|---|---|---|
| `rival_agresivo` | Expectimax a 2 plies que captura y avanza | 16–0 |
| `rival_defensivo` | Cuida su material y su casa | 14–2 |
| `rival_corredor` | Mete peones a la meta lo antes posible | 12–4 |
| `rival_ataque` | Nuestro motor con pesos inclinados al ataque | 8–8 |
| `rival_defensa` | Nuestro motor con pesos inclinados a la defensa | 9–7 |
| `rival_mcfuerte` | Montecarlo con el doble de tiempo y más simulaciones | 34–14 |

Los rivales que empatan son los que usan nuestro propio motor. Eso indica que lo que más pesa
es la búsqueda, no el estilo. Con m7, contra el montecarlo fuerte, el resultado subió a 42–6.

## Comparación con los otros dos equipos

Según lo que cada equipo reporta en su informe, en Docker contra `montecarlo`:

| Equipo | Resultado |
|---|---|
| Nosotros | 80–16 (83 %) con m7; 77–11 (87.5 %) con mezcla 30 % |
| pamplinotas | 19–5 (79 %) |
| t_regex | 15–5 (75 %) |

Las muestras de los otros equipos son pequeñas, así que la comparación es aproximada.
