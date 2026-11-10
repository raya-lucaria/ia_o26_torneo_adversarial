# Informe: torneo_breakthrough

## 1. Integrantes

- Israel Sánchez Falcón, clave única 220461.
- Alejandro Ozymandias Cepeda Beltrán, clave única 219451.

## 2. Idea en un párrafo

Nuestro bot usa **expectiminimax con poda alfa-beta**. Mira varias jugadas hacia adelante: en sus
jugadas toma el máximo, en las del rival el mínimo y en los ataques el **promedio**, porque esos los
decide un dado. Como el árbol no cabe, **corta y evalúa** con una función de evaluación (EVAL)
aprendida con datos, que estima la probabilidad de ganar. La idea que guió el diseño salió de medir:
**el juego es una carrera con defensa**. Los peones solo avanzan, así que un corredor que rebasa a un
defensor ya no puede ser atacado por él. Por eso la EVAL mira sobre todo corredores, la defensa de la
fila de casa y el marcador, más que contar peones. Cada cambio entró solo si ganó partidas en
nuestras pruebas.

## 3. Búsqueda (algoritmo y por qué)

Elegimos el algoritmo con las cuatro preguntas de *Elegir el algoritmo*:

| Pregunta | En Breakthrough de combate | Consecuencia |
|---|---|---|
| ¿Cabe el árbol? | No: hay ≈28 jugadas posibles por turno | Minimax con corte |
| ¿Hay dados? | Sí: un ataque se gana con probabilidad p | Nodos de azar: expectiminimax |
| ¿Hay una EVAL buena? | No venía dada; la aprendimos (§4) | Cortar y evaluar, en vez de simular |
| ¿Hay reloj? | Sí: 30 s por partida | Profundización iterativa |

- **Expectiminimax con alfa-beta.** Un ataque es un nodo de azar y vale
  p·V(gana) + (1 − p)·V(pierde). Como todo valor está entre −1 (perder) y +1 (ganar), también se
  poda dentro de un ataque: si tras el primer resultado del dado ya no puede cambiar la decisión, no
  se busca el otro (Star1; Ballard 1983).
- **Orden de jugadas y tabla de transposición.** Alfa-beta poda más si prueba primero las mejores
  jugadas. Prueba primero la mejor que ya conoce, luego las que anotan o atacan y luego las que
  cortaron en otras ramas. La tabla guarda cada posición ya valorada con su valor, piso o techo.
- **Búsqueda de quietud**, contra el efecto horizonte. En las hojas no evalúa a media pelea: sigue
  hasta 4 jugadas más, solo las que anotan y los ataques con p ≥ 0.5.
- **Rápido y verificado.** Usa bitboards en numba: ≈7 M nodos por segundo, 6–7 jugadas de
  profundidad más la quietud. Coincide con el motor oficial en 263 487 jugadas, y la poda da el mismo
  valor que expectiminimax sin poda (probado a profundidad 2 y 3).
- **Por qué no MCTS.** Según el curso, Monte Carlo y UCT convienen cuando no hay una EVAL buena.
  Nosotros sí la tenemos, y con el mismo reloj alfa-beta le ganó 60 % a nuestro MCTS con nodos de
  azar.

## 4. Función de evaluación

**Qué estima.** La probabilidad de que gane quien mueve, P = σ(z) = 1 / (1 + e^(−z)); la búsqueda usa
V = 2P − 1. Es una probabilidad porque expectiminimax promedia, y con azar importan las distancias
entre valores, no solo su orden.

**Qué mira.** z es una suma ponderada (modelo lineal, sin redes neuronales):
- **46 rasgos:** peones por fila, apoyos, peones atacables, corredores (peones que ya nadie alcanza)
  según su distancia a la meta, la penúltima fila, las orillas, los huecos de la defensa, el marcador
  y el turno.
- **Pesos según el marcador:** cada bando usa los de su situación (sus puntos y los del rival). Un
  peón libre en la penúltima fila vale +0.25 en 0-0 y +2.70 en 2-2.
- **Patrones de 3×3:** un peso por cada forma posible de una ventana de 3×3 casillas, según su fila y
  el marcador. Así ve combinaciones de apoyos que los rasgos no ven.

**Cómo se aprendió**, con regresión logística:
- **Qué se quiere aprender:** f(posición) = probabilidad de ganar.
- **Datos:** 42 000 partidas del bot contra sí mismo.
- **Etiquetas.** Primero, quién ganó (método de Texel). Pero ese resultado depende de dados que aún no
  salen y es muy ruidoso. Después, el valor de una búsqueda mucho más profunda desde esa posición, de
  50 000 y luego 500 000 nodos (TreeStrap).
- **Ajuste.** Minimizamos la log-pérdida con regularización L2 para limitar la capacidad, y medimos el
  error en partidas apartadas.
- **El error que importa son las partidas.** Una versión que predecía mejor el resultado jugaba peor.
  Por eso cada cambio se decidió jugando.

| Ronda | Qué cambió | Resultado |
|---|---|---|
| 1 | Aprender del resultado (16 000 partidas) | montecarlo: 68.5 → 78.1 % |
| 2 | 13 rasgos de corredores (de analizar derrotas) | +56 Elo |
| 3 | Imitar búsquedas de 50 000 nodos | +23 Elo |
| 4 | Pesos por marcador y patrones; búsquedas de 500 000 nodos | +78 Elo (gana 61 %) |
| final | Patrones por marcador; 200 000 posiciones más | montecarlo: 87.0 → 90.5 % |

## 5. Precomputación (qué y cómo regenerarla)

- **Qué hay en `data/`.** `pesos.npy` tiene los 1 102 662 pesos de la EVAL (9 × 46 de los rasgos y 4
  tablas de patrones). `pesos.json` tiene los 9 × 46 con sus nombres. `pesos_46.npy`/`.json` son los
  46 pesos de la ronda 3, con los que se hicieron las búsquedas de las etiquetas.
- **`preparar`** carga `pesos.npy`, lo pasa a enteros de 16 bits y compila numba.
- **Cómo se regenera.** Con el código de `herramientas/`, desde la raíz del repo y con `.venv` (≈10 h
  en una Mac de 10 núcleos). Las partidas son del bot contra sí mismo, con aperturas al azar.

```bash
H=estudiantes/torneo_breakthrough/herramientas
# Ronda 2: regresión logística sobre 16 000 partidas del bot de la ronda 1
python $H/ajustar_pesos.py partidas_lineal_16k.npz --guardar-en pesos_iter2.npy
# Partidas de la ronda 2: 10 000 para entrenar y 6 000 para validar
python $H/generar_partidas.py --partidas 10000 --semilla 10 --pesos pesos_iter2.npy --salida partidas_iter2_b.npz
python $H/generar_partidas.py --partidas 6000 --semilla 9 --pesos pesos_iter2.npy --salida partidas_iter2_a.npz
# Ronda 3: etiquetas de búsquedas de 50 000 nodos
python $H/etiquetas_busqueda.py --nodos 50000 --posiciones 400000 --lam 0.3 --procesos 10 \
    --pesos-busqueda pesos_iter2.npy --validar partidas_iter2_a.npz --guardar-en pesos_busqueda.npy \
    partidas_lineal_16k.npz partidas_iter2_b.npz partidas_v2_16k.npz
cp pesos_busqueda_lam0.30.npy estudiantes/torneo_breakthrough/data/pesos_46.npy
# Rondas 4 y 5: partidas de la ronda 3 y etiquetas de búsquedas de 500 000 nodos
python $H/generar_partidas.py --partidas 10000 --semilla 11 --procesos 10 \
    --pesos estudiantes/torneo_breakthrough/data/pesos_46.npy --salida partidas_ronda4.npz
python $H/etiquetas_busqueda.py --nodos 500000 --posiciones 400000 --lam 0 0.3 --procesos 10 \
    --semilla 5 --pesos-busqueda estudiantes/torneo_breakthrough/data/pesos_46.npy \
    --validar partidas_iter2_a.npz --guardar-en pesos_profundas.npy partidas_ronda4.npz partidas_iter2_b.npz
# (lo mismo con 200 000 posiciones de nuestras partidas de prueba → pesos_lab_etiquetas.npz)
python $H/ajustar_marcador.py pesos_profundas_etiquetas.npz pesos_lab_etiquetas.npz --guardar
```

Con los mismos `.npz` salen exactamente los mismos pesos (lo comprobamos). Las 32 000 partidas
iniciales (`partidas_v2_16k.npz`, `partidas_lineal_16k.npz`) las jugaron versiones anteriores del
bot, así que desde cero salen pesos equivalentes, no idénticos.

## 6. Manejo del reloj

- **Profundización iterativa.** Busca a profundidad 1, 2, 3… y entrega la jugada de la última
  búsqueda completa. No empieza otra si ya usó la mitad del tiempo de la jugada, y corta una que se
  pase del doble.
- **Tiempo por jugada:** (tiempo restante − 4 s) / máx(12, 42 − jugadas hechas), y nunca más del 15 %
  de lo disponible. Una partida típica tiene 37–40 jugadas nuestras.
- **Reserva de 4 s** que nunca se gasta, por la comunicación con el árbitro y porque la máquina de la
  liga puede ser más lenta. Con 3 s llegamos a 26.9 s, al borde del aviso.
- **Medido.** En 2 000 partidas de prueba, máximo 25.9 s y ninguna derrota por tiempo. La versión
  final usó como máximo 25.4 s en Docker y 24.6 s en la prueba de GitHub Actions (la máquina de la
  liga por omisión). `preparar` tarda 2.9–6.3 s de sus 20, y la memoria es ≈100 MB de 1024.

## 7. Resultados vs random y montecarlo

| Rival | Versión | Partidas y condiciones | Resultado |
|---|---|---|---|
| random | final | 6: Docker (4) y GitHub Actions (2) | **6–0**, todas 3–0 |
| montecarlo | final | 4: Docker, `./torneo jugar`, semilla 6 | **4–0** (3–2, 3–0, 3–2, 3–1) |
| montecarlo | final | 2: prueba de GitHub Actions | 1–1 (3–1 y 1–3) |
| montecarlo | a mano / ronda 1 / ronda 2 | 400 c/u: pruebas propias, reloj de 30 s | 68.5 / 78.1 / 82.3 % |
| montecarlo | ronda 3 / 4 / final | 400 c/u: pruebas propias, reloj de la liga | 84.0 / 87.0 / **90.5 ± 1.9 %** |
| ronda 4 | final | 500: pruebas propias, reloj de 30 s | 50.1 ± 2.1 % |

- **Cómo hicimos las pruebas propias.** Con dados, pocas partidas engañan. Jugamos en pares, con los
  mismos dados y la misma apertura y los colores invertidos, y restamos la suerte de cada combate (lo
  que salió menos lo esperado según p). Decidimos con una prueba secuencial (SPRT) o con cientos de
  pares. El «reloj de la liga» imita su máquina, más lenta: montecarlo juega con 13 s y nosotros con
  15.3 s.
- **Llegamos a 90 % contra montecarlo**, pero justo: el intervalo va de 88.6 % a 92.4 %. Casi todas
  las derrotas llegan con los dados en contra (78 de 79 en una corrida de 400 partidas).
- **Contra la versión anterior** la final queda pareja (50.1 %); la elegimos porque contra montecarlo
  sube de 87.0 % a 90.5 % con los mismos dados. Con la ronda 3, contra nuestros prototipos (alfa-beta
  con EVAL a mano, MCTS y un híbrido) ganamos 74–84 %, y 62–82 % aun dándoles 4× tiempo.

## 8. Qué intentamos y no funcionó

- **Una EVAL «de libro» (material y avance):** 4 % contra montecarlo. Faltaba valorar la fila de
  casa; con ella, 54 %. El juego no es de material.
- **MCTS con simulaciones al azar:** 44 % contra montecarlo. Un híbrido MCTS–alfa-beta: 34–38 %
  contra alfa-beta en su mejor variante de 7.
- **Búsqueda de quietud sin poda en los ataques:** se comía el 98 % de los nodos.
- **Podar o extender más** (jugada nula, reducir jugadas tardías, futilidad, extensiones): ninguna
  ganó. La fuerza vino de la EVAL: buscar 15× más nodos vale solo +109 Elo.
- **Una tabla de «ventaja» contra montecarlo:** solo parecía ganar porque la medimos con los dados
  con los que la ajustamos. Lección: medir siempre con datos apartados.
- **Libro de aperturas, más tiempo en jugadas críticas y rasgos de «quién anota primero»:** sin
  mejora. Con el reloj real (≈2.5 M nodos por jugada) la búsqueda ya corrige los errores chicos de la
  EVAL; solo se notaron las mejoras grandes.

## 9. Fuentes y uso de IA

**Del curso.** Juegos: *Cuando decide un dado*, *Cortar y evaluar*, *Hasta dónde mirar*, *MCTS con
UCT* y *Elegir el algoritmo*. Aprendizaje de máquina: qué se quiere aprender, el error que importa y
cómo limitar la capacidad.

**Artículos.** Búsqueda con dados: Ballard (1983) y Hauk, Buro & Schaeffer (2006). Tabla de
transposición con azar: Veness & Blair (2007). Alfa-beta: Marsland & Campbell (1982), Zobrist (1970),
Akl & Newborn (1977) y Schaeffer (1989). Breakthrough: Lorentz & Horey (2014) y Saffidine, Jouandeau
& Cazenave (2012). MCTS: Lanctot et al. (2014) y Baier & Winands (2018). EVAL aprendida: Österlund
(2014), Veness, Silver, Uther & Blair (2009) y Buro (1998). Pruebas: Wald (1945) y Zinkevich et al.
(2006).

**Código.** No copiamos código de internet. Escribimos las reglas en bitboards a partir del motor
oficial y usamos sus tablas Zobrist, para que nuestras claves coincidan con `estado.clave`.

**Uso de IA.** Usamos Claude Code (Claude Opus 5.5).
- **Qué hizo.** Investigó la literatura, escribió el código y corrió las mediciones con los criterios
  que fijamos nosotros. Lo verificamos jugada por jugada contra el motor oficial y cada cambio con
  partidas.
- **Errores suyos que encontramos al medir.** Una EVAL que perdía 96 % contra montecarlo, una clave
  Zobrist con el tipo equivocado y pruebas inválidas (por el reloj o por medir con los dados del
  ajuste). Las repetimos.
- **Quién decidió.** Nosotros: alfa-beta y no MCTS, aprender la EVAL, los criterios para aceptar un
  cambio y qué versión entregar.

**Participación de cada integrante.** Los dos decidimos juntos la estrategia, los criterios para
aceptar un cambio y qué versión entregar, y revisamos las mediciones y el informe. Israel Sánchez
Falcón llevó el trabajo con Claude Code; Alejandro Ozymandias Cepeda Beltrán probó la versión final
en su máquina.
