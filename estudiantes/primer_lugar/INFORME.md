# Informe del equipo primer_lugar

## 1. Integrantes

Paris López y Alejandro Martínez. Sus claves únicas están en `equipo.toml`.

## 2. Idea en un párrafo

El bot combina búsqueda adversarial con azar explícito, una evaluación manual
del tablero y una colección pequeña de finales resueltos. Su objetivo es ganar:
terminales 1/0 y sin descuento temporal. Conserva tablero completo, turno y ambos
marcadores. La versión entregable mantiene los pesos manuales porque los
experimentos de aprendizaje no demostraron una mejora suficiente para sustituirlos.
El evaluador estima la calidad de la posición; su salida no es una probabilidad
de victoria calibrada. No utiliza redes neuronales.

## 3. Búsqueda (algoritmo y por qué)

Se usa expectiminimax con profundización iterativa. El valor se expresa desde
quien mueve; después de cambiar el turno, el valor sucesor se convierte con
`1 - V`. Cada acción se respalda con `Q(s,a) = suma p(s'|s,a) * (1 - V(s'))`
y se elige el máximo. Los combates conservan ambos resultados y sus probabilidades
oficiales. Si pierde el atacante, se elimina al atacante, no al defensor.

Esto aplica inducción hacia atrás sobre las ramas exploradas. El grafo del juego
es acíclico: avanzar, capturar o anotar reduce la suma de distancias restantes a
la meta. Sin embargo, la búsqueda se corta por profundidad y tiempo; usar una
heurística en sus hojas no resuelve el juego completo. Sólo los finales calculados
con todas sus dependencias tienen valores resueltos.

Las ramas deterministas usan ventanas alfa-beta. Los hijos de azar se buscan con
ventana completa; una acción se descarta cuando su contribución máxima restante
ya no puede superar una alternativa. Se detectan victorias inmediatas ciertas en
la raíz. La tabla de transposición tiene 50 000 entradas y 2 050 000 bytes de
almacenamiento: verifica hash, identidad completa de 22 bytes y profundidad.
No guarda cortes incompletos como valores exactos. Se devuelve la mejor acción
de la última iteración completa, con una acción legal de respaldo si falta tiempo.

## 4. Función de evaluación

Se calcula una combinación lineal de rasgos normalizados y una sigmoide sobre
su resultado, acotado antes a `[-20,20]`. El formato admite 150 rasgos; sólo
36 tienen peso distinto de cero: 18 propios y sus equivalentes rivales con
signo opuesto. Los demás siguen en cero y no contribuyen.

Los pesos propios son: puntos 5; material 1; los tres avances máximos 0.7, 0.6
y 0.7; movilidad 0.5; apoyos 0.5; anotaciones disponibles 3; tercer punto 5;
ataques 0.2; mejor ataque 0.1; riesgo -1; amenazados -0.3; bloqueados -0.3;
contacto frontal -0.2; corredores estáticos 0.5; avance de corredor 0.5; y
avance teniendo dos puntos 1. No se ajustan durante una partida.

Se normaliza por 3 puntos, 18 peones, 8 filas de avance, 54 movimientos o
36 apoyos/ataques, según el rasgo. Anotaciones y tercer punto ponderan las
posibilidades de anotar por su probabilidad. El riesgo usa, por peón, el mayor
ataque rival disponible. Los corredores describen caminos con el tablero
actual: no certifican que el rival permanezca inmóvil ni que lleguen a la meta.

## 5. Precomputación (qué y cómo regenerarla)

`data/modelo.json` contiene los pesos definidos en `geometria.pesos_iniciales_v4`.
`data/finales.npz` contiene 10 161 estados y sus valores `float64`, con identidad
completa. Se consulta sólo con hasta tres peones y coincidencia exacta; una
posición parecida no hereda una solución. `neutral.json` y `referencia.json`
conservan referencias históricas de pesos; esta estrategia no las carga.

`herramientas/estados_finales.npy` fija las identidades del corpus sin valores
objetivo. El generador comprueba que el corpus sea cerrado y vuelve a resolver
todas las acciones y los resultados de azar, memoizando dependencias. No convierte
un corte por presupuesto en una solución. Con el motor oficial instalado:

```bash
python herramientas/regenerar.py --verificar
python herramientas/regenerar.py --salida datos_regenerados
```

La verificación compara los valores bit a bit. El hash lógico de los arreglos
está en `herramientas/origen_datos.json`; el ZIP puede cambiar de fecha sin
cambiar sus valores. El mismo generador reproduce las dos referencias históricas.

## 6. Manejo del reloj

El reloj es de 30 segundos por jugador y partida. `preparar`, incluida la
importación, tiene un límite independiente de 20 segundos según la regla vigente;
ahí se cargan datos y se calientan las funciones Numba. Las comparaciones antiguas
se hicieron con el límite anterior de 10 segundos de preparación.

Se reservan tres segundos del reloj. En cada turno, el presupuesto es el mínimo
entre 1.2 segundos, el 10% del disponible y el disponible dividido entre
`max(12,48-turnos)`. Este último término se multiplica por 1.5 si alguien tiene
dos puntos. La búsqueda respeta el plazo y conserva sólo iteraciones completas.
El bot usa un CPU, no piensa durante el turno rival, no escribe archivos y no
accede a la red en las partidas.

## 7. Resultados vs random y montecarlo

En el experimento cerrado de Monte Carlo guiado, con 416 partidas entre ambas
versiones y cero fallos, esta manual ganó 128/200 contra Monte Carlo (64%) y
8/8 contra Random. La variante guiada ganó 105/200 y 8/8. La diferencia guiada
menos manual tuvo intervalo bootstrap pareado del 95% de -19 a -3.5 puntos
porcentuales. Ambas usaron Docker, 30 segundos de juego, un CPU por bot y el
límite entonces vigente de 10 segundos para preparar.

La comparación posterior con búsqueda compilada se detuvo en 348/416 registros
por un fallo de preparación de la manual. Excluyendo el bloque completo afectado
de ambas versiones, quedaron 86 semillas con ambos colores: manual 117/172
(68.0%) y compilada 116/172 (67.4%). No se ejecutó Random y la validación quedó
incompleta; no justificó sustituir la manual.

Dos pilotos directos posteriores, de 20 partidas cada uno, terminaron sin fallos:
manual 16/20 contra V1 y 16/20 contra V2. Usaron 30 segundos y preparación de
20 segundos. Ese 80% es contra nuestras versiones antiguas, no contra Monte Carlo.
Las muestras se informan por separado, sin elegir la más favorable ni mezclar
turnos como observaciones independientes. No garantizan el rendimiento de la liga.

## 8. Qué intentamos y no funcionó

Los valores aprendidos con fuentes mezcladas no superaron la manual. El ajuste
SPSA empató 21/40 contra Monte Carlo con su control. Caché, ordenamiento y Star1
ahorraron cálculo, pero su confirmación terminó 135/200 para ambas versiones.
El reloj adaptativo obtuvo 126/200 frente a 127/200. El Monte Carlo guiado
retrocedió. La candidata Numba aceleró búsquedas, pero su validación incompleta
no probó más victorias. Por eso conservamos el paquete manual, sin entrenamiento
adicional ni cambios basados en los pilotos contra V1/V2.

## 9. Fuentes y uso de IA

Base de reglas, motor y ejecución: [repositorio del profesor](https://github.com/raya-lucaria/ia_o26_torneo_adversarial).
Base conceptual: [apuntes de juegos y Markov de la clase](https://github.com/raya-lucaria/ia_o26).
La inducción hacia atrás y el corte con evaluación aproximada proceden de esos temas.
Se utilizó [Numba](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
para compilar rasgos numéricos, sin `fastmath`.

Para los experimentos descartados se consultaron Lanctot et al.,
[Implicit Minimax Backups](https://mlanctot.info/files/papers/cig14-immcts.pdf);
Cazenave, [Sequential Halving applied to Trees](https://www.lamsade.dauphine.fr/~cazenave/papers/shot.pdf);
y Tesauro y Galperin, [On-line Policy Improvement using Monte-Carlo Search](https://proceedings.neurips.cc/paper_files/paper/1996/file/996009f2374006606f4c0b0fda878af1-Paper.pdf).
Orientaron pruebas, no implementaciones completas de esos artículos. Se utilizó
Codex para investigación, programación, pruebas, análisis y redacción. El informe
distingue lo implementado de lo explorado y limita las conclusiones a los registros.
