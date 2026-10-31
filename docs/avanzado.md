# Avanzado

Para cuando tu bot ya juega y quieres que piense más rápido o mejor. Nada de esto es obligatorio.

## Usar el motor en tu editor

Dentro de `./torneo` el motor ya está instalado. Para que tu editor lo reconozca (autocompletado,
`from motor import ...` sin errores) y para correr tus scripts de `herramientas/`, crea una vez el
entorno `.venv/` desde la raíz de tu repo y actívalo en cada terminal nueva (detalles en
[docs/empezar.md](empezar.md#el-entorno-de-python-venv)):

```bash
./torneo instalar
source .venv/bin/activate          # en cada terminal nueva
```

Luego elige ese Python en tu editor (en VS Code: **Python: Select Interpreter** → `.venv`).
En **Codespaces** ya está hecho: `.venv/` se crea al abrirlo y las terminales nuevas lo activan.

Con eso también puedes usar `--sin-docker` (ver [docs/probar.md](probar.md)).

## Velocidad

Tienes 30 s por partida y unas 40 jugadas propias: menos de 1 s por jugada. Antes de optimizar,
mide.

- **Mide** con `reportar(nodos=..., prof=...)` y `-v`: cuántos nodos por segundo, qué profundidad.
- **Reparte el reloj.** No todas las jugadas valen lo mismo: piensa más cuando importa y guarda
  reserva para el final. `tiempo_restante` siempre te dice cuánto queda. Con profundización
  iterativa siempre tienes una respuesta lista cuando se acaba tu tiempo.
- **Cuida la reserva.** El reloj incluye la comunicación con el árbitro (unos milisegundos por
  jugada) y la máquina de la liga puede ser algo más lenta que la tuya. Mira la columna
  **Reloj máx** y los avisos ⚠.
- **Usa lo rápido del motor.** `simular_al_azar` ya está compilado con numba. `estado.clave`
  sirve como llave de una tabla de transposición (un `dict`).
- **Python puro es lento** para recorrer millones de nodos. Si tu búsqueda es el cuello de botella,
  considera numba.

## numba

numba compila funciones de Python con números y arreglos de numpy a código máquina. Está
preinstalado y se vale (es la única forma permitida de compilar).

```python
import numpy as np
from numba import njit

@njit(cache=True)
def suma_avance(filas: np.ndarray) -> int:
    total = 0
    for f in filas:
        total += f
    return total
```

- **La primera llamada compila**, y compilar tarda (a veces segundos). En la liga cada partida es un
  proceso nuevo y tu carpeta es de solo lectura, así que **tu código numba se compila en cada
  partida**.
- Por eso, **llama tus funciones numba una vez en `preparar`** (20 s, fuera del reloj), como hace
  `montecarlo` con `simular_al_azar`. Lo que no compiles ahí se compila en tu turno y te cuesta
  reloj.
- Pero `preparar` también tiene límite: si compilar todo tarda más de 20 s, pierdes la partida.
  Mide cuánto tarda con `./torneo validar` y la columna **Preparar máx** de `./torneo jugar` (con
  Docker), y lee [Numba que compila rápido](#numba-que-compila-rápido).
- `numba` solo entiende un subconjunto de Python: números, arreglos de numpy, tuplas, ciclos. No
  entiende los objetos `Estado`. Si quieres tu propia representación rápida, mira
  [Modificar el motor](#modificar-el-motor).

## Numba que compila rápido

Compilar es lo más lento de `preparar`, y en la liga tarda más que en tu laptop:

- **En la liga tu código numba se compila desde cero en cada partida.** El motor (`simular_al_azar`
  y lo demás) ya viene compilado, pero la caché de numba empieza vacía para tu código: todo lo que
  compila tu `preparar` cuenta en sus 20 s. `cache=True` no ayuda ahí.
- **Sin Docker parece más rápido de lo que es:** tu carpeta se puede escribir, así que con
  `cache=True` numba guarda lo compilado y las partidas siguientes ya no compilan. Mide con Docker
  (o con la prueba de GitHub) y fíjate en la primera partida.
- **Las máquinas de la liga varían.** La liga corre en máquinas de GitHub: unas son más lentas que
  otras, y al compilar numba pueden ser hasta ~2,5× más lentas que una laptop rápida. Si tu
  `preparar` tarda 10 s en tu máquina, en la liga puede pasar de 20 s. Por eso `./torneo validar`
  avisa (⚠) cuando tu `preparar` pasa de la mitad del límite.
- **Usa el reloj, no una profundidad fija.** Por la misma razón, una búsqueda a profundidad fija que
  en tu máquina tarda 0.5 s por jugada puede tardar bastante más en la liga. Decide cuánto pensar con
  `tiempo_restante` (profundización iterativa) y así tu bot se ajusta a la máquina que le toque.

Para que compile rápido:

- **No crees arreglos dentro de `@njit` en el camino caliente** (`np.empty`, `np.zeros`, rebanadas
  como `tablero[a:b]`, tuplas que se arman en cada llamada). Además de ser lentos al correr, hacen
  más grande lo que numba compila. Preasigna los búferes en Python (en `preparar`) y pásalos como
  argumentos.
- **Una sola firma por función.** numba compila una versión por cada combinación de tipos con la
  que llamas una función. Llámala siempre con los mismos tipos: no mezcles arreglos de solo lectura
  y escribibles, `int32` e `int64`, ni números de Python (literales) con enteros de numpy.
- **En `preparar` compila exactamente lo que vas a usar:** llama cada función con argumentos de los
  mismos tipos que usarás al jugar. Si en tu turno la llamas con otros tipos, se compila otra vez y
  te cuesta reloj.
- **Mide:** `./torneo validar` te dice cuánto tardó tu `preparar` y avisa si estás cerca del
  límite; en `./torneo jugar` mira la columna **Preparar máx**.
- **No metas una caché precompilada de numba en `data/`** (archivos `.nbi`/`.nbc` u otro código
  compilado): se considera código nativo propio, que está prohibido (ver
  [docs/calificacion.md](calificacion.md#qué-es-trampa)). `data/` es solo para datos.

## Precomputar

Lo que tu bot sepa de antemano lo calculas **en tu computadora, antes de entregar**, y lo guardas
en `data/`:

- libros de aperturas (qué jugar en las primeras jugadas);
- tablas de finales;
- pesos de tu función de evaluación, ajustados con partidas;
- tablas de patrones.

Reglas:

- El código que genera `data/` va en `herramientas/` y tu informe explica cómo regenerarlo
  (sección 5). `herramientas/` no se ejecuta en la liga.
- `data/` se sube con tu carpeta: 100 MB en total y 50 MB por archivo. Cualquier formato
  (`.npy`, `.npz`, `.json`, `pickle`…).
- `data/` son **datos**, no código escondido.
- En la liga se lee en `preparar`, que tiene 20 s. Mide cuánto tarda cargar tus archivos.

```python
import numpy as np

def preparar(config):
    global pesos
    pesos = np.load(config.carpeta_datos / "pesos.npy")
```

**Offline todo se vale:** en tu computadora puedes usar cualquier librería, todos tus núcleos y
todo el tiempo que quieras para generar `data/`. Lo único que se limita es lo que corre en la liga.
Lo que corre en la liga no puede ser una red neuronal, aunque la hayas entrenado offline.

## Librerías (`requirements.txt`)

Ya vienen instaladas, con la misma versión para todos: **numpy, scipy y numba**. No las pongas en
`requirements.txt`.

Si tu bot necesita otra librería **en la liga**, declárala en `estudiantes/<tu_equipo>/requirements.txt`:

```text
# una por línea, con versión exacta
sortedcontainers==2.4.0
```

- Una librería por línea, con **versión exacta**: `paquete==1.2.3`. Sin rangos (`>=`), sin
  opciones, sin URLs. Las líneas que empiezan con `#` son comentarios.
- Solo de PyPI y solo paquetes ya compilados (*wheels*): no se compila nada al instalar.
- A lo más **300 MB** instalados, contando sus dependencias.
- No puede cambiar la versión de numpy, scipy, numba ni llvmlite (la usa numba).
- **Lista negra** (no se instalan, ni como dependencia de otra):
  `torch`, `tensorflow`, `keras`, `jax`, `jaxlib`, `onnxruntime` (redes neuronales),
  `cython`, `pythran`, `mypyc` (otros compiladores), `ray`, `dask`, `requests`, `httpx`,
  `urllib3`, `aiohttp`.
- En la liga no hay red: todo se instala antes, al construir la imagen de tu equipo.
  `./torneo` la reconstruye sola cuando cambias `requirements.txt`.
- **Sin Docker** (Codespaces o `--sin-docker`): vuelve a correr `./torneo instalar` cada vez que
  cambies tu `requirements.txt`, para instalarla en `.venv/`.

¿Quieres una librería que no se puede? Pídela. Si se aprueba, queda para todos.

## Modificar el motor

Puedes usar, modificar o reescribir el motor. Lo que **no** puedes es cambiar el `motor` instalado:
ese es el del árbitro, y el PR solo puede tocar tu carpeta.

Para tener tu propia versión, **cópiala a tu carpeta con otro nombre**:

```bash
cp sistema/motor/nucleo.py estudiantes/<tu_equipo>/mi_motor.py
```

y en tu `strategy.py`:

```python
import mi_motor            # tu copia: cámbiala como quieras
from motor import Estado   # el oficial sigue disponible
```

No la llames `motor`: chocaría con el oficial. Tu bot puede importar cualquier `.py` de su carpeta.

Tu copia es para pensar más rápido. La jugada que regresas la revisa siempre el árbitro con el
motor oficial: si tu copia tiene un error y propone una jugada ilegal, pierdes la partida. `jugar`
puede regresar el texto de la jugada (`"e4-f5"`), así que no necesitas la clase `Jugada` del motor
oficial.

`sistema/motor/nucleo.py` tiene las reglas en funciones numba sobre un arreglo de 81 casillas
(`fila * 9 + columna`). Es un buen punto de partida.
