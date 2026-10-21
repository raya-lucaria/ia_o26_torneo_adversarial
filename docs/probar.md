# Probar y depurar tu bot

Todo se corre con `./torneo` desde la carpeta de tu repo (o cualquier subcarpeta). Con Docker
(lo normal), cada bot corre aislado con los límites de la liga. Agrega `--sin-docker` a cualquier
comando para correr con el Python de `.venv/` (se prepara solo la primera vez; ver
[docs/empezar.md](empezar.md#sin-docker)): el reloj y `preparar` sí cuentan, pero **no hay límite de
memoria ni de CPU**. En **Codespaces** no hay Docker y `--sin-docker` se agrega solo. Para medir con
los límites de la liga, usa Docker o la [prueba de GitHub](#pruebas-en-github).

## Comandos

| Comando | Qué hace |
|---|---|
| `./torneo jugar <a> <b>` | 4 partidas entre `a` y `b` (2 con cada color) |
| `./torneo mini <tu_equipo>` | Mini torneo contra `random` y `montecarlo`, con tabla y la calificación que tendrías |
| `./torneo validar <tu_equipo>` | Revisa tu carpeta mientras trabajas: un informe sin terminar solo es aviso ⚠ |
| `./torneo validar <tu_equipo> --entrega` | Revisa exactamente lo que revisará tu PR (ver [docs/entregar.md](entregar.md#qué-revisa-el-validador)) |
| `./torneo jugar humano montecarlo` | Juegas tú en la terminal: la mejor forma de aprender el juego |
| `./torneo liga <a> <b> <c> ...` | Liga todos contra todos entre los participantes que digas |
| `./torneo repetir <archivo.json>` | Repite en la terminal una partida guardada en `resultados/` |
| `./torneo visor <carpeta>` | Vuelve a generar el `index.html` de una carpeta de `resultados/` |
| `./torneo diagnostico` | Revisa tu máquina, `.venv/`, git y tu carpeta, y dice qué hacer. No cambia nada |
| `./torneo instalar` | Crea o repara `.venv/`: el Python del torneo que usan `--sin-docker`, tu editor y tus scripts. Después: `source .venv/bin/activate` en cada terminal |
| `./torneo actualizar` | Trae la última versión del curso (nunca toca tu carpeta) y hace el commit |
| `./torneo construir` | Reconstruye desde cero las imágenes de Docker del torneo y la caché de numba (unos minutos). Úsalo si algo raro pasa con Docker o después de actualizar el sistema |
| `./torneo --help` | Ayuda de todos los comandos (y `./torneo jugar --help`, etc.) |

Ejemplos:

```bash
./torneo jugar los_alfa_beta montecarlo
./torneo jugar los_alfa_beta montecarlo --partidas 10
./torneo jugar los_alfa_beta random --semilla 7 -v
./torneo jugar los_alfa_beta montecarlo --ver
./torneo mini los_alfa_beta
./torneo validar los_alfa_beta
./torneo validar los_alfa_beta --entrega
./torneo liga los_alfa_beta random montecarlo --juegos-dados 3
./torneo jugar los_alfa_beta montecarlo --sin-docker
```

Opciones de `jugar`:

- `--partidas N`: cuántas partidas (4 si no lo dices). Los colores se alternan.
- `--semilla N`: fija los dados. Con la misma semilla, los dados salen igual. Tu bot puede jugar
  distinto si depende del reloj (piensa más o menos según la velocidad de la máquina).
- `-v`: una línea por jugada (ver abajo).
- `--ver`: el tablero en vivo, jugada a jugada.

`mini` no acepta `--partidas`: siempre juega 2 partidas por pareja (una con cada color). `mini <tu_equipo>` juega 6 (3 parejas × 2); `mini` a secas (random contra montecarlo) juega 2.

### Cuánto tarda

Cada bot tiene 30 s de reloj por partida, así que una partida puede durar hasta un minuto.

| Comando | Tiempo aproximado |
|---|---|
| `./torneo jugar <a> <b>` (4 partidas) | 2 a 4 minutos |
| `./torneo mini <tu_equipo>` | 3 a 10 minutos |

- **Mientras programas:** usa `--partidas 1` o `--partidas 2` para ver rápido si algo se rompió.
- **Para comparar dos versiones:** usa `--partidas 10` o más. Con pocas partidas, quién gana es
  sobre todo suerte de los dados.

La primera vez tarda más: construye las imágenes en tu máquina (la primera vez unos minutos) y
prepara numba
([docs/empezar.md](empezar.md#7-crea-tu-carpeta-y-juega-tu-primera-partida)).

Para comparar dos versiones de tu bot, copia tu carpeta con otro nombre
(`cp -r estudiantes/los_alfa_beta estudiantes/los_alfa_beta_v1`) y juega una contra otra.
Juega al menos 10 partidas (`--partidas 10`). **No subas la copia** (no la agregues con `git add`): la prueba de GitHub valida cada carpeta de
`estudiantes/`, y la copia falla porque su `equipo.toml` dice otro nombre. Si quieres subirla,
cambia `nombre` en su `equipo.toml` al nombre de la copia. Tu pull request solo puede traer tu
carpeta.

### El resultado de `mini` es corto

`mini` juega solo 2 partidas contra cada rival. Con tan pocas partidas el resultado es sobre todo
azar: una copia sin cambios de `montecarlo` puede ganarle o perderle. Para comparar de verdad, usa
más partidas:

```bash
./torneo jugar los_alfa_beta montecarlo --partidas 10
```

## Leer la salida

```text
Partidas: los_alfa_beta vs montecarlo · 4 partidas · semilla 81723
Reloj: 30 s por partida para cada bot · preparar: 10 s · con Docker

#  Blancas        Negras         Ganador        Puntos  Jugadas   Reloj máx
1  los_alfa_beta  montecarlo     los_alfa_beta     3–1       77  21.3 s (B)
2  montecarlo     los_alfa_beta  montecarlo        3–0       64  28.1 s (N)
...
⚠ partida 2: los_alfa_beta usó 28.1 de 30 s
Resultado: los_alfa_beta 3 – 1 montecarlo
Repeticiones: …/resultados/2026-10-20_17-03-11/index.html  (ábrelo en tu navegador)
Para repetir exactamente estas partidas: --semilla 81723
```

- **Puntos:** puntos de Blancas–Negras al final de la partida.
- **Reloj máx:** el mayor reloj que usó un bot en esa partida, y de qué color era (`B` o `N`).
- **⚠:** tu bot usó más del 90 % de su reloj. En la liga, un poco más y pierdes.
- **✗:** alguien perdió por algo que no es el juego (reloj, jugada ilegal, error, memoria,
  `preparar`). El mensaje dice qué pasó, en qué jugada y qué hacer, y muestra lo último que imprimió
  ese bot.

## `-v`: una línea por jugada

```text
jug  quién          jugada  prof  nodos   eval     t  reloj
  1  los_alfa_beta  e2-d3      4  12034   0.52  0.71    0.7
  2  montecarlo     c8-c7      -      -  0.413  0.84    0.8  muestras=120
```

- `t`: segundos que tardó esa jugada. `reloj`: reloj usado en total hasta ahí.
- `prof`, `nodos`, `eval`: las tres columnas fijas. Muestran lo que tu bot pasó a `reportar(...)`
  con esos nombres en ese turno. Si no reporta ese dato, sale `-`. Repórtalo: te ayuda a entender
  qué piensa tu bot.
- Cualquier otro dato que pases a `reportar` (por ejemplo `muestras=...`) aparece al final de la
  línea, como información extra, y también en el visor.

```python
from motor import reportar

def jugar(estado, tiempo_restante):
    ...
    reportar(prof=profundidad, nodos=contador, eval=valor)
    return mejor
```

Con `-v` también ves el resultado de cada combate y, aparte, lo que imprimió cada bot con `print`.

## El visor

Cada corrida guarda sus partidas en `resultados/<fecha_hora>/`. Abre `index.html` con doble clic
(no necesitas instalar nada): tiene la lista de partidas y un enlace a cada una.

<!-- TODO(profesor): agrega docs/img/capturas/visor.png y quita este comentario.
![Captura: Visor de una partida con el tablero, los botones de avance, la lista de jugadas y la gráfica de evaluación](img/capturas/visor.png) -->

En el visor de una partida:

- **⏮ ◀ ▶ ⏭** y las flechas del teclado para avanzar o retroceder. **Reproducir** (barra
  espaciadora) con velocidad ajustable.
- **Siguiente combate** (tecla `C`) salta al próximo ataque.
- Capas: **Apoyo** de cada peón, **p de cada ataque** posible y **Jugadas legales**.
- La lista de jugadas con cada combate (su `p` y si ganó o perdió), el reloj de cada bot y lo que
  reportó tu bot en cada turno.
- La gráfica de `eval` si tu bot la reporta.

Cada partida también queda en un `.json`. Para verla otra vez en la terminal:

```bash
./torneo repetir resultados/2026-10-20_17-03-11/0_0_los_alfa_beta_vs_montecarlo.json
```

`mini` y `liga` guardan además `parejas/N.json` (las partidas de cada pareja de equipos). Cada
partida de ahí también se repite:

```bash
./torneo repetir resultados/2026-10-20_17-03-11/parejas/0.json
```

## Cómo depurar

1. **Lee el mensaje ✗.** Dice por qué perdió tu bot, en qué jugada y lo último que imprimió.
2. **Usa `print`.** Lo que imprimes no rompe el juego; lo ves con `-v` (hasta 64 KB por partida).
3. **Prueba sin el árbitro.** Arma una posición y llama tu función directo en Python:

   ```python
   from motor import Estado
   import strategy                     # corre esto desde estudiantes/<tu_equipo>/

   from pathlib import Path

   class Config:                       # lo mismo que recibe preparar en la liga
       color, semilla, reloj_s = "blancas", 1, 30.0
       carpeta_datos = Path("data")

   strategy.preparar(Config())
   e = Estado.desde_dict({"blancas": ["e4", "d3"], "negras": ["f5"], "turno": "blancas"})
   print(e)
   print(strategy.jugar(e, 30.0))
   ```

   Para que `from motor import ...` funcione fuera de `./torneo`, instala el motor una vez
   ([docs/avanzado.md](avanzado.md#usar-el-motor-en-tu-editor)).
4. **Repite los dados** con `--semilla N` para volver a ver el mismo problema.
5. **Juega tú contra tu bot** (`./torneo jugar humano <tu_equipo>`) y llévalo a posiciones raras.
6. **Mide con Docker.** `--sin-docker` no limita memoria ni CPU: un bot que ahí va bien puede
   pasarse de reloj o de memoria en la liga. En Codespaces, mide con la
   [prueba de GitHub](#pruebas-en-github).

Motivos de derrota y qué hacer:

| Motivo | Qué pasó | Qué hacer |
|---|---|---|
| reloj | Se acabaron tus 30 s | Usa `tiempo_restante` para repartir tu tiempo y deja reserva |
| ilegal | Regresaste algo que no está en `estado.jugadas()` | Regresa una `Jugada` de `estado.jugadas()` o su texto `"e2-e3"` |
| error | Tu código lanzó una excepción o tu proceso se cerró | Lee el traceback que muestra el mensaje ✗ |
| memoria | Pasaste de 1024 MB | Limita el tamaño de tus tablas y árboles |
| preparar | `preparar` más la importación de `strategy.py` tardaron más de 10 s | Carga menos o más rápido; no hagas trabajo pesado al importar |

## Pruebas en GitHub

Tu repo privado trae una prueba automática. Cada *push* a `main` que cambie algo en
`estudiantes/` la lanza: valida tu carpeta (como `./torneo validar`, sin `--entrega`) y juega partidas contra `random` y `montecarlo` con los
límites de la liga. Tarda unos minutos. También puedes lanzarla a mano: **Actions → prueba → Run
workflow**.

Para ver el resultado:

1. En tu repo, abre la pestaña **Actions**.
2. Entra a la corrida más reciente.
3. Baja al **resumen** (*Summary*). Tiene una tabla con ✅ y ⚠️:
   - **Validación:** si tu carpeta pasa el validador.
   - **Carga:** si `preparar` terminó a tiempo.
   - **Contra random** y **Contra montecarlo:** cuántas partidas ganaste (2 contra cada uno).
     ✅ contra `random` si ganaste todas; contra `montecarlo`, si ganaste más de la mitad.
   - **Reloj máx:** el mayor reloj que usaste en una partida.
   - **Memoria pico (informativa):** cuánta memoria usó tu bot.
4. Abajo, en **Artifacts**, descarga `repeticiones`: es un `.zip` con los visores de esas
   partidas (abre su `index.html`).

<!-- TODO(profesor): agrega docs/img/capturas/actions_resumen.png y quita este comentario.
![Captura: Resumen de una corrida de la prueba en la pestaña Actions, con la tabla de validación y resultados](img/capturas/actions_resumen.png) -->

Si la corrida sale en rojo (✗), abre el paso que falló: el mensaje dice qué arreglar. Si dice
**«No encontré tu carpeta en estudiantes/»**, tu carpeta no está en ese commit de `main`: casi
siempre falta el `git add` / `commit` / `push` de `estudiantes/<tu_equipo>/` (revísalo con
`git ls-files estudiantes/`), o la subiste a otra rama. Las cuentas
personales tienen minutos gratis de Actions al mes para repos privados; cada corrida usa unos
pocos minutos.
