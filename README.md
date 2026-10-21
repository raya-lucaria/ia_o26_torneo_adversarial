# Breakthrough de combate — Torneo IA O26

Vas a programar, en equipo de 2, un bot que juega **Breakthrough de combate**: un juego de peones
en un tablero de 9 × 9 donde cada ataque se decide con un dado. No existe en internet: no hay motor
ni libro que copiar. Tu bot compite en una **liga** todos contra todos con los bots de tus
compañeros y con los nuestros. Tu calificación depende de a cuáles de nuestros bots les sacas
estrictamente más puntos. Tú piensas la estrategia; el motor del juego, el árbitro y las herramientas ya están hechos.

## El juego en un minuto

Cada bando tiene 18 peones. Blancas empiezan en las filas 1 y 2; Negras, en las 8 y 9.
Por turno mueves **un peón una casilla hacia adelante**: recto a una casilla vacía, o en diagonal
a una casilla vacía o atacando a un peón rival. Nunca hacia atrás ni de lado, y no se ataca recto.

![Un peón avanza recto o en diagonal a una casilla vacía, o ataca en diagonal a un rival](docs/img/moverse.svg)

**Combate.** Atacar es tirar un dado. Cada peón tiene un **apoyo**: cuántos peones propios tiene
en sus dos diagonales de atrás (0, 1 o 2). La probabilidad de ganar el ataque es
`p = 0.5 + 0.15 × (apoyo del atacante − apoyo del defensor)`. Si ganas, el defensor sale del
tablero y tu peón ocupa su casilla. Si pierdes, sale **tu** peón.

![Combate: el atacante tiene apoyo 2, el defensor apoyo 1, así que p = 0.65](docs/img/combate.svg)

**Puntos.** Un peón que llega a la última fila rival anota 1 punto y sale del tablero.
Gana quien llega a **3 puntos** o deja al rival **sin jugadas**. No hay empates.

![Un peón blanco llega a la fila 9, anota un punto y sale del tablero](docs/img/anotar.svg)

Reglas completas, con ejemplos resueltos: [docs/reglas.md](docs/reglas.md).

## Empieza en 3 pasos

> [!IMPORTANT]
> **Trabaja en un repo privado.** Un *fork* de un repo público es público: cualquiera vería tu
> código. Crea tu repo con «Use this template» y márcalo **Private**. No compartas código ni
> `data/` con otros equipos. El *fork* y el *pull request* se hacen **solo** el día de la entrega.
> **Fechas: por definirse** (las anunciará el profesor); ver [docs/entregar.md](docs/entregar.md).

1. **Crea tu repo privado.** Arriba en esta página: **Use this template → Create a new
   repository**, elige **Private** y clónalo en tu computadora (o ábrelo en Codespaces).
   ¿Nunca has usado GitHub o Docker? Sigue [docs/empezar.md](docs/empezar.md) paso a paso.
2. **Copia el participante `montecarlo` a tu carpeta** y edita `equipo.toml` (nombre del equipo
   y los dos integrantes) e `INFORME.md`:

   ```bash
   cp -r estudiantes/montecarlo estudiantes/<tu_equipo>
   ```

   El nombre de tu carpeta usa solo minúsculas sin acentos, dígitos y `_` (de 3 a 30
   caracteres), por ejemplo `los_alfa_beta`.
3. **Juega contra `montecarlo`:**

   ```bash
   ./torneo jugar <tu_equipo> montecarlo
   ```

   La primera vez construye las imágenes de Docker en tu máquina (la primera vez unos minutos).
   Verás una tabla con las partidas y la ruta a un visor (`index.html`) para verlas jugada a
   jugada. Desde ahí, mejora tu `strategy.py`.

   **¿Dónde trabajar?** En orden: tu computadora con Mac o Linux, Windows con WSL2, o Codespaces
   (en el navegador, sin Docker: ahí `./torneo` corre con `--sin-docker` solo). Para tu editor y
   tu asistente de IA, activa el entorno del torneo en cada terminal nueva:
   `source .venv/bin/activate` (se crea con `./torneo instalar`; ver
   [docs/empezar.md](docs/empezar.md#el-entorno-de-python-venv)). ¿Algo falla? `./torneo diagnostico`.

## Tu carpeta

```text
estudiantes/<tu_equipo>/
├── equipo.toml        nombre del equipo + los 2 integrantes con su clave única
├── strategy.py        tu bot: lo único que se ejecuta (puede importar otros .py de tu carpeta)
├── INFORME.md         obligatorio: qué hace tu bot y por qué (9 secciones)
├── requirements.txt   opcional: librerías extra
├── data/              lo que precomputes (tablas, pesos, aperturas); en la liga es de solo lectura
└── herramientas/      el código con el que generaste data/ (no se ejecuta en la liga)
```

Todo lo demás del repo (motor, árbitro, Docker) vive en `sistema/`. No necesitas abrirlo.

## Entrada y salida

Tu bot se reduce a dos funciones en `strategy.py`:

```python
from motor import Estado, Jugada

def preparar(config):
    """Una vez por partida, antes de empezar (10 s, fuera del reloj).
    config.color ("blancas" o "negras"), config.carpeta_datos (tu data/),
    config.semilla (para tu propio azar), config.reloj_s (30)."""

def jugar(estado: Estado, tiempo_restante: float) -> Jugada:
    """Te toca. Regresa una jugada de estado.jugadas()."""
```

- **Entrada:** el `estado` completo (tablero, puntos y turno) y los segundos que te quedan de reloj.
  Cuando se llama `jugar`, siempre es tu turno: `estado.turno == config.color`.
  No necesitas recordar nada entre turnos.
- **Salida:** una jugada legal (una `Jugada` de `estado.jugadas()` o su texto, como `"e2-e3"`).
- **Lo de en medio es tuyo:** búsqueda, evaluación, manejo del azar y del reloj, precomputación.
  El motor te ayuda con `estado.jugadas()`, `estado.siguiente(j)` (resultados posibles con su
  probabilidad), `estado.prob_ataque(j)` y `simular_al_azar(...)`. Detalles en
  [docs/reglas.md](docs/reglas.md#el-motor).

## Los participantes

`random` y `montecarlo` **no son ejemplos aparte: son participantes como tú**. Viven en
`estudiantes/`, juegan la liga con las mismas reglas y sirven para probar tu bot.

- `random` elige una jugada legal al azar.
- `montecarlo` hace Monte Carlo plano: para cada jugada simula partidas al azar y elige la de mejor
  promedio. Es tu punto de partida; sus comentarios marcan dónde puedes mejorar.
- **Tier 1 a Tier 5:** bots más fuertes, de menor a mayor. Los detalles se anuncian en clase.
  Su código no se publica: aparecen el día de la liga y juegan con las mismas reglas y límites
  que tú.

## Cómo se califica

Liga todos contra todos: contra cada participante juegas 4 partidas (2 con Blancas, 2 con Negras).
Cada victoria vale 1 punto. Tu calificación es la del **mejor bot de referencia al que le ganas**
(`random`, `montecarlo` o un tier), y «ganarle» es terminar con **estrictamente más puntos** que
él. Empatar no es ganarle.

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

Sin `INFORME.md` completo o sin `equipo.toml` válido, tu PR no se acepta y no tienes calificación. Desempates y ejemplos:
[docs/calificacion.md](docs/calificacion.md).

## Reglas que no puedes romper

> **Misma entrada, misma salida; en medio, lo tuyo.** Tres condiciones: nada de redes
> neuronales, nada de trampas y código legible.

- **Tu carpeta es de solo lectura** durante la partida: puedes leer `data/`, pero no escribir
  archivos. Cada partida es un proceso nuevo: nada pasa de una partida a otra. No hay red.
- **Límites** (iguales en tu máquina y en la liga):

<!-- limites:inicio -->
| Recurso | Límite |
|---|---|
| Reloj | 30 s por partida, estilo ajedrez: solo corre en tu turno |
| `preparar` (incluye importar `strategy.py`) | 10 s, fuera del reloj |
| CPU | 1 núcleo |
| Memoria | 1024 MB |
| Tu carpeta | 100 MB en total y 50 MB por archivo |
| Lo que imprimes (`print`) | se guarda hasta 64 KB por partida |
| Librerías de `requirements.txt` | 300 MB instalados |
<!-- limites:fin -->

- **Pierdes esa partida** (y sigues en la liga) si se acaba tu reloj, regresas una jugada ilegal,
  lanzas una excepción, te pasas de memoria o `preparar` tarda más de 10 s.
- **Prohibido:** redes neuronales, código ofuscado, código nativo propio (C, Cython, `.so`),
  salir del aislamiento, adivinar o manipular los dados, copiar o compartir código o datos con otro
  equipo, ponerse de acuerdo con otro equipo. Lista completa en
  [docs/calificacion.md](docs/calificacion.md#qué-es-trampa).
- **Trampa o intento de trampa = 0 en el torneo para todo el equipo.**
- **Tu pull request solo puede cambiar archivos dentro de `estudiantes/<tu_equipo>/`.** Si cambia
  cualquier otra cosa (README, `docs/`, `sistema/`, workflows, carpetas de otros equipos, archivos
  de la raíz), la revisión automática `revisar-entrega` falla y el PR no se puede aceptar. Todo lo
  demás lo mantiene el equipo del curso.
- **Cada entrega la revisa el validador y además un agente de IA**: código, `data/` e informe.
- ¿Duda si algo se vale? Pregunta **antes** de entregar. La respuesta se publica para todos.

## Equipos de 2

Cada equipo tiene exactamente 2 integrantes, los dos en `equipo.toml` con su clave única.
Otro número solo con permiso del profesor.

## Herramientas recomendadas

Usa un agente de IA para programar: **Antigravity**, **Cursor**, **Claude Code** o **GitHub
Copilot**. **Que la IA escriba código; tú piensa la estrategia.** Pídele que lea
[`AGENTS.md`](AGENTS.md) (cómo instalar, correr y qué no puede tocar), este README y
`docs/reglas.md` antes de empezar. Un bot que una IA hace sola en una noche no llega lejos: lo que
gana es tu idea, medida con partidas.

## Para leer cuando lo necesites

| Guía | Qué tiene |
|---|---|
| [docs/reglas.md](docs/reglas.md) | Reglas completas, ejemplos resueltos, el motor y las reglas de ejecución |
| [docs/empezar.md](docs/empezar.md) | GitHub desde cero, repo privado, Docker, Codespaces, Windows |
| [docs/probar.md](docs/probar.md) | Comandos, visor, cómo depurar, pruebas en GitHub |
| [docs/entregar.md](docs/entregar.md) | Carpeta, `equipo.toml`, `INFORME.md`, registro del SHA y pull request |
| [docs/calificacion.md](docs/calificacion.md) | Liga, desempates, calificación, qué es trampa |
| [docs/avanzado.md](docs/avanzado.md) | Velocidad, numba, precomputar, `requirements.txt` |
| [docs/preguntas.md](docs/preguntas.md) | Preguntas frecuentes |

## Fe de erratas

Correcciones publicadas después del inicio del curso. Para traerlas a tu repo:
[¿Cómo actualizo el sistema?](docs/preguntas.md#cómo-actualizo-el-sistema-cuando-el-curso-publica-una-corrección)

- **20 de octubre de 2026 — Codespaces y entorno de Python.**
  - Codespaces ya no intenta usar Docker (fallaba al crearse y abría en «modo de recuperación», sin
    pip; por eso numba y llvmlite no se podían instalar). Ahora corre siempre sin Docker, con
    `.venv/` listo al abrirlo.
  - Nuevos comandos: `./torneo instalar` (crea o repara `.venv/` con Python 3.12, sin compilar
    nada), `./torneo diagnostico` (dice qué falta y cómo arreglarlo) y `./torneo actualizar` (trae
    estas correcciones sin tocar tu carpeta).
  - Sin Docker, `.venv/` se prepara solo la primera vez y los bots usan un solo hilo, como en la
    liga.
  - La prueba de GitHub explica mejor cuándo no encuentra tu carpeta (casi siempre falta el
    `git push`).
  - Guía nueva: [El entorno de Python](docs/empezar.md#el-entorno-de-python-venv) (cómo activarlo)
    e instrucciones para asistentes de IA en [`AGENTS.md`](AGENTS.md).
