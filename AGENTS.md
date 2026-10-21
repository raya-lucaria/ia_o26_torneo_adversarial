# Instrucciones para asistentes de IA

Este repo es el torneo «Breakthrough de combate» (curso IA O26). Ayudas a un equipo de 2 alumnos a
programar su bot. Responde en español. Lee `README.md` y `docs/reglas.md` antes de proponer código.

## Dónde se trabaja

- El equipo trabaja **solo** en `estudiantes/<su_equipo>/`. **No modifiques** nada fuera de esa
  carpeta (`sistema/`, `docs/`, `.github/`, `torneo`, archivos de la raíz, otras carpetas de
  `estudiantes/`): su pull request falla si cambia algo más.
- Si aún no tienen carpeta: `cp -r estudiantes/montecarlo estudiantes/<su_equipo>` (nombre: solo
  minúsculas sin acentos, dígitos y `_`, de 3 a 30 caracteres) y editar `equipo.toml`.
- El bot es `strategy.py` con `preparar(config)` y `jugar(estado, tiempo_restante)`. Usa el motor
  (`from motor import Estado, Jugada, simular_al_azar`); no reimplementes las reglas.

## Cómo instalar y correr (no inventes otro camino)

Todo se corre con `./torneo` desde la raíz del repo. Si algo falla, **empieza siempre por**
`./torneo diagnostico`: dice qué falta y qué comando lo arregla.

- **Con Docker** (Mac, Linux o Windows con WSL2): `./torneo jugar <su_equipo> montecarlo`. No hay
  que instalar nada más; las imágenes se construyen solas.
- **Sin Docker** (Codespaces, donde `--sin-docker` se pone solo, o `--sin-docker` a mano): usa el
  entorno `.venv/` de la raíz, que `./torneo` prepara solo la primera vez (Python 3.12 con uv, el
  motor, numpy, scipy y numba en las versiones de la liga).
- **Para el editor, los scripts y tú mismo**, activa ese entorno en cada terminal nueva:

  ```bash
  ./torneo instalar          # crea o repara .venv/ (repetirlo no rompe nada)
  source .venv/bin/activate  # la terminal muestra (.venv)
  ```

- **Nunca** uses `pip install` fuera de `.venv/`, ni crees otros entornos, ni compiles numba o
  llvmlite. Si el equipo agrega `estudiantes/<su_equipo>/requirements.txt`, corre
  `./torneo instalar`.
- Windows: solo dentro de Ubuntu (WSL2), con VS Code conectado a WSL (abajo a la izquierda dice
  `WSL: Ubuntu`) y el repo clonado dentro de Ubuntu, o en Codespaces. Nunca en PowerShell ni Git
  Bash (aunque sea la terminal de VS Code): `uname -s` debe decir `Linux`. Mac Intel: solo con
  Docker o Codespaces.
- Otros comandos: `./torneo validar <su_equipo>`, `./torneo mini <su_equipo>`, `./torneo --help`.
  Guía completa: `docs/probar.md`.
- Sin Docker cuentan el reloj (30 s por partida) y `preparar` (10 s), pero no la memoria ni la CPU.
  Lo oficial es con Docker o la prueba de GitHub Actions (cada *push* a `main` que cambie
  `estudiantes/`).

## Git

- Guardar y subir: `git add estudiantes/<su_equipo> && git commit -m "..." && git push` (a `main`:
  la prueba de GitHub solo corre ahí).
- Traer lo del compañero: `git pull`. Correcciones del curso: `./torneo actualizar` y luego
  `git push` (nunca toca la carpeta del equipo; ver `docs/preguntas.md`).

## Reglas que no puedes romper (o el equipo saca 0)

- Nada de redes neuronales (ni escritas a mano), código ofuscado ni código nativo propio
  (C, Cython, `.so`).
- Durante la partida la carpeta es de solo lectura (se puede leer `data/`), no hay red y cada
  partida es un proceso nuevo.
- No adivinar ni manipular los dados, no salir del aislamiento, no copiar código ni datos de otro
  equipo.
- Código legible. Límites y lista completa: `README.md` y `docs/calificacion.md#qué-es-trampa`.

El equipo piensa la estrategia; tú ayudas a escribirla y a medirla con partidas.
