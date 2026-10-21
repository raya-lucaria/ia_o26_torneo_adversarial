# Empezar desde cero

Esta guía te lleva de no tener nada a ver tu primera partida. Quién hace cada paso:

- **Los dos integrantes:** pasos 1, 4, 5 y 6 (cada quien en su computadora).
- **Solo uno:** pasos 2, 3 y 7. Después, el otro baja los cambios con `git pull`.

Si algo falla, revisa [docs/preguntas.md](preguntas.md).

Puedes trabajar en cualquiera de estos lugares. Están en orden de recomendación:

| Dónde | Qué necesitas | Cómo corre tu bot |
|---|---|---|
| **1. Tu computadora, Mac o Linux** | Docker | Con los límites de la liga. Lo más rápido para trabajar todos los días |
| **2. Tu computadora, Windows** | WSL2 (Ubuntu dentro de Windows) y Docker | Igual que en Mac o Linux |
| **3. Codespaces** | Solo un navegador | Sin Docker: cuentan el reloj y `preparar`, pero no la memoria ni la CPU |

Además, cada *push* a `main` que cambie tu carpeta lo prueba **GitHub Actions** con los límites
de la liga, sin instalar nada: es la referencia oficial ([docs/probar.md](probar.md#pruebas-en-github)).

¿Algo no funciona? Corre `./torneo diagnostico`: revisa tu máquina, tu entorno y tu carpeta, y te
dice qué hacer. Si pides ayuda, manda lo que imprime.

## 1. Crea tu cuenta de GitHub

1. Entra a <https://github.com/signup> y crea una cuenta (usa tu correo de la universidad si
   quieres pedir después los beneficios de estudiante en <https://education.github.com>).
2. Cada integrante del equipo necesita su propia cuenta.

## 2. Crea el repo privado de tu equipo

Solo **un** integrante hace este paso.

1. Abre el repo del torneo:
   <https://github.com/raya-lucaria/ia_o26_torneo_adversarial>.
2. Arriba a la derecha, pulsa **Use this template → Create a new repository**.

   <!-- TODO(profesor): agrega docs/img/capturas/use_this_template.png y quita este comentario.
   ![Captura: Botón verde «Use this template» abierto, con la opción «Create a new repository»](img/capturas/use_this_template.png) -->

3. En **Owner** elige tu cuenta. En **Repository name** escribe un nombre (por ejemplo
   `torneo_los_alfa_beta`).
4. Marca **Private**. Esto es lo importante: tu código solo lo ven tú y quien invites.

   <!-- TODO(profesor): agrega docs/img/capturas/repo_privado.png y quita este comentario.
   ![Captura: Formulario «Create a new repository» con la opción Private marcada](img/capturas/repo_privado.png) -->

5. Pulsa **Create repository**.

> [!WARNING]
> **No uses «Fork» para trabajar.** Un *fork* de un repo público es público: cualquiera vería tu
> código. El *fork* se hace solo el día de la entrega ([docs/entregar.md](entregar.md)).

## 3. Invita a tu compañero y al profesor

Solo quien creó el repo. En tu repo nuevo: **Settings → Collaborators → Add people**.

1. Agrega a tu compañero de equipo.
2. Agrega también al profesor (`uumami`). Lo necesita para revisar tu entrega: compara tu pull
   request con el commit que registres.

<!-- TODO(profesor): agrega docs/img/capturas/colaboradores.png y quita este comentario.
![Captura: Página Settings → Collaborators con el botón «Add people»](img/capturas/colaboradores.png) -->

Tu compañero recibe un correo y debe aceptar la invitación.

## 4. Elige dónde vas a trabajar

1. **Mac o Linux:** sigue con el paso 5 (instalar Docker).
2. **Windows:** sigue con [Windows: WSL2](#windows-wsl2) y luego con el paso 6, **dentro de Ubuntu**.
   No uses PowerShell ni Git Bash para `./torneo`.
3. **Si no puedes instalar Docker** (o tu computadora no da): salta a [Codespaces](#codespaces).

## 5. Instala Docker

Docker corre cada bot en una «caja» aislada con los mismos límites que la liga (1 núcleo, 1024 MB,
reloj de 30 s). Así lo que ves en tu computadora es lo que pasará en la liga.

### Mac (Intel o Apple Silicon)

1. Descarga **Docker Desktop** de <https://www.docker.com/products/docker-desktop/>. Elige la
   versión de tu procesador: **Apple Silicon** (M1, M2, M3, M4…) o **Intel**. Si no sabes cuál
   tienes: menú Apple → **Acerca de esta Mac** → **Chip** o **Procesador**.
2. Abre el `.dmg`, arrastra Docker a Aplicaciones y ábrelo. Acepta los permisos.
3. Espera a que el ícono de la ballena en la barra de arriba diga **Docker Desktop is running**.

   <!-- TODO(profesor): agrega docs/img/capturas/docker_mac_running.png y quita este comentario.
   ![Captura: Menú de la ballena de Docker en la barra de Mac, con el texto «Docker Desktop is running»](img/capturas/docker_mac_running.png) -->

4. En **Settings → Resources**, dale a Docker al menos **4 GB** de memoria y **2 CPU**.
5. Abre la **Terminal** y comprueba:

   ```bash
   docker info
   ```

   Si imprime información sin errores, listo. Si dice que no puede conectarse, Docker Desktop no
   está abierto.

Instala también `git` si no lo tienes: en la Terminal escribe `git --version`; si no lo tienes, la
Mac te ofrece instalarlo.

En una Mac con procesador **Intel**, trabaja siempre con Docker: sin Docker el torneo no corre ahí.

### Linux

1. Instala Docker Engine siguiendo la guía de tu distribución en
   <https://docs.docker.com/engine/install/> (Ubuntu, Debian, Fedora…). Instala también el
   paquete `docker-compose-plugin`.
2. Para usar Docker sin `sudo`:

   ```bash
   sudo usermod -aG docker $USER
   ```

   Cierra sesión y vuelve a entrar. Si no quieres cerrar sesión, en esa terminal escribe
   `newgrp docker` (vale solo para esa terminal).
3. Comprueba:

   ```bash
   docker info
   docker compose version
   ```

### Windows: WSL2

En Windows el torneo corre dentro de **WSL2** (un Ubuntu dentro de Windows).

1. Abre **PowerShell como administrador** y escribe:

   ```powershell
   wsl --install
   ```

   Reinicia cuando te lo pida. Al volver, se abre Ubuntu: crea tu usuario y contraseña.
2. Instala **Docker Desktop para Windows** de
   <https://www.docker.com/products/docker-desktop/>. En la instalación deja marcada la opción
   **Use WSL 2**.
3. En Docker Desktop: **Settings → Resources → WSL integration**, activa **Ubuntu**.

   <!-- TODO(profesor): agrega docs/img/capturas/docker_wsl_integration.png y quita este comentario.
   ![Captura: Docker Desktop en Windows, Settings → Resources → WSL integration con Ubuntu activado](img/capturas/docker_wsl_integration.png) -->

4. Abre **Ubuntu** (desde el menú Inicio) y comprueba:

   ```bash
   docker info
   ```

5. **Trabaja siempre dentro de Ubuntu**, en tu carpeta de Linux (`cd ~`), no en `C:\` ni en
   `/mnt/c/`: ahí todo es mucho más lento y fallan permisos y finales de línea. Clona tu repo ahí
   (paso 6).
6. **Conecta VS Code a Ubuntu.** Instala en VS Code la extensión **WSL** (de Microsoft). Luego
   **Ctrl+Shift+P → WSL: Connect to WSL**, y abre tu repo con **File → Open Folder** (está en
   `/home/<tu_usuario>/`). También puedes abrirlo desde Ubuntu: entra a la carpeta y escribe
   `code .`.

   Abajo a la izquierda de VS Code debe decir **`WSL: Ubuntu`**. Si no lo dice, tu terminal no es
   Ubuntu.

> [!WARNING]
> **En Windows, la terminal de VS Code abre por defecto PowerShell o Git Bash**, y ahí `./torneo`
> no funciona (por ejemplo, Docker dice `the working directory is invalid 'C:/Program
> Files/Git/torneo'`). Comprueba en qué terminal estás:
>
> ```bash
> uname -s
> ```
>
> Debe decir `Linux`. Si dice `MINGW64_NT…` (Git Bash) o da error (PowerShell), conecta VS Code a
> WSL (punto 6). Cambiar solo el tipo de terminal no basta si tu repo está en `C:\`: clónalo
> dentro de Ubuntu.

## 6. Clona tu repo

Clonar es bajar una copia de tu repo a tu computadora. La forma más fácil con un repo privado es
con **GitHub CLI** (`gh`). **En Windows, haz todo este paso dentro de Ubuntu (WSL2)**, empezando
con `cd ~`, no en PowerShell ni en Git Bash.

1. Instala `gh`:
   - **Mac:** si tienes Homebrew, `brew install gh`; si no, descarga el instalador `.pkg` de
     <https://cli.github.com>.
   - **Ubuntu o WSL:**

     ```bash
     sudo apt update
     sudo apt install gh git
     ```

   - **Otro Linux:** sigue <https://cli.github.com>.
2. Inicia sesión una vez y sigue las instrucciones. Elige **GitHub.com**, **HTTPS**, responde
   **Yes** a «Authenticate Git with your GitHub credentials?» y elige **Login with a web browser**:

   ```bash
   gh auth login
   ```

3. Dile a git quién eres y cómo juntar tu trabajo con el de tu compañero (una sola vez; tu
   nombre aparece en tus commits):

   ```bash
   git config --global user.name "Ana Pérez"
   git config --global user.email "tu_correo@ejemplo.com"
   git config --global pull.rebase false
   ```

4. Clona tu repo (cambia `tu_usuario/tu_repo` por el tuyo) y entra a la carpeta:

   ```bash
   gh repo clone tu_usuario/tu_repo
   cd tu_repo
   ```

## 7. Crea tu carpeta y juega tu primera partida

Solo un integrante hace los pasos 1 a 3 y 5. El otro hace `git pull` (paso 6). Los dos pueden
jugar.

1. Copia el participante `montecarlo` con el nombre de tu equipo. El nombre usa solo minúsculas sin
   acentos, dígitos y `_` (3 a 30 caracteres). No puede ser `random`, `montecarlo`, `humano`,
   empezar con `tier_` ni con `_`.

   ```bash
   cp -r estudiantes/montecarlo estudiantes/los_alfa_beta
   ```

2. Abre `estudiantes/los_alfa_beta/equipo.toml` y pon el nombre de tu carpeta y a los dos
   integrantes con su clave única:

   ```toml
   nombre = "los_alfa_beta"
   integrantes = [
     { nombre = "Ana Pérez",  clave_unica = "000123456" },
     { nombre = "Luis Gómez", clave_unica = "000654321" },
   ]
   ```

3. Juega contra `montecarlo`:

   ```bash
   ./torneo jugar los_alfa_beta montecarlo
   ```

   **La primera vez tarda unos minutos antes de empezar:** construye las imágenes de Docker en
   tu máquina y prepara numba (en Codespaces no hay imágenes: solo numba). Si ves `Preparando la caché de numba…`, espera: es normal (≈1 min). Las
   siguientes veces empieza en segundos.

   Las 4 partidas tardan unos 2 a 4 minutos (cada bot tiene 30 s de reloj por partida). Al final
   verás algo así:

   ```text
   #  Blancas        Negras         Ganador        Puntos  Jugadas   Reloj máx
   1  los_alfa_beta  montecarlo     montecarlo        0–3       60  14.6 s (N)
   ...
   Resultado: montecarlo 2 – 2 los_alfa_beta
   Repeticiones: …/resultados/2026-10-20_17-03-11/index.html  (ábrelo en tu navegador)
   ```

4. Abre ese `index.html` con doble clic: es el visor, para ver cada partida jugada a jugada.

5. Guarda tu trabajo en GitHub:

   ```bash
   git add estudiantes/los_alfa_beta
   git commit -m "Nuestra carpeta"
   git push
   ```

   Cada *push* a `main` que cambie `estudiantes/` también prueba tu bot en GitHub (ver
   [docs/probar.md](probar.md#pruebas-en-github)).

6. Tu compañero baja la carpeta a su computadora:

   ```bash
   git pull
   ```

   Antes de empezar a trabajar, cada quien hace `git pull` para tener lo último del otro. Al
   terminar, `git add`, `git commit` y `git push`. `./torneo diagnostico` te dice si tu carpeta
   ya está en GitHub.

No tienes que terminar `INFORME.md` ahora: escríbelo mientras mejoras tu bot. Mientras siga
igual al de `montecarlo`, `./torneo validar` y la prueba de GitHub te lo recuerdan con un aviso ⚠.
En la entrega debe describir **tu** bot ([docs/entregar.md](entregar.md#informemd)).

## Codespaces

Codespaces es una computadora de GitHub con VS Code en tu navegador. Sirve en cualquier sistema.
**No tiene Docker:** ahí `./torneo` corre solo [sin Docker](#sin-docker), con el entorno de Python
`.venv/` que se prepara al crear el codespace.

1. En tu repo privado: **Code → Codespaces → Create codespace on main**.

   <!-- TODO(profesor): agrega docs/img/capturas/codespaces_crear.png y quita este comentario.
   ![Captura: Menú «Code» abierto en la pestaña Codespaces con el botón «Create codespace on main»](img/capturas/codespaces_crear.png) -->

2. La primera vez tarda unos minutos: prepara `.venv/` con el motor y sus librerías. Espera a que
   la terminal diga `✓ Listo`.
3. Abre una **terminal nueva**: menú **☰ → Terminal → New Terminal**. La línea debe empezar con
   `(.venv)`. Si no, actívalo (ver [El entorno de Python](#el-entorno-de-python-venv)):

   ```bash
   source .venv/bin/activate
   ```

4. Comprueba que todo esté bien:

   ```bash
   ./torneo diagnostico
   ```

5. Sigue desde el [paso 7](#7-crea-tu-carpeta-y-juega-tu-primera-partida). Los comandos son los
   mismos: `./torneo` agrega `--sin-docker` por ti.
6. Para ver el visor, descarga **la carpeta completa** de la corrida, no solo el `index.html`
   (sin las demás páginas, sus enlaces no abren). Primero empaquétala en la terminal (cambia la
   fecha por la de tu corrida):

   ```bash
   python3 -m zipfile -c visor.zip resultados/2026-10-20_17-03-11
   ```

   Luego, en el explorador de archivos de la izquierda, clic derecho en `visor.zip` →
   **Download**. En tu computadora descomprímelo y abre su `index.html`.

> [!WARNING]
> Si al abrir el codespace ves un aviso de **recovery mode** («modo de recuperación»), o
> `./torneo diagnostico` lo dice, el codespace abrió en un contenedor de emergencia sin lo del
> torneo. Sube tu trabajo (`git add estudiantes`, `git commit -m "Mi trabajo"`, `git push`) y luego
> **Ctrl+Shift+P → Codespaces: Rebuild Container** (o borra el codespace y crea uno nuevo).

**Cómo se usa el tiempo de Codespaces:**

- **Tus archivos se guardan.** Puedes cerrar la pestaña: al volver, todo sigue ahí (también
  `.venv/` y lo que no hayas subido). No es como Colab.
- **Se detiene solo** tras unos 30 minutos sin que lo uses, y lo que estuviera corriendo se corta.
  No dejes una liga larga corriendo para irte.
- **Un codespace sin usar por un mes se borra**, con lo que no hayas subido. Al terminar cada
  sesión: `git add`, `git commit` y `git push`.
- **Tu cuenta tiene horas gratis cada mes** (más si te das de alta como estudiante en
  [GitHub Education](https://education.github.com/)). **Detén el codespace cuando no lo uses**
  (<https://github.com/codespaces> → **⋯ → Stop codespace**) para no gastarlas. Sin tarjeta
  registrada no te cobran nada: al acabarse las horas, se detiene hasta el mes siguiente.
- Sin Docker no hay límite de memoria ni de CPU: para medir tu bot con los límites de la liga, haz
  *push* y mira la [prueba de GitHub](probar.md#pruebas-en-github).

## El entorno de Python (`.venv/`)

`.venv/` es una carpeta en la raíz de tu repo con **el Python del torneo**: Python 3.12, el motor,
numpy, scipy y numba, en las mismas versiones que la liga. No se sube a git: cada quien tiene el
suyo.

**Cuándo lo necesitas:**

- **Sin Docker** (Codespaces, o `--sin-docker`): siempre. `./torneo` lo prepara solo la primera vez.
- **Con Docker:** solo para que tu editor y tu asistente de IA reconozcan `from motor import ...`,
  y para correr tus scripts de `herramientas/`.

**Crearlo o repararlo** (desde la raíz de tu repo; repetirlo no rompe nada):

```bash
./torneo instalar
```

No necesitas tener Python 3.12: si tu computadora no lo tiene, `./torneo instalar` lo descarga
para `.venv/` (con [uv](https://docs.astral.sh/uv/)). No instala nada fuera de tu repo y no toca el
Python de tu sistema.

**Activarlo, en cada terminal nueva** (Mac, Linux, Ubuntu en WSL2 y Codespaces):

```bash
source .venv/bin/activate
```

- Sabes que está activo porque la línea de tu terminal empieza con `(.venv)`, y `which python`
  termina en `.venv/bin/python`.
- Cada terminal nueva empieza sin activar (salvo en Codespaces, que lo activa solo). Si cierras la
  terminal, vuelve a activarlo.
- Para desactivarlo: `deactivate`.
- `./torneo` usa `.venv/` aunque no lo actives. Activarlo sirve para lo demás: `python`, tus
  scripts y tu asistente de IA usan entonces el Python del torneo.

**En tu editor:** en VS Code, **Ctrl+Shift+P → Python: Select Interpreter** → elige el que dice
`.venv`. En Codespaces ya está elegido.

**Con tu asistente de IA** (Copilot, Claude, Cursor…): pídele que lea `AGENTS.md`. Ahí está este
mismo camino. Si te propone `pip install` fuera de `.venv/` o crear otro entorno, dile que no: usa
`./torneo instalar`.

## Sin Docker

Agrega `--sin-docker` a cualquier comando (en Codespaces se agrega solo):

```bash
./torneo jugar los_alfa_beta montecarlo --sin-docker
```

La primera vez prepara [`.venv/`](#el-entorno-de-python-venv) solo (unos minutos). Si agregas un
`requirements.txt` a tu carpeta, corre `./torneo instalar` para instalarlo en `.venv/`.

Sin Docker **cuentan el reloj (30 s) y `preparar` (10 s)**, pero **no hay límite de memoria ni de
CPU ni aislamiento**. Antes de confiar en un resultado, pruébalo con Docker o en GitHub.

No funciona en Mac con procesador Intel (numba no existe ahí sin Docker): usa Docker o Codespaces.

## Siguiente paso

- Cómo probar, ver partidas y depurar: [docs/probar.md](probar.md).
- Cómo mejorar tu bot: lee `estudiantes/montecarlo/strategy.py` y busca `AQUÍ PUEDES MEJORAR`.
