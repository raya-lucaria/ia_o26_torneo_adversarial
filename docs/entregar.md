# Entregar

> [!IMPORTANT]
> **Hora límite: lunes 9 de noviembre de 2026, 10:00, hora de la Ciudad de México (UTC−6).** Es la única hora límite de la
> entrega. Antes de esa hora debes **(1) registrar el SHA** de tu commit final y **(2) abrir tu
> pull request** con tu carpeta exactamente como en ese SHA. **Después de esa hora no se acepta
> nada:** ni registro, ni PR, ni cambios.

La entrega tiene **dos pasos, los dos antes de la hora límite**:

1. **Primero registras el SHA** del último commit de tu repo privado. Tu código sigue privado.
2. **Enseguida** haces un *fork* del repo del torneo, copias tu carpeta **exactamente** como estaba
   en ese SHA y abres un *pull request* (PR).

Haz los dos pasos con tiempo y en ese orden: registra el SHA primero y abre el PR enseguida, no
lo dejes para el último minuto. El SHA registrado fija tu bot: tu PR debe coincidir con él, así que
nadie puede cambiar su bot después de registrarlo. Copiar código de un PR ajeno es copiar código de
otro equipo, y eso es trampa ([docs/calificacion.md](calificacion.md#qué-es-trampa)).

> [!IMPORTANT]
> **Tu pull request solo puede cambiar archivos dentro de tu carpeta,
> `estudiantes/<tu_equipo>/`.** Si cambia cualquier otra cosa (el README, `docs/`, `sistema/`, los
> workflows, carpetas de otros equipos o archivos de la raíz), la revisión automática
> `revisar-entrega` falla y el PR no se puede aceptar. Todo lo demás lo mantiene el equipo del
> curso.

## Tu carpeta

```text
estudiantes/<tu_equipo>/
├── equipo.toml        obligatorio
├── strategy.py        obligatorio: define preparar(config) y jugar(estado, tiempo_restante)
├── INFORME.md         obligatorio
├── requirements.txt   opcional
├── data/              opcional: tus precomputados
├── herramientas/      opcional: el código con que generaste data/
└── *.py               opcional: otros módulos que importe strategy.py
```

- El nombre de la carpeta: de 3 a 30 caracteres, solo minúsculas sin acentos, dígitos y `_`.
  No puede ser `random`, `montecarlo`, `humano`, empezar con `tier_` ni con `_`.
- **`data/` va dentro de tu carpeta y se sube con tu commit** (y en tu PR), respetando los límites:
  100 MB en total y 50 MB por archivo. En la liga es de solo lectura.
- Las carpetas `__pycache__/` se ignoran: el torneo no usa tus `.pyc`. Puedes borrarlas.
- No pongas binarios (`.so`, `.dll`, ejecutables) ni enlaces simbólicos.

## `equipo.toml`

```toml
nombre = "los_alfa_beta"
integrantes = [
  { nombre = "Ana Pérez",  clave_unica = "000123456" },
  { nombre = "Luis Gómez", clave_unica = "000654321" },
]
```

- `nombre` es igual al nombre de tu carpeta.
- **Exactamente 2 integrantes**, cada uno con su nombre y su clave única **real**, entre comillas
  (de 6 a 12 dígitos). Otro número de integrantes solo con permiso del profesor.

## `INFORME.md`

Obligatorio, de 2 a 4 páginas. Sin `INFORME.md` completo o sin `equipo.toml` válido, tu PR no se acepta y no tienes calificación. Tiene estas 9
secciones, cada una como título numerado (por ejemplo `## 3. Búsqueda (algoritmo y por qué)`):

1. **Integrantes.**
2. **Idea en un párrafo:** qué hace tu bot.
3. **Búsqueda (algoritmo y por qué):** qué algoritmo usas y por qué ese.
4. **Función de evaluación:** cómo mides una posición.
5. **Precomputación (qué y cómo regenerarla):** qué hay en `data/` y cómo se vuelve a generar con
   tu código de `herramientas/`.
6. **Manejo del reloj:** cómo repartes tus 30 s.
7. **Resultados vs random y montecarlo:** números, con cuántas partidas y con qué reloj.
8. **Qué intentamos y no funcionó.**
9. **Fuentes y uso de IA:** artículos, código de internet y cómo usaron agentes de IA.

`estudiantes/montecarlo/INFORME.md` es un ejemplo de formato. El tuyo habla de **tu** bot.

Mientras trabajas, un informe sin terminar no te detiene: `./torneo validar` y la prueba de GitHub
solo muestran un aviso ⚠ si todavía tiene texto del ejemplo o es casi una copia. **En la entrega
(el PR) es un error:** el informe debe estar completo y ser tuyo.

## Antes de la hora límite

1. **Valida tu carpeta:**

   ```bash
   ./torneo validar <tu_equipo> --entrega
   ```

   Con `--entrega` revisa exactamente lo que revisará tu PR (sin `--entrega`, lo pendiente del
   informe solo es aviso). Debe terminar con `✓ <tu_equipo> pasa la validación.` Revisa también los
   avisos (⚠).
2. **Revisa la prueba de GitHub** de tu último *push* (pestaña **Actions**, ver
   [docs/probar.md](probar.md#pruebas-en-github)).
3. **Sube tu último commit** y copia su SHA (el identificador largo del commit):

   ```bash
   git push
   git rev-parse HEAD
   ```

   También lo ves en GitHub: pestaña **Commits** de tu repo, botón para copiar el SHA completo.
4. **Registra el SHA antes de la hora límite** (**lunes 9 de noviembre de 2026, 10:00, hora de la Ciudad de México (UTC−6)**) donde indique
   el profesor, con el nombre de tu equipo y el enlace a tu repo privado.
5. **Comprueba que el profesor (`uumami`) tiene acceso a tu repo privado** (Settings →
   Collaborators). Lo usa para comparar tu PR con el SHA registrado.

Después de registrar, no cambies nada: lo que cuenta es ese SHA. Y abre tu PR enseguida (abajo):
también tiene que estar abierto antes de la hora límite.

## Enseguida, antes de la hora límite: fork y pull request

Hazlo desde la carpeta que **contiene** tu repo privado (en el ejemplo, tu repo se llama
`tu_repo`). Cambia `los_alfa_beta` por tu equipo y `<SHA>` por el que registraste.

1. Haz el *fork* del repo del torneo y clónalo:

   ```bash
   gh repo fork raya-lucaria/ia_o26_torneo_adversarial --clone
   cd ia_o26_torneo_adversarial
   git checkout -b entrega-los_alfa_beta
   ```

2. Copia tu carpeta **exactamente como estaba en el SHA registrado** (este comando la saca de tu
   repo privado tal cual):

   ```bash
   git -C ../tu_repo archive <SHA> estudiantes/los_alfa_beta | tar -x
   git status        # solo debe aparecer estudiantes/los_alfa_beta/
   ```

   Por ejemplo, si el SHA registrado es `3f9c2a7e1b4d8c6f0a5e9b2d7c1f4a8e6b3d0c9f`:

   ```bash
   git -C ../tu_repo archive 3f9c2a7e1b4d8c6f0a5e9b2d7c1f4a8e6b3d0c9f estudiantes/los_alfa_beta | tar -x
   ```

   Usa el SHA completo (40 caracteres), sin `<` ni `>`.

3. Sube la rama y abre el PR contra el repo del torneo:

   ```bash
   git add estudiantes/los_alfa_beta
   git commit -m "Entrega los_alfa_beta"
   git push -u origin entrega-los_alfa_beta
   gh pr create --repo raya-lucaria/ia_o26_torneo_adversarial \
     --title "Entrega los_alfa_beta" --body "SHA registrado: <SHA>"
   ```

4. En la página del PR, espera a que termine la revisión automática `revisar-entrega`. Si sale en
   rojo, abre el detalle: dice qué falla.

Un PR cuyo contenido no coincide con el SHA registrado se rechaza, y un PR abierto después de la
hora límite no se acepta.

## Qué revisa el validador

Lo mismo que `./torneo validar <tu_equipo> --entrega`, más la revisión del PR:

- **Solo tu carpeta:** el PR no toca nada fuera de `estudiantes/<tu_equipo>/`.
- **Nombre** de la carpeta: formato válido y no reservado.
- **Archivos:** sin enlaces simbólicos, sin binarios ni código compilado (`.so`, `.dll`, `.exe`,
  `.pyc` sueltos…), 100 MB en total, 50 MB por archivo y a lo más 2000 archivos.
- **`equipo.toml`:** bien escrito, `nombre` igual a la carpeta, 2 integrantes con claves únicas
  reales y distintas.
- **`INFORME.md`:** las 9 secciones, ninguna vacía, sin texto del ejemplo y que no sea una copia del
  de `montecarlo` (sin `--entrega`, estas dos últimas son solo aviso).
- **Código:** `strategy.py` existe y define `preparar` y `jugar`; todo `.py` es Python válido; no
  importa librerías de redes neuronales ni código del torneo (`arbitro`, `liga`, `validador`, `cli`,
  `visor`) ni de `estudiantes`. Marca para revisión manual el código que importa o ejecuta código de
  forma indirecta (`exec`, `eval`, `__import__`…).
- **`requirements.txt`:** una librería por línea con versión exacta, ninguna de la lista negra, que
  no cambie numpy, scipy ni numba, y a lo más 300 MB instalados.
- **Partidas de prueba:** 2 partidas contra `random`. Si tu bot pierde por error, jugada ilegal,
  memoria o `preparar`, es un error que debes arreglar.

## Qué revisa el agente de IA

Además del validador, un agente de IA revisa cada entrega (código, `data/` e informe) y marca lo
sospechoso para que lo decida el profesor:

- redes neuronales, aunque estén escritas a mano;
- código ofuscado o ilegible, o datos en `data/` que en realidad son código;
- intentos de salir del aislamiento, de adivinar los dados o de tocar el árbitro;
- que el informe, el código y `data/` cuenten la misma historia (lo que dices que hiciste es lo que
  hay);
- código copiado de otro equipo.

El profesor decide cada caso marcado antes de poner un 0.

## Si algo técnico falla

Después de la hora límite no se acepta nada por iniciativa tuya. Solo si **el profesor te avisa** que
tu entrega tiene un error técnico (por ejemplo, el validador falla por un archivo de más), tienes
**una sola corrección**, dentro de las 24 horas siguientes a ese aviso, con el **cambio mínimo** para
arreglarlo. Se revisa a mano. No se aceptan cambios de estrategia. Si sigue fallando, tu bot no
entra a la liga (calificación 0).
