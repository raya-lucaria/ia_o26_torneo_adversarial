# Cómo regenerar `data/`

`data/` es de solo lectura durante la liga; estos son los generadores.
Ninguno escribe rutas privadas de esta máquina: todos reciben sus entradas
por argumento de línea de comandos.

## `data/pesos_eval3_ligero.npz`

`herramientas/fase5_reproducir_activo.py` (+ `fase5_features.py`): IRLS
(25 iteraciones máx., λ=100 hacia una base de 6 pesos históricos, columnas
puntos/patrones/filas/marcador, escalado por desviación estándar sin
centrar, partición de validación por `SHA256(id_partida + "_principal")`).

```bash
cd estudiantes/iacasitegano
python herramientas/fase5_reproducir_activo.py \
    --dataset <dataset.npz con tablero/puntos/turno/ganador_blancas/partida_origen> \
    --pesos-antiguos <pesos.npy de 6 valores (ridge histórico)> \
    --referencia data/pesos_eval3_ligero.npz \
    --salida <carpeta de salida NUEVA>
```

Compara el resultado contra `--referencia` con tolerancia `1e-8` y falla
(`exit 1`) si no coincide. Ya se reprodujo así una vez con error máximo
~1.8e-15 (ver `obsidian/01-Estado.md`, sección 5C del plan). El dataset y
los pesos antiguos son artefactos de entrenamiento (de `resultados/` o
equivalente, fuera del repo por tamaño); no se incluyen aquí porque
`resultados/` está en `.gitignore`.

## `data/libro.json`

El libro **entregado** lo generó `herramientas/libro_apertura.py`:
self-play con el buscador/evaluación ANTERIORES a eval3
(`busqueda_numba.py` + `eval.py`, pesos lineales de 6 valores), registrando
qué jugada ganó más seguido en las primeras `PLIES_LIBRO=4` jugadas propias.

```bash
cd estudiantes/iacasitegano
python herramientas/libro_apertura.py   # escribe data/libro.json
```

`herramientas/fase7a_libro.py` es un generador **distinto y más nuevo**,
preparado en la fase 7A para anotar propuestas con el buscador/pesos
actuales (`busqueda_eval3.py` + `eval3.py`) y nodos por posición, con
manifiesto/procedencia por entrada:

```bash
cd estudiantes/iacasitegano
python herramientas/fase7a_libro.py \
    --buscador busqueda_eval3.py --pesos data/pesos_eval3_ligero.npz \
    --posiciones <poblacion.jsonl> --poblacion <nombre> \
    --profundidad <n> --nodos <n> --semillas <csv> \
    --max-posiciones <n> --salida <carpeta NUEVA>
```

**Este generador nunca se usó para producir el `data/libro.json` que se
entrega**: la fase 7B (adaptar el libro al modelo eval3 final) quedó
pendiente (ver `obsidian/07-Plan-definitivo.md`, fase 7). Se deja aquí
documentado y verificado (importa sin error) para quien continúe esa fase,
no como la receta del archivo actual.

**No borrar `busqueda_numba.py` ni `eval.py` de `estudiantes/iacasitegano/`**
aunque `strategy.py` ya no los importe (son del bot "anytime" anterior, no
del activo `busqueda_eval3`/`eval3`): `herramientas/libro_apertura.py` los
importa directamente (`import busqueda_numba as bn`, `import eval as ev`)
para regenerar `data/libro.json` con la misma receta que produjo el
archivo entregado. `busqueda.py` sí es código muerto (ni `strategy.py` ni
`libro_apertura.py` lo usan); `fase2_limpia_supervisor.py` también lo era
y ya se borró (nada lo importaba).

## Verificado en esta sesión (2026-10-31)

Solo se confirmó que ambos scripts de pesos y los dos de libro **importan
sin error** con `NUMBA_DISABLE_JIT=1`; no se ejecutaron completos (piden
datasets/poblaciones que no viven en el repo). No se repitió la
reproducción de pesos con error 1.8e-15 en esta sesión; esa cifra es la
medición previa ya registrada en `obsidian/01-Estado.md`.
