# Laboratorio CPU y validación de fase 5

Paquete de laboratorio, no entrega del torneo. El principal no se sustituye. Los experimentos N y las reservas anteriores permanecen cerrados/intactos.

## Primera vez en Windows

1. Ejecutar `diagnostico_windows.ps1` en PowerShell para obtener CPU/RAM/Windows/WSL. Si falta WSL, instalar Ubuntu24.04 con `wsl --install -d Ubuntu-24.04` como administrador; reiniciar cuando lo solicite. No se hace desde esta Mac.
2. Instalar Docker Desktop para Windows con backend WSL2 si se van a medir partidas oficiales. La generación y el entrenamiento de este paquete no lo necesitan. No instalar soporte de GPU ni Kubernetes como requisito.
3. Copiar el tar y comprobar su SHA-256. Extraer dentro del home de Ubuntu en carpeta NUEVA, por ejemplo `~/breakthrough-lab-pc`. No sobreescribir el repositorio de entrega. No transportar cachés de Mac.
4. Desde esa carpeta: `bash estudiantes/iacasitegano/herramientas/pc/preparar_wsl.sh`. Necesita Python>=3.12 y python3-venv. Instala solo wheels NumPy/SciPy/Numba con versiones del curso en `.venv`. Requiere red durante instalación, no durante generación.
5. Reproducir primero el activo con el comando de LEEME de receta o el de abajo. Conectar corriente y evitar suspensión de Windows mientras corre; apagar pantalla sí se puede. Este paquete no modifica la configuración de energía.

Todos los comandos siguientes parten de la raíz extraída. Usar el Python de `.venv`; anteponer `rtk proxy` si RTK está instalado.

```sh
.venv/bin/python -B estudiantes/iacasitegano/herramientas/fase5_reproducir_activo.py --dataset entrada/posiciones_historicas.npz --pesos-antiguos estudiantes/iacasitegano/data/pesos.npy --referencia estudiantes/iacasitegano/data/pesos_eval3_ligero.npz --salida salida/receta_pc
```

## Piloto y lotes recuperables

El generador NUEVO añade Monte Carlo plano de presupuesto fijo (128 muestras×20 rollouts por acción) y UCT4096; conserva dos estilos sencillos como diversidad. Profesor U: profundidad máxima7 y3M nodos por iteración, conserva la última profundidad completa. Estos presupuestos NO prueban que el profesor sea óptimo ni que el rival sea de nivel tier. Las etiquetas son resultados reales de esas políticas.

Primero probar2 trabajadores diez minutos:

```sh
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/lanzar.py --salida salida/piloto2 --workers 2 --partidas 8 --minutos 10 --semilla 61000000
```

Luego4, en salida y semillas distintas:

```sh
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/lanzar.py --salida salida/piloto4 --workers 4 --partidas 8 --minutos 10 --semilla 63000000
```

Comparar partidas/posiciones totales producidas y RAM en la PC. El piloto no es una comparación de fuerza. No asumir aceleración4×. Un hilo BLAS/Numba por trabajador. Si4 no compensa o la RAM se agota, elegir2 trabajadores con128 episodios cada uno y otro contrato de producción, antes de generar el lote definitivo.

Lote completo previsto:4×64=256 episodios,128 grupos, ambos colores. Sesión de una hora:

```sh
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/lanzar.py --salida salida/lote256 --workers 4 --partidas 64 --minutos 60 --semilla 53000000
```

Para seguir siete horas, repetir EXACTAMENTE la misma orden cambiando solo `--minutos 420`; para24h,1440. Reanuda por checkpoint de cada ply, omite partidas completas y verifica fuentes/presupuestos. Si completa256 se detiene; no inventa más partidas para ocupar las horas. Para un lote mayor, fijar desde el inicio `--partidas 256` (1024 total) en otra salida y semillas no superpuestas, con contrato nuevo. No ampliar una muestra mirando resultados.

`Ctrl+C` detiene hijos conservando checkpoints. Tras apagado brusco puede quedar `ACTIVO.json`: comprobar que sus procesos y generadores ya no viven antes de retirar SOLO ese lock; nunca borrar resultados/manifest. No lanzar dos supervisores sobre la misma salida. Estado en `progreso_pc.json` y logs por trabajador. Un error o pausa no significa lote completo.

## Etapa5V obligatoria: preparar, seleccionar y examinar

Antes del ajuste, se puede auditar un lote COMPLETO y ver la calibración del modelo congelado sin seleccionarlo ni modificarlo. Cada lote debe tener su `VERIFICACION.json` vigente:

```sh
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/auditar_piloto.py --lotes salida/lote256/worker_0 salida/lote256/worker_1 salida/lote256/worker_2 salida/lote256/worker_3 --salida salida/auditoria_lote.json
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/calibracion_piloto.py --auditoria salida/auditoria_lote.json --modelo estudiantes/iacasitegano/data/pesos_eval3_ligero.npz --salida salida/calibracion_base.json
```

Los informes publican grupos, colores/fases/rivales, profundidad realmente completada y costo de la última iteración. Son diagnósticos; un piloto pequeño no acredita fuerza ni sustituye las puertas siguientes. Para un examen independiente futuro, congelar la elección de modelo antes de inspeccionar su test; esta calibración de desarrollo no lo sustituye.

Solo tras completar y verificar todos los lotes:

```sh
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/preparar_lotes.py --lotes salida/lote256/worker_0 salida/lote256/worker_1 salida/lote256/worker_2 salida/lote256/worker_3 --salida salida/datos_nuevos.npz
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/validacion.py preparar --datos salida/datos_nuevos.npz --historico entrada/historico_preparado.npz --base estudiantes/iacasitegano/data/pesos_eval3_ligero.npz --salida salida/validacion --tipo nuevo
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/entrenar_replay.py --directorio salida/validacion --salida salida/candidato/replay.npz
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/validacion.py seleccionar --directorio salida/validacion --id replay --candidato salida/candidato/replay.npz
```

Si selección devuelve código2, se RECHAZA: revisar RESULTADO.json, no ejecutar test. Si pasa:

```sh
.venv/bin/python -B estudiantes/iacasitegano/herramientas/pc/validacion.py examinar --directorio salida/validacion --id replay
```

UCT entero queda en test; demás grupos75/25 train/validation; se publican proporciones reales y se purgan estados/espejos cruzados. Antes de entrenar también se eliminan de validation/test los estados presentes en replay/retención históricos, y de train los presentes en retención; el contrato registra los descartes. Al menos24 grupos de entrenamiento,12 validación,12 test y4 por estrato. Transformaciones se calculan con entrenamiento. Una selección con2 grupos falla. No hay todavía CV de varios pliegues implementada: estas son tres particiones y estratos explícitos.

Hipótesis NUEVA fijada antes de entrenar: logística λ100 con prior activo y replay80%histórico/20%nuevo, peso igual por grupo dentro de cada origen. No reutiliza N ni cambia búsqueda/rasgos. Mejora mínima de Brier validación .002, logloss no peor; retención histórica +.005 máximo global, +.01 por fase; colores +.005 máximo. Test solo tras seleccionar, una exposición, no peor global y límites por estratos. Contrato, hashes de datos/modelo/representación/entrenador congelados. Máximo2 candidatos en selección.

El histórico ya fue visto por el activo: es control de retención, no examen independiente. Un resultado etiquetado `historico` nunca acredita datos nuevos. Un test aprobado SOLO permite pasar a controles técnicos/costo y partidas oficiales nuevas; nunca promueve, copia pesos al activo ni garantiza fuerza. Devolver hashes, resultados, logs y versiones al Mac. Reservas anteriores sin tocar. Fase9 seguirá como examen final.

## Kubernetes opcional

Kubernetes no aporta núcleos a una sola PC. El lote CPU usa una cola local más ligera. La Mac tiene kind/kubectl y un clúster de pruebas independiente, detenido tras verificar cuatro tareas; consultar la nota de infraestructura del proyecto. No ejecutar mediciones oficiales con control-plane, compilación o entrenamiento consumiendo la misma máquina.
