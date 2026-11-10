# Validación de fase5 en Kubernetes

Uso offline. No cambia estrategia/pesos ni reemplaza el árbitro. Cada tarea realiza varias verificaciones y recibe como máximo1CPU/1536MiB. El checkout se copia a una imagen; los pods no montan el repositorio para escritura ni reciben credenciales Kubernetes.

`suite.py` implementa cuatro tareas: tests de validación/modelos, reproducción de receta completa, generación funcional/reanudación/replay/particiones y controles retrospectivos de retención/replay histórico. `recoger.py` verifica un archivo comprimido exportado por los logs y conserva resultados locales; evita perderlos cuando termina el pod.

Protocolo, contrato de entradas, Dockerfile congelado, imagen y JSON de trabajos: `resultados/medicion/fase5_mac_k8s_20261029/`. Las tareas0/1 pasaron en el trabajo original. La tarea3 intentó guardar un informe en la copia de solo lectura; se corrigió usando `/tmp`. La tarea2 fue cancelada por ese fallo. Los reintentos independientes2/3 pasaron con los mismos presupuestos y fuente de datos. Ver `INCIDENCIA_R1.json` y `RESUMEN_KUBERNETES.json`. No repetir las tareas correctas para inflar resultados.

Para una nueva batería, usar nombres/salida nuevos, congelar entradas antes de lanzar y comprobar que no haya mediciones oficiales en el host. La primera imagen se construyó desde la imagen local de dependencias SHA `79983872dd22263467bfb3f1e19a8619edb15cacfb79f2d9aef4f03b2d19d188`; conservar la identidad en IMAGEN.json, no suponer que una etiqueta mutable identifica siempre las mismas dependencias.

Ejemplos de recuperación de la batería ya ejecutada (clúster debe estar iniciado):

```sh
rtk proxy python3 estudiantes/iacasitegano/herramientas/pc/kubernetes/recoger.py --kubeconfig "$HOME/.kube/breakthrough-lab.config" --salida resultados/medicion/fase5_mac_k8s_20261029/exportacion
rtk proxy python3 estudiantes/iacasitegano/herramientas/pc/kubernetes/recoger.py --kubeconfig "$HOME/.kube/breakthrough-lab.config" --salida resultados/medicion/fase5_mac_k8s_20261029/reintento2 --job fase5-validacion-20261029-tarea2-r1 --esperadas 1
```

No ejecutar `apply` sobre un Job ya terminado esperando reanudarlo. Las ejecuciones futuras requieren otro nombre y su manifiesto. Antes de apagar kind, exportar y verificar todos los resultados; después `rtk proxy docker stop breakthrough-lab-control-plane`. Las partidas funcionales de profundidad2 no se incorporan al entrenamiento de producción.
