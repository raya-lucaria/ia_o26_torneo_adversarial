"""Controlador autónomo de una corrida UCT de 14 horas en Kubernetes.

Lanza un Job paralelo, espera su desenlace y, solo si todos los workers
terminan correctamente, consolida/entrena/valida en el host. Es seguro dejarlo
ejecutándose sin Codex abierto.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def run(cmd, **kwargs):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def out(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main(args):
    root = Path(args.root).resolve()
    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)
    kubectl = shutil.which("kubectl") or str(Path.home() / ".kube/bin/kubectl-desktop")
    if not Path(kubectl).exists() and shutil.which(kubectl) is None:
        raise SystemExit("No encuentro kubectl; configura Docker Desktop Kubernetes y vuelve a correr este comando.")
    run([kubectl, "cluster-info"], stdout=subprocess.DEVNULL)
    context = subprocess.check_output([kubectl, "config", "current-context"], text=True).strip()
    if context != "docker-desktop" and not context.startswith("kind-"):
        raise SystemExit(f"Contexto Kubernetes inesperado: {context}. Usa docker-desktop o un kind local.")
    workers = args.workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    job_id = "uct14h-" + uuid.uuid4().hex[:8]
    namespace = "breakthrough-uct14h"
    manifest = salida / "job.yaml"
    relative_output = str(salida.relative_to(root))
    worker_script = "/workspace/estudiantes/iacasitegano/herramientas/fase5_uct14h_worker.py"
    image = "torneo-bot-uct14h:fixed-dirs"
    base_image = "torneo-bot-base:local"
    if not args.dry_run:
        if subprocess.run(["docker", "image", "inspect", base_image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            run(["docker", "build", "-f", str(root / "sistema/Dockerfile"), "--target", "bot", "-t", base_image, str(root)])
        if subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            run(["docker", "build", "-f", str(root / "estudiantes/iacasitegano/herramientas/Dockerfile.uct14h"), "-t", image, str(root)])
    manifest.write_text(f'''apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_id}
  namespace: {namespace}
  labels:
    app: breakthrough-uct14h
spec:
  completions: {workers}
  parallelism: {workers}
  completionMode: Indexed
  backoffLimit: 0
  activeDeadlineSeconds: {args.minutes * 60 + 300:.0f}
  template:
    metadata:
      labels:
        app: breakthrough-uct14h
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: {image}
          imagePullPolicy: Never
          securityContext:
            runAsUser: 0
          env:
            - name: PYTHONPATH
              value: /workspace/sistema:/workspace/estudiantes/iacasitegano:/workspace/estudiantes/iacasitegano/herramientas
            - name: NUMBA_CACHE_DIR
              value: /tmp/numba
            - name: OPENBLAS_NUM_THREADS
              value: "1"
            - name: OMP_NUM_THREADS
              value: "1"
            - name: NUMBA_NUM_THREADS
              value: "1"
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
          command: ["sh", "-c"]
          args:
            - >-
              exec python -B {worker_script} --root /workspace
              --salida /workspace/{relative_output}/worker-$JOB_COMPLETION_INDEX
              --minutos {args.minutes} --uct-iteraciones 4096
              --limite-jugada-s {args.move_seconds}
              --semilla $(( {args.seed} + JOB_COMPLETION_INDEX * 1000000 ))
          resources:
            requests:
              cpu: "1"
              memory: "1Gi"
            limits:
              cpu: "1"
              memory: "2Gi"
          volumeMounts:
            - name: salida
              mountPath: /workspace/{relative_output}
      volumes:
        - name: salida
          emptyDir: {{}}
''')
    out(salida / "CONTROL.json", {
        "inicio": dt.datetime.now().astimezone().isoformat(), "job": job_id,
        "namespace": namespace, "context": context, "workers": workers,
        "minutes": args.minutes, "rival": "UCT4096", "seed": args.seed,
        "move_seconds": args.move_seconds, "active_model_untouched": True,
        "image": image, "salida_en_pod": "/workspace/" + relative_output,
        "transferencia": "kubectl cp al finalizar; sin hostPath del macOS",
        "manifest": str(manifest),
    })
    if args.dry_run:
        print(json.dumps({"job": job_id, "namespace": namespace, "workers": workers, "manifest": str(manifest)}, ensure_ascii=False, indent=2))
        return
    run([kubectl, "apply", "-f", str(manifest)])
    while True:
        status = json.loads(subprocess.check_output([kubectl, "get", "job", job_id, "-n", namespace, "-o", "json"], text=True))
        st = status.get("status", {})
        out(salida / "K8S_STATUS.json", {"at": dt.datetime.now().astimezone().isoformat(), "status": st})
        if st.get("succeeded", 0) >= workers:
            break
        if st.get("failed", 0) > 0:
            out(salida / "CONTROL_FINAL.json", {"estado": "workers_failed", "status": st, "job": job_id})
            raise SystemExit("Uno o más workers fallaron; no se ejecuta entrenamiento automático.")
        time.sleep(60)
    run([kubectl, "logs", "job/" + job_id, "-n", namespace, "--all-containers=true"], stdout=(salida / "k8s_logs.txt").open("w"), check=True)
    pods = json.loads(subprocess.check_output([
        kubectl, "get", "pods", "-n", namespace, "-l", "job-name=" + job_id, "-o", "json"
    ], text=True))["items"]
    for pod in pods:
        labels = pod.get("metadata", {}).get("labels", {})
        index = labels.get("batch.kubernetes.io/job-completion-index")
        if index is None:
            raise SystemExit("No encuentro el índice de completado de un worker; no se consolida.")
        destino = salida / ("worker-" + str(index))
        origen = f"{namespace}/{pod['metadata']['name']}:/workspace/{relative_output}/worker-{index}"
        run([kubectl, "cp", origen, str(destino)])
    post = root / "estudiantes/iacasitegano/herramientas/fase5_uct14h_post.py"
    run([sys.executable, "-B", str(post), "--root", str(root), "--salida", str(salida)])
    out(salida / "CONTROL_FINAL.json", {
        "estado": "completo_y_validado_internamente",
        "fin": dt.datetime.now().astimezone().isoformat(), "job": job_id,
        "modelo_activo_tocado": False,
        "siguiente": "revisar VALIDACION_INTERNA.json y ejecutar controles funcionales/partidas solo con protocolo aprobado",
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--minutes", type=int, default=840)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--move-seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=202610300)
    p.add_argument("--dry-run", action="store_true")
    main(p.parse_args())
