#!/usr/bin/env bash
# Ejecutar dentro de Ubuntu/WSL, desde el paquete extraído en carpeta nueva.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo "Este instalador es para Ubuntu/WSL. En Mac se verificó con el entorno existente."
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lab_root="$(cd -- "$script_dir/../../../.." && pwd)"
cd "$lab_root"
python3 -c 'import sys; assert sys.version_info >= (3,12), "Se necesita Python >=3.12"'
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || { echo 'Instala python3-venv en Ubuntu y repite.'; exit 1; }
fi
.venv/bin/python -m pip install --only-binary=:all: -r "$script_dir/requirements-cpu.txt"
.venv/bin/python -B verificar_paquete.py
.venv/bin/python -B "$script_dir/test_validacion.py"
echo 'Entorno listo. Sigue LEEME_PC.md: piloto de 10 minutos antes de una sesión larga.'
