#!/usr/bin/env bash
# Se corre una vez al crear el Codespace: deja listo .venv/ para ./torneo.
# Codespaces no tiene Docker: ./torneo corre ahí con --sin-docker (se pone solo).
# Después, en la terminal: ./torneo jugar mi_equipo montecarlo
set -euo pipefail
cd "$(dirname "$0")/.."

if ! ./torneo instalar; then
    echo "⚠ No se pudo preparar .venv/. En la terminal, corre: ./torneo instalar"
    exit 0 # el Codespace abre igual; se puede repetir a mano
fi

echo
echo "✓ Listo. En una terminal NUEVA (ya trae .venv/ activado), prueba:"
echo "    ./torneo jugar random montecarlo"
echo "  Si tu terminal no muestra (.venv) al inicio, actívalo con: source .venv/bin/activate"
