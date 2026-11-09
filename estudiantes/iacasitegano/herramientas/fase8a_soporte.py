"""Utilidades locales de 8A; sin dependencias del flujo de fase5."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'resultados/medicion/fase8A_preintegracion_20261029'
TEAM = ROOT / 'estudiantes/iacasitegano'


class Budget:
    def __init__(self):
        self.wall, self.cpu = time.monotonic(), time.process_time()

    def wait(self):
        while True:
            delay = (time.process_time() - self.cpu) / 0.12 - (time.monotonic() - self.wall)
            if delay <= 0:
                return
            time.sleep(min(delay, 0.25))


def sha(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            if budget:
                budget.wait()
    return digest.hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def engine():
    for name in ['NUMBA_DISABLE_JIT', 'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']:
        os.environ[name] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'sistema'))
    from motor import Estado, Jugada
    return Estado, Jugada
