#!/usr/bin/env bash
# Run inside the Conda environment created from environment.yml. Set PYTHON_BIN
# explicitly when calling from a shell where Conda activation is unavailable.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "Python was not found. Activate the Conda environment or set PYTHON_BIN." >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install -e .
"$PYTHON_BIN" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m concept_retrofit.cli synthetic-smoke \
  --steps 500 \
  --out runs/smoke/synthetic-report.json
