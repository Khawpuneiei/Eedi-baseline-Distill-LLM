#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
fi

"$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$PROJECT_ROOT/requirements-cu121.txt"
"$PROJECT_ROOT/.venv/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT"

echo 'Environment installed. Activate with: source .venv/bin/activate'
echo 'Check CUDA with: python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
