#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${EXPOLICY_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x /data/ppnm/miniconda3/envs/bishop/bin/python ]]; then
    PYTHON=/data/ppnm/miniconda3/envs/bishop/bin/python
  else
    PYTHON="$(command -v python3)"
  fi
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON" -m venv --system-site-packages "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pip install omegaconf pytest pytest-cov ruff || true
python -m pip install faiss-cpu || echo "faiss-cpu optional; numpy index will be used"

python -m expolicy.cli.main setup --seed 13 --world-path "$ROOT/data/world/helion.json"
echo "ExPolicy setup complete. Activate with: source $ROOT/.venv/bin/activate"
