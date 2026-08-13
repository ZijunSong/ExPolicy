#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUFFER="${1:-outputs/full_expolicy/utility_records.jsonl}"
python -m expolicy.cli.main train-policy --buffer "$BUFFER" --out outputs/experience_policy.npz --steps 200
