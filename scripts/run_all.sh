#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

"$PYTHON" "$ROOT/scripts/download_data.py"
"$PYTHON" "$ROOT/scripts/run_cleaning.py"
"$PYTHON" "$ROOT/scripts/run_dedup.py"
"$PYTHON" "$ROOT/scripts/run_multimodal.py"
