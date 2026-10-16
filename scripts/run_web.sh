#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

PY="${CONDA_PREFIX:-/home/zyserver/miniconda3/envs/rs2sam2}/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="/home/zyserver/miniconda3/envs/rs2sam2/bin/python"
fi

cd "$ROOT/web"
echo "Starting annotator at http://0.0.0.0:7865"
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 7865
