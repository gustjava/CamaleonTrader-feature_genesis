#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if ! command -v python >/dev/null 2>&1; then
  echo "[run_backend] Python não encontrado no PATH." >&2
  exit 1
fi

if ! python -m pip show uvicorn >/dev/null 2>&1; then
  echo "[run_backend] Instalando dependências do backend (requirements.txt)..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
fi

echo "[run_backend] Iniciando FastAPI em http://0.0.0.0:${BACKTEST_PORT:-8000}"
exec python -m uvicorn backtesting.service:create_app --host 0.0.0.0 --port "${BACKTEST_PORT:-8000}"
