#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FRONT_DIR="$REPO_ROOT/frontend/backtest-ui"

if ! command -v npm >/dev/null 2>&1; then
  echo "[run_frontend] npm não encontrado no PATH. Instale Node.js 18+." >&2
  exit 1
fi

cd "$FRONT_DIR"

echo "[run_frontend] Instalando dependências do frontend..."
rm -f package-lock.json
npm install

echo "[run_frontend] Iniciando servidor Angular (http://localhost:4200)"
npm start
