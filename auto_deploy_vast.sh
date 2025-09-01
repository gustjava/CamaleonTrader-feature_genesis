#!/usr/bin/env bash
set -euo pipefail

# Auto-runner for deploy_to_vast.sh with retries until success
# Usage: AUTO=1 bash auto_deploy_vast.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="$SCRIPT_DIR/deploy_to_vast.sh"

if [[ ! -x "$DEPLOY_SCRIPT" ]]; then
  echo "Erro: deploy_to_vast.sh não encontrado ou não executável em $DEPLOY_SCRIPT" >&2
  exit 1
fi

ATTEMPT=0
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-0}  # 0 = infinito

echo "Iniciando auto-deploy (tentativas ilimitadas: $([[ "$MAX_ATTEMPTS" -eq 0 ]] && echo sim || echo não))"

while :; do
  ATTEMPT=$((ATTEMPT+1))
  echo "\n==== Tentativa #$ATTEMPT - $(date -Is) ===="
  set +e
  AUTO=1 bash "$DEPLOY_SCRIPT"
  EXIT_CODE=$?
  set -e

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "✅ deploy_to_vast.sh finalizado com sucesso na tentativa #$ATTEMPT"
    break
  fi

  echo "❌ deploy_to_vast.sh retornou código $EXIT_CODE"
  if [[ "$MAX_ATTEMPTS" -gt 0 && "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]]; then
    echo "⚠️  Atingido número máximo de tentativas ($MAX_ATTEMPTS). Abortando."
    exit "$EXIT_CODE"
  fi

  echo "⏳ Aguardando $SLEEP_SECONDS segundos antes de tentar novamente..."
  sleep "$SLEEP_SECONDS"
done

echo "🎉 Auto-deploy concluído."

