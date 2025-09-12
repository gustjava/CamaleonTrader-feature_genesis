#!/bin/bash

# Identificar instância ativa
echo '🔍 Identificando instância ativa...'
VAST_BIN=''
if command -v vastai &>/dev/null; then
  VAST_BIN=$(command -v vastai)
elif command -v vast &>/dev/null; then
  VAST_BIN=$(command -v vast)
else
  echo '❌ CLI da vast.ai não encontrado'
  exit 1
fi

INSTANCES_RAW="$($VAST_BIN show instances --raw)"
INSTANCE_ID=$(echo "$INSTANCES_RAW" | jq -r '[.[] | select(.actual_status=="running")][0].id // empty')

if [[ -z "$INSTANCE_ID" ]]; then
  echo '❌ Nenhuma instância ativa encontrada.'
  exit 1
fi

echo "✅ Instância encontrada: $INSTANCE_ID"

# Extrair informações SSH
INSTANCE_INFO=$(echo "$INSTANCES_RAW" | jq -r ".[] | select(.id == $INSTANCE_ID)")
SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.ssh_host // empty')
SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ssh_port // empty')

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
  echo '❌ Erro: Não foi possível obter informações SSH'
  exit 1
fi

echo "📋 Conectando em: $SSH_HOST:$SSH_PORT"

# Executar pkill via SSH
echo '🛑 Executando pkill no servidor remoto...'
ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST 'pkill -9 -f orchestration/main.py && echo "✅ Processo orchestration/main.py morto" || echo "⚠️  Processo não encontrado ou já estava morto"'

echo '🔍 Verificando se o processo foi morto...'
ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST 'ps aux | grep orchestration/main.py | grep -v grep || echo "✅ Nenhum processo encontrado"'

echo '🎯 Comando concluído!'
