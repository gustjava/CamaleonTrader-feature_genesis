#!/bin/bash

# Script para atualizar a configuração no servidor remoto sem reiniciar o pipeline

echo "🔄 Atualizando configuração no servidor remoto..."

# Identificar instância ativa
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

# Sincronizar apenas os arquivos de configuração
echo "🔄 Sincronizando arquivos de configuração..."
rsync -avz --progress -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519" \
  config/study/stageA.yaml \
  orchestration/main.py \
  "root@$SSH_HOST:/workspace/feature_genesis/"

echo "✅ Configuração atualizada no servidor remoto!"
echo "💡 O pipeline continuará com as novas configurações nos próximos trials."
