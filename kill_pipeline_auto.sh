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
echo '🛑 Executando limpeza completa no servidor remoto...'
ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST '
echo "🔪 Matando processos Python do pipeline..."
pkill -9 -f "orchestration/main.py" 2>/dev/null || true
pkill -9 -f "python.*main.py" 2>/dev/null || true

echo "🔪 Matando processos Dask..."
pkill -9 -f "dask" 2>/dev/null || true
pkill -9 -f "distributed" 2>/dev/null || true

echo "🔪 Matando processos CuDF/Rapids..."
pkill -9 -f "cudf" 2>/dev/null || true
pkill -9 -f "rapids" 2>/dev/null || true

echo "🧹 Limpando portas UCX..."
lsof -ti:8889 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:8888 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:8890 2>/dev/null | xargs -r kill -9 2>/dev/null || true

echo "🧹 Limpando memória GPU..."
nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true

echo "✅ Limpeza completa finalizada"
'

echo '🔍 Verificando se todos os processos foram mortos...'
ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST '
echo "📊 Processos Python restantes:"
ps aux | grep python | grep -v grep || echo "✅ Nenhum processo Python encontrado"
echo "📊 Processos Dask restantes:"
ps aux | grep dask | grep -v grep || echo "✅ Nenhum processo Dask encontrado"
echo "📊 Portas UCX em uso:"
lsof -i:8889 2>/dev/null || echo "✅ Porta 8889 livre"
lsof -i:8888 2>/dev/null || echo "✅ Porta 8888 livre"
lsof -i:8890 2>/dev/null || echo "✅ Porta 8890 livre"
'

echo '🎯 Comando concluído!'
