#!/bin/bash

# Função para aguardar com timeout
wait_for_condition() {
    local condition_cmd="$1"
    local timeout_seconds="$2"
    local description="$3"
    local interval=2
    
    echo "⏳ Aguardando: $description (timeout: ${timeout_seconds}s)"
    
    for ((i=0; i<timeout_seconds; i+=interval)); do
        if eval "$condition_cmd" 2>/dev/null; then
            echo "✅ $description - OK"
            return 0
        fi
        echo -n "."
        sleep $interval
    done
    
    echo "❌ $description - TIMEOUT após ${timeout_seconds}s"
    return 1
}

# Função para verificar se porta está livre
check_port_free() {
    local port="$1"
    ! lsof -i:$port >/dev/null 2>&1
}

# Função para aguardar porta ficar livre
wait_for_port_free() {
    local port="$1"
    local timeout=30
    wait_for_condition "check_port_free $port" $timeout "Porta $port ficar livre"
}

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

# Executar limpeza completa via SSH com aguardamento
echo '🛑 Executando limpeza completa no servidor remoto...'
ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST '
echo "🔍 Verificando processos existentes..."
EXISTING_PROCESSES=$(ps aux | grep -E "(python.*orchestration/main\.py|dask-worker|dask-scheduler|python.*main\.py)" | grep -v grep | wc -l)
if [ $EXISTING_PROCESSES -gt 0 ]; then
    echo "⚠️  Encontrados $EXISTING_PROCESSES processo(s) ativo(s)"
    ps aux | grep -E "(python.*orchestration/main\.py|dask-worker|dask-scheduler|python.*main\.py)" | grep -v grep
else
    echo "✅ Nenhum processo ativo encontrado"
fi

echo "🔪 Finalizando processos Python do pipeline..."
pkill -TERM -f "orchestration/main.py" 2>/dev/null || true
pkill -TERM -f "python.*main.py" 2>/dev/null || true

echo "🔪 Finalizando processos Dask..."
pkill -TERM -f "dask-worker" 2>/dev/null || true
pkill -TERM -f "dask-scheduler" 2>/dev/null || true
pkill -TERM -f "distributed" 2>/dev/null || true

echo "🔪 Finalizando processos CuDF/Rapids..."
pkill -TERM -f "cudf" 2>/dev/null || true
pkill -TERM -f "rapids" 2>/dev/null || true

echo "⏳ Aguardando processos terminarem graciosamente (10s)..."
sleep 10

echo "🔪 Forçando término de processos remanescentes..."
pkill -9 -f "orchestration/main.py" 2>/dev/null || true
pkill -9 -f "python.*main.py" 2>/dev/null || true
pkill -9 -f "dask-worker" 2>/dev/null || true
pkill -9 -f "dask-scheduler" 2>/dev/null || true
pkill -9 -f "distributed" 2>/dev/null || true
pkill -9 -f "cudf" 2>/dev/null || true
pkill -9 -f "rapids" 2>/dev/null || true

echo "🧹 Limpando portas UCX..."
for port in 8888 8889 8890; do
    PIDS=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "  Matando processos na porta $port: $PIDS"
        echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
    fi
done

echo "⏳ Aguardando portas ficarem livres..."
for port in 8888 8889 8890; do
    for i in {1..15}; do
        if ! lsof -i:$port >/dev/null 2>&1; then
            echo "  ✅ Porta $port livre"
            break
        fi
        if [ $i -eq 15 ]; then
            echo "  ⚠️  Porta $port ainda em uso após 15 tentativas"
        fi
        sleep 1
    done
done

echo "🧹 Limpando memória GPU..."
nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true

echo "🧹 Limpando cache CUDA..."
python3 -c "
import ctypes
try:
    libcudart = ctypes.CDLL('\''libcudart.so'\'')
    libcudart.cudaDeviceReset()
    print('\''✅ Cache CUDA limpo'\'')
except:
    print('\''⚠️  Não foi possível limpar cache CUDA'\'')
" 2>/dev/null || true

echo "✅ Limpeza completa finalizada"
'

echo '🔍 Verificação final de limpeza...'
ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST '
echo "📊 Processos Python restantes:"
REMAINING_PYTHON=$(ps aux | grep python | grep -v grep | grep -v "grep python" || true)
if [ -n "$REMAINING_PYTHON" ]; then
    echo "$REMAINING_PYTHON"
else
    echo "✅ Nenhum processo Python encontrado"
fi

echo "📊 Processos Dask restantes:"
REMAINING_DASK=$(ps aux | grep dask | grep -v grep | grep -v "grep dask" || true)
if [ -n "$REMAINING_DASK" ]; then
    echo "$REMAINING_DASK"
else
    echo "✅ Nenhum processo Dask encontrado"
fi

echo "📊 Portas UCX em uso:"
for port in 8888 8889 8890; do
    PORT_USAGE=$(lsof -i:$port 2>/dev/null || true)
    if [ -n "$PORT_USAGE" ]; then
        echo "  Porta $port:"
        echo "$PORT_USAGE"
    else
        echo "  ✅ Porta $port livre"
    fi
done

echo "📊 Status GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "⚠️  nvidia-smi não disponível"
'

echo '🎯 Comando concluído!'
