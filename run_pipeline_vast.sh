#!/usr/bin/env bash
set -eo pipefail

# ====================================================================================
# SCRIPT PARA EXECUÇÃO DIRETA DO PIPELINE NA VAST.AI
#
# Este script automatiza o processo de:
# 1. Conectar-se a uma instância JÁ EXISTENTE na vast.ai.
# 2. Conectar-se à instância remota.
# 3. Sincronizar seu código local para a instância remota.
# 4. Sincronizar os dados do R2 para a instância remota.
# 5. Executar o pipeline de features.
# ====================================================================================

# ----------------------------- CONFIGURAÇÕES GERAIS ---------------------------------
# Diretórios do projeto
LOCAL_PROJECT_DIR="$(pwd)" # Assume que está rodando da raiz do projeto 'feature_genesis'
REMOTE_PROJECT_DIR="/workspace/feature_genesis"
REMOTE_DATA_DIR="/data" # Diretório para os parquets na instância remota

# SSH
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"

# MySQL tunnel removed - no longer needed

# -------------------------- FUNÇÕES AUXILIARES/VERIFICAÇÕES -------------------------
need_cmd() { command -v "$1" &>/dev/null || { echo "Erro: '$1' não encontrado. Por favor, instale-o."; exit 1; }; }

echo "Verificando dependências: jq, ssh, rsync, nc..."
need_cmd jq
need_cmd ssh
need_cmd rsync
need_cmd nc
echo "Dependências OK."

# ------------------------------ CLI VAST.AI -----------------------------------------
VAST_BIN=""
if command -v vastai &>/dev/null; then
  VAST_BIN="$(command -v vastai)"
elif command -v vast &>/dev/null; then
  VAST_BIN="$(command -v vast)"
else
  echo "Erro: CLI da vast.ai ('vast' ou 'vastai') não encontrado no seu PATH."
  exit 1
fi
echo "Usando vast CLI: $VAST_BIN"
"$VAST_BIN" show user >/dev/null # Força o login se necessário

# --------------------------- SELEÇÃO/CONEXÃO INSTÂNCIA ------------------------------
echo -e "\n--- Instâncias ativas ---"
INSTANCES_RAW="$("$VAST_BIN" show instances --raw)"
INSTANCE_ID=$(echo "$INSTANCES_RAW" | jq -r '[.[] | select(.actual_status=="running")][0].id // empty')
if [[ -z "$INSTANCE_ID" ]]; then
  echo "❌ Nenhuma instância ativa encontrada."
  exit 1
fi
echo "✅ Instância selecionada automaticamente: $INSTANCE_ID"

# Extrair todas as informações da instância de uma vez
echo "📋 Coletando informações da instância..."
INSTANCE_INFO=$(echo "$INSTANCES_RAW" | jq -r ".[] | select(.id == $INSTANCE_ID)")
SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.ssh_host // empty')
SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ssh_port // empty')

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
  echo "❌ Erro: Não foi possível obter informações SSH da instância $INSTANCE_ID"
  echo "SSH_HOST: '$SSH_HOST'"
  echo "SSH_PORT: '$SSH_PORT'"
  exit 1
fi

echo "📋 Informações da instância:"
echo "  ID: $INSTANCE_ID"
echo "  SSH Host: $SSH_HOST"
echo "  SSH Port: $SSH_PORT"

# --------------------------- CONEXÃO SSH --------------------------------------------
echo "Aguardando SSH da instância $INSTANCE_ID ficar disponível..."
for i in {1..120}; do
  if nc -z -w5 "$SSH_HOST" "$SSH_PORT"; then
    echo "✅ SSH pronto em $SSH_HOST:$SSH_PORT"
    break
  fi
  echo -n "."
  sleep 5
done
echo ""

# Verificar se conseguiu conectar
if ! nc -z -w5 "$SSH_HOST" "$SSH_PORT"; then
  echo "❌ Erro fatal: Timeout ao esperar pela conexão SSH da instância."
  echo "Tentando conectar em: $SSH_HOST:$SSH_PORT"
  exit 1
fi

# --- SINCRONIZAÇÃO E EXECUÇÃO ---
SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i $SSH_KEY_PATH"
# SSH tunnel options removed - no longer needed

# Garantir que o diretório de destino existe na instância remota
echo -e "\n🔄  Preparando diretório remoto..."
ssh $SSH_OPTS "root@$SSH_HOST" "mkdir -p $REMOTE_PROJECT_DIR"
echo "✅ Diretório remoto pronto."

# MySQL tunnel removed - no longer needed

# --- CRIAR TÚNEL PARA DASHBOARD DASK ---
echo -e "\n🔗  Criando túnel SSH para dashboard Dask..."
DASHBOARD_TUNNEL_PID_FILE="/tmp/vast_dashboard_tunnel_${INSTANCE_ID}.pid"
DASHBOARD_LOCAL_PORT="8888"
DASHBOARD_REMOTE_PORT="8888"

# Mata qualquer túnel de dashboard anterior para esta instância
if [[ -f "$DASHBOARD_TUNNEL_PID_FILE" ]]; then
    OLD_DASHBOARD_PID=$(cat "$DASHBOARD_TUNNEL_PID_FILE")
    if kill -0 "$OLD_DASHBOARD_PID" 2>/dev/null; then
        echo "Matando túnel de dashboard anterior (PID: $OLD_DASHBOARD_PID)..."
        kill "$OLD_DASHBOARD_PID"
        sleep 2
    fi
    rm -f "$DASHBOARD_TUNNEL_PID_FILE"
fi

# Verifica se a porta local já está em uso
if nc -z -w5 127.0.0.1 "$DASHBOARD_LOCAL_PORT"; then
    echo "⚠️  Porta $DASHBOARD_LOCAL_PORT já está em uso. Tentando porta 8889..."
    DASHBOARD_LOCAL_PORT="8889"
    if nc -z -w5 127.0.0.1 "$DASHBOARD_LOCAL_PORT"; then
        echo "⚠️  Porta $DASHBOARD_LOCAL_PORT também está em uso. Tentando porta 8890..."
        DASHBOARD_LOCAL_PORT="8890"
    fi
fi

# Cria o túnel do dashboard em background com nohup
nohup ssh $SSH_OPTS -L $DASHBOARD_LOCAL_PORT:localhost:$DASHBOARD_REMOTE_PORT -N "root@$SSH_HOST" > /tmp/vast_dashboard_tunnel_${INSTANCE_ID}.log 2>&1 &
DASHBOARD_TUNNEL_PID=$!
echo "$DASHBOARD_TUNNEL_PID" > "$DASHBOARD_TUNNEL_PID_FILE"

# Aguarda um pouco para o túnel se estabelecer
echo "Aguardando túnel do dashboard se estabelecer..."
sleep 3

# Verifica se o túnel do dashboard está funcionando
if nc -z -w5 127.0.0.1 "$DASHBOARD_LOCAL_PORT"; then
    echo "✅ Túnel SSH para dashboard Dask criado (PID: $DASHBOARD_TUNNEL_PID)"
    echo "📝 Logs do túnel dashboard: /tmp/vast_dashboard_tunnel_${INSTANCE_ID}.log"
    echo "🌐 Dashboard disponível em: http://localhost:$DASHBOARD_LOCAL_PORT"
else
    echo "⚠️  Túnel do dashboard não conseguiu se estabelecer, mas continuando..."
    echo "📝 Logs do túnel dashboard: /tmp/vast_dashboard_tunnel_${INSTANCE_ID}.log"
fi

# --- SINCRONIZAÇÃO DE CÓDIGO ---
echo -e "\n🔄  Sincronizando código local com a instância remota via rsync..."
rsync -avz --delete -e "ssh $SSH_OPTS" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='data/' --exclude='logs/' \
  --exclude='output/' --exclude='*.sqlite' --exclude='*.db' \
  --exclude='trial_progress_*.json' --exclude='study_snapshot_*.json' \
  "$LOCAL_PROJECT_DIR/" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
echo "✅ Sincronização de código completa."

# Arquivos de programa já sincronizados pelo rsync principal acima

# --- LIMPEZA LOCAL DE PORTAS ---
echo -e "\n🧹 Limpando portas locais antes de iniciar..."
./clean_local_ports.sh

# --- VERIFICAÇÃO DE PORTAS ANTES DE INICIAR ---
echo -e "\n🔍 Verificando disponibilidade de portas no servidor remoto..."
ssh $SSH_OPTS "root@$SSH_HOST" '
echo "Verificando portas UCX no servidor remoto..."
PORTS_FREE=true
for port in 8888 8889 8890; do
    if lsof -i:$port >/dev/null 2>&1; then
        echo "❌ Porta $port está em uso:"
        lsof -i:$port
        PORTS_FREE=false
    else
        echo "✅ Porta $port está livre"
    fi
done

if [ "$PORTS_FREE" = false ]; then
    echo "⚠️  Algumas portas estão em uso. Executando limpeza..."
    echo "🔪 Finalizando processos nas portas..."
    for port in 8888 8889 8890; do
        PIDS=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            echo "  Matando processos na porta $port: $PIDS"
            echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
        fi
    done
    
    echo "⏳ Aguardando portas ficarem livres..."
    for port in 8888 8889 8890; do
        for i in {1..10}; do
            if ! lsof -i:$port >/dev/null 2>&1; then
                echo "  ✅ Porta $port livre"
                break
            fi
            if [ $i -eq 10 ]; then
                echo "  ⚠️  Porta $port ainda em uso após 10 tentativas"
            fi
            sleep 1
        done
    done
else
    echo "✅ Todas as portas estão livres!"
fi
'

# --- EXECUÇÃO DO PIPELINE COM TMUX DUAL TERMINAL ---
echo -e "\n🚀  Executando pipeline remotamente com monitoramento dual..."

# Environment variables for pipeline execution
REMOTE_ENV_EXPORTS=$(cat <<EOF
export LOG_LEVEL=INFO
export DEBUG=false
export CUDA_VISIBLE_DEVICES=0
export R2_ACCOUNT_ID=ac68ac775ba99b267edee7f9b4b3bc4e
export R2_ACCESS_KEY=0e315105695707ca4fe1e5f83a38f807
export R2_SECRET_KEY=5fbf8a2121f48807fdd3abc1c63c28cae6b67424f01e8d20a9cc68b1d47ca515
export R2_BUCKET_NAME=camaleon
export R2_ENDPOINT_URL=https://ac68ac775ba99b267edee7f9b4b3bc4e.r2.cloudflarestorage.com
EOF
)

# Comando de execução do pipeline
PIPELINE_CMD="
set -e
echo '--- [REMOTO] LIMPEZA COMPLETA DE PROCESSOS E PORTAS...'

# Função para aguardar com timeout
wait_for_condition() {
    local condition_cmd=\"\$1\"
    local timeout_seconds=\"\$2\"
    local description=\"\$3\"
    local interval=2
    
    echo \"⏳ Aguardando: \$description (timeout: \${timeout_seconds}s)\"
    
    for ((i=0; i<timeout_seconds; i+=interval)); do
        if eval \"\$condition_cmd\" 2>/dev/null; then
            echo \"✅ \$description - OK\"
            return 0
        fi
        echo -n \".\"
        sleep \$interval
    done
    
    echo \"❌ \$description - TIMEOUT após \${timeout_seconds}s\"
    return 1
}

# Função para verificar se porta está livre
check_port_free() {
    local port=\"\$1\"
    ! lsof -i:\$port >/dev/null 2>&1
}

echo '🔍 Verificando processos existentes...'
EXISTING_PROCESSES=\$(ps aux | grep -E \"(python.*orchestration/main\\.py|dask-worker|dask-scheduler|python.*main\\.py)\" | grep -v grep | wc -l)
if [ \$EXISTING_PROCESSES -gt 0 ]; then
    echo \"⚠️  Encontrados \$EXISTING_PROCESSES processo(s) ativo(s)\"
    ps aux | grep -E \"(python.*orchestration/main\\.py|dask-worker|dask-scheduler|python.*main\\.py)\" | grep -v grep
    
    # Verificar se há estudos Optuna ativos antes de matar processos
    echo \"🔍 Verificando estudos Optuna ativos...\"
    if [ -d \"\$REMOTE_PROJECT_DIR/output/optuna_stage\" ]; then
        echo \"📊 Estudos Optuna encontrados. Preservando estado...\"
        echo \"💡 Os estudos existentes serão continuados automaticamente.\"
    else
        echo \"📊 Nenhum estudo Optuna ativo encontrado.\"
    fi
    
    echo \"🔪 Finalizando processos Python do pipeline...\"
    pkill -TERM -f \"orchestration/main.py\" 2>/dev/null || true
    pkill -TERM -f \"python.*main.py\" 2>/dev/null || true
    
    echo \"🔪 Finalizando processos Dask...\"
    pkill -TERM -f \"dask-worker\" 2>/dev/null || true
    pkill -TERM -f \"dask-scheduler\" 2>/dev/null || true
    pkill -TERM -f \"distributed\" 2>/dev/null || true
    
    echo \"🔪 Finalizando processos CuDF/Rapids...\"
    pkill -TERM -f \"cudf\" 2>/dev/null || true
    pkill -TERM -f \"rapids\" 2>/dev/null || true
    
    echo \"⏳ Aguardando processos terminarem graciosamente (10s)...\"
    sleep 10
    
    echo \"🔪 Forçando término de processos remanescentes...\"
    pkill -9 -f \"orchestration/main.py\" 2>/dev/null || true
    pkill -9 -f \"python.*main.py\" 2>/dev/null || true
    pkill -9 -f \"dask-worker\" 2>/dev/null || true
    pkill -9 -f \"dask-scheduler\" 2>/dev/null || true
    pkill -9 -f \"distributed\" 2>/dev/null || true
    pkill -9 -f \"cudf\" 2>/dev/null || true
    pkill -9 -f \"rapids\" 2>/dev/null || true
    
    echo \"🧹 Limpando portas UCX...\"
    for port in 8888 8889 8890; do
        PIDS=\$(lsof -ti:\$port 2>/dev/null || true)
        if [ -n \"\$PIDS\" ]; then
            echo \"  Matando processos na porta \$port: \$PIDS\"
            echo \"\$PIDS\" | xargs -r kill -9 2>/dev/null || true
        fi
    done
    
    echo \"⏳ Aguardando portas ficarem livres...\"
    for port in 8888 8889 8890; do
        for i in {1..15}; do
            if ! lsof -i:\$port >/dev/null 2>&1; then
                echo \"  ✅ Porta \$port livre\"
                break
            fi
            if [ \$i -eq 15 ]; then
                echo \"  ⚠️  Porta \$port ainda em uso após 15 tentativas\"
            fi
            sleep 1
        done
    done
    
    echo \"🧹 Limpando memória GPU...\"
    nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true
    
    echo \"🧹 Limpando cache CUDA...\"
    python3 -c \"
import ctypes
try:
    libcudart = ctypes.CDLL('libcudart.so')
    libcudart.cudaDeviceReset()
    print('✅ Cache CUDA limpo')
except:
    print('⚠️  Não foi possível limpar cache CUDA')
\" 2>/dev/null || true
    
    echo \"✅ Limpeza completa finalizada\"
else
    echo '✅ Nenhum processo ativo encontrado'
fi

echo '🔍 Verificação final de limpeza...'
echo \"📊 Processos Python restantes:\"
REMAINING_PYTHON=\$(ps aux | grep python | grep -v grep | grep -v \"grep python\" || true)
if [ -n \"\$REMAINING_PYTHON\" ]; then
    echo \"\$REMAINING_PYTHON\"
else
    echo \"✅ Nenhum processo Python encontrado\"
fi

echo \"📊 Processos Dask restantes:\"
REMAINING_DASK=\$(ps aux | grep dask | grep -v grep | grep -v \"grep dask\" || true)
if [ -n \"\$REMAINING_DASK\" ]; then
    echo \"\$REMAINING_DASK\"
else
    echo \"✅ Nenhum processo Dask encontrado\"
fi

echo \"📊 Portas UCX em uso:\"
for port in 8888 8889 8890; do
    PORT_USAGE=\$(lsof -i:\$port 2>/dev/null || true)
    if [ -n \"\$PORT_USAGE\" ]; then
        echo \"  Porta \$port:\"
        echo \"\$PORT_USAGE\"
    else
        echo \"  ✅ Porta \$port livre\"
    fi
done

echo \"📊 Status GPU:\"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo \"⚠️  nvidia-smi não disponível\"

echo '--- [REMOTO] Configurando ambiente...'
cd $REMOTE_PROJECT_DIR
source /opt/conda/etc/profile.d/conda.sh

# Verificar se o ambiente dynamic-stage0 existe, senão usar base
if conda env list | grep -q 'dynamic-stage0'; then
    echo '✅ Ativando ambiente dynamic-stage0...'
    conda activate dynamic-stage0
else
    echo '⚠️  Usando ambiente base (RAPIDS já instalado)...'
    # Não ativar nenhum ambiente específico, usar o base
fi

eval "$REMOTE_ENV_EXPORTS"

echo '--- [REMOTO] Iniciando ESTUDO OPTUNA (Stage A) com 1200 trials...'
echo '🔬 Modo: stageA (preprocess_selection)'
echo '📊 Trials: 1200 (configurado em stageA.yaml)'
echo '🎯 Objetivo: Otimização de features para EURUSD'

# Verificação final das portas antes de iniciar o pipeline
echo '--- [REMOTO] Verificação final das portas antes de iniciar o pipeline...'
PORTS_FREE=true
for port in 8888 8889 8890; do
    if lsof -i:\$port >/dev/null 2>&1; then
        echo \"❌ Porta \$port ainda está em uso:\"
        lsof -i:\$port
        PORTS_FREE=false
    else
        echo \"✅ Porta \$port livre\"
    fi
done

if [ \"\$PORTS_FREE\" = false ]; then
    echo \"⚠️  ALGUMAS PORTAS AINDA ESTÃO EM USO! Tentando limpeza final...\"
    for port in 8888 8889 8890; do
        PIDS=\$(lsof -ti:\$port 2>/dev/null || true)
        if [ -n \"\$PIDS\" ]; then
            echo \"  Forçando limpeza da porta \$port: \$PIDS\"
            echo \"\$PIDS\" | xargs -r kill -9 2>/dev/null || true
            sleep 2
        fi
    done
    
    # Verificação final
    echo \"🔍 Verificação final após limpeza forçada:\"
    for port in 8888 8889 8890; do
        if lsof -i:\$port >/dev/null 2>&1; then
            echo \"❌ Porta \$port AINDA em uso após limpeza forçada!\"
            lsof -i:\$port
        else
            echo \"✅ Porta \$port finalmente livre\"
        fi
    done
else
    echo \"✅ Todas as portas estão livres! Pronto para iniciar o pipeline.\"
fi

# Verificar estado do estudo Optuna
echo '--- [REMOTO] Verificando estado do estudo Optuna...'
if [ -f \"\$REMOTE_PROJECT_DIR/output/optuna_stage/study_a.sqlite\" ]; then
    echo '📊 Banco de dados do estudo encontrado. Optuna continuará de onde parou.'
    echo '💡 Para forçar um novo estudo, delete: output/optuna_stage/study_a.sqlite'
else
    echo '🆕 Nenhum estudo anterior encontrado. Iniciando novo estudo.'
fi

# Executar o estudo Optuna com debug completo
echo '--- [REMOTO] Iniciando pipeline Python...'
HYDRA_FULL_ERROR=1 python orchestration/main.py
"

# Criar arquivo de log para o pipeline
PIPELINE_LOG_FILE="/tmp/vast_pipeline_${INSTANCE_ID}.log"
echo "📝 Logs do pipeline: $PIPELINE_LOG_FILE"

# Executa o comando via SSH diretamente
echo "🔄 Iniciando pipeline..."
echo "📋 Para acompanhar os logs em tempo real, execute em outro terminal:"
echo "  tail -f $PIPELINE_LOG_FILE"
echo ""

# Executa o pipeline e salva os logs
ssh $SSH_OPTS "root@$SSH_HOST" "$PIPELINE_CMD" 2>&1 | tee "$PIPELINE_LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]:-0}
echo ""
echo "📋 RESULTADO FINAL:"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline concluído com sucesso!"
else
    echo "❌ Pipeline falhou com código de saída: $EXIT_CODE"
fi

echo "📝 Logs completos salvos em: $PIPELINE_LOG_FILE"

echo -e "\n🔗 TÚNEIS SSH ATIVOS:"
if [[ -f "$DASHBOARD_TUNNEL_PID_FILE" ]]; then
    DASHBOARD_PID=$(cat "$DASHBOARD_TUNNEL_PID_FILE" 2>/dev/null || echo "N/A")
    echo "   • Dashboard Dask: localhost:$DASHBOARD_LOCAL_PORT → remoto:$DASHBOARD_REMOTE_PORT (PID: $DASHBOARD_PID)"
    echo "   • Acesse o dashboard em: http://localhost:$DASHBOARD_LOCAL_PORT"
else
    echo "   • Dashboard Dask: Não disponível"
fi

echo -e "\n💡 COMANDOS ÚTEIS:"
echo "   • Verificar túneis: ps aux | grep 'ssh.*$SSH_HOST'"
if [[ -f "$DASHBOARD_TUNNEL_PID_FILE" ]]; then
    echo "   • Parar túnel Dashboard: kill \$(cat $DASHBOARD_TUNNEL_PID_FILE)"
    echo "   • Ver logs Dashboard: tail -f /tmp/vast_dashboard_tunnel_${INSTANCE_ID}.log"
fi
