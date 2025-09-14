#!/usr/bin/env bash
set -euo pipefail

# ====================================================================================
# SCRIPT PARA SESSÃO DE DESENVOLVIMENTO REMOTO NA VAST.AI
#
# Este script automatiza o processo de:
# 1. Conectar-se a uma instância JÁ EXISTENTE na vast.ai.
# 2. Criar um túnel SSH reverso para seu banco de dados MySQL local.
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
SSH_USER="root"

# Configurações do Túnel MySQL
LOCAL_MYSQL_PORT="3010"
REMOTE_MYSQL_PORT="3010"


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
echo "$INSTANCES_RAW" | jq -r '(["ID","Imagem","GPUs","Status"], (.[] | select(.actual_status=="running") | [.id, .image, .num_gpus, .status_msg])) | @tsv'
echo "----------------------------------------------------------------------------------"

# Seleciona a primeira instância em execução por padrão, caso o usuário apenas pressione Enter
FIRST_INSTANCE_ID="$(echo "$INSTANCES_RAW" | jq -r '[.[] | select(.actual_status=="running")][0].id // empty')"
if [[ -z "$FIRST_INSTANCE_ID" ]]; then
  echo "❌ Nenhuma instância em execução encontrada."
  exit 1
fi

if [[ -n "${AUTO:-}" || ! -t 0 ]]; then
  # Modo não interativo ou AUTO: usa a primeira instância
  INSTANCE_ID="${FIRST_INSTANCE_ID}"
  echo "AUTO=1 ou stdin não interativo: usando a primeira instância ($INSTANCE_ID)"
else
  # Prompt com timeout de 5s; default para a primeira instância
  read -t 5 -rp "Digite o ID da instância vast.ai (Enter para usar a primeira: $FIRST_INSTANCE_ID): " INSTANCE_ID || true
fi
INSTANCE_ID="${INSTANCE_ID:-$FIRST_INSTANCE_ID}"
[[ -z "${INSTANCE_ID}" ]] && { echo "Erro: ID da instância vazio."; exit 1; }

echo "✅ Instância selecionada: $INSTANCE_ID"

echo "Aguardando SSH da instância $INSTANCE_ID ficar disponível..."
SSH_HOST=""; SSH_PORT=""
for i in {1..120}; do
  # Usar vast show instances em vez de vast ssh-url para evitar inconsistências
  INSTANCE_INFO=$("$VAST_BIN" show instances --raw | jq -r ".[] | select(.id == $INSTANCE_ID) | {ssh_host, ssh_port}" 2>/dev/null || echo "")
  if [[ -n "$INSTANCE_INFO" && "$INSTANCE_INFO" != "null" ]]; then
    SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.ssh_host' 2>/dev/null || echo "")
    SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ssh_port' 2>/dev/null || echo "")
    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" && "$SSH_HOST" != "null" && "$SSH_PORT" != "null" ]]; then
      if nc -z -w5 "$SSH_HOST" "$SSH_PORT"; then
        echo "✅ SSH pronto em $SSH_HOST:$SSH_PORT"
        break
      fi
    fi
  fi
  echo -n "."
  sleep 5
done
echo ""
[[ -z "$SSH_HOST" || -z "$SSH_PORT" ]] && { echo "Erro fatal: Timeout ao esperar pela conexão SSH da instância."; exit 1; }

# --- SINCRONIZAÇÃO E EXECUÇÃO ---
SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i $SSH_KEY_PATH"
SSH_TUNNEL_OPTS="-R $REMOTE_MYSQL_PORT:127.0.0.1:$LOCAL_MYSQL_PORT"

# **CORREÇÃO**: Garantir que o diretório de destino existe na instância remota
echo -e "\n🔄  Preparando diretório remoto..."
ssh $SSH_OPTS "root@$SSH_HOST" "mkdir -p $REMOTE_PROJECT_DIR"
echo "✅ Diretório remoto pronto."

# --- CRIAR TÚNEL PERSISTENTE COM NOHUP ---
echo -e "\n🔗  Criando túnel SSH persistente com nohup..."
TUNNEL_PID_FILE="/tmp/vast_tunnel_${INSTANCE_ID}.pid"

# Mata qualquer túnel anterior para esta instância
if [[ -f "$TUNNEL_PID_FILE" ]]; then
    OLD_PID=$(cat "$TUNNEL_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Matando túnel anterior (PID: $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f "$TUNNEL_PID_FILE"
fi

# Cria o túnel em background com nohup
nohup ssh $SSH_OPTS $SSH_TUNNEL_OPTS -N "root@$SSH_HOST" > /tmp/vast_tunnel_${INSTANCE_ID}.log 2>&1 &
TUNNEL_PID=$!
echo "$TUNNEL_PID" > "$TUNNEL_PID_FILE"

# Aguarda um pouco para o túnel se estabelecer
echo "Aguardando túnel se estabelecer..."
sleep 3

# Verifica se o túnel está funcionando
if ! nc -z -w5 127.0.0.1 "$LOCAL_MYSQL_PORT"; then
    echo "❌ Erro: Túnel não conseguiu se estabelecer. Verificando logs..."
    cat "/tmp/vast_tunnel_${INSTANCE_ID}.log"
    exit 1
fi

echo "✅ Túnel SSH persistente criado (PID: $TUNNEL_PID)"
echo "📝 Logs do túnel: /tmp/vast_tunnel_${INSTANCE_ID}.log"
echo "💡 Para parar o túnel: kill \$(cat $TUNNEL_PID_FILE)"

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

echo -e "\n🔄  Sincronizando código local com a instância remota via rsync..."
rsync -avz --delete -e "ssh $SSH_OPTS" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='data/' --exclude='logs/' \
  "$LOCAL_PROJECT_DIR/" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
echo "✅ Sincronização de código completa."

# Variáveis de ambiente para R2 e MySQL a serem usadas no comando remoto
REMOTE_ENV_EXPORTS=$(cat <<EOF
export R2_ACCOUNT_ID=ac68ac775ba99b267edee7f9b4b3bc4e
export R2_ACCESS_KEY=0e315105695707ca4fe1e5f83a38f807
export R2_SECRET_KEY=5fbf8a2121f48807fdd3abc1c63c28cae6b67424f01e8d20a9cc68b1d47ca515
export R2_BUCKET_NAME=camaleon
export R2_ENDPOINT_URL=https://ac68ac775ba99b267edee7f9b4b3bc4e.r2.cloudflarestorage.com
export R2_REGION=auto
export MYSQL_HOST=127.0.0.1
export MYSQL_PORT=${REMOTE_MYSQL_PORT}
export MYSQL_DATABASE=dynamic_stage0_db
export MYSQL_USERNAME=root
export MYSQL_PASSWORD=root
export LOG_LEVEL=INFO
export DEBUG=false
# Configurações CUDA para evitar warnings deprecated
export CUDA_USE_DEPRECATED_API=0
export RMM_USE_NEW_CUDA_BINDINGS=1
export CUDF_USE_NEW_CUDA_BINDINGS=1
EOF
)

echo -e "\n🚀  Executando o pipeline remotamente (com túnel para MySQL local)..."

# Comando final, agora incluindo a sincronização de dados
REMOTE_EXEC_CMD="
set -e
echo '--- [REMOTO] Configurando ambiente...'
cd $REMOTE_PROJECT_DIR
source /opt/conda/etc/profile.d/conda.sh

# Verificar recursos do sistema
echo '--- [REMOTO] Verificando recursos do sistema...'
echo \"Memória disponível: \$(free -h | grep Mem | awk '{print \$7}')\"
echo \"Espaço em disco: \$(df -h / | tail -1 | awk '{print \$4}')\"
echo \"GPUs disponíveis: \$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo '0')\"

# Verificar e ativar ambiente (já criado pelo onstart.sh)
echo '--- [REMOTO] Verificando ambiente conda...'

echo '--- [REMOTO] Verificando ambiente conda...'
if conda env list | grep -q 'dynamic-stage0'; then
    echo '✅ Ambiente dynamic-stage0 encontrado, ativando...'
    conda activate dynamic-stage0
elif conda env list | grep -q 'feature-genesis'; then
    echo '✅ Ambiente feature-genesis encontrado, ativando...'
    conda activate feature-genesis
else
    echo '⚠️  Nenhum dos ambientes esperados (dynamic-stage0/feature-genesis) encontrado!'
    echo 'Usando ambiente base (RAPIDS já instalado, se aplicável)...'
    # Não ativar nenhum ambiente específico, usar o base
fi

# --- Garantir bibliotecas de seleção (CatBoost/LightGBM) instaladas ---
echo '--- [REMOTO] Instalando dependências adicionais (CatBoost/LightGBM para seleção de features)...'
conda install -y lightgbm || true
conda install -c conda-forge -y catboost || true

echo '--- [REMOTO] Instalando dependências adicionais para Stage 3/4...'
# LightGBM (árvores), scikit-learn (LassoCV/TSS), XGBoost (GPU opcional), matplotlib (plots Stage 4)
conda install -c conda-forge -y \
  sqlalchemy pymysql cryptography \
  scikit-learn lightgbm catboost xgboost matplotlib || true

echo '--- [REMOTO] Garantindo instalação da biblioteca EMD (signal processing)...'
pip install --no-cache-dir emd || true

echo '--- [REMOTO] Verificando se rclone está instalado...'
if ! command -v rclone &> /dev/null; then
    echo 'rclone não encontrado, instalando...'
    curl https://rclone.org/install.sh | bash
else
    echo 'rclone já está instalado'
fi

echo '--- [REMOTO] Configurando rclone com credenciais seguras...'
# Configurar rclone de forma segura
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << 'EOF'
[R2]
type = s3
provider = Cloudflare
access_key_id = 0e315105695707ca4fe1e5f83a38f807
secret_access_key = 5fbf8a2121f48807fdd3abc1c63c28cae6b67424f01e8d20a9cc68b1d47ca515
endpoint = https://ac68ac775ba99b267edee7f9b4b3bc4e.r2.cloudflarestorage.com
EOF

echo '--- [REMOTO] Sincronizando dados do R2...'
$REMOTE_ENV_EXPORTS

# Criar diretório de dados se não existir
mkdir -p \"$REMOTE_DATA_DIR\"

# Sync data from R2
rclone sync \"R2:\$R2_BUCKET_NAME\" \"$REMOTE_DATA_DIR\" --progress

# Check what files were synced
echo '--- [REMOTO] Verificando arquivos sincronizados...'
ls -la \"$REMOTE_DATA_DIR\" || echo \"Diretório $REMOTE_DATA_DIR não existe\"

# Check for master_features files specifically
echo '--- [REMOTO] Verificando arquivos master_features...'
find \"$REMOTE_DATA_DIR\" -name \"*_master_features.parquet\" -type f 2>/dev/null | head -10 || echo \"Nenhum arquivo master_features encontrado\"

# If no master_features files found, try to find any parquet files
echo '--- [REMOTO] Verificando outros arquivos parquet...'
find \"$REMOTE_DATA_DIR\" -name \"*.parquet\" -type f 2>/dev/null | head -10 || echo \"Nenhum arquivo parquet encontrado\"

# Simple check if directory has any data files
echo '--- [REMOTO] Verificando se há arquivos de dados...'
if ! ls \"$REMOTE_DATA_DIR\"/*.parquet 1>/dev/null 2>&1; then
    echo \"⚠️  AVISO: Nenhum arquivo de dados encontrado. O pipeline pode falhar.\"
    echo \"   Verifique se o R2 bucket contém os arquivos necessários.\"
    echo \"   Arquivos esperados: *_master_features.parquet\"
fi

echo '--- [REMOTO] Verificando processos existentes...'
EXISTING_PIDS=\$(ps -eo pid,command | grep -E '(python .*orchestration/main\\.py|dask-worker|dask-scheduler)' | grep -v grep | awk '{print \$1}')
if [ -n \"\$EXISTING_PIDS\" ]; then
    COUNT=\$(echo \"\$EXISTING_PIDS\" | wc -w)
    echo \"⚠️  ATENÇÃO: \$COUNT PROCESSO(S) EXISTENTE(S) DETECTADO(S)!\"
    ps -fp \$EXISTING_PIDS 2>/dev/null || true
    echo \"⚠️  Processos existentes detectados. Verifique se deseja continuar.\"
    echo \"Aguardando 10 segundos antes de continuar...\"
    sleep 10
else
    echo '✅ Nenhum processo do pipeline detectado.'
fi

echo '--- [REMOTO] Iniciando pipeline...'
python orchestration/main.py
"

# Executa o comando final via SSH (túnel já está rodando em background)
ssh $SSH_OPTS "root@$SSH_HOST" "$REMOTE_EXEC_CMD"

echo -e "\n✅ Sessão de desenvolvimento finalizada."
echo -e "\n📋 RESUMO:"
echo "   • Túnel SSH MySQL: ATIVO (PID: $TUNNEL_PID)"
echo "   • Arquivo PID MySQL: $TUNNEL_PID_FILE"
echo "   • Logs do túnel MySQL: /tmp/vast_tunnel_${INSTANCE_ID}.log"
echo "   • Porta local MySQL: $LOCAL_MYSQL_PORT → Porta remota: $REMOTE_MYSQL_PORT"
if [[ -f "$DASHBOARD_TUNNEL_PID_FILE" ]]; then
    DASHBOARD_PID=$(cat "$DASHBOARD_TUNNEL_PID_FILE" 2>/dev/null || echo "N/A")
    echo "   • Túnel SSH Dashboard: ATIVO (PID: $DASHBOARD_PID)"
    echo "   • Arquivo PID Dashboard: $DASHBOARD_TUNNEL_PID_FILE"
    echo "   • Logs do túnel Dashboard: /tmp/vast_dashboard_tunnel_${INSTANCE_ID}.log"
    echo "   • Dashboard disponível em: http://localhost:$DASHBOARD_LOCAL_PORT"
fi
echo ""
echo "💡 COMANDOS ÚTEIS:"
echo "   • Verificar se túneis estão ativos: ps aux | grep 'ssh.*$SSH_HOST'"
echo "   • Parar túnel MySQL: kill \$(cat $TUNNEL_PID_FILE)"
if [[ -f "$DASHBOARD_TUNNEL_PID_FILE" ]]; then
    echo "   • Parar túnel Dashboard: kill \$(cat $DASHBOARD_TUNNEL_PID_FILE)"
fi
echo "   • Ver logs do túnel MySQL: tail -f /tmp/vast_tunnel_${INSTANCE_ID}.log"
if [[ -f "$DASHBOARD_TUNNEL_PID_FILE" ]]; then
    echo "   • Ver logs do túnel Dashboard: tail -f /tmp/vast_dashboard_tunnel_${INSTANCE_ID}.log"
fi
echo "   • Gerenciar túneis: ./manage_tunnels.sh list|stop|stop-all"
echo ""
echo "⚠️  IMPORTANTE: Os túneis continuarão rodando mesmo após fechar esta sessão!"
