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
"$VAST_BIN" show instances --raw | jq -r '(["ID","Imagem","GPUs","Status"], (.[] | select(.actual_status=="running") | [.id, .image, .num_gpus, .status_msg])) | @tsv'
echo "----------------------------------------------------------------------------------"

read -rp "Digite o ID da instância vast.ai que deseja usar: " INSTANCE_ID
[[ -z "${INSTANCE_ID}" ]] && { echo "Erro: ID da instância vazio."; exit 1; }

echo "✅ Instância selecionada: $INSTANCE_ID"

echo "Aguardando SSH da instância $INSTANCE_ID ficar disponível..."
SSH_HOST=""; SSH_PORT=""
for i in {1..120}; do
  SSH_URL=$("$VAST_BIN" ssh-url "$INSTANCE_ID" 2>/dev/null || echo "")
  if [[ -n "$SSH_URL" ]]; then
    SSH_HOST=$(sed -n 's#ssh://root@\([^:]*\):.*#\1#p' <<<"$SSH_URL")
    SSH_PORT=$(sed -n 's#ssh://.*:\([0-9]*\).*#\1#p' <<<"$SSH_URL")
    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
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

echo -e "\n🔄  Sincronizando código local com a instância remota via rsync..."
rsync -avz --delete -e "ssh $SSH_OPTS" \
  --exclude='__pycache__/' --exclude='data/' --exclude='logs/' \
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

# Limpar cache do conda se necessário
echo '--- [REMOTO] Limpando cache do conda...'
conda clean --all -y || true

# Verificar se mamba está disponível, se não, instalar
if ! command -v mamba &> /dev/null; then
    echo 'Instalando mamba...'
    conda install mamba -n base -c conda-forge -y
fi

# Tentar criar/atualizar ambiente com diferentes abordagens
echo '--- [REMOTO] Criando/atualizando ambiente...'

# Tentar criar/atualizar ambiente com diferentes abordagens
echo '--- [REMOTO] Criando/atualizando ambiente...'

# Tentar criar/atualizar ambiente
if conda env list | grep -q 'dynamic-stage0'; then
    echo 'Atualizando ambiente existente...'
    # Tentar mamba primeiro
    if mamba env update -f environment.yml --prune; then
        echo '✅ Mamba atualizou ambiente com sucesso.'
    else
        echo 'Mamba falhou, tentando conda...'
        if conda env update -f environment.yml --prune; then
            echo '✅ Conda atualizou ambiente com sucesso.'
        else
            echo '❌ Falha na atualização do ambiente.'
        fi
    fi
else
    echo 'Criando novo ambiente...'
    # Tentar mamba primeiro
    if mamba env create -f environment.yml; then
        echo '✅ Mamba criou ambiente com sucesso.'
    else
        echo 'Mamba falhou, tentando conda...'
        if conda env create -f environment.yml; then
            echo '✅ Conda criou ambiente com sucesso.'
        else
            echo '❌ Falha na criação do ambiente. Tentando abordagem alternativa...'
            # Tentar criar ambiente básico e depois instalar RAPIDS
            conda create -n dynamic-stage0 python=3.10 -y
            conda activate dynamic-stage0
            conda install -c rapidsai -c conda-forge -c nvidia rapids=24.06 cuda-version=12.5 -y
            conda install -c conda-forge jupyterlab ipykernel pytest black flake8 -y
            echo '✅ Ambiente criado com abordagem alternativa.'
        fi
    fi
fi





# Ativar ambiente
conda activate dynamic-stage0

echo '--- [REMOTO] Instalando dependências adicionais...'
conda install -c conda-forge sqlalchemy pymysql -y

echo '--- [REMOTO] Sincronizando dados do R2...'
$REMOTE_ENV_EXPORTS

# **CORREÇÃO**: Cria o arquivo de config do rclone manualmente.
# Este método é mais robusto e compatível com versões mais antigas do rclone
# que não possuem a flag --non-interactive.
mkdir -p ~/.config/rclone
echo \"[R2]
type = s3
provider = Cloudflare
access_key_id = \$R2_ACCESS_KEY
secret_access_key = \$R2_SECRET_KEY
endpoint = \$R2_ENDPOINT_URL\" > ~/.config/rclone/rclone.conf

# Sync data from R2
rclone sync \"R2:\$R2_BUCKET_NAME\" \"$REMOTE_DATA_DIR\" --progress

# Check what files were synced
echo '--- [REMOTO] Verificando arquivos sincronizados...'
ls -la \"$REMOTE_DATA_DIR\" || echo \"Diretório $REMOTE_DATA_DIR não existe\"

# Check for master_features files specifically
echo '--- [REMOTO] Verificando arquivos master_features...'
MASTER_FEATURES_FILES=$(find \"$REMOTE_DATA_DIR\" -name \"*_master_features.parquet\" -type f 2>/dev/null | head -10)
if [[ -n \"$MASTER_FEATURES_FILES\" ]]; then
    echo \"Arquivos master_features encontrados:\"
    echo \"$MASTER_FEATURES_FILES\"
else
    echo \"Nenhum arquivo master_features encontrado\"
fi

# If no master_features files found, try to find any parquet files
echo '--- [REMOTO] Verificando outros arquivos parquet...'
PARQUET_FILES=$(find \"$REMOTE_DATA_DIR\" -name \"*.parquet\" -type f 2>/dev/null | head -10)
if [[ -n \"$PARQUET_FILES\" ]]; then
    echo \"Arquivos parquet encontrados:\"
    echo \"$PARQUET_FILES\"
else
    echo \"Nenhum arquivo parquet encontrado\"
fi

# Check if we have any data files to work with
if [[ -z \"$MASTER_FEATURES_FILES\" && -z \"$PARQUET_FILES\" ]]; then
    echo \"⚠️  AVISO: Nenhum arquivo de dados encontrado. O pipeline pode falhar.\"
    echo \"   Verifique se o R2 bucket contém os arquivos necessários.\"
    echo \"   Arquivos esperados: *_master_features.parquet\"
fi

echo '--- [REMOTO] Iniciando pipeline...'
# Suppress CUDA deprecation warnings
export PYTHONWARNINGS="ignore::FutureWarning:distributed.diagnostics.rmm,ignore::FutureWarning:rmm"
python orchestration/main.py
"

# Executa o comando final via SSH (túnel já está rodando em background)
ssh $SSH_OPTS "root@$SSH_HOST" "$REMOTE_EXEC_CMD"

echo -e "\n✅ Sessão de desenvolvimento finalizada."
echo -e "\n📋 RESUMO:"
echo "   • Túnel SSH persistente: ATIVO (PID: $TUNNEL_PID)"
echo "   • Arquivo PID: $TUNNEL_PID_FILE"
echo "   • Logs do túnel: /tmp/vast_tunnel_${INSTANCE_ID}.log"
echo "   • Porta local: $LOCAL_MYSQL_PORT → Porta remota: $REMOTE_MYSQL_PORT"
echo ""
echo "💡 COMANDOS ÚTEIS:"
echo "   • Verificar se túnel está ativo: ps aux | grep 'ssh.*$SSH_HOST'"
echo "   • Parar túnel: kill \$(cat $TUNNEL_PID_FILE)"
echo "   • Ver logs do túnel: tail -f /tmp/vast_tunnel_${INSTANCE_ID}.log"
echo "   • Gerenciar túneis: ./manage_tunnels.sh list|stop|stop-all"
echo ""
echo "⚠️  IMPORTANTE: O túnel continuará rodando mesmo após fechar esta sessão!"

