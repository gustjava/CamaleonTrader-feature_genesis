#!/usr/bin/env bash
set -euo pipefail

# ====================================================================================
# SCRIPT PARA EXECUÇÃO DIRETA DO PIPELINE NA VAST.AI
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
INSTANCES_RAW="$("$VAST_BIN" show instances --raw)"
INSTANCE_ID=$(echo "$INSTANCES_RAW" | jq -r '[.[] | select(.actual_status=="running")][0].id // empty')
if [[ -z "$INSTANCE_ID" ]]; then
  echo "❌ Nenhuma instância ativa encontrada."
  exit 1
fi
echo "✅ Instância selecionada automaticamente: $INSTANCE_ID"

# --------------------------- CONEXÃO SSH --------------------------------------------
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

# Garantir que o diretório de destino existe na instância remota
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

# --- SINCRONIZAÇÃO DE CÓDIGO ---
echo -e "\n🔄  Sincronizando código local com a instância remota via rsync..."
rsync -avz --delete -e "ssh $SSH_OPTS" \
  --exclude='__pycache__/' --exclude='data/' --exclude='logs/' \
  "$LOCAL_PROJECT_DIR/" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
echo "✅ Sincronização de código completa."

# --- SINCRONIZAÇÃO DE ARQUIVOS DE PROGRAMA ---
echo -e "\n🔄  Sincronizando arquivos de programa..."
# Sincronizar arquivos específicos que podem ter sido modificados
rsync -avz -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/deploy_to_vast.sh" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
rsync -avz -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/run_pipeline_vast.sh" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
rsync -avz -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/environment.yml" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
rsync -avz -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/onstart.sh" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
echo "✅ Sincronização de arquivos de programa completa."

# --- EXECUÇÃO DO PIPELINE ---
echo -e "\n🚀  Executando pipeline remotamente..."

# Variáveis de ambiente para MySQL
REMOTE_ENV_EXPORTS=$(cat <<EOF
export MYSQL_HOST=127.0.0.1
export MYSQL_PORT=${REMOTE_MYSQL_PORT}
export MYSQL_DATABASE=dynamic_stage0_db
export MYSQL_USERNAME=root
export MYSQL_PASSWORD=root
export LOG_LEVEL=INFO
export DEBUG=false
EOF
)

# Comando de execução do pipeline
PIPELINE_CMD="
set -e
echo '--- [REMOTO] Verificando processos existentes...'
EXISTING_PIDS=\$(ps -eo pid,command | grep -E '(python .*orchestration/main\\.py|dask-worker|dask-scheduler)' | grep -v grep | sed -E 's/^[[:space:]]*([0-9]+).*/\1/')
if [ -n \"\$EXISTING_PIDS\" ]; then
    COUNT=\$(echo \"\$EXISTING_PIDS\" | wc -w)
    echo \"⚠️  ATENÇÃO: \$COUNT PROCESSO(S) EXISTENTE(S) DETECTADO(S)!\"
    ps -fp \$EXISTING_PIDS 2>/dev/null || true
    echo \"⚠️  Processos existentes detectados. Verifique se deseja continuar.\"
    echo \"Aguardando 10 segundos antes de continuar...\"
    sleep 10
else
    echo '✅ Nenhum processo do pipeline detectado. Continuando...'
fi

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

$REMOTE_ENV_EXPORTS

echo '--- [REMOTO] Iniciando pipeline...'
python orchestration/main.py
"

# Criar arquivo de log para o pipeline
PIPELINE_LOG_FILE="/tmp/vast_pipeline_${INSTANCE_ID}.log"
echo "📝 Logs do pipeline: $PIPELINE_LOG_FILE"

# Executa o comando via SSH em background e salva o PID
echo "🔄 Iniciando pipeline em background..."
ssh $SSH_OPTS "root@$SSH_HOST" "$PIPELINE_CMD" > "$PIPELINE_LOG_FILE" 2>&1 &
PIPELINE_PID=$!
echo "$PIPELINE_PID" > "/tmp/vast_pipeline_${INSTANCE_ID}.pid"

echo "✅ Pipeline iniciado (PID: $PIPELINE_PID)"
echo "📡 Acompanhe os logs em tempo real abaixo (Ctrl+C para parar o tail):"

# Tail em tempo real do log local gerado pelo SSH
touch "$PIPELINE_LOG_FILE"
TAIL_PID=""
tail -n +1 -F "$PIPELINE_LOG_FILE" &
TAIL_PID=$!

# Garante limpeza do tail ao sair
cleanup() {
  if [ -n "${TAIL_PID:-}" ]; then
    kill "$TAIL_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM EXIT

# Aguarda o término do processo SSH (pipeline remoto)
PIPELINE_EXIT_CODE=1
if [ -n "${PIPELINE_PID:-}" ]; then
  wait "$PIPELINE_PID"
  PIPELINE_EXIT_CODE=$?
fi

# Para o tail e imprime resumo final
cleanup

echo "📋 RESULTADO FINAL (código $PIPELINE_EXIT_CODE):"
echo "--- Últimas 50 linhas do log ---"
tail -n 50 "$PIPELINE_LOG_FILE" || true
echo "---"

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
  echo "✅ Pipeline finalizado com sucesso."
else
  echo "❌ Pipeline finalizado com erro (código $PIPELINE_EXIT_CODE)."
fi

# (monitor removido)

# Mostra apenas o resultado final resumido
if [ -f "$PIPELINE_LOG_FILE" ]; then
    echo "📋 RESULTADO FINAL:"
    tail -n 10 "$PIPELINE_LOG_FILE" | grep -E "SUCCESS|FAILURE|ERROR|CRITICAL" || echo "Pipeline concluído"
fi

# Verifica se o processo ainda está rodando
if [ -n "${PIPELINE_PID:-}" ] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
    echo "🔄 Pipeline ainda em execução. Para parar: kill $PIPELINE_PID"
else
    echo "✅ Pipeline finalizado."
fi

echo "✅ Pipeline finalizado. Logs: $PIPELINE_LOG_FILE"
