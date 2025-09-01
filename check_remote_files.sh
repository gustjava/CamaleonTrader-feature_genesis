#!/usr/bin/env bash
set -euo pipefail

# ====================================================================================
# SCRIPT PARA VERIFICAR ARQUIVOS DE DADOS NO SERVIDOR REMOTO
#
# Este script verifica se os arquivos de saída do pipeline foram criados
# corretamente no servidor remoto e mostra informações sobre eles.
# ====================================================================================

# ----------------------------- CONFIGURAÇÕES GERAIS ---------------------------------
# Diretórios do projeto
LOCAL_PROJECT_DIR="$(pwd)"
REMOTE_PROJECT_DIR="/workspace/feature_genesis"
REMOTE_OUTPUT_DIR="/workspace/feature_genesis/output"

# SSH
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"

# -------------------------- FUNÇÕES AUXILIARES/VERIFICAÇÕES -------------------------
need_cmd() { command -v "$1" &>/dev/null || { echo "Erro: '$1' não encontrado. Por favor, instale-o."; exit 1; }; }

echo "Verificando dependências: jq, ssh, nc..."
need_cmd jq
need_cmd ssh
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

# Seleciona a primeira instância em execução por padrão
FIRST_INSTANCE_ID="$(echo "$INSTANCES_RAW" | jq -r '[.[] | select(.actual_status=="running")][0].id // empty')"
if [[ -z "$FIRST_INSTANCE_ID" ]]; then
  echo "❌ Nenhuma instância em execução encontrada."
  exit 1
fi

if [[ -n "${AUTO:-}" || ! -t 0 ]]; then
  INSTANCE_ID="${FIRST_INSTANCE_ID}"
  echo "AUTO=1 ou stdin não interativo: usando a primeira instância ($INSTANCE_ID)"
else
  read -t 5 -rp "Digite o ID da instância vast.ai (Enter para usar a primeira: $FIRST_INSTANCE_ID): " INSTANCE_ID || true
fi
INSTANCE_ID="${INSTANCE_ID:-$FIRST_INSTANCE_ID}"
[[ -z "${INSTANCE_ID}" ]] && { echo "Erro: ID da instância vazio."; exit 1; }

echo "✅ Instância selecionada: $INSTANCE_ID"

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

# --- CONFIGURAÇÃO SSH ---
SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i $SSH_KEY_PATH"

# --- COMANDO DE VERIFICAÇÃO REMOTA ---
echo -e "\n🔍 Verificando arquivos de saída no servidor remoto..."

VERIFICATION_CMD="
set -e
echo '=== VERIFICAÇÃO DE ARQUIVOS DE SAÍDA ==='
echo 'Data/Hora: '\$(date)
echo ''

# 1. Verificar se o diretório de saída existe
echo '1. VERIFICANDO DIRETÓRIO DE SAÍDA...'
if [ -d \"$REMOTE_OUTPUT_DIR\" ]; then
    echo \"✅ Diretório de saída existe: $REMOTE_OUTPUT_DIR\"
    ls -la \"$REMOTE_OUTPUT_DIR\"
else
    echo \"❌ Diretório de saída NÃO existe: $REMOTE_OUTPUT_DIR\"
    echo \"   Verificando se existe em outros locais...\"
    find /workspace -name \"output\" -type d 2>/dev/null | head -5 || echo \"   Nenhum diretório 'output' encontrado\"
    find /data -name \"output\" -type d 2>/dev/null | head -5 || echo \"   Nenhum diretório 'output' encontrado em /data\"
fi
echo ''

# 2. Procurar por arquivos Feather em todo o workspace
echo '2. PROCURANDO ARQUIVOS FEATHER...'
echo \"Procurando em $REMOTE_PROJECT_DIR...\"
find \"$REMOTE_PROJECT_DIR\" -name \"*.feather\" -type f 2>/dev/null | head -20 || echo \"   Nenhum arquivo .feather encontrado\"

echo \"Procurando em /data...\"
find /data -name \"*.feather\" -type f 2>/dev/null | head -20 || echo \"   Nenhum arquivo .feather encontrado\"

echo \"Procurando em todo o sistema...\"
find / -name \"*.feather\" -type f 2>/dev/null | grep -E '(feature_genesis|output|data)' | head -20 || echo \"   Nenhum arquivo .feather relevante encontrado\"
echo ''

# 3. Verificar estrutura de diretórios do projeto
echo '3. ESTRUTURA DO PROJETO...'
if [ -d \"$REMOTE_PROJECT_DIR\" ]; then
    echo \"Estrutura de $REMOTE_PROJECT_DIR:\"
    ls -la \"$REMOTE_PROJECT_DIR\" | head -20
    echo ''
    
    # Verificar se há logs
    if [ -d \"$REMOTE_PROJECT_DIR/logs\" ]; then
        echo \"Logs disponíveis:\"
        ls -la \"$REMOTE_PROJECT_DIR/logs/\" | head -10
    fi
else
    echo \"❌ Diretório do projeto não existe: $REMOTE_PROJECT_DIR\"
fi
echo ''

# 4. Verificar processos em execução
echo '4. PROCESSOS EM EXECUÇÃO...'
echo \"Processos Python relacionados ao pipeline:\"
ps aux | grep -E '(python.*orchestration|dask-worker|dask-scheduler)' | grep -v grep || echo \"   Nenhum processo do pipeline encontrado\"
echo ''

# 5. Verificar uso de disco e memória
echo '5. RECURSOS DO SISTEMA...'
echo \"Uso de disco:\"
df -h | grep -E '(Filesystem|/workspace|/data)' || df -h | head -5
echo ''
echo \"Uso de memória:\"
free -h
echo ''

# 6. Verificar se há arquivos temporários ou parciais
echo '6. ARQUIVOS TEMPORÁRIOS OU PARCIAIS...'
echo \"Arquivos .tmp ou .part:\"
find \"$REMOTE_PROJECT_DIR\" -name \"*.tmp\" -o -name \"*.part\" 2>/dev/null | head -10 || echo \"   Nenhum arquivo temporário encontrado\"
echo ''

# 7. Verificar logs do sistema
echo '7. LOGS DO SISTEMA...'
echo \"Últimas linhas do syslog (se disponível):\"
tail -n 20 /var/log/syslog 2>/dev/null || echo \"   Syslog não disponível\"
echo ''

# 8. Verificar se o pipeline foi executado recentemente
echo '8. HISTÓRICO DE EXECUÇÃO...'
echo \"Arquivos modificados nas últimas 24h:\"
find \"$REMOTE_PROJECT_DIR\" -type f -mtime -1 2>/dev/null | head -10 || echo \"   Nenhum arquivo modificado recentemente\"
echo ''

# 9. Verificar configuração do projeto
echo '9. CONFIGURAÇÃO DO PROJETO...'
if [ -f \"$REMOTE_PROJECT_DIR/config/config.yaml\" ]; then
    echo \"Configuração de saída:\"
    grep -A 5 -B 2 \"output:\" \"$REMOTE_PROJECT_DIR/config/config.yaml\" || echo \"   Seção output não encontrada\"
else
    echo \"❌ Arquivo de configuração não encontrado\"
fi
echo ''

# 10. Resumo final
echo '=== RESUMO FINAL ==='
echo \"Diretório de saída configurado: $REMOTE_OUTPUT_DIR\"
echo \"Diretório existe: \$([ -d \"$REMOTE_OUTPUT_DIR\" ] && echo 'SIM' || echo 'NÃO')\"
echo \"Arquivos .feather encontrados: \$(find \"$REMOTE_PROJECT_DIR\" -name \"*.feather\" 2>/dev/null | wc -l)\"
echo \"Processos do pipeline ativos: \$(ps aux | grep -E '(python.*orchestration|dask-worker|dask-scheduler)' | grep -v grep | wc -l)\"
echo \"Espaço livre em disco: \$(df -h /workspace | tail -1 | awk '{print \$4}')\"
echo \"Memória livre: \$(free -h | grep Mem | awk '{print \$7}')\"
echo ''
echo '=== FIM DA VERIFICAÇÃO ==='
"

# Executa o comando de verificação
echo "Executando verificação remota..."
ssh $SSH_OPTS "root@$SSH_HOST" "$VERIFICATION_CMD"

echo -e "\n✅ Verificação concluída!"
echo -e "\n💡 DICAS:"
echo "   • Se não houver arquivos de saída, verifique se o pipeline foi executado"
echo "   • Verifique os logs em $REMOTE_PROJECT_DIR/logs/"
echo "   • Execute o pipeline novamente se necessário: ./run_pipeline_vast.sh"
echo "   • Use o script de validação local: python validate_output.py"
