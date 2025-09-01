#!/usr/bin/env bash
set -euo pipefail

# ====================================================================================
# SCRIPT COMPLETO PARA VERIFICAR ARQUIVOS DE SAÍDA DO PIPELINE
#
# Este script:
# 1. Conecta ao servidor remoto
# 2. Verifica se os arquivos de saída existem
# 3. Executa validação detalhada dos arquivos
# 4. Gera relatório completo
# ====================================================================================

# ----------------------------- CONFIGURAÇÕES GERAIS ---------------------------------
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

# --- SINCRONIZAÇÃO DE SCRIPTS DE VALIDAÇÃO ---
echo -e "\n🔄 Sincronizando scripts de validação..."
rsync -avz -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/check_output_files.sh" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
rsync -avz -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/validate_remote_output.py" "root@$SSH_HOST:$REMOTE_PROJECT_DIR/"
echo "✅ Scripts sincronizados."

# --- COMANDO DE VERIFICAÇÃO COMPLETA ---
echo -e "\n🔍 Executando verificação completa dos arquivos de saída..."

VERIFICATION_CMD="
set -e
echo '=== VERIFICAÇÃO COMPLETA DOS ARQUIVOS DE SAÍDA ==='
echo 'Data/Hora: '\$(date)
echo ''

# 1. Verificação básica com script bash
echo '1. VERIFICAÇÃO BÁSICA...'
if [ -f \"$REMOTE_PROJECT_DIR/check_output_files.sh\" ]; then
    chmod +x \"$REMOTE_PROJECT_DIR/check_output_files.sh\"
    \"$REMOTE_PROJECT_DIR/check_output_files.sh\"
else
    echo \"❌ Script de verificação básica não encontrado\"
fi
echo ''

# 2. Verificação detalhada com Python
echo '2. VERIFICAÇÃO DETALHADA COM PYTHON...'
cd \"$REMOTE_PROJECT_DIR\"

# Verificar se o ambiente conda está disponível
if command -v conda &> /dev/null; then
    echo 'Ativando ambiente conda...'
    source /opt/conda/etc/profile.d/conda.sh
    
    if conda env list | grep -q 'dynamic-stage0'; then
        conda activate dynamic-stage0
        echo '✅ Ambiente dynamic-stage0 ativado'
    else
        echo '⚠️  Usando ambiente base'
    fi
else
    echo '⚠️  Conda não encontrado, usando Python do sistema'
fi

# Verificar se pandas está disponível
if python -c \"import pandas\" 2>/dev/null; then
    echo '✅ Pandas disponível'
    
    if [ -f \"$REMOTE_PROJECT_DIR/validate_remote_output.py\" ]; then
        chmod +x \"$REMOTE_PROJECT_DIR/validate_remote_output.py\"
        echo 'Executando validação Python...'
        python \"$REMOTE_PROJECT_DIR/validate_remote_output.py\"
    else
        echo \"❌ Script de validação Python não encontrado\"
    fi
else
    echo '❌ Pandas não disponível, instalando...'
    pip install pandas pyarrow numpy || echo '⚠️  Falha na instalação do pandas'
fi
echo ''

# 3. Verificação manual adicional
echo '3. VERIFICAÇÃO MANUAL ADICIONAL...'

# Verificar se há arquivos de saída
if [ -d \"$REMOTE_OUTPUT_DIR\" ]; then
    echo \"Diretório de saída encontrado: $REMOTE_OUTPUT_DIR\"
    
    # Contar arquivos por tipo
    feather_count=\$(find \"$REMOTE_OUTPUT_DIR\" -name \"*.feather\" -type f 2>/dev/null | wc -l)
    parquet_count=\$(find \"$REMOTE_OUTPUT_DIR\" -name \"*.parquet\" -type f 2>/dev/null | wc -l)
    
    echo \"Arquivos .feather: \$feather_count\"
    echo \"Arquivos .parquet: \$parquet_count\"
    
    if [ \$feather_count -gt 0 ]; then
        echo \"Primeiros arquivos .feather:\"
        find \"$REMOTE_OUTPUT_DIR\" -name \"*.feather\" -type f 2>/dev/null | head -5
        
        # Verificar tamanho dos arquivos
        echo \"Tamanhos dos arquivos:\"
        for file in \$(find \"$REMOTE_OUTPUT_DIR\" -name \"*.feather\" -type f 2>/dev/null | head -5); do
            size_mb=\$(du -h \"\$file\" | cut -f1)
            echo \"  \$(basename \$file): \$size_mb\"
        done
    fi
else
    echo \"❌ Diretório de saída não encontrado: $REMOTE_OUTPUT_DIR\"
fi
echo ''

# 4. Verificar logs do pipeline
echo '4. VERIFICANDO LOGS DO PIPELINE...'
if [ -d \"$REMOTE_PROJECT_DIR/logs\" ]; then
    echo \"Logs disponíveis:\"
    ls -la \"$REMOTE_PROJECT_DIR/logs/\" | head -10
    
    # Verificar último log
    latest_log=\$(ls -t \"$REMOTE_PROJECT_DIR/logs/\"*.log 2>/dev/null | head -1)
    if [ -n \"\$latest_log\" ]; then
        echo \"Último log: \$latest_log\"
        echo \"Últimas 10 linhas:\"
        tail -n 10 \"\$latest_log\" 2>/dev/null || echo \"   Erro ao ler log\"
    fi
else
    echo \"❌ Diretório de logs não encontrado\"
fi
echo ''

# 5. Verificar processos em execução
echo '5. VERIFICANDO PROCESSOS...'
pipeline_processes=\$(ps aux | grep -E '(python.*orchestration|dask-worker|dask-scheduler)' | grep -v grep || true)
if [ -n \"\$pipeline_processes\" ]; then
    echo \"Processos do pipeline em execução:\"
    echo \"\$pipeline_processes\"
else
    echo \"Nenhum processo do pipeline em execução\"
fi
echo ''

# 6. Resumo final
echo '=== RESUMO FINAL ==='
echo \"Diretório de saída: $REMOTE_OUTPUT_DIR\"
echo \"Existe: \$([ -d \"$REMOTE_OUTPUT_DIR\" ] && echo 'SIM' || echo 'NÃO')\"
echo \"Arquivos .feather: \$(find \"$REMOTE_OUTPUT_DIR\" -name \"*.feather\" 2>/dev/null | wc -l)\"
echo \"Processos ativos: \$(ps aux | grep -E '(python.*orchestration|dask-worker|dask-scheduler)' | grep -v grep | wc -l)\"
echo \"Espaço livre: \$(df -h /workspace | tail -1 | awk '{print \$4}')\"
echo \"Memória livre: \$(free -h | grep Mem | awk '{print \$7}')\"
echo ''
echo '=== FIM DA VERIFICAÇÃO ==='
"

# Executa o comando de verificação
echo "Executando verificação completa..."
ssh $SSH_OPTS "root@$SSH_HOST" "$VERIFICATION_CMD"

echo -e "\n✅ Verificação completa concluída!"
echo -e "\n💡 PRÓXIMOS PASSOS:"
echo "   • Se arquivos foram encontrados: ✅ Pipeline funcionou!"
echo "   • Se nenhum arquivo foi encontrado: ❌ Execute o pipeline novamente"
echo "   • Para executar o pipeline: ./run_pipeline_vast.sh"
echo "   • Para ver logs detalhados: ssh -p $SSH_PORT -i $SSH_KEY_PATH root@$SSH_HOST 'tail -f $REMOTE_PROJECT_DIR/logs/pipeline_execution.log'"
