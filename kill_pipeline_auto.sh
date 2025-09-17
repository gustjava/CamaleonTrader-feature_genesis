#!/bin/bash

# Configurações
MAX_RETRIES=3
RETRY_DELAY=2
FORCE_KILL=${1:-false}  # Aceita --force como parâmetro
VERBOSE=${2:-false}     # Aceita --verbose como parâmetro

# Função de logging
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%H:%M:%S')
    
    case "$level" in
        "INFO")  echo "[$timestamp] ℹ️  $message" ;;
        "WARN")  echo "[$timestamp] ⚠️  $message" ;;
        "ERROR") echo "[$timestamp] ❌ $message" ;;
        "SUCCESS") echo "[$timestamp] ✅ $message" ;;
        "DEBUG") 
            if [[ "$VERBOSE" == "true" ]]; then
                echo "[$timestamp] 🔍 $message"
            fi
            ;;
    esac
}

# Função para executar comando com retry
execute_with_retry() {
    local cmd="$1"
    local description="$2"
    local max_attempts=${3:-$MAX_RETRIES}
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "DEBUG" "Tentativa $attempt/$max_attempts: $description"
        
        if eval "$cmd"; then
            log "SUCCESS" "$description - Sucesso"
            return 0
        else
            log "WARN" "$description - Falhou (tentativa $attempt/$max_attempts)"
            if [[ $attempt -lt $max_attempts ]]; then
                log "INFO" "Aguardando ${RETRY_DELAY}s antes da próxima tentativa..."
                sleep $RETRY_DELAY
            fi
        fi
        
        ((attempt++))
    done
    
    log "ERROR" "$description - Falhou após $max_attempts tentativas"
    return 1
}

# Função para verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para conectar SSH com retry
ssh_connect() {
    local ssh_cmd="$1"
    local description="$2"
    local max_attempts=3
    local ignore_exit_code=${3:-false}  # Novo parâmetro para ignorar exit codes
    
    for attempt in $(seq 1 $max_attempts); do
        log "DEBUG" "Tentativa SSH $attempt/$max_attempts: $description"
        
        local ssh_output
        local ssh_exit_code
        
        # Executar comando SSH e capturar output e exit code
        ssh_output=$(ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST "$ssh_cmd" 2>&1)
        ssh_exit_code=$?
        
        # Se ignore_exit_code=true, sempre considerar sucesso se a conexão SSH funcionou
        if [[ "$ignore_exit_code" == "true" ]]; then
            log "SUCCESS" "SSH: $description - Sucesso (ignorando exit code: $ssh_exit_code)"
            return 0
        fi
        
        # Verificar se é erro de conexão SSH (exit codes 255, 1) vs comando executado com sucesso
        if [[ $ssh_exit_code -eq 0 ]]; then
            log "SUCCESS" "SSH: $description - Sucesso"
            return 0
        elif [[ $ssh_exit_code -eq 255 ]]; then
            log "WARN" "SSH: $description - Erro de conexão SSH (tentativa $attempt/$max_attempts)"
            if [[ $attempt -lt $max_attempts ]]; then
                log "INFO" "Aguardando ${RETRY_DELAY}s antes da próxima tentativa SSH..."
                sleep $RETRY_DELAY
            fi
        else
            # Exit code diferente de 0 e 255 - comando executado mas falhou
            log "WARN" "SSH: $description - Comando falhou com exit code $ssh_exit_code (tentativa $attempt/$max_attempts)"
            if [[ $attempt -lt $max_attempts ]]; then
                log "INFO" "Aguardando ${RETRY_DELAY}s antes da próxima tentativa SSH..."
                sleep $RETRY_DELAY
            fi
        fi
    done
    
    log "ERROR" "SSH: $description - Falhou após $max_attempts tentativas"
    return 1
}

# Verificar dependências
log "INFO" "Verificando dependências..."
if ! command_exists jq; then
    log "ERROR" "jq não encontrado. Instale com: apt-get install jq"
    exit 1
fi

# Identificar instância ativa
log "INFO" "Identificando instância ativa..."
VAST_BIN=''
if command_exists vastai; then
    VAST_BIN=$(command -v vastai)
elif command_exists vast; then
    VAST_BIN=$(command -v vast)
else
    log "ERROR" "CLI da vast.ai não encontrado"
    exit 1
fi

log "DEBUG" "Usando CLI: $VAST_BIN"

# Obter instâncias com retry
if ! execute_with_retry "$VAST_BIN show instances --raw" "Obter lista de instâncias"; then
    log "ERROR" "Falha ao obter lista de instâncias"
    exit 1
fi

INSTANCES_RAW="$($VAST_BIN show instances --raw)"
INSTANCE_ID=$(echo "$INSTANCES_RAW" | jq -r '[.[] | select(.actual_status=="running")][0].id // empty')

if [[ -z "$INSTANCE_ID" ]]; then
    log "ERROR" "Nenhuma instância ativa encontrada"
    exit 1
fi

log "SUCCESS" "Instância encontrada: $INSTANCE_ID"

# Extrair informações SSH
INSTANCE_INFO=$(echo "$INSTANCES_RAW" | jq -r ".[] | select(.id == $INSTANCE_ID)")
SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.ssh_host // empty')
SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ssh_port // empty')

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    log "ERROR" "Não foi possível obter informações SSH"
    exit 1
fi

log "INFO" "Conectando em: $SSH_HOST:$SSH_PORT"

# Testar conexão SSH
log "INFO" "Testando conexão SSH..."
if ! ssh_connect "echo 'Conexão SSH OK'" "Teste de conectividade"; then
    log "ERROR" "Falha na conexão SSH"
    exit 1
fi

# Função para kill de processos com diferentes estratégias
kill_processes() {
    local pattern="$1"
    local description="$2"
    local force="$3"
    
    log "INFO" "Matando $description..."
    
    # Verificar se há processos antes de tentar matar
    local has_processes=$(ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST "ps aux | grep '$pattern' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    
    if [[ "$has_processes" -gt 0 ]]; then
        log "DEBUG" "Encontrados $has_processes processos para $description"
        
        # Estratégia 1: SIGTERM (graceful)
        if ! ssh_connect "pkill -f '$pattern' 2>/dev/null || true" "SIGTERM para $description" "true"; then
            log "WARN" "SIGTERM falhou para $description"
        fi
        
        # Aguardar um pouco
        sleep 1
        
        # Estratégia 2: SIGKILL (forçado)
        if ! ssh_connect "pkill -9 -f '$pattern' 2>/dev/null || true" "SIGKILL para $description" "true"; then
            log "WARN" "SIGKILL falhou para $description"
        fi
    else
        log "DEBUG" "Nenhum processo encontrado para $description - pulando kill"
    fi
    
    # Estratégia 3: Kill por PID específico (se force=true)
    if [[ "$force" == "true" ]]; then
        log "INFO" "Modo FORÇADO: Matando $description por PID..."
        ssh_connect "
            ps aux | grep '$pattern' | grep -v grep | awk '{print \$2}' | while read pid; do
                if [[ -n \"\$pid\" ]]; then
                    kill -9 \$pid 2>/dev/null || true
                    echo \"Matado PID: \$pid\"
                fi
            done
        " "Kill forçado por PID para $description"
    fi
}

# Função para liberar portas
free_ports() {
    local ports=("$@")
    local description="$1"
    
    log "INFO" "Liberando portas: ${ports[*]}..."
    
    for port in "${ports[@]}"; do
        ssh_connect "
            pids=\$(lsof -ti:$port 2>/dev/null)
            if [[ -n \"\$pids\" ]]; then
                echo \"Matando processos na porta $port: \$pids\"
                echo \"\$pids\" | xargs -r kill -9 2>/dev/null || true
            else
                echo \"Porta $port já está livre\"
            fi
        " "Liberar porta $port" "true"
    done
}

# Função para verificar processos
check_processes() {
    local pattern="$1"
    local description="$2"
    
    ssh_connect "
        processes=\$(ps aux | grep '$pattern' | grep -v grep)
        if [[ -n \"\$processes\" ]]; then
            echo \"❌ $description ainda ativos:\"
            echo \"\$processes\"
            return 1
        else
            echo \"✅ $description - Nenhum processo encontrado\"
            return 0
        fi
    " "Verificar $description" "true"
}

# Função para verificar portas
check_ports() {
    local port="$1"
    local description="$2"
    
    ssh_connect "
        usage=\$(lsof -i:$port 2>/dev/null)
        if [[ -n \"\$usage\" ]]; then
            echo \"❌ $description ainda em uso:\"
            echo \"\$usage\"
            return 1
        else
            echo \"✅ $description - Porta livre\"
            return 0
        fi
    " "Verificar porta $port" "true"
}

# Executar limpeza principal
log "INFO" "Iniciando limpeza completa do pipeline..."

# Matar processos principais
kill_processes "orchestration/main.py" "Processos do pipeline principal" "$FORCE_KILL"
kill_processes "python.*main.py" "Processos Python main.py" "$FORCE_KILL"
kill_processes "dask" "Processos Dask" "$FORCE_KILL"
kill_processes "distributed" "Processos Distributed" "$FORCE_KILL"
kill_processes "cudf" "Processos CuDF/Rapids" "$FORCE_KILL"
kill_processes "catboost" "Processos CatBoost" "$FORCE_KILL"

# Liberar portas principais
free_ports "8888" "8889" "8890" "8786" "8787"

# Limpeza de memória GPU
log "INFO" "Limpando memória GPU..."
ssh_connect "
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true
        echo 'Reset GPU executado'
    else
        echo 'nvidia-smi não disponível'
    fi
" "Limpeza de memória GPU"

# Limpeza de arquivos temporários
log "INFO" "Limpando arquivos temporários..."
ssh_connect "
    temp_files=\$(find /tmp -name '*dask*' -o -name '*cudf*' -o -name '*pipeline*' 2>/dev/null | wc -l)
    if [[ \$temp_files -gt 0 ]]; then
        echo \"Removendo \$temp_files arquivos temporários...\"
        find /tmp -name '*dask*' -o -name '*cudf*' -o -name '*pipeline*' -delete 2>/dev/null || true
        echo 'Arquivos temporários removidos'
    else
        echo 'Nenhum arquivo temporário encontrado'
    fi
" "Limpeza de arquivos temporários"

# Aguardar estabilização
log "INFO" "Aguardando estabilização do sistema..."
sleep 3

# Verificação completa
log "INFO" "Executando verificação completa..."

ssh_connect '
echo "=========================================="
echo "🔍 VERIFICAÇÃO COMPLETA DE LIMPEZA"
echo "=========================================="

# Verificações de processos
echo "📊 Verificando processos..."
failed_checks=0

if ! ps aux | grep "orchestration/main.py" | grep -v grep >/dev/null; then
    echo "✅ Processos do pipeline principal - Limpos"
else
    echo "❌ Processos do pipeline principal - Ainda ativos"
    failed_checks=$((failed_checks + 1))
fi

if ! ps aux | grep "python.*main.py" | grep -v grep >/dev/null; then
    echo "✅ Processos Python main.py - Limpos"
else
    echo "❌ Processos Python main.py - Ainda ativos"
    failed_checks=$((failed_checks + 1))
fi

if ! ps aux | grep "dask" | grep -v grep >/dev/null; then
    echo "✅ Processos Dask - Limpos"
else
    echo "❌ Processos Dask - Ainda ativos"
    failed_checks=$((failed_checks + 1))
fi

if ! ps aux | grep "distributed" | grep -v grep >/dev/null; then
    echo "✅ Processos Distributed - Limpos"
else
    echo "❌ Processos Distributed - Ainda ativos"
    failed_checks=$((failed_checks + 1))
fi

if ! ps aux | grep "cudf" | grep -v grep >/dev/null; then
    echo "✅ Processos CuDF/Rapids - Limpos"
else
    echo "❌ Processos CuDF/Rapids - Ainda ativos"
    failed_checks=$((failed_checks + 1))
fi

if ! ps aux | grep "catboost" | grep -v grep >/dev/null; then
    echo "✅ Processos CatBoost - Limpos"
else
    echo "❌ Processos CatBoost - Ainda ativos"
    failed_checks=$((failed_checks + 1))
fi

# Verificações de portas
echo ""
echo "📊 Verificando portas..."
port_failed_checks=0

for port in 8888 8889 8890 8786 8787; do
    if ! lsof -i:$port >/dev/null 2>&1; then
        echo "✅ Porta $port - Livre"
    else
        echo "❌ Porta $port - Ainda em uso"
        port_failed_checks=$((port_failed_checks + 1))
    fi
done

# Verificação de memória GPU
echo ""
echo "📊 Verificando memória GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "$gpu_memory" ]]; then
        if [[ "$gpu_memory" -gt 100 ]]; then
            echo "⚠️  Memória GPU ainda em uso: ${gpu_memory}MB"
        else
            echo "✅ Memória GPU liberada: ${gpu_memory}MB"
        fi
    else
        echo "✅ GPU não detectada ou limpa"
    fi
else
    echo "ℹ️  nvidia-smi não disponível"
fi

# Resumo final
echo ""
echo "=========================================="
echo "📊 RESUMO FINAL DA LIMPEZA"
echo "=========================================="

remaining_processes=$(ps aux | grep -E "(orchestration/main.py|python.*main.py|dask|distributed|cudf|catboost)" | grep -v grep | wc -l)
remaining_ports=0

for port in 8888 8889 8890 8786 8787; do
    if lsof -i:$port >/dev/null 2>&1; then
        remaining_ports=$((remaining_ports + 1))
    fi
done

echo "📈 Estatísticas finais:"
echo "   • Processos restantes: $remaining_processes"
echo "   • Portas ainda em uso: $remaining_ports"
echo "   • Memória GPU: ${gpu_memory:-N/A}MB"

if [[ $remaining_processes -eq 0 && $remaining_ports -eq 0 ]]; then
    echo ""
    echo "🎉 SUCESSO TOTAL! Todos os serviços foram limpos com sucesso!"
    echo "✅ Sistema pronto para nova execução do pipeline"
    exit 0
else
    echo ""
    echo "⚠️  ATENÇÃO: Ainda há recursos em uso"
    if [[ $remaining_processes -gt 0 ]]; then
        echo "   • $remaining_processes processos ainda ativos"
    fi
    if [[ $remaining_ports -gt 0 ]]; then
        echo "   • $remaining_ports portas ainda em uso"
    fi
    echo "💡 Considere usar --force ou reiniciar a instância"
    exit 1
fi
' "Verificação completa de limpeza"

# Verificar resultado da verificação
if [[ $? -eq 0 ]]; then
    log "SUCCESS" "Limpeza completa realizada com sucesso!"
else
    log "WARN" "Limpeza realizada, mas alguns recursos ainda podem estar em uso"
    if [[ "$FORCE_KILL" == "false" ]]; then
        log "INFO" "Dica: Use --force para tentar uma limpeza mais agressiva"
    fi
fi

log "INFO" "Comando concluído!"