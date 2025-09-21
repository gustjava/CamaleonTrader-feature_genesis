#!/bin/bash

# Script para verificar o status do estudo Optuna no servidor remoto

echo "🔍 Verificando status do estudo Optuna no servidor remoto..."

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

# Verificar status do estudo Optuna
ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i ~/.ssh/id_ed25519 root@$SSH_HOST '
echo "🔍 Verificando status do estudo Optuna..."

STUDY_DB_PATH="/workspace/feature_genesis/output/optuna_stage/study_a.sqlite"

if [ ! -f "$STUDY_DB_PATH" ]; then
    echo "❌ Banco de dados do estudo não encontrado:"
    echo "   $STUDY_DB_PATH"
    echo "💡 Execute o pipeline primeiro para criar o estudo."
    exit 1
fi

echo "✅ Banco de dados encontrado: $STUDY_DB_PATH"

# Verificar se sqlite3 está disponível
if ! command -v sqlite3 &> /dev/null; then
    echo "❌ sqlite3 não está disponível no servidor remoto"
    exit 1
fi

# Verificar estudos disponíveis
echo "📊 Estudos disponíveis:"
sqlite3 "$STUDY_DB_PATH" "SELECT study_id, study_name FROM studies;"

# Para cada estudo, verificar trials
echo ""
echo "📈 Status dos trials:"
sqlite3 "$STUDY_DB_PATH" "
SELECT 
    s.study_name,
    CASE t.state 
        WHEN 0 THEN '\''RUNNING'\''
        WHEN 1 THEN '\''COMPLETE'\''
        WHEN 2 THEN '\''PRUNED'\''
        WHEN 3 THEN '\''FAIL'\''
        ELSE '\''UNKNOWN('\'' || t.state || '\'')'\''
    END as state_name,
    COUNT(*) as count
FROM studies s
LEFT JOIN trials t ON s.study_id = t.study_id
GROUP BY s.study_id, s.study_name, t.state
ORDER BY s.study_id, t.state;
"

echo ""
echo "🏆 Últimos 5 trials completos:"
sqlite3 "$STUDY_DB_PATH" "
SELECT 
    s.study_name,
    t.number,
    t.datetime_complete
FROM studies s
JOIN trials t ON s.study_id = t.study_id
WHERE t.state = 1
ORDER BY t.number DESC
LIMIT 5;
"

echo ""
echo "🥇 Melhor trial por estudo (por número):"
sqlite3 "$STUDY_DB_PATH" "
SELECT 
    s.study_name,
    MAX(t.number) as latest_trial_number
FROM studies s
JOIN trials t ON s.study_id = t.study_id
WHERE t.state = 1
GROUP BY s.study_id;
"

echo ""
echo "📊 Resumo por estudo:"
sqlite3 "$STUDY_DB_PATH" "
SELECT 
    s.study_name,
    COUNT(t.trial_id) as total_trials,
    SUM(CASE WHEN t.state = 1 THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN t.state = 2 THEN 1 ELSE 0 END) as pruned,
    SUM(CASE WHEN t.state = 3 THEN 1 ELSE 0 END) as failed,
    MAX(t.number) as latest_trial_number
FROM studies s
LEFT JOIN trials t ON s.study_id = t.study_id
GROUP BY s.study_id, s.study_name;
"
'

echo "🎯 Verificação concluída!"
