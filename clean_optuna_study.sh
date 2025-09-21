#!/bin/bash
# =============================================================================
# Script para limpeza segura de estudos Optuna
# =============================================================================

set -e

PROJECT_DIR="${1:-$(pwd)}"
STUDY_NAME="${2:-study_a}"

echo "🧹 Limpeza de estudo Optuna"
echo "📁 Diretório: $PROJECT_DIR"
echo "🔬 Estudo: $STUDY_NAME"

# Verificar se o diretório existe
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Diretório não encontrado: $PROJECT_DIR"
    exit 1
fi

# Caminhos dos arquivos de estudo
STUDY_DB="$PROJECT_DIR/output/optuna_stage/${STUDY_NAME}.sqlite"
STUDY_DIR="$PROJECT_DIR/output/optuna_stage"

echo ""
echo "🔍 Verificando arquivos de estudo..."

if [ -f "$STUDY_DB" ]; then
    echo "📊 Banco de dados encontrado: $STUDY_DB"
    
    # Mostrar informações do estudo
    echo "📈 Informações do estudo:"
    python3 -c "
import sqlite3
import sys
try:
    conn = sqlite3.connect('$STUDY_DB')
    cursor = conn.cursor()
    
    # Contar trials
    cursor.execute('SELECT COUNT(*) FROM trials')
    total_trials = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trials WHERE state = 1')  # COMPLETE
    completed_trials = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trials WHERE state = 2')  # PRUNED
    pruned_trials = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trials WHERE state = 3')  # FAIL
    failed_trials = cursor.fetchone()[0]
    
    print(f'  Total de trials: {total_trials}')
    print(f'  Trials completados: {completed_trials}')
    print(f'  Trials podados: {pruned_trials}')
    print(f'  Trials falharam: {failed_trials}')
    
    conn.close()
except Exception as e:
    print(f'  Erro ao ler banco: {e}')
    sys.exit(1)
"
    
    echo ""
    read -p "⚠️  Deseja realmente deletar este estudo? (digite 'DELETE' para confirmar): " confirmation
    
    if [ "$confirmation" = "DELETE" ]; then
        echo "🗑️  Removendo banco de dados do estudo..."
        rm -f "$STUDY_DB"
        echo "✅ Banco de dados removido"
    else
        echo "❌ Operação cancelada"
        exit 0
    fi
else
    echo "📊 Nenhum banco de dados encontrado"
fi

# Verificar outros arquivos relacionados
if [ -d "$STUDY_DIR" ]; then
    echo ""
    echo "🔍 Verificando outros arquivos de estudo..."
    
    # Listar arquivos relacionados
    find "$STUDY_DIR" -name "*${STUDY_NAME}*" -o -name "trial_progress_*.json" -o -name "study_snapshot_*.json" | while read file; do
        if [ -f "$file" ]; then
            echo "📄 $file"
        fi
    done
    
    echo ""
    read -p "🗑️  Deseja remover todos os arquivos relacionados ao estudo? (y/N): " remove_files
    
    if [ "$remove_files" = "y" ] || [ "$remove_files" = "Y" ]; then
        echo "🗑️  Removendo arquivos relacionados..."
        find "$STUDY_DIR" -name "*${STUDY_NAME}*" -o -name "trial_progress_*.json" -o -name "study_snapshot_*.json" -delete
        echo "✅ Arquivos relacionados removidos"
    fi
fi

echo ""
echo "✅ Limpeza concluída"
echo "💡 Para iniciar um novo estudo, execute o pipeline novamente"
