#!/usr/bin/env bash
set -euo pipefail

# ====================================================================================
# SCRIPT PARA VERIFICAR ARQUIVOS DE SAÍDA DO PIPELINE
# 
# Este script deve ser executado diretamente no servidor remoto para verificar
# se os arquivos de dados foram criados corretamente pelo pipeline.
# ====================================================================================

echo "=== VERIFICAÇÃO DE ARQUIVOS DE SAÍDA DO PIPELINE ==="
echo "Data/Hora: $(date)"
echo ""

# Configurações
PROJECT_DIR="/workspace/feature_genesis"
OUTPUT_DIR="$PROJECT_DIR/output"
DATA_DIR="/data"

echo "1. VERIFICANDO DIRETÓRIO DE SAÍDA..."
if [ -d "$OUTPUT_DIR" ]; then
    echo "✅ Diretório de saída existe: $OUTPUT_DIR"
    echo "Conteúdo:"
    ls -la "$OUTPUT_DIR"
    
    # Contar arquivos por tipo
    feather_count=$(find "$OUTPUT_DIR" -name "*.feather" -type f 2>/dev/null | wc -l)
    parquet_count=$(find "$OUTPUT_DIR" -name "*.parquet" -type f 2>/dev/null | wc -l)
    
    echo ""
    echo "📊 Estatísticas:"
    echo "   Arquivos .feather: $feather_count"
    echo "   Arquivos .parquet: $parquet_count"
    
    if [ $feather_count -gt 0 ]; then
        echo ""
        echo "📁 Arquivos .feather encontrados:"
        find "$OUTPUT_DIR" -name "*.feather" -type f 2>/dev/null | head -10
    fi
else
    echo "❌ Diretório de saída NÃO existe: $OUTPUT_DIR"
    echo "   Verificando se existe em outros locais..."
    find /workspace -name "output" -type d 2>/dev/null | head -5 || echo "   Nenhum diretório 'output' encontrado"
    find /data -name "output" -type d 2>/dev/null | head -5 || echo "   Nenhum diretório 'output' encontrado em /data"
fi
echo ""

echo "2. PROCURANDO ARQUIVOS FEATHER EM TODO O SISTEMA..."
echo "Procurando em $PROJECT_DIR..."
find "$PROJECT_DIR" -name "*.feather" -type f 2>/dev/null | head -20 || echo "   Nenhum arquivo .feather encontrado"

echo "Procurando em $DATA_DIR..."
find "$DATA_DIR" -name "*.feather" -type f 2>/dev/null | head -20 || echo "   Nenhum arquivo .feather encontrado"

echo "Procurando em todo o sistema (relevantes)..."
find / -name "*.feather" -type f 2>/dev/null | grep -E "(feature_genesis|output|data)" | head -20 || echo "   Nenhum arquivo .feather relevante encontrado"
echo ""

echo "3. VERIFICANDO PROCESSOS DO PIPELINE..."
echo "Processos Python relacionados ao pipeline:"
ps aux | grep -E "(python.*orchestration|dask-worker|dask-scheduler)" | grep -v grep || echo "   Nenhum processo do pipeline encontrado"
echo ""

echo "4. VERIFICANDO LOGS..."
if [ -d "$PROJECT_DIR/logs" ]; then
    echo "Logs disponíveis em $PROJECT_DIR/logs/:"
    ls -la "$PROJECT_DIR/logs/" | head -10
    
    # Mostrar últimas linhas do log principal
    if [ -f "$PROJECT_DIR/logs/pipeline_execution.log" ]; then
        echo ""
        echo "Últimas 20 linhas do log principal:"
        tail -n 20 "$PROJECT_DIR/logs/pipeline_execution.log"
    fi
else
    echo "❌ Diretório de logs não existe: $PROJECT_DIR/logs"
fi
echo ""

echo "5. VERIFICANDO CONFIGURAÇÃO..."
if [ -f "$PROJECT_DIR/config/config.yaml" ]; then
    echo "Configuração de saída:"
    grep -A 5 -B 2 "output:" "$PROJECT_DIR/config/config.yaml" || echo "   Seção output não encontrada"
else
    echo "❌ Arquivo de configuração não encontrado"
fi
echo ""

echo "6. VERIFICANDO RECURSOS DO SISTEMA..."
echo "Uso de disco:"
df -h | grep -E "(Filesystem|/workspace|/data)" || df -h | head -5
echo ""
echo "Uso de memória:"
free -h
echo ""

echo "7. VERIFICANDO ARQUIVOS MODIFICADOS RECENTEMENTE..."
echo "Arquivos modificados nas últimas 24h:"
find "$PROJECT_DIR" -type f -mtime -1 2>/dev/null | head -10 || echo "   Nenhum arquivo modificado recentemente"
echo ""

echo "=== RESUMO FINAL ==="
echo "Diretório de saída configurado: $OUTPUT_DIR"
echo "Diretório existe: $([ -d "$OUTPUT_DIR" ] && echo 'SIM' || echo 'NÃO')"
echo "Arquivos .feather encontrados: $(find "$PROJECT_DIR" -name "*.feather" 2>/dev/null | wc -l)"
echo "Processos do pipeline ativos: $(ps aux | grep -E "(python.*orchestration|dask-worker|dask-scheduler)" | grep -v grep | wc -l)"
echo "Espaço livre em disco: $(df -h /workspace | tail -1 | awk '{print $4}')"
echo "Memória livre: $(free -h | grep Mem | awk '{print $7}')"
echo ""
echo "=== FIM DA VERIFICAÇÃO ==="
