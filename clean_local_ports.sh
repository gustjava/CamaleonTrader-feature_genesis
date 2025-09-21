#!/bin/bash

# Script para limpar portas locais antes de executar o pipeline
# Evita conflitos com túneis SSH ou outros processos locais

echo "🧹 Limpando portas locais..."

PORTS=(8888 8889 8890)
CLEANED=false

for port in "${PORTS[@]}"; do
    PIDS=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "🔪 Matando processos na porta $port: $PIDS"
        echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
        CLEANED=true
    else
        echo "✅ Porta $port já está livre"
    fi
done

if [ "$CLEANED" = true ]; then
    echo "⏳ Aguardando portas ficarem livres..."
    sleep 3
    
    echo "🔍 Verificação final:"
    for port in "${PORTS[@]}"; do
        if lsof -i:$port >/dev/null 2>&1; then
            echo "❌ Porta $port ainda em uso:"
            lsof -i:$port
        else
            echo "✅ Porta $port livre"
        fi
    done
else
    echo "✅ Todas as portas já estavam livres!"
fi

echo "🎯 Limpeza local concluída!"
