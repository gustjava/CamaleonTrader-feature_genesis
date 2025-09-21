#!/bin/bash

# Script para verificar se as portas necessárias estão disponíveis
# antes de iniciar o pipeline Dask

echo "🔍 Verificando disponibilidade de portas..."

PORTS=(8888 8889 8890)
ALL_PORTS_FREE=true

for port in "${PORTS[@]}"; do
    if lsof -i:$port >/dev/null 2>&1; then
        echo "❌ Porta $port está em uso:"
        lsof -i:$port
        ALL_PORTS_FREE=false
    else
        echo "✅ Porta $port está livre"
    fi
done

if [ "$ALL_PORTS_FREE" = true ]; then
    echo ""
    echo "🎯 Todas as portas estão livres! Pronto para iniciar o pipeline."
    exit 0
else
    echo ""
    echo "⚠️  Algumas portas estão em uso. Execute o script de limpeza:"
    echo "   ./kill_pipeline_auto.sh"
    exit 1
fi
