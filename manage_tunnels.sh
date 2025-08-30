#!/usr/bin/env bash
set -euo pipefail

# ====================================================================================
# SCRIPT PARA GERENCIAR TÚNEIS SSH PERSISTENTES DA VAST.AI
#
# Uso:
#   ./manage_tunnels.sh list     - Lista todos os túneis ativos
#   ./manage_tunnels.sh stop ID  - Para o túnel de uma instância específica
#   ./manage_tunnels.sh stop-all - Para todos os túneis ativos
# ====================================================================================

TUNNEL_DIR="/tmp"
TUNNEL_PREFIX="vast_tunnel_"

list_tunnels() {
    echo "🔍 Procurando túneis ativos..."
    echo "=================================="
    
    found=false
    for pid_file in "$TUNNEL_DIR"/${TUNNEL_PREFIX}*.pid; do
        if [[ -f "$pid_file" ]]; then
            instance_id=$(basename "$pid_file" .pid | sed "s/$TUNNEL_PREFIX//")
            pid=$(cat "$pid_file")
            
            if kill -0 "$pid" 2>/dev/null; then
                echo "✅ Instância: $instance_id (PID: $pid)"
                echo "   PID File: $pid_file"
                echo "   Log File: /tmp/vast_tunnel_${instance_id}.log"
                echo ""
                found=true
            else
                echo "❌ Instância: $instance_id (PID: $pid) - PROCESSO MORTO"
                echo "   Removendo arquivo PID órfão..."
                rm -f "$pid_file"
                echo ""
            fi
        fi
    done
    
    if [[ "$found" == false ]]; then
        echo "Nenhum túnel ativo encontrado."
    fi
}

stop_tunnel() {
    local instance_id="$1"
    local pid_file="$TUNNEL_DIR/${TUNNEL_PREFIX}${instance_id}.pid"
    
    if [[ ! -f "$pid_file" ]]; then
        echo "❌ Erro: Nenhum túnel encontrado para a instância $instance_id"
        exit 1
    fi
    
    local pid=$(cat "$pid_file")
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "🛑 Parando túnel da instância $instance_id (PID: $pid)..."
        kill "$pid"
        sleep 2
        
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚠️  Processo não parou, forçando..."
            kill -9 "$pid"
        fi
        
        rm -f "$pid_file"
        echo "✅ Túnel da instância $instance_id parado com sucesso."
    else
        echo "⚠️  Processo já estava morto. Removendo arquivo PID..."
        rm -f "$pid_file"
    fi
}

stop_all_tunnels() {
    echo "🛑 Parando todos os túneis ativos..."
    echo "=================================="
    
    for pid_file in "$TUNNEL_DIR"/${TUNNEL_PREFIX}*.pid; do
        if [[ -f "$pid_file" ]]; then
            instance_id=$(basename "$pid_file" .pid | sed "s/$TUNNEL_PREFIX//")
            stop_tunnel "$instance_id"
        fi
    done
    
    echo "✅ Todos os túneis foram parados."
}

show_usage() {
    echo "Uso: $0 {list|stop <instance_id>|stop-all}"
    echo ""
    echo "Comandos:"
    echo "  list                    - Lista todos os túneis ativos"
    echo "  stop <instance_id>      - Para o túnel de uma instância específica"
    echo "  stop-all                - Para todos os túneis ativos"
    echo ""
    echo "Exemplos:"
    echo "  $0 list"
    echo "  $0 stop 12345"
    echo "  $0 stop-all"
}

# Main logic
case "${1:-}" in
    "list")
        list_tunnels
        ;;
    "stop")
        if [[ -z "${2:-}" ]]; then
            echo "❌ Erro: ID da instância não fornecido"
            show_usage
            exit 1
        fi
        stop_tunnel "$2"
        ;;
    "stop-all")
        stop_all_tunnels
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
