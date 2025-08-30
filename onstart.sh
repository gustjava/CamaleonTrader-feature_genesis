#!/bin/bash
set -euo pipefail

# ====================================================================================
# SCRIPT ONSTART OTIMIZADO PARA VAST.AI
#
# Este script executa uma vez na criação da instância. Ele combina as boas práticas
# da template da vast.ai com nossas necessidades específicas.
# ====================================================================================

echo "--- [onstart.sh] Iniciando configuração da instância ---"

# 1. Propagar variáveis de ambiente (boa prática da template vast.ai)
# Isso garante que as variáveis definidas na UI da vast.ai fiquem
# disponíveis para futuras sessões SSH.
echo "Propagando variáveis de ambiente para /etc/environment..."
env >> /etc/environment
echo "✅ Variáveis de ambiente propagadas."

# 2. Atualizar pacotes e instalar ferramentas essenciais
# rsync é necessário para a sincronização de código que virá do seu script local.
# rclone também será usado pelo seu script local, mas já o deixamos pré-instalado.
echo "Instalando ferramentas essenciais (rsync, rclone)..."
apt-get update -qq
apt-get install -y rsync rclone -qq
echo "✅ Ferramentas essenciais instaladas."

# 3. Mantém o contêiner rodando indefinidamente
# O script que você roda do seu computador ('deploy_production_instance.sh')
# irá se conectar e executar o trabalho real.
echo "--- [onstart.sh] Configuração finalizada. Instância pronta para conexões. ---"
tail -f /dev/null

    