#!/bin/bash

echo "Iniciando configuracao da instancia..."

env >> /etc/environment

apt-get update -qq
apt-get install -y rsync rclone wget curl -qq

if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
    rm /tmp/miniconda.sh
fi

export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh

if ! command -v mamba &> /dev/null; then
    conda install mamba -n base -c conda-forge -y
fi

mkdir -p /data
mkdir -p /workspace/feature_genesis/logs

if ! conda env list | grep -q 'dynamic-stage0'; then
    echo "Criando ambiente conda dynamic-stage0..."
    
    if [ -d "/workspace/feature_genesis" ]; then
        cd /workspace/feature_genesis
        if [ -f "environment.yml" ]; then
            echo "Usando environment.yml do workspace..."
            if mamba env create -f environment.yml; then
                echo "Ambiente criado com mamba"
            else
                echo "Mamba falhou, tentando com conda..."
                conda env create -f environment.yml
                echo "Ambiente criado com conda"
            fi
        else
            echo "environment.yml não encontrado em /workspace/feature_genesis"
            exit 1
        fi
    else
        echo "Diretório /workspace/feature_genesis não existe"
        exit 1
    fi
else
    echo "Ambiente dynamic-stage0 já existe"
fi

cat >> /etc/environment << 'EOF'
CUDA_USE_DEPRECATED_API=0
RMM_USE_NEW_CUDA_BINDINGS=1
CUDF_USE_NEW_CUDA_BINDINGS=1
EOF

mkdir -p ~/.config/rclone
echo "Rclone será configurado pelo script de deploy com credenciais seguras"

echo "Configuracao finalizada com sucesso!"
echo "Ambientes conda disponíveis:"
conda env list

echo "Container configurado e pronto para uso."

    