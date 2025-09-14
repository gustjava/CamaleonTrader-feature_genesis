#!/bin/bash

echo "Iniciando configuracao da instancia..."

# Verificar GPUs disponíveis
echo "Verificando GPUs disponíveis..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "GPUs detectadas: $GPU_COUNT"
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "AVISO: Nenhuma GPU detectada!"
    elif [ "$GPU_COUNT" -eq 1 ]; then
        echo "1 GPU detectada - configuração adequada para instância single-GPU"
    else
        echo "$GPU_COUNT GPUs detectadas"
    fi
else
    echo "nvidia-smi não encontrado - GPUs podem não estar disponíveis"
fi

env >> /etc/environment

apt-get update -qq

# Instalar dependências de sistema essenciais (apenas as realmente necessárias)
echo "Instalando dependências de sistema essenciais..."
apt-get install -y \
    rsync rclone wget curl \
    build-essential gcc g++ make \
    libblas-dev liblapack-dev \
    libfreetype6-dev libpng-dev \
    libssl-dev libffi-dev \
    git vim nano htop \
    -qq

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

# Detecta/Cria ambiente Conda para o pipeline
ENV_NAME=""
if conda env list | grep -q 'dynamic-stage0'; then
    ENV_NAME="dynamic-stage0"
    echo "Ambiente conda já existe: $ENV_NAME"
elif conda env list | grep -q 'feature-genesis'; then
    ENV_NAME="feature-genesis"
    echo "Ambiente conda já existe: $ENV_NAME"
else
    echo "Criando ambiente conda dynamic-stage0 a partir do environment.yml..."
    if [ -d "/workspace/feature_genesis" ]; then
        cd /workspace/feature_genesis
        if [ -f "environment.yml" ]; then
            echo "Usando environment.yml do workspace..."
            if command -v mamba &> /dev/null; then
                if mamba env create -n dynamic-stage0 -f environment.yml; then
                    ENV_NAME="dynamic-stage0"
                    echo "Ambiente criado com mamba: $ENV_NAME"
                else
                    echo "Mamba falhou, tentando com conda..."
                    conda env create -n dynamic-stage0 -f environment.yml
                    ENV_NAME="dynamic-stage0"
                    echo "Ambiente criado com conda: $ENV_NAME"
                fi
            else
                conda env create -n dynamic-stage0 -f environment.yml
                ENV_NAME="dynamic-stage0"
                echo "Ambiente criado com conda: $ENV_NAME"
            fi
        else
            echo "environment.yml não encontrado em /workspace/feature_genesis"
            exit 1
        fi
    else
        echo "Diretório /workspace/feature_genesis não existe"
        exit 1
    fi
fi

# Garante dependências do Stage 3/4 (LightGBM, scikit-learn, XGBoost, matplotlib)
echo "Garantindo dependências do Stage 3/4 no ambiente '$ENV_NAME'..."
if [ -n "$ENV_NAME" ]; then
    if command -v mamba &> /dev/null; then
        mamba install -n "$ENV_NAME" -c conda-forge -y scikit-learn lightgbm xgboost matplotlib || true
    else
        conda install -n "$ENV_NAME" -c conda-forge -y scikit-learn lightgbm xgboost matplotlib || true
    fi
else
    echo "Aviso: ENV_NAME vazio; pulando instalação de dependências específicas do ambiente."
fi

# Instalar apenas dependências Python ESSENCIAIS (não-opcionais)
echo "Instalando dependências Python essenciais..."
if [ -n "$ENV_NAME" ]; then
    # Ativar o ambiente antes de instalar
    conda activate "$ENV_NAME"
    
    # Instalar apenas dependências CORE (não marcadas como Optional)
    pip install --no-cache-dir \
        pyyaml python-dotenv click \
        structlog rich python-json-logger psutil \
        sqlalchemy pymysql cryptography \
        boto3 s3fs pyarrow \
        statsmodels arch \
        dcor emd memory-profiler gputil \
        joblib multiprocessing-logging \
        || echo "Algumas dependências falharam, continuando..."
    
    echo "Dependências Python essenciais instaladas no ambiente '$ENV_NAME'"
else
    echo "Aviso: ENV_NAME vazio; pulando instalação de dependências Python."
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

# 4. Pré-instalar dependências pesadas do Conda e Pip (CatBoost, EMD)
echo "--- [onstart.sh] Pré-instalando dependências do pipeline (CatBoost, EMD)... ---"
source /opt/conda/etc/profile.d/conda.sh
conda activate base
conda install -c conda-forge -y catboost || true
pip install --no-cache-dir emd || true
echo "✅ Dependências pré-instaladas."

# Mantém o contêiner rodando indefinidamente (ambientação Vast)
echo "--- [onstart.sh] Configuração finalizada. Instância pronta para conexões. ---"
tail -f /dev/null


