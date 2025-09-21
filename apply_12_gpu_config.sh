#!/bin/bash
# =============================================================================
# Script para aplicar configuração otimizada para 12 GPUs
# =============================================================================

set -e

echo "🚀 Aplicando configuração para 12 GPUs..."

# Backup da configuração atual
echo "📋 Fazendo backup da configuração atual..."
cp config/study/stageA.yaml config/study/stageA.yaml.backup
cp config/config.yaml config/config.yaml.backup

# Aplicar configurações otimizadas
echo "⚙️ Aplicando configurações otimizadas..."

# 1. Atualizar stageA.yaml para 12 GPUs
cat > config/study/stageA.yaml << 'EOF'
package: /study

# =============================================================================
# Hydra Study Configuration - Stage A (Preprocessing & Feature Selection)
# OPTIMIZED FOR 12 GPUs
# =============================================================================

mode: preprocess_selection
name: stageA_preprocess_selection

# Optuna configuration optimized for 12 GPUs
optuna:
  direction: maximize
  n_trials: 1200               # Increased for comprehensive exploration
  timeout: null                # Optional: set seconds limit instead of n_trials
  seed: 42
  sampler: tpe                 # TPESampler is a good default for continuous spaces
  pruner: hyperband            # Hyperband for better early stopping
  hyperband_max_resource: 100  # Maximum resource for hyperband
  hyperband_min_resource: 10   # Minimum resource for hyperband
  # Parallel execution control - OPTIMIZED FOR 12 GPUs
  max_concurrent_trials: 12    # Use all 12 GPUs
  use_dask_parallelization: true  # Enable Dask parallelization

# Output artifacts for the handoff between Study A -> Study B.
outputs:
  dir: ${output.output_path}/optuna_stage
  best_stage12_path: ${study.outputs.dir}/best_stage12.json
  storage_db: ${study.outputs.dir}/study_a.sqlite

dataset:
  path: /data/EURUSD_master_features.parquet  # Path to training data from R2 sync
  format: parquet                      # parquet|csv
  target: ${features.selection_target_column}
  target_candidates: [
    y_ret_fwd_1m,
    y_ret_fwd_3m,
    y_ret_fwd_5m,
    y_ret_fwd_10m,
    y_ret_fwd_15m,
    y_ret_fwd_20m,
    y_ret_fwd_30m,
    y_ret_fwd_60m,
    y_ret_fwd_120m,
    y_ret_fwd_240m
  ]
  symbol: EURUSD
  timeframe: 60m
  sample_rows: 500000                  # Increased for EURUSD optimization
  
# EURUSD-specific optimizations
eurusd_config:
  volatility_regime_detection: true    # Enable volatility regime detection
  session_aware_features: true         # Enable session-aware feature engineering
  macro_economic_features: true        # Enable macro economic features
  cross_currency_features: true        # Enable cross-currency features
  time_decay_features: true            # Enable time decay features

# Lightweight proxy model optimized for EURUSD preprocessing evaluation
proxy_model:
  backend: catboost
  iterations: 400                 # Increased for EURUSD complexity
  depth: 8                       # Increased depth
  learning_rate: 0.05            # Lower learning rate for EURUSD
  early_stopping_rounds: 50      # More patience
  random_state: 42
  loss_function: Huber           # More robust loss for EURUSD
  huber_delta: 1.0               # Delta parameter for Huber loss
  l2_leaf_reg: 5.0              # Added regularization
  subsample: 0.8                # Added subsampling

# Search space optimized for EURUSD forex data characteristics
search_space:
  frac_diff:
    d:
      low: 0.05                # Lower bound for more aggressive stationarization
      high: 0.6                # Higher bound for EURUSD volatility
      step: 0.025              # Finer granularity
    threshold:
      low: 1e-8                # Much more relaxed threshold
      high: 1e-2               # Higher upper bound
      log: true
    max_lag:
      choices: [200, 400, 600, 800, 1000, 1200]  # More options including lower lags

  dcor:
    threshold:
      low: 0.001               # Much more permissive threshold  
      high: 0.2                # Lower upper bound
    min_percentile:
      low: 0.0
      high: 0.1                # Much lower percentile
    top_k:
      low: 50                  # Higher minimum to keep more features
      high: 200                # Keep more features

  vif:
    threshold:
      low: 2.0                 # Higher VIF threshold (less aggressive)
      high: 20                 # Higher tolerance for multicollinearity

  mi:
    threshold:
      low: 0.0001              # Much lower MI threshold
      high: 0.2                # Lower upper bound
    bins:
      choices: [8, 16, 32, 48, 64, 96]  # More bin options
    chunk_size:
      choices: [8, 16, 32, 48, 64, 96]  # More chunk size options

  stage3:
    top_n:
      low: 30                  # Higher minimum to preserve more features
      high: 150                # Much higher bound for EURUSD
    catboost_iterations:
      low: 100                 # Lower bound for faster trials
      high: 2000               # Higher bound for EURUSD complexity

  sampling:
    selection_max_rows:
      low: 100000              # Lower bound for faster trials
      high: 300000             # Higher bound for EURUSD data size

  # EURUSD-specific parameters
  eurusd_specific:
    volatility_scaling:
      choices: [true, false]   # Test volatility scaling for EURUSD
    session_aware:
      choices: [true, false]   # Test session-aware features
    macro_features:
      choices: [true, false]   # Test macro economic features

# Persistence + caching controls for Study A runs.
storage:
  backend: sqlite
  url: sqlite:///${study.outputs.storage_db}
  clean_on_start: false

cache:
  fingerprint_enabled: true
  reuse_intermediate_artifacts: true
EOF

# 2. Atualizar config.yaml para 12 GPUs
echo "📊 Atualizando configuração do Dask para 12 GPUs..."

# Atualizar apenas a seção dask do config.yaml
sed -i '/^dask:/,/^[a-zA-Z]/ {
  /^dask:/,/^[a-zA-Z]/ {
    /workers_per_gpu:/ s/workers_per_gpu: 1/workers_per_gpu: 1/
    /threads_per_worker:/ s/threads_per_worker: 4/threads_per_worker: 6/
    /memory_limit_fraction:/ s/memory_limit_fraction: 0.80/memory_limit_fraction: 0.75/
    /rmm_pool_fraction:/ s/rmm_pool_fraction: 0.50/rmm_pool_fraction: 0.60/
    /rmm_initial_pool_fraction:/ s/rmm_initial_pool_fraction: 0.25/rmm_initial_pool_fraction: 0.30/
    /rmm_maximum_pool_fraction:/ s/rmm_maximum_pool_fraction: 0.70/rmm_maximum_pool_fraction: 0.80/
    /memory_target_fraction:/ s/memory_target_fraction: 0.70/memory_target_fraction: 0.65/
    /memory_spill_fraction:/ s/memory_spill_fraction: 0.85/memory_spill_fraction: 0.80/
  }
}' config/config.yaml

echo "✅ Configuração para 12 GPUs aplicada com sucesso!"
echo ""
echo "📋 Resumo das mudanças:"
echo "  - max_concurrent_trials: 12 (usar todas as GPUs)"
echo "  - threads_per_worker: 6 (otimizado para 12 GPUs)"
echo "  - memory_limit_fraction: 0.75 (mais conservador)"
echo "  - RMM otimizado para múltiplas GPUs"
echo ""
echo "🚀 Para executar com 12 GPUs:"
echo "  ./run_pipeline_vast.sh"
echo ""
echo "💡 Para reverter as mudanças:"
echo "  cp config/study/stageA.yaml.backup config/study/stageA.yaml"
echo "  cp config/config.yaml.backup config/config.yaml"
