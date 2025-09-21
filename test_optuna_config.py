#!/usr/bin/env python3
"""
Script para testar a configuração do Optuna e verificar se a paralelização está funcionando corretamente.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

def test_optuna_config():
    """Testa a configuração do Optuna."""
    print("🔍 Testando configuração do Optuna...")
    
    try:
        # Carregar configuração do Dask
        dask_config_path = "config/config.yaml"
        if os.path.exists(dask_config_path):
            with open(dask_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            dask_config = config.get('dask', {})
            print(f"📊 Configuração Dask:")
            print(f"  - workers_per_gpu: {dask_config.get('workers_per_gpu', 'N/A')}")
            print(f"  - threads_per_worker: {dask_config.get('threads_per_worker', 'N/A')}")
            print(f"  - memory_limit_fraction: {dask_config.get('memory_limit_fraction', 'N/A')}")
        else:
            print(f"❌ Arquivo de configuração não encontrado: {dask_config_path}")
        
        # Carregar configuração do estudo
        study_config_path = "config/study/stageA.yaml"
        if os.path.exists(study_config_path):
            with open(study_config_path, 'r') as f:
                study_config = yaml.safe_load(f)
            
            optuna_config = study_config.get('optuna', {})
            print(f"🔬 Configuração Optuna:")
            print(f"  - n_trials: {optuna_config.get('n_trials', 'N/A')}")
            print(f"  - max_concurrent_trials: {optuna_config.get('max_concurrent_trials', 'N/A')}")
            print(f"  - use_dask_parallelization: {optuna_config.get('use_dask_parallelization', 'N/A')}")
            print(f"  - sampler: {optuna_config.get('sampler', 'N/A')}")
            print(f"  - pruner: {optuna_config.get('pruner', 'N/A')}")
        else:
            print(f"❌ Arquivo de configuração não encontrado: {study_config_path}")
        
        # Verificar se Optuna está disponível
        try:
            import optuna
            print(f"✅ Optuna disponível: versão {optuna.__version__}")
        except ImportError:
            print("❌ Optuna não está disponível")
        
        # Verificar se Dask está disponível
        try:
            import dask
            import dask.distributed
            print(f"✅ Dask disponível: versão {dask.__version__}")
        except ImportError:
            print("❌ Dask não está disponível")
        
        print("\n💡 Recomendações:")
        print("  - Para execução sequencial: use_dask_parallelization: false")
        print("  - Para paralelização controlada: max_concurrent_trials: 4-8")
        print("  - Para paralelização máxima: max_concurrent_trials: 0 (usa todos os workers)")
        
    except Exception as e:
        print(f"❌ Erro ao testar configuração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optuna_config()
