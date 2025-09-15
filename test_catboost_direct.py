#!/usr/bin/env python3
"""
Teste direto do CatBoost para validar as correções
sem depender do pipeline completo.
"""
import sys
import os
import numpy as np
import pandas as pd
import cudf

# Add the project root to Python path
sys.path.insert(0, '/workspace/feature_genesis')

from features.statistical_tests.feature_selection import FeatureSelection
from utils.logging_utils import get_logger

logger = get_logger(__name__, "test_catboost")

def test_catboost_direct():
    """Teste direto do CatBoost com dados sintéticos."""
    logger.info("🚀 Iniciando teste direto do CatBoost...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 5000
    n_features = 20
    
    # Create synthetic features
    X_data = np.random.randn(n_samples, n_features).astype(np.float32)
    y_data = (X_data[:, 0] + 0.5 * X_data[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Convert to cuDF
    X_df = cudf.DataFrame(X_data, columns=feature_names)
    y_series = cudf.Series(y_data, name='target')
    
    logger.info(f"📊 Dados sintéticos criados: {X_df.shape} features, {len(y_series)} samples")
    logger.info(f"🎯 Distribuição do target: {y_series.value_counts().to_pandas().to_dict()}")
    
    # Initialize FeatureSelection
    selector = FeatureSelection()
    # Set the CatBoost parameters manually
    selector.stage3_catboost_iterations = 100
    selector.stage3_catboost_learning_rate = 0.1
    selector.stage3_catboost_depth = 4
    selector.stage3_catboost_task_type = 'GPU'
    selector.stage3_catboost_devices = '0'
    
    try:
        # Test the corrected method
        logger.info("🔬 Testando _stage3_selectfrommodel com correções...")
        
        selected_features, importances, backend, score, metrics = selector._stage3_selectfrommodel(
            X_df, y_series, feature_names
        )
        
        logger.info(f"✅ Teste concluído com sucesso!")
        logger.info(f"📈 Backend usado: {backend}")
        logger.info(f"📊 Score do modelo: {score:.6f}")
        logger.info(f"🏆 Features selecionadas: {len(selected_features)}")
        logger.info(f"🎖️ Top 5 features: {selected_features[:5]}")
        
        # Check importances
        logger.info("🏅 Top 10 importâncias:")
        for i, (feat, imp) in enumerate(list(importances.items())[:10]):
            logger.info(f"  {i+1:2d}. {feat}: {imp:.6f}")
        
        # Verify not all uniform
        importance_values = list(importances.values())
        unique_importances = len(set(importance_values))
        logger.info(f"🔍 Importâncias únicas: {unique_importances}/{len(importance_values)}")
        
        if unique_importances > 1:
            logger.info("✅ SUCESSO: Importâncias não são uniformes!")
            return True
        else:
            logger.error("❌ FALHA: Importâncias ainda são uniformes!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro durante teste: {e}")
        import traceback
        logger.error(f"📜 Traceback completo:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_catboost_direct()
    sys.exit(0 if success else 1)
