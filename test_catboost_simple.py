#!/usr/bin/env python3
"""
Teste direto simplificado do CatBoost.
"""
import sys
import os
import numpy as np
import pandas as pd
import cudf

# Add the project root to Python path
sys.path.insert(0, '/workspace/feature_genesis')

def test_catboost_simple():
    """Teste direto do CatBoost com dados sintéticos."""
    print("🚀 Iniciando teste direto do CatBoost...")
    
    try:
        from features.statistical_tests.feature_selection import FeatureSelection
        print("✅ Importação do FeatureSelection bem-sucedida")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Create synthetic features
        X_data = np.random.randn(n_samples, n_features).astype(np.float32)
        y_data = (X_data[:, 0] + 0.5 * X_data[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Convert to cuDF
        X_df = cudf.DataFrame(X_data, columns=feature_names)
        y_series = cudf.Series(y_data, name='target')
        
        print(f"📊 Dados sintéticos criados: {X_df.shape} features, {len(y_series)} samples")
        print(f"🎯 Distribuição do target: {y_series.value_counts().to_pandas().to_dict()}")
        
        # Initialize FeatureSelection
        selector = FeatureSelection()
        selector.stage3_catboost_iterations = 50
        selector.stage3_catboost_learning_rate = 0.1
        selector.stage3_catboost_depth = 4
        selector.stage3_catboost_task_type = 'GPU'
        selector.stage3_catboost_devices = '0'
        
        print("🔬 Testando _stage3_selectfrommodel com correções...")
        
        # Test the corrected method
        selected_features, importances, backend, score, metrics = selector._stage3_selectfrommodel(
            X_df, y_series, feature_names
        )
        
        print(f"✅ Teste concluído com sucesso!")
        print(f"📈 Backend usado: {backend}")
        print(f"📊 Score do modelo: {score:.6f}")
        print(f"🏆 Features selecionadas: {len(selected_features)}")
        print(f"🎖️ Top 5 features: {selected_features[:5]}")
        
        # Check importances
        print("🏅 Top 5 importâncias:")
        for i, (feat, imp) in enumerate(list(importances.items())[:5]):
            print(f"  {i+1:2d}. {feat}: {imp:.6f}")
        
        # Verify not all uniform
        importance_values = list(importances.values())
        unique_importances = len(set(importance_values))
        print(f"🔍 Importâncias únicas: {unique_importances}/{len(importance_values)}")
        
        if unique_importances > 1:
            print("✅ SUCESSO: Importâncias não são uniformes!")
            return True
        else:
            print("❌ FALHA: Importâncias ainda são uniformes!")
            # Show some values for debugging
            print(f"🔍 Primeiras 5 importâncias: {importance_values[:5]}")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        print(f"📜 Traceback completo:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_catboost_simple()
    print(f"🏁 Resultado final: {'SUCESSO' if success else 'FALHA'}")
    sys.exit(0 if success else 1)
