#!/usr/bin/env python3
"""
Script para gerar dados de teste simulados para demonstração do pipeline

Este script cria dados financeiros simulados com características
realísticas para testar o pipeline de otimização avançado.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Configurações
RANDOM_SEED = 42
N_SAMPLES = 15000  # ~6 meses de dados de 5 minutos
N_FEATURES = 120
TARGET_VOLATILITY = 0.02  # 2% volatilidade diária

np.random.seed(RANDOM_SEED)

def generate_realistic_features(n_samples: int, n_features: int) -> np.ndarray:
    """Gerar features com características financeiras realísticas"""
    
    # Base features com correlações temporais
    n_base = min(25, n_features // 3)
    X_base = np.random.randn(n_samples, n_base)
    
    # Adicionar autocorrelação temporal (memória)
    for i in range(1, min(10, n_samples)):
        X_base[i:, :] += 0.05 * X_base[:-i, :]
    
    # Features técnicas (médias móveis simuladas)
    n_technical = min(30, n_features // 3)
    X_technical = np.zeros((n_samples, n_technical))
    
    # Simular médias móveis de diferentes períodos
    windows = [5, 10, 20, 50, 100]
    for i, window in enumerate(windows):
        if i < n_technical:
            # Média móvel simples
            for j in range(window, n_samples):
                X_technical[j, i] = np.mean(X_base[j-window:j, 0])
            
            # Média móvel exponencial
            if i + 5 < n_technical:
                alpha = 2 / (window + 1)
                ema = np.zeros(n_samples)
                ema[0] = X_base[0, 0]
                for j in range(1, n_samples):
                    ema[j] = alpha * X_base[j, 0] + (1 - alpha) * ema[j-1]
                X_technical[:, i + 5] = ema
    
    # Features de volatilidade (rolling std)
    for i, window in enumerate([10, 20, 50]):
        if i + 10 < n_technical:
            for j in range(window, n_samples):
                X_technical[j, i + 10] = np.std(X_base[j-window:j, 0])
    
    # Features de momentum
    for i, lag in enumerate([1, 5, 10, 20]):
        if i + 15 < n_technical:
            X_technical[lag:, i + 15] = X_base[lag:, 0] - X_base[:-lag, 0]
    
    # Features de ruído (não informativas)
    n_noise = n_features - n_base - n_technical
    X_noise = np.random.randn(n_samples, n_noise) * 0.5
    
    # Combinar todas as features
    X = np.column_stack([X_base, X_technical, X_noise])
    
    return X


def generate_realistic_target(X: np.ndarray, target_vol: float) -> np.ndarray:
    """Gerar target com relação não-linear realística"""
    
    # Usar algumas features como base para o target
    signal = (
        0.3 * X[:, 0] * X[:, 1] +  # Interação
        0.2 * np.tanh(X[:, 2]) +   # Não-linearidade
        0.15 * (X[:, 3] ** 2 - 1) +  # Parabólica
        0.1 * np.sin(X[:, 4] * 2) +  # Cíclica
        0.05 * np.sign(X[:, 5]) * np.sqrt(np.abs(X[:, 5]))  # Assimétrica
    )
    
    # Adicionar ruído correlacionado
    noise = np.random.randn(len(X)) * 0.8
    
    # Combinar sinal e ruído
    target_raw = signal + noise
    
    # Normalizar para volatilidade desejada
    target_std = np.std(target_raw)
    target = (target_raw / target_std) * target_vol
    
    # Adicionar algumas características financeiras
    # Clustering de volatilidade (GARCH-like)
    for i in range(1, len(target)):
        if np.abs(target[i-1]) > target_vol * 1.5:
            target[i] *= 1.2  # Volatilidade persiste
    
    return target


def add_timestamp(n_samples: int) -> pd.Series:
    """Adicionar timestamps realísticos (5 minutos)"""
    start_date = pd.Timestamp('2024-01-01 09:00:00')
    
    # 5 minutos entre observações, apenas em horário de mercado
    timestamps = []
    current_time = start_date
    
    for _ in range(n_samples):
        timestamps.append(current_time)
        current_time += pd.Timedelta(minutes=5)
        
        # Pular fins de semana (simplificado)
        if current_time.weekday() >= 5:  # Sábado ou domingo
            current_time += pd.Timedelta(days=2)
            current_time = current_time.replace(hour=9, minute=0)
    
    return pd.Series(timestamps)


def main():
    """Gerar dataset de teste"""
    print(f"Gerando dados de teste...")
    print(f"Amostras: {N_SAMPLES}, Features: {N_FEATURES}")
    
    # Gerar features
    X = generate_realistic_features(N_SAMPLES, N_FEATURES)
    
    # Gerar target
    y = generate_realistic_target(X, TARGET_VOLATILITY)
    
    # Criar feature names
    feature_names = [f'feature_{i:03d}' for i in range(N_FEATURES)]
    
    # Criar timestamps
    timestamps = add_timestamp(N_SAMPLES)
    
    # Criar DataFrame
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    data['timestamp'] = timestamps
    
    # Remover NaN (se houver)
    data = data.dropna()
    
    # Estatísticas
    print(f"\nEstatísticas do dataset:")
    print(f"Shape final: {data.shape}")
    print(f"Target - Média: {data['target'].mean():.6f}, Std: {data['target'].std():.6f}")
    print(f"Features - Média: {data[feature_names].mean().mean():.3f}")
    
    # Salvar dados
    output_dir = Path('data_test')
    output_dir.mkdir(exist_ok=True)
    
    # Parquet (recomendado)
    parquet_path = output_dir / 'eurusd_features_test.parquet'
    data.to_parquet(parquet_path, index=False)
    
    # CSV (backup)
    csv_path = output_dir / 'eurusd_features_test.csv'
    data.to_csv(csv_path, index=False)
    
    # Feature names para referência
    features_df = pd.DataFrame({'feature_name': feature_names})
    features_df.to_csv(output_dir / 'feature_names.csv', index=False)
    
    print(f"\nDados salvos em:")
    print(f"  - {parquet_path}")
    print(f"  - {csv_path}")
    print(f"  - {output_dir / 'feature_names.csv'}")
    
    print(f"\nPara testar o pipeline, execute:")
    print(f"python run_advanced_optimization.py --data-path {parquet_path} --n-trials 50 --sample-size 5000")


if __name__ == "__main__":
    main()
