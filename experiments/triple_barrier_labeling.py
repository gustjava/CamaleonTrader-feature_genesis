#!/usr/bin/env python3
"""
Experimento 2: Triple-Barrier Labeling Implementation
Implementa rotulagem baseada em take-profit, stop-loss e tempo limite
para remover ruído e focar em movimentos significativos de mercado
"""

import numpy as np
import pandas as pd
import cupy as cp
from numba import jit
from typing import Tuple, Optional
import warnings

class TripleBarrierLabeler:
    """
    Implementa o método Triple-Barrier para rotulagem de dados financeiros.
    
    O método define três "barreiras":
    1. Take Profit: barreira superior (lucro)
    2. Stop Loss: barreira inferior (perda) 
    3. Time Limit: barreira temporal (timeout)
    
    O primeiro evento a ser atingido determina o label.
    """
    
    def __init__(
        self,
        take_profit_pct: float = 0.002,  # 0.2% take profit
        stop_loss_pct: float = 0.001,    # 0.1% stop loss  
        time_limit_periods: int = 60,    # 60 períodos (minutos)
        min_periods_ahead: int = 5       # Mínimo 5 períodos no futuro
    ):
        """
        Args:
            take_profit_pct: Percentual de lucro para barreira superior
            stop_loss_pct: Percentual de perda para barreira inferior  
            time_limit_periods: Número máximo de períodos para esperar
            min_periods_ahead: Mínimo de períodos à frente para considerar
        """
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.time_limit_periods = time_limit_periods
        self.min_periods_ahead = min_periods_ahead
        
        # Validate parameters
        if take_profit_pct <= 0 or stop_loss_pct <= 0:
            raise ValueError("Take profit and stop loss must be positive")
        if time_limit_periods < min_periods_ahead:
            raise ValueError("Time limit must be >= min_periods_ahead")
    
    def create_labels(
        self, 
        prices: pd.Series,
        use_gpu: bool = True
    ) -> pd.DataFrame:
        """
        Cria labels usando o método Triple-Barrier
        
        Args:
            prices: Série temporal de preços (close prices)
            use_gpu: Se True, usa CuPy para acelerar cálculos
            
        Returns:
            DataFrame com colunas:
            - label: -1 (stop loss), 0 (timeout), 1 (take profit)
            - barrier_hit: qual barreira foi atingida primeiro
            - periods_to_barrier: quantos períodos até atingir a barreira
            - return_achieved: retorno real obtido
            - take_profit_barrier: valor da barreira de lucro
            - stop_loss_barrier: valor da barreira de perda
        """
        
        if len(prices) < self.time_limit_periods + self.min_periods_ahead:
            raise ValueError(f"Need at least {self.time_limit_periods + self.min_periods_ahead} prices")
        
        # Convert to numpy for faster processing
        prices_array = prices.values
        
        if use_gpu and cp:
            return self._create_labels_gpu(prices_array, prices.index)
        else:
            return self._create_labels_cpu(prices_array, prices.index)
    
    def _create_labels_gpu(self, prices: np.ndarray, index: pd.Index) -> pd.DataFrame:
        """GPU-accelerated version using CuPy"""
        
        try:
            # Transfer to GPU
            gpu_prices = cp.asarray(prices)
            n_samples = len(prices) - self.time_limit_periods - self.min_periods_ahead
            
            # Pre-allocate result arrays
            labels = cp.zeros(n_samples, dtype=cp.int8)
            barriers_hit = cp.zeros(n_samples, dtype=cp.int8)  # 1=TP, -1=SL, 0=Time
            periods_to_barrier = cp.zeros(n_samples, dtype=cp.int16)
            returns_achieved = cp.zeros(n_samples, dtype=cp.float32)
            tp_barriers = cp.zeros(n_samples, dtype=cp.float32)
            sl_barriers = cp.zeros(n_samples, dtype=cp.float32)
            
            # Process each sample
            for i in range(n_samples):
                entry_price = gpu_prices[i]
                
                # Define barriers
                tp_barrier = entry_price * (1 + self.take_profit_pct)
                sl_barrier = entry_price * (1 - self.stop_loss_pct)
                
                tp_barriers[i] = tp_barrier
                sl_barriers[i] = sl_barrier
                
                # Look ahead for barrier hits
                end_idx = min(i + self.time_limit_periods + 1, len(gpu_prices))
                future_prices = gpu_prices[i + self.min_periods_ahead:end_idx]
                
                # Find first barrier hit
                tp_hits = future_prices >= tp_barrier
                sl_hits = future_prices <= sl_barrier
                
                tp_hit_idx = cp.where(tp_hits)[0]
                sl_hit_idx = cp.where(sl_hits)[0]
                
                if len(tp_hit_idx) > 0 and len(sl_hit_idx) > 0:
                    # Both barriers hit, take the first one
                    if tp_hit_idx[0] <= sl_hit_idx[0]:
                        # Take profit hit first
                        labels[i] = 1
                        barriers_hit[i] = 1
                        periods_to_barrier[i] = tp_hit_idx[0] + self.min_periods_ahead
                        final_price = future_prices[tp_hit_idx[0]]
                    else:
                        # Stop loss hit first
                        labels[i] = -1
                        barriers_hit[i] = -1
                        periods_to_barrier[i] = sl_hit_idx[0] + self.min_periods_ahead
                        final_price = future_prices[sl_hit_idx[0]]
                        
                elif len(tp_hit_idx) > 0:
                    # Only take profit hit
                    labels[i] = 1
                    barriers_hit[i] = 1
                    periods_to_barrier[i] = tp_hit_idx[0] + self.min_periods_ahead
                    final_price = future_prices[tp_hit_idx[0]]
                    
                elif len(sl_hit_idx) > 0:
                    # Only stop loss hit
                    labels[i] = -1
                    barriers_hit[i] = -1
                    periods_to_barrier[i] = sl_hit_idx[0] + self.min_periods_ahead
                    final_price = future_prices[sl_hit_idx[0]]
                    
                else:
                    # Timeout - no barrier hit
                    labels[i] = 0
                    barriers_hit[i] = 0
                    periods_to_barrier[i] = len(future_prices) - 1 + self.min_periods_ahead
                    final_price = future_prices[-1]
                
                # Calculate actual return achieved
                returns_achieved[i] = (final_price - entry_price) / entry_price
            
            # Transfer back to CPU
            result_data = {
                'label': cp.asnumpy(labels),
                'barrier_hit': cp.asnumpy(barriers_hit),
                'periods_to_barrier': cp.asnumpy(periods_to_barrier),
                'return_achieved': cp.asnumpy(returns_achieved),
                'take_profit_barrier': cp.asnumpy(tp_barriers),
                'stop_loss_barrier': cp.asnumpy(sl_barriers)
            }
            
        except Exception as e:
            warnings.warn(f"GPU processing failed: {e}. Falling back to CPU.")
            return self._create_labels_cpu(prices, index)
        
        # Create DataFrame with original index alignment
        result_index = index[:n_samples]
        return pd.DataFrame(result_data, index=result_index)
    
    def _create_labels_cpu(self, prices: np.ndarray, index: pd.Index) -> pd.DataFrame:
        """CPU version using NumPy"""
        
        n_samples = len(prices) - self.time_limit_periods - self.min_periods_ahead
        
        # Pre-allocate result arrays
        labels = np.zeros(n_samples, dtype=np.int8)
        barriers_hit = np.zeros(n_samples, dtype=np.int8)
        periods_to_barrier = np.zeros(n_samples, dtype=np.int16)
        returns_achieved = np.zeros(n_samples, dtype=np.float32)
        tp_barriers = np.zeros(n_samples, dtype=np.float32)
        sl_barriers = np.zeros(n_samples, dtype=np.float32)
        
        # Process each sample (vectorized where possible)
        for i in range(n_samples):
            entry_price = prices[i]
            
            # Define barriers
            tp_barrier = entry_price * (1 + self.take_profit_pct)
            sl_barrier = entry_price * (1 - self.stop_loss_pct)
            
            tp_barriers[i] = tp_barrier
            sl_barriers[i] = sl_barrier
            
            # Look ahead for barrier hits
            end_idx = min(i + self.time_limit_periods + 1, len(prices))
            future_prices = prices[i + self.min_periods_ahead:end_idx]
            
            # Find first barrier hit
            tp_hits = np.where(future_prices >= tp_barrier)[0]
            sl_hits = np.where(future_prices <= sl_barrier)[0]
            
            if len(tp_hits) > 0 and len(sl_hits) > 0:
                # Both barriers hit, take the first one
                if tp_hits[0] <= sl_hits[0]:
                    # Take profit hit first
                    labels[i] = 1
                    barriers_hit[i] = 1
                    periods_to_barrier[i] = tp_hits[0] + self.min_periods_ahead
                    final_price = future_prices[tp_hits[0]]
                else:
                    # Stop loss hit first
                    labels[i] = -1
                    barriers_hit[i] = -1
                    periods_to_barrier[i] = sl_hits[0] + self.min_periods_ahead
                    final_price = future_prices[sl_hits[0]]
                    
            elif len(tp_hits) > 0:
                # Only take profit hit
                labels[i] = 1
                barriers_hit[i] = 1
                periods_to_barrier[i] = tp_hits[0] + self.min_periods_ahead
                final_price = future_prices[tp_hits[0]]
                
            elif len(sl_hits) > 0:
                # Only stop loss hit
                labels[i] = -1
                barriers_hit[i] = -1
                periods_to_barrier[i] = sl_hits[0] + self.min_periods_ahead
                final_price = future_prices[sl_hits[0]]
                
            else:
                # Timeout - no barrier hit
                labels[i] = 0
                barriers_hit[i] = 0
                periods_to_barrier[i] = len(future_prices) - 1 + self.min_periods_ahead
                final_price = future_prices[-1]
            
            # Calculate actual return achieved
            returns_achieved[i] = (final_price - entry_price) / entry_price
        
        result_data = {
            'label': labels,
            'barrier_hit': barriers_hit,
            'periods_to_barrier': periods_to_barrier,
            'return_achieved': returns_achieved,
            'take_profit_barrier': tp_barriers,
            'stop_loss_barrier': sl_barriers
        }
        
        # Create DataFrame with original index alignment
        result_index = index[:n_samples]
        return pd.DataFrame(result_data, index=result_index)
    
    def analyze_labels(self, labels_df: pd.DataFrame) -> dict:
        """
        Analisa a distribuição e qualidade dos labels gerados
        
        Returns:
            Dict com estatísticas dos labels
        """
        
        analysis = {
            'total_samples': len(labels_df),
            'label_distribution': {
                'take_profit': int((labels_df['label'] == 1).sum()),
                'stop_loss': int((labels_df['label'] == -1).sum()),
                'timeout': int((labels_df['label'] == 0).sum())
            },
            'label_percentages': {
                'take_profit_pct': float((labels_df['label'] == 1).mean() * 100),
                'stop_loss_pct': float((labels_df['label'] == -1).mean() * 100),
                'timeout_pct': float((labels_df['label'] == 0).mean() * 100)
            },
            'average_periods_to_barrier': {
                'take_profit': float(labels_df[labels_df['label'] == 1]['periods_to_barrier'].mean()),
                'stop_loss': float(labels_df[labels_df['label'] == -1]['periods_to_barrier'].mean()),
                'timeout': float(labels_df[labels_df['label'] == 0]['periods_to_barrier'].mean())
            },
            'average_returns': {
                'take_profit': float(labels_df[labels_df['label'] == 1]['return_achieved'].mean()),
                'stop_loss': float(labels_df[labels_df['label'] == -1]['return_achieved'].mean()),
                'timeout': float(labels_df[labels_df['label'] == 0]['return_achieved'].mean())
            },
            'settings': {
                'take_profit_pct': self.take_profit_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'time_limit_periods': self.time_limit_periods,
                'min_periods_ahead': self.min_periods_ahead
            }
        }
        
        return analysis


def demo_triple_barrier():
    """Demonstração do Triple-Barrier Labeling"""
    
    print("🎯 Triple-Barrier Labeling Demo")
    print("=" * 50)
    
    # Generate sample price data
    np.random.seed(42)
    n_periods = 10000
    
    # Create realistic price series with some trend and volatility
    returns = np.random.normal(0.0001, 0.01, n_periods)  # Small positive drift
    prices = 1.0 * np.exp(np.cumsum(returns))
    
    price_series = pd.Series(
        prices, 
        index=pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    )
    
    print(f"📊 Generated {len(price_series)} price observations")
    print(f"💰 Price range: {price_series.min():.4f} - {price_series.max():.4f}")
    
    # Create labeler with different configurations
    configs = [
        {"name": "Conservative", "tp": 0.001, "sl": 0.0005, "time": 30},
        {"name": "Moderate", "tp": 0.002, "sl": 0.001, "time": 60},
        {"name": "Aggressive", "tp": 0.005, "sl": 0.002, "time": 120}
    ]
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']} Configuration:")
        print(f"   TP: {config['tp']*100:.2f}% | SL: {config['sl']*100:.2f}% | Time: {config['time']} periods")
        
        labeler = TripleBarrierLabeler(
            take_profit_pct=config['tp'],
            stop_loss_pct=config['sl'],
            time_limit_periods=config['time']
        )
        
        # Create labels
        labels_df = labeler.create_labels(price_series, use_gpu=True)
        
        # Analyze results
        analysis = labeler.analyze_labels(labels_df)
        
        print(f"   📈 Results:")
        print(f"      Take Profit: {analysis['label_percentages']['take_profit_pct']:.1f}% "
              f"({analysis['label_distribution']['take_profit']} trades)")
        print(f"      Stop Loss:   {analysis['label_percentages']['stop_loss_pct']:.1f}% "
              f"({analysis['label_distribution']['stop_loss']} trades)")
        print(f"      Timeout:     {analysis['label_percentages']['timeout_pct']:.1f}% "
              f"({analysis['label_distribution']['timeout']} trades)")
        
        # Calculate profitability metrics
        tp_return = analysis['average_returns']['take_profit'] * 100
        sl_return = analysis['average_returns']['stop_loss'] * 100
        timeout_return = analysis['average_returns']['timeout'] * 100
        
        print(f"   💰 Average Returns:")
        print(f"      Take Profit: {tp_return:.3f}%")
        print(f"      Stop Loss:   {sl_return:.3f}%")
        print(f"      Timeout:     {timeout_return:.3f}%")
        
        # Overall expected return
        expected_return = (
            analysis['label_percentages']['take_profit_pct']/100 * tp_return +
            analysis['label_percentages']['stop_loss_pct']/100 * sl_return +
            analysis['label_percentages']['timeout_pct']/100 * timeout_return
        )
        
        print(f"   🎯 Expected Return: {expected_return:.4f}%")
        
        # Balance check
        win_rate = analysis['label_percentages']['take_profit_pct']
        print(f"   ⚖️ Win Rate: {win_rate:.1f}%")
        
        if win_rate > 50:
            print(f"   ✅ Positive win rate with {config['name'].lower()} settings")
        else:
            print(f"   ⚠️ Low win rate - consider adjusting TP/SL ratio")


if __name__ == "__main__":
    demo_triple_barrier()
