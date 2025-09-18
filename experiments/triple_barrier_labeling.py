#!/usr/bin/env python3
"""
Triple-Barrier Labeling for Trading Strategy Development
Implements the triple-barrier method for creating trading labels
"""

import numpy as np
import pandas as pd
import cupy as cp
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

class TripleBarrierLabeler:
    """
    Triple-barrier labeling for creating trading signals
    Labels are: 1 (take profit), -1 (stop loss), 0 (timeout)
    """
    
    def __init__(
        self,
        take_profit_pct: float = 0.002,
        stop_loss_pct: float = 0.001,
        time_limit_periods: int = 60,
        use_gpu: bool = True
    ):
        """
        Initialize triple-barrier labeler
        
        Args:
            take_profit_pct: Take profit threshold (e.g., 0.002 = 0.2%)
            stop_loss_pct: Stop loss threshold (e.g., 0.001 = 0.1%)
            time_limit_periods: Maximum periods to hold position
            use_gpu: Whether to use GPU acceleration
        """
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.time_limit_periods = time_limit_periods
        self.use_gpu = use_gpu and cp is not None
        
    def create_labels(
        self, 
        price_series: pd.Series, 
        use_gpu: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Create triple-barrier labels for a price series
        
        Args:
            price_series: Price series with datetime index
            use_gpu: Override GPU usage setting
            
        Returns:
            DataFrame with labels and metadata
        """
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu
        
        if use_gpu:
            return self._create_labels_gpu(price_series)
        else:
            return self._create_labels_cpu(price_series)
    
    def _create_labels_cpu(self, price_series: pd.Series) -> pd.DataFrame:
        """CPU implementation of triple-barrier labeling"""
        
        prices = price_series.values
        n = len(prices)
        
        # Initialize arrays
        labels = np.zeros(n, dtype=int)
        barrier_hit = np.zeros(n, dtype=int)  # 1=TP, -1=SL, 0=timeout
        hit_periods = np.full(n, self.time_limit_periods, dtype=int)
        returns = np.zeros(n, dtype=float)
        
        for i in range(n - self.time_limit_periods):
            entry_price = prices[i]
            
            # Calculate barriers
            tp_price = entry_price * (1 + self.take_profit_pct)
            sl_price = entry_price * (1 - self.stop_loss_pct)
            
            # Check each period until time limit
            for j in range(1, self.time_limit_periods + 1):
                if i + j >= n:
                    break
                    
                current_price = prices[i + j]
                period_return = (current_price - entry_price) / entry_price
                
                # Check barriers
                if current_price >= tp_price:
                    labels[i] = 1
                    barrier_hit[i] = 1
                    hit_periods[i] = j
                    returns[i] = period_return
                    break
                elif current_price <= sl_price:
                    labels[i] = -1
                    barrier_hit[i] = -1
                    hit_periods[i] = j
                    returns[i] = period_return
                    break
                elif j == self.time_limit_periods:
                    # Timeout
                    labels[i] = 0
                    barrier_hit[i] = 0
                    hit_periods[i] = j
                    returns[i] = period_return
                    break
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'timestamp': price_series.index,
            'price': prices,
            'label': labels,
            'barrier_hit': barrier_hit,
            'hit_periods': hit_periods,
            'return': returns
        })
        
        return result_df
    
    def _create_labels_gpu(self, price_series: pd.Series) -> pd.DataFrame:
        """GPU implementation of triple-barrier labeling"""
        
        # Convert to GPU arrays
        prices_gpu = cp.asarray(price_series.values)
        n = len(prices_gpu)
        
        # Initialize GPU arrays
        labels_gpu = cp.zeros(n, dtype=cp.int32)
        barrier_hit_gpu = cp.zeros(n, dtype=cp.int32)
        hit_periods_gpu = cp.full(n, self.time_limit_periods, dtype=cp.int32)
        returns_gpu = cp.zeros(n, dtype=cp.float32)
        
        # Process in chunks to avoid memory issues
        chunk_size = min(10000, n)
        
        for start_idx in range(0, n - self.time_limit_periods, chunk_size):
            end_idx = min(start_idx + chunk_size, n - self.time_limit_periods)
            
            # Process chunk
            for i in range(start_idx, end_idx):
                entry_price = prices_gpu[i]
                
                # Calculate barriers
                tp_price = entry_price * (1 + self.take_profit_pct)
                sl_price = entry_price * (1 - self.stop_loss_pct)
                
                # Check each period until time limit
                for j in range(1, self.time_limit_periods + 1):
                    if i + j >= n:
                        break
                        
                    current_price = prices_gpu[i + j]
                    period_return = (current_price - entry_price) / entry_price
                    
                    # Check barriers
                    if current_price >= tp_price:
                        labels_gpu[i] = 1
                        barrier_hit_gpu[i] = 1
                        hit_periods_gpu[i] = j
                        returns_gpu[i] = period_return
                        break
                    elif current_price <= sl_price:
                        labels_gpu[i] = -1
                        barrier_hit_gpu[i] = -1
                        hit_periods_gpu[i] = j
                        returns_gpu[i] = period_return
                        break
                    elif j == self.time_limit_periods:
                        # Timeout
                        labels_gpu[i] = 0
                        barrier_hit_gpu[i] = 0
                        hit_periods_gpu[i] = j
                        returns_gpu[i] = period_return
                        break
        
        # Convert back to CPU
        labels = cp.asnumpy(labels_gpu)
        barrier_hit = cp.asnumpy(barrier_hit_gpu)
        hit_periods = cp.asnumpy(hit_periods_gpu)
        returns = cp.asnumpy(returns_gpu)
        prices = cp.asnumpy(prices_gpu)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'timestamp': price_series.index,
            'price': prices,
            'label': labels,
            'barrier_hit': barrier_hit,
            'hit_periods': hit_periods,
            'return': returns
        })
        
        return result_df
    
    def analyze_labels(self, labels_df: pd.DataFrame) -> Dict:
        """
        Analyze the generated labels
        
        Args:
            labels_df: DataFrame with labels from create_labels
            
        Returns:
            Dictionary with analysis results
        """
        
        # Basic statistics
        total_labels = len(labels_df)
        take_profit_count = (labels_df['label'] == 1).sum()
        stop_loss_count = (labels_df['label'] == -1).sum()
        timeout_count = (labels_df['label'] == 0).sum()
        
        # Percentages
        tp_pct = (take_profit_count / total_labels) * 100
        sl_pct = (stop_loss_count / total_labels) * 100
        timeout_pct = (timeout_count / total_labels) * 100
        
        # Average returns by label type
        tp_returns = labels_df[labels_df['label'] == 1]['return']
        sl_returns = labels_df[labels_df['label'] == -1]['return']
        timeout_returns = labels_df[labels_df['label'] == 0]['return']
        
        avg_tp_return = tp_returns.mean() if len(tp_returns) > 0 else 0
        avg_sl_return = sl_returns.mean() if len(sl_returns) > 0 else 0
        avg_timeout_return = timeout_returns.mean() if len(timeout_returns) > 0 else 0
        
        # Average hit periods
        avg_tp_periods = labels_df[labels_df['label'] == 1]['hit_periods'].mean() if len(tp_returns) > 0 else 0
        avg_sl_periods = labels_df[labels_df['label'] == -1]['hit_periods'].mean() if len(sl_returns) > 0 else 0
        avg_timeout_periods = labels_df[labels_df['label'] == 0]['hit_periods'].mean() if len(timeout_returns) > 0 else 0
        
        # Overall statistics
        overall_return = labels_df['return'].mean()
        overall_std = labels_df['return'].std()
        sharpe_ratio = overall_return / overall_std if overall_std > 0 else 0
        
        # Win rate (profitable trades)
        profitable_trades = (labels_df['return'] > 0).sum()
        win_rate = (profitable_trades / total_labels) * 100
        
        analysis = {
            'total_labels': total_labels,
            'label_counts': {
                'take_profit': take_profit_count,
                'stop_loss': stop_loss_count,
                'timeout': timeout_count
            },
            'label_percentages': {
                'take_profit_pct': tp_pct,
                'stop_loss_pct': sl_pct,
                'timeout_pct': timeout_pct
            },
            'average_returns': {
                'take_profit': avg_tp_return,
                'stop_loss': avg_sl_return,
                'timeout': avg_timeout_return
            },
            'average_periods': {
                'take_profit': avg_tp_periods,
                'stop_loss': avg_sl_periods,
                'timeout': avg_timeout_periods
            },
            'overall_metrics': {
                'mean_return': overall_return,
                'std_return': overall_std,
                'sharpe_ratio': sharpe_ratio,
                'win_rate_pct': win_rate
            }
        }
        
        return analysis
    
    def get_optimal_parameters(
        self, 
        price_series: pd.Series,
        tp_range: Tuple[float, float] = (0.0005, 0.005),
        sl_range: Tuple[float, float] = (0.0005, 0.005),
        time_range: Tuple[int, int] = (30, 120),
        n_tests: int = 20
    ) -> Dict:
        """
        Find optimal triple-barrier parameters using grid search
        
        Args:
            price_series: Price series to optimize on
            tp_range: Take profit range (min, max)
            sl_range: Stop loss range (min, max)
            time_range: Time limit range (min, max)
            n_tests: Number of parameter combinations to test
            
        Returns:
            Dictionary with optimal parameters and results
        """
        
        # Generate parameter combinations
        tp_values = np.linspace(tp_range[0], tp_range[1], int(np.sqrt(n_tests)))
        sl_values = np.linspace(sl_range[0], sl_range[1], int(np.sqrt(n_tests)))
        time_values = np.linspace(time_range[0], time_range[1], int(np.sqrt(n_tests)), dtype=int)
        
        best_score = -np.inf
        best_params = None
        best_analysis = None
        
        results = []
        
        for tp in tp_values:
            for sl in sl_values:
                for time_limit in time_values:
                    # Create labeler with current parameters
                    labeler = TripleBarrierLabeler(
                        take_profit_pct=tp,
                        stop_loss_pct=sl,
                        time_limit_periods=time_limit,
                        use_gpu=self.use_gpu
                    )
                    
                    # Generate labels
                    labels_df = labeler.create_labels(price_series)
                    analysis = labeler.analyze_labels(labels_df)
                    
                    # Calculate score (simple: win rate * average return)
                    score = analysis['overall_metrics']['win_rate_pct'] * analysis['overall_metrics']['mean_return']
                    
                    results.append({
                        'tp': tp,
                        'sl': sl,
                        'time_limit': time_limit,
                        'score': score,
                        'analysis': analysis
                    })
                    
                    # Update best
                    if score > best_score:
                        best_score = score
                        best_params = {'tp': tp, 'sl': sl, 'time_limit': time_limit}
                        best_analysis = analysis
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_analysis': best_analysis,
            'all_results': results[:10]  # Top 10 results
        }


def main():
    """Main execution function for testing"""
    
    print("🎯 Triple-Barrier Labeling System")
    print("=" * 40)
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 10000
    returns = np.random.normal(0.0001, 0.01, n_periods)
    prices = 1.0 * np.exp(np.cumsum(returns))
    
    price_series = pd.Series(
        prices, 
        index=pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    )
    
    print(f"Generated {n_periods} periods of sample price data")
    
    # Test different configurations
    configs = [
        {"name": "Conservative", "tp": 0.001, "sl": 0.0005, "time": 30},
        {"name": "Moderate", "tp": 0.002, "sl": 0.001, "time": 60},
        {"name": "Aggressive", "tp": 0.003, "sl": 0.0015, "time": 90}
    ]
    
    for config in configs:
        print(f"\n📊 Testing {config['name']} configuration:")
        print(f"   TP: {config['tp']:.3f}, SL: {config['sl']:.3f}, Time: {config['time']}")
        
        labeler = TripleBarrierLabeler(
            take_profit_pct=config['tp'],
            stop_loss_pct=config['sl'],
            time_limit_periods=config['time']
        )
        
        labels_df = labeler.create_labels(price_series)
        analysis = labeler.analyze_labels(labels_df)
        
        print(f"   Win Rate: {analysis['overall_metrics']['win_rate_pct']:.1f}%")
        print(f"   Mean Return: {analysis['overall_metrics']['mean_return']:.6f}")
        print(f"   Sharpe Ratio: {analysis['overall_metrics']['sharpe_ratio']:.3f}")
        print(f"   TP: {analysis['label_percentages']['take_profit_pct']:.1f}%, "
              f"SL: {analysis['label_percentages']['stop_loss_pct']:.1f}%, "
              f"Timeout: {analysis['label_percentages']['timeout_pct']:.1f}%")
    
    print(f"\n🏁 Testing completed!")


if __name__ == "__main__":
    main()
