#!/usr/bin/env python3
"""
Trading-specific metrics for model evaluation
Implements proper baselines, skill scores, and statistical tests
"""

import numpy as np
import pandas as pd
import cupy as cp
import cudf
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score
from typing import Any, Dict, List, Tuple, Optional, Union
import warnings

class TradingMetrics:
    """
    Comprehensive trading metrics with proper baselines and statistical tests
    """
    
    def __init__(self, config=None, use_gpu: bool = True):
        """Initialize trading metrics calculator"""
        self.use_gpu = use_gpu and cp is not None
        self.config = config
        
    def compute_comprehensive_metrics(
        self, 
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        volatility: Optional[Union[np.ndarray, pd.Series]] = None,
        fold_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive trading metrics with proper baselines
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            volatility: Volatility estimates for vol-scaling (optional)
            fold_id: Fold identifier for logging
            
        Returns:
            Dictionary with comprehensive metrics
        """
        
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        if volatility is not None and isinstance(volatility, pd.Series):
            volatility = volatility.values
            
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if volatility is not None:
            volatility_clean = volatility[mask]
        else:
            volatility_clean = None
            
        # Basic statistics
        y_std = np.std(y_true_clean)
        y_mean = np.mean(y_true_clean)
        
        # MSE calculations
        mse_model = np.mean((y_true_clean - y_pred_clean) ** 2)
        mse_naive = np.mean(y_true_clean ** 2)  # Baseline: predict 0
        mse_mean = np.mean((y_true_clean - y_mean) ** 2)  # Baseline: predict mean
        
        # Skill scores
        skill_vs_naive = 1 - (mse_model / mse_naive) if mse_naive > 0 else 0
        skill_vs_mean = 1 - (mse_model / mse_mean) if mse_mean > 0 else 0
        
        # RMSE and normalized metrics
        rmse_model = np.sqrt(mse_model)
        nrmse = rmse_model / y_std if y_std > 0 else np.inf
        
        # R² calculations
        r2_vs_naive = skill_vs_naive  # Same as skill vs naive
        r2_vs_mean = skill_vs_mean    # Traditional R²
        
        # Information Coefficient (IC) with block bootstrap
        ic_stats = self._compute_ic_with_bootstrap(y_true_clean, y_pred_clean)
        
        # Directional accuracy
        directional_acc = self._compute_directional_accuracy(y_true_clean, y_pred_clean)
        
        # Diebold-Mariano test vs baseline
        dm_stats = self._diebold_mariano_test(y_true_clean, y_pred_clean, mse_naive)
        
        # Vol-adjusted metrics if volatility provided
        vol_metrics = {}
        if volatility_clean is not None:
            vol_metrics = self._compute_vol_adjusted_metrics(
                y_true_clean, y_pred_clean, volatility_clean
            )
        
        # Compile comprehensive metrics
        metrics = {
            # Basic statistics
            'y_std': y_std,
            'y_mean': y_mean,
            'n_samples': len(y_true_clean),
            
            # MSE metrics
            'mse_model': mse_model,
            'mse_naive': mse_naive,
            'mse_mean': mse_mean,
            
            # Skill scores
            'skill_vs_naive': skill_vs_naive,
            'skill_vs_mean': skill_vs_mean,
            
            # RMSE and normalized
            'rmse_model': rmse_model,
            'nrmse': nrmse,
            
            # R² metrics
            'r2_vs_naive': r2_vs_naive,
            'r2_vs_mean': r2_vs_mean,
            
            # Information Coefficient
            'ic_mean': ic_stats['ic_mean'],
            'ic_std': ic_stats['ic_std'],
            'ic_t_stat': ic_stats['ic_t_stat'],
            'ic_p_value': ic_stats['ic_p_value'],
            'ic_confidence_95': ic_stats['ic_confidence_95'],
            
            # Directional accuracy
            'directional_accuracy': directional_acc,
            
            # Diebold-Mariano
            'dm_statistic': dm_stats['dm_statistic'],
            'dm_p_value': dm_stats['dm_p_value'],
        }
        
        # Add vol-adjusted metrics
        metrics.update(vol_metrics)
        
        # Add fold information
        if fold_id:
            metrics['fold_id'] = fold_id
            
        return metrics
    
    def _compute_ic_with_bootstrap(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        n_bootstrap: int = 1000,
        block_size: int = 50
    ) -> Dict[str, float]:
        """
        Compute Information Coefficient with block bootstrap confidence intervals
        """
        
        # Original IC
        ic_original = stats.spearmanr(y_true, y_pred)[0]
        if np.isnan(ic_original):
            ic_original = 0.0
            
        # Block bootstrap
        n_samples = len(y_true)
        n_blocks = n_samples // block_size
        
        if n_blocks < 10:  # Too few blocks for reliable bootstrap
            return {
                'ic_mean': ic_original,
                'ic_std': 0.0,
                'ic_t_stat': 0.0,
                'ic_p_value': 1.0,
                'ic_confidence_95': (ic_original, ic_original)
            }
        
        ic_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            
            # Reconstruct time series from blocks
            bootstrap_indices = []
            for block_idx in block_indices:
                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, n_samples)
                bootstrap_indices.extend(range(start_idx, end_idx))
            
            # Trim to original length
            bootstrap_indices = bootstrap_indices[:n_samples]
            
            # Calculate IC for bootstrap sample
            y_true_bootstrap = y_true[bootstrap_indices]
            y_pred_bootstrap = y_pred[bootstrap_indices]
            
            ic_bootstrap_sample = stats.spearmanr(y_true_bootstrap, y_pred_bootstrap)[0]
            if not np.isnan(ic_bootstrap_sample):
                ic_bootstrap.append(ic_bootstrap_sample)
        
        if len(ic_bootstrap) == 0:
            ic_bootstrap = [0.0]
            
        ic_bootstrap = np.array(ic_bootstrap)
        
        # Statistics
        ic_mean = np.mean(ic_bootstrap)
        ic_std = np.std(ic_bootstrap)
        
        # T-test vs zero
        if ic_std > 0:
            ic_t_stat = ic_mean / (ic_std / np.sqrt(len(ic_bootstrap)))
            ic_p_value = 2 * (1 - stats.t.cdf(abs(ic_t_stat), len(ic_bootstrap) - 1))
        else:
            ic_t_stat = 0.0
            ic_p_value = 1.0
        
        # Confidence interval
        ic_confidence_95 = np.percentile(ic_bootstrap, [2.5, 97.5])
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_t_stat': ic_t_stat,
            'ic_p_value': ic_p_value,
            'ic_confidence_95': tuple(ic_confidence_95)
        }
    
    def _compute_directional_accuracy(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Compute directional accuracy (sign prediction)"""
        
        # Remove zeros to avoid ambiguity
        mask = (y_true != 0) & (y_pred != 0)
        if mask.sum() == 0:
            return 0.5
            
        y_true_sign = np.sign(y_true[mask])
        y_pred_sign = np.sign(y_pred[mask])
        
        return np.mean(y_true_sign == y_pred_sign)
    
    def _diebold_mariano_test(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        mse_baseline: float
    ) -> Dict[str, float]:
        """
        Diebold-Mariano test comparing model vs naive baseline
        """
        
        # Loss differences
        loss_model = (y_true - y_pred) ** 2
        loss_baseline = y_true ** 2  # Naive baseline
        
        loss_diff = loss_model - loss_baseline
        
        # DM statistic
        d_mean = np.mean(loss_diff)
        
        # Newey-West HAC standard error (simple version)
        n = len(loss_diff)
        h = int(np.floor(4 * (n / 100) ** (2/9)))  # Bandwidth
        
        # Auto-covariances
        gamma_0 = np.var(loss_diff)
        gamma_sum = 0
        
        for j in range(1, h + 1):
            if j < n:
                gamma_j = np.cov(loss_diff[:-j], loss_diff[j:])[0, 1]
                gamma_sum += 2 * gamma_j
        
        # HAC variance
        var_d = (gamma_0 + gamma_sum) / n
        
        if var_d <= 0:
            return {'dm_statistic': 0.0, 'dm_p_value': 1.0}
        
        # DM statistic
        dm_stat = d_mean / np.sqrt(var_d)
        
        # P-value (two-tailed)
        dm_p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return {
            'dm_statistic': dm_stat,
            'dm_p_value': dm_p_value
        }
    
    def _compute_vol_adjusted_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        volatility: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute vol-adjusted metrics: ỹ = y/σ̂_t
        """
        
        # Avoid division by zero
        vol_safe = np.maximum(volatility, 1e-6)
        
        # Vol-adjusted targets and predictions
        y_true_adj = y_true / vol_safe
        y_pred_adj = y_pred / vol_safe
        
        # Vol-adjusted metrics
        mse_vol_adj = np.mean((y_true_adj - y_pred_adj) ** 2)
        mse_naive_vol_adj = np.mean(y_true_adj ** 2)
        
        skill_vol_adj = 1 - (mse_vol_adj / mse_naive_vol_adj) if mse_naive_vol_adj > 0 else 0
        
        rmse_vol_adj = np.sqrt(mse_vol_adj)
        y_std_vol_adj = np.std(y_true_adj)
        nrmse_vol_adj = rmse_vol_adj / y_std_vol_adj if y_std_vol_adj > 0 else np.inf
        
        # Vol-adjusted IC
        ic_vol_adj = stats.spearmanr(y_true_adj, y_pred_adj)[0]
        if np.isnan(ic_vol_adj):
            ic_vol_adj = 0.0
        
        return {
            'mse_vol_adj': mse_vol_adj,
            'skill_vol_adj': skill_vol_adj,
            'rmse_vol_adj': rmse_vol_adj,
            'nrmse_vol_adj': nrmse_vol_adj,
            'ic_vol_adj': ic_vol_adj
        }
    
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict[str, float]:
        """
        Compute classification metrics for triple-barrier labeling
        
        Args:
            y_true: True labels (-1, 0, 1)
            y_pred_proba: Predicted probabilities for each class
            thresholds: Precision thresholds to evaluate
        """
        
        metrics = {}
        
        # Convert to binary (profitable vs not)
        y_true_binary = (y_true == 1).astype(int)
        
        if y_pred_proba.ndim == 2:
            # Multi-class probabilities
            y_pred_proba_positive = y_pred_proba[:, -1]  # Probability of class 1
        else:
            # Assume single probability for positive class
            y_pred_proba_positive = y_pred_proba
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true_binary, y_pred_proba_positive)
        except:
            auc_roc = 0.5
        
        # Precision-Recall curve
        try:
            precision, recall, pr_thresholds = precision_recall_curve(
                y_true_binary, y_pred_proba_positive
            )
            auc_pr = np.trapz(precision, recall)
        except:
            auc_pr = np.mean(y_true_binary)  # Baseline
            precision = np.array([np.mean(y_true_binary)])
            recall = np.array([1.0])
        
        metrics.update({
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        })
        
        # Precision at thresholds
        for threshold in thresholds:
            precision_at_k = self._precision_at_threshold(
                y_true_binary, y_pred_proba_positive, threshold
            )
            metrics[f'precision_at_{int(threshold*100)}pct'] = precision_at_k
        
        return metrics
    
    def _precision_at_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float
    ) -> float:
        """Compute precision at given threshold"""
        
        # Sort by score (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        
        # Take top threshold fraction
        n_selected = max(1, int(len(y_scores) * threshold))
        selected_indices = sorted_indices[:n_selected]
        
        # Compute precision
        if len(selected_indices) == 0:
            return 0.0
        
        return np.mean(y_true[selected_indices])
    
    def log_comprehensive_metrics(
        self,
        metrics: Dict[str, float],
        context: str = "",
        logger=None
    ) -> None:
        """Log comprehensive metrics in a structured way"""
        
        # Use internal logger or provided logger
        log_func = logger.info if logger else print
        prefix = f"[TradingMetrics]{' ' + context if context else ''}"
        
        log_func(f"{prefix} 📊 COMPREHENSIVE EVALUATION METRICS:")
        
        # Basic statistics
        log_func(f"{prefix}   📈 Dataset: n={metrics.get('n_samples', 0):,}, "
                    f"std(y)={metrics.get('y_std', 0):.6f}, "
                    f"mean(y)={metrics.get('y_mean', 0):.6f}")
        
        # MSE baselines
        log_func(f"{prefix}   🎯 MSE: model={metrics.get('mse_model', 0):.8f}, "
                    f"naive={metrics.get('mse_naive', 0):.8f}, "
                    f"mean={metrics.get('mse_mean', 0):.8f}")
        
        # Skill scores
        log_func(f"{prefix}   💪 Skill: vs_naive={metrics.get('skill_vs_naive', 0):.4f}, "
                    f"vs_mean={metrics.get('skill_vs_mean', 0):.4f}")
        
        # RMSE metrics
        log_func(f"{prefix}   📐 RMSE: {metrics.get('rmse_model', 0):.6f}, "
                    f"NRMSE={metrics.get('nrmse', 0):.4f}")
        
        # Information Coefficient
        ic_mean = metrics.get('ic_mean', 0)
        ic_std = metrics.get('ic_std', 0)
        ic_p_value = metrics.get('ic_p_value', 1)
        log_func(f"{prefix}   🔗 IC: {ic_mean:.4f}±{ic_std:.4f} "
                    f"(t={metrics.get('ic_t_stat', 0):.2f}, p={ic_p_value:.4f})")
        
        # Statistical significance
        significance = "***" if ic_p_value < 0.01 else "**" if ic_p_value < 0.05 else "*" if ic_p_value < 0.1 else ""
        if significance:
            log_func(f"{prefix}   ⭐ Statistical Significance: {significance}")
        
        # Diebold-Mariano
        dm_stat = metrics.get('dm_statistic', 0)
        dm_p = metrics.get('dm_p_value', 1)
        log_func(f"{prefix}   🔬 Diebold-Mariano: stat={dm_stat:.3f}, p={dm_p:.4f}")
        
        # Directional accuracy
        log_func(f"{prefix}   🎲 Directional Accuracy: {metrics.get('directional_accuracy', 0.5):.4f}")
        
        # Vol-adjusted metrics (if available)
        if 'skill_vol_adj' in metrics:
            log_func(f"{prefix}   🌪️ Vol-Adjusted: skill={metrics.get('skill_vol_adj', 0):.4f}, "
                        f"IC={metrics.get('ic_vol_adj', 0):.4f}, "
                        f"NRMSE={metrics.get('nrmse_vol_adj', 0):.4f}")
        
        # Classification metrics (if available)
        if 'auc_roc' in metrics:
            log_func(f"{prefix}   🎯 Classification: AUC-ROC={metrics.get('auc_roc', 0.5):.4f}, "
                        f"AUC-PR={metrics.get('auc_pr', 0):.4f}")
            
            # Precision at thresholds
            precision_metrics = [k for k in metrics.keys() if k.startswith('precision_at_')]
            if precision_metrics:
                precision_str = ", ".join([f"{k.split('_')[-1]}={metrics[k]:.3f}" for k in precision_metrics])
                log_func(f"{prefix}   📊 Precision@: {precision_str}")


class GPUPostTrainingMetrics:
    """
    Post-training metrics fully computed on GPU using RAPIDS.
    
    Key improvements for numerical stability and correctness:
    - Rank-based buckets to avoid Q5-Q1=0 from tied predictions
    - Float64 precision for Sortino, MDD, and bucket calculations  
    - Standardized trade definition: 1 trade = signal change (entry OR exit)
    - Consistent MDD calculation using cuDF.cummax() for running maximum
    """

    def __init__(
        self,
        cost_per_trade: float = 0.0,
        annual_factor: float = float(np.sqrt(252 * 24)),
        window_size: int = 480,
        bucket_count: int = 5
    ) -> None:
        self.cost_per_trade = float(cost_per_trade)
        self.annual_factor = float(annual_factor)
        self.window_size = max(int(window_size), 1)
        self.bucket_count = max(int(bucket_count), 2)

    def compute_metrics(self, y_true, y_pred) -> Dict[str, Any]:
        """Compute global and windowed GPU metrics."""

        y_true_cp, y_pred_cp = self._prepare_arrays(y_true, y_pred)
        n_obs = int(y_true_cp.size)
        if n_obs == 0:
            return {
                'global': self._empty_global(),
                'windows': [],
                'stability': self._empty_stability(),
                'bucket_means': {}
            }

        ic_value = self._spearman_ic(y_pred_cp, y_true_cp)
        hit_rate, z_score, hit_vector = self._hit_rate_and_vector(y_pred_cp, y_true_cp)
        pnl_result = self._pnl_metrics(y_pred_cp, y_true_cp)
        bucket_result = self._bucket_stats(y_pred_cp, y_true_cp)

        window_metrics, window_summary = self._window_metrics(
            y_pred_cp,
            y_true_cp,
            pnl_result['pnl_net'],
            hit_vector
        )

        global_metrics = {
            'IC': float(ic_value),
            'ICIR': float(window_summary['icir']),
            'hit': float(hit_rate),
            'z_score': float(z_score),
            'sharpe_liq': float(pnl_result['sharpe']),
            'sortino_liq': float(pnl_result['sortino']),
            'mdd_liq': float(pnl_result['mdd']),
            'q5_minus_q1': float(bucket_result['spread']),
            'tstat_q5q1': float(bucket_result['tstat']),
            'bucket_monotonicity': float(bucket_result.get('bucket_monotonicity', 0.0)),
            'turnover': float(pnl_result['turnover']),
            'trades_total': int(pnl_result['trades_total'])
        }

        stability = {
            'ic_positive_pct': float(window_summary['ic_positive_pct']),
            'sharpe_positive_pct': float(window_summary['sharpe_positive_pct']),
            'ic_mean': float(window_summary['ic_mean']),
            'icir': float(window_summary['icir']),
            'worst_sharpe': float(window_summary['worst_sharpe']),
            'worst_sharpe_window': int(window_summary['worst_sharpe_window']) if window_summary['worst_sharpe_window'] is not None else -1,
            'worst_mdd': float(window_summary['worst_mdd']),
            'worst_mdd_window': int(window_summary['worst_mdd_window']) if window_summary['worst_mdd_window'] is not None else -1
        }

        return {
            'global': global_metrics,
            'windows': window_metrics,
            'stability': stability,
            'bucket_means': bucket_result['bucket_means']
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_arrays(self, y_true, y_pred) -> Tuple[cp.ndarray, cp.ndarray]:
        y_true_cp = self._to_cupy_array(y_true)
        y_pred_cp = self._to_cupy_array(y_pred)
        mask = cp.isfinite(y_true_cp) & cp.isfinite(y_pred_cp)
        y_true_cp = y_true_cp[mask]
        y_pred_cp = y_pred_cp[mask]
        return y_true_cp, y_pred_cp

    def _to_cupy_array(self, data) -> cp.ndarray:
        """Convert data to CuPy array with default float32 precision."""
        if isinstance(data, cp.ndarray):
            return data.astype(cp.float32, copy=False)
        if hasattr(data, 'to_cupy'):
            return data.to_cupy().astype(cp.float32, copy=False)
        return cp.asarray(data, dtype=cp.float32)
    
    def _to_cupy_array_f64(self, data) -> cp.ndarray:
        """Convert data to CuPy array with float64 precision for sensitive calculations."""
        if isinstance(data, cp.ndarray):
            return data.astype(cp.float64, copy=False)
        if hasattr(data, 'to_cupy'):
            return data.to_cupy().astype(cp.float64, copy=False)
        return cp.asarray(data, dtype=cp.float64)

    def _spearman_ic(self, y_pred_cp: cp.ndarray, y_true_cp: cp.ndarray) -> float:
        if y_pred_cp.size == 0:
            return 0.0
        pred_rank = cudf.Series(y_pred_cp).rank(method="average")
        true_rank = cudf.Series(y_true_cp).rank(method="average")
        corr_matrix = cp.corrcoef(pred_rank.to_cupy(), true_rank.to_cupy())
        ic_value = float(corr_matrix[0, 1].item()) if not cp.isnan(corr_matrix[0, 1]) else 0.0
        return ic_value

    def _hit_rate_and_vector(
        self,
        y_pred_cp: cp.ndarray,
        y_true_cp: cp.ndarray
    ) -> Tuple[float, float, cp.ndarray]:
        hits = (cp.sign(y_pred_cp) == cp.sign(y_true_cp)).astype(cp.float32)
        hit_rate = float(hits.mean().item()) if hits.size else 0.5
        n = hits.size if hits.size else 1
        z_score = float(((hit_rate - 0.5) / cp.sqrt(0.25 / n)).item()) if n > 0 else 0.0
        return hit_rate, z_score, hits

    def _pnl_metrics(
        self,
        y_pred_cp: cp.ndarray,
        y_true_cp: cp.ndarray
    ) -> Dict[str, Any]:
        pnl = y_pred_cp * y_true_cp
        signal = cp.sign(y_pred_cp)
        
        # Standardized trade definition: 1 trade = signal change (entry OR exit)
        shifted = cp.concatenate([signal[:1], signal[:-1]])
        trades_indicator = (signal != shifted).astype(cp.float64)  # Use float64 for precision
        trades_indicator[0] = 0.0  # No trade on first observation
        
        pnl_net = pnl - trades_indicator * self.cost_per_trade
        
        # MDD calculation in float64 for consistency and precision
        equity_curve = cp.cumsum(pnl_net.astype(cp.float64))  # Force float64 for equity
        running_max = cudf.Series(equity_curve).cummax().to_cupy()  # Use cuDF cummax
        drawdown = running_max - equity_curve
        mdd = float(drawdown.max().item()) if drawdown.size else 0.0

        # Sharpe calculation (maintain existing precision for compatibility)
        mean_pnl = pnl_net.mean() if pnl_net.size else cp.float32(0.0)
        std_pnl = pnl_net.std() + 1e-12
        sharpe = float((mean_pnl / std_pnl * self.annual_factor).item())

        # Sortino with float64 stabilization in denominator
        downside = cp.where(pnl_net < 0, pnl_net, 0.0).astype(cp.float64)  # Force float64
        downside_std = cp.sqrt((downside ** 2).mean()) + 1e-12
        sortino = float((mean_pnl / downside_std * self.annual_factor).item())

        # Turnover/Trades: standardized definition
        trades_total = int(trades_indicator.sum().item())
        turnover = float((trades_indicator.sum() / max(pnl_net.size, 1)).item()) if pnl_net.size else 0.0

        return {
            'pnl_net': pnl_net,
            'sharpe': sharpe,
            'sortino': sortino,
            'mdd': mdd,
            'trades_total': trades_total,
            'turnover': turnover
        }

    def _compute_rank_buckets_gpu(
        self,
        df: cudf.DataFrame,
        ypred_col: str = "y_pred",
        ytrue_col: str = "y_true",
        q: int = 5,
        use_jitter: bool = False,
        jitter_eps: float = 1e-9,
        jitter_seed: int = 7,
    ):
        """
        Calcula buckets por rank de y_pred (Q1→Qq), média de y_true por bucket,
        spread (Qq - Q1) e t-stat do spread. 100% em GPU.

        Parâmetros:
          - df: cudf.DataFrame com colunas y_pred e y_true
          - q: número de quantis (5 = quintis)
          - use_jitter: se True, aplica jitter minúsculo e determinístico apenas para o corte
          - jitter_eps: escala do jitter (multiplica o std de y_pred)
          - jitter_seed: seed fixa do jitter (reprodutível)

        Retorna:
          - spread_q: float (Qq - Q1)
          - tstat_spread: float
          - bucket_means: cudf.Series (média de y_true em cada bucket, ordenado)
          - bucket_stats: cudf.DataFrame com colunas ['mean','var','count'] por bucket
        """

        # ---- Preparação (float64 para estabilidade) ----
        df = df.copy(deep=False)
        df[ypred_col] = df[ypred_col].astype("float64")
        df[ytrue_col] = df[ytrue_col].astype("float64")

        # ---- Rank-buckets (evita empates sem mudar o sinal) ----
        # rank(method='first') é determinístico; também dá para usar 'average'
        ranks = df[ypred_col].rank(method="first")  # GPU
        if use_jitter:
            # jitter minúsculo e determinístico APENAS para o corte (não para o PnL)
            cp.random.seed(jitter_seed)
            yp_cp = df[ypred_col].to_cupy()
            std_yp = float(df[ypred_col].std())
            eps = jitter_eps * (std_yp if std_yp > 0 else 1.0)
            yp_cut = yp_cp + eps * cp.random.standard_normal(yp_cp.shape[0])
            # corta pelos ranks do vetor com jitter
            # (mantemos 'ranks' em cuDF para qcut)
            ranks = cudf.Series(yp_cut).rank(method="first")

        # Corta quantis em GPU
        df["bucket"] = cudf.qcut(ranks, q)  # IntervalIndex ordenado

        # ---- Agregações por bucket (GPU) ----
        # means, vars, counts 100% em GPU
        bucket_stats = df.groupby("bucket")[ytrue_col].agg(["mean", "var", "count"])
        # garantir ordenação de buckets
        bucket_stats = bucket_stats.sort_index()

        # Séries auxiliares (GPU)
        mean_s = bucket_stats["mean"]
        var_s = bucket_stats["var"]
        cnt_s = bucket_stats["count"]

        # ---- Spread Qq - Q1 e t-stat (GPU) ----
        # Q1 = primeiro bucket; Qq = último bucket
        q1_mean = mean_s.iloc[0].astype("float64")
        qk_mean = mean_s.iloc[-1].astype("float64")
        spread_q = float(qk_mean - q1_mean)

        # Erro-padrão do spread (independência aproximada): sqrt(var(Qq)/nq + var(Q1)/n1)
        v1 = var_s.iloc[0].astype("float64").to_cupy()
        n1 = cnt_s.iloc[0].astype("float64").to_cupy()
        vk = var_s.iloc[-1].astype("float64").to_cupy()
        nk = cnt_s.iloc[-1].astype("float64").to_cupy()

        se = cp.sqrt(v1 / (n1 + 1e-12) + vk / (nk + 1e-12)) + 1e-12
        tstat_spread = float((qk_mean - q1_mean).astype("float64").to_cupy() / se)

        # ---- Saídas ----
        bucket_means = mean_s  # já em GPU e ordenado
        return spread_q, tstat_spread, bucket_means, bucket_stats

    def _bucket_stats(
        self,
        y_pred_cp: cp.ndarray,
        y_true_cp: cp.ndarray
    ) -> Dict[str, Any]:
        try:
            # Create DataFrame for rank-buckets computation
            frame = cudf.DataFrame({'y_pred': y_pred_cp, 'y_true': y_true_cp})
            frame = frame.dropna()
            if len(frame) < self.bucket_count:
                return {'spread': 0.0, 'tstat': 0.0, 'bucket_means': {}}

            # Use plug-and-play rank-buckets implementation
            spread_q, tstat_spread, bucket_means, bucket_stats = self._compute_rank_buckets_gpu(
                df=frame,
                ypred_col="y_pred",
                ytrue_col="y_true",
                q=self.bucket_count,
                use_jitter=False  # Use rank-buckets without jitter by default
            )

            # Convert bucket_means to dictionary format
            bucket_means_dict = {
                f"Q{i + 1}": float(bucket_means.iloc[i])
                for i in range(len(bucket_means))
            }

            # Optional: Check bucket monotonicity (for validation)
            try:
                idx = cudf.Series(cp.arange(1, len(bucket_means) + 1, dtype=cp.float64))
                rho = cp.corrcoef(idx.to_cupy(), bucket_means.astype('float64').to_cupy())[0, 1]
                bucket_monotonicity = float(rho)
            except:
                bucket_monotonicity = 0.0

            return {
                'spread': spread_q,
                'tstat': tstat_spread,
                'bucket_means': bucket_means_dict,
                'bucket_monotonicity': bucket_monotonicity
            }

        except Exception as e:
            # Log the specific error for debugging
            import traceback
            print(f"Bucket stats error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {'spread': 0.0, 'tstat': 0.0, 'bucket_means': {}, 'bucket_monotonicity': 0.0}

    def _window_metrics(
        self,
        y_pred_cp: cp.ndarray,
        y_true_cp: cp.ndarray,
        pnl_net_cp: cp.ndarray,
        hit_vector: cp.ndarray
    ) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
        window_metrics: List[Dict[str, float]] = []
        ic_values: List[float] = []
        sharpe_values: List[float] = []
        mdd_values: List[float] = []
        hit_values: List[float] = []

        total_obs = int(y_pred_cp.size)
        window_size = self.window_size
        n_windows = (total_obs + window_size - 1) // window_size

        for idx in range(n_windows):
            start = idx * window_size
            end = min((idx + 1) * window_size, total_obs)
            if end - start < 2:
                continue

            pred_slice = y_pred_cp[start:end]
            true_slice = y_true_cp[start:end]
            pnl_slice = pnl_net_cp[start:end]
            hits_slice = hit_vector[start:end]

            ic_val = self._spearman_ic(pred_slice, true_slice)
            sharpe_val = self._sharpe_from_slice(pnl_slice)
            mdd_val = self._mdd_from_slice(pnl_slice)
            hit_val = float(hits_slice.mean().item()) if hits_slice.size else 0.0
            bucket_slice = self._bucket_stats(pred_slice, true_slice)

            ic_values.append(ic_val)
            sharpe_values.append(sharpe_val)
            mdd_values.append(mdd_val)
            hit_values.append(hit_val)

            window_metrics.append({
                'window_id': idx,
                'IC': ic_val,
                'hit': hit_val,
                'sharpe_liq': sharpe_val,
                'mdd_liq': mdd_val,
                'q5_minus_q1': bucket_slice['spread']
            })

        if window_metrics:
            ic_cp = cp.asarray(ic_values, dtype=cp.float32)
            sharpe_cp = cp.asarray(sharpe_values, dtype=cp.float32)
            mdd_cp = cp.asarray(mdd_values, dtype=cp.float32)

            ic_mean = float(ic_cp.mean().item())
            ic_std = float(ic_cp.std().item())
            icir = ic_mean / (ic_std + 1e-12)
            ic_positive_pct = float(((ic_cp > 0).mean()).item())
            sharpe_positive_pct = float(((sharpe_cp > 0).mean()).item())

            worst_sharpe_idx = int(cp.argmin(sharpe_cp).item())
            worst_mdd_idx = int(cp.argmax(mdd_cp).item())
            worst_sharpe_val = float(sharpe_cp[worst_sharpe_idx].item())
            worst_mdd_val = float(mdd_cp[worst_mdd_idx].item())
        else:
            icir = 0.0
            ic_mean = 0.0
            ic_positive_pct = 0.0
            sharpe_positive_pct = 0.0
            worst_sharpe_idx = None
            worst_mdd_idx = None
            worst_sharpe_val = 0.0
            worst_mdd_val = 0.0

        summary = {
            'icir': icir,
            'ic_mean': ic_mean,
            'ic_positive_pct': ic_positive_pct,
            'sharpe_positive_pct': sharpe_positive_pct,
            'worst_sharpe': worst_sharpe_val,
            'worst_sharpe_window': worst_sharpe_idx,
            'worst_mdd': worst_mdd_val,
            'worst_mdd_window': worst_mdd_idx
        }

        return window_metrics, summary

    def _sharpe_from_slice(self, pnl_slice: cp.ndarray) -> float:
        if pnl_slice.size == 0:
            return 0.0
        mean_pnl = pnl_slice.mean()
        std_pnl = pnl_slice.std() + 1e-12
        return float((mean_pnl / std_pnl * self.annual_factor).item())

    def _mdd_from_slice(self, pnl_slice: cp.ndarray) -> float:
        if pnl_slice.size == 0:
            return 0.0
        # Use float64 for consistency with main MDD calculation
        equity = cp.cumsum(pnl_slice.astype(cp.float64))
        running_max = cudf.Series(equity).cummax().to_cupy()  # Use cuDF cummax
        drawdown = running_max - equity
        return float(drawdown.max().item()) if drawdown.size else 0.0

    def _empty_global(self) -> Dict[str, Any]:
        return {
            'IC': 0.0,
            'ICIR': 0.0,
            'hit': 0.0,
            'z_score': 0.0,
            'sharpe_liq': 0.0,
            'sortino_liq': 0.0,
            'mdd_liq': 0.0,
            'q5_minus_q1': 0.0,
            'tstat_q5q1': 0.0,
            'turnover': 0.0,
            'trades_total': 0
        }

    def _empty_stability(self) -> Dict[str, Any]:
        return {
            'ic_positive_pct': 0.0,
            'sharpe_positive_pct': 0.0,
            'ic_mean': 0.0,
            'icir': 0.0,
            'worst_sharpe': 0.0,
            'worst_sharpe_window': -1,
            'worst_mdd': 0.0,
            'worst_mdd_window': -1
        }


def log_comprehensive_metrics(
    metrics: Dict[str, float],
    logger,
    prefix: str = "[TradingMetrics]"
) -> None:
    """Log comprehensive metrics in a structured way"""
    
    logger.info(f"{prefix} 📊 COMPREHENSIVE EVALUATION METRICS:")
    
    # Basic statistics
    logger.info(f"{prefix}   📈 Dataset: n={metrics.get('n_samples', 0):,}, "
                f"std(y)={metrics.get('y_std', 0):.6f}, "
                f"mean(y)={metrics.get('y_mean', 0):.6f}")
    
    # MSE baselines
    logger.info(f"{prefix}   🎯 MSE: model={metrics.get('mse_model', 0):.8f}, "
                f"naive={metrics.get('mse_naive', 0):.8f}, "
                f"mean={metrics.get('mse_mean', 0):.8f}")
    
    # Skill scores
    logger.info(f"{prefix}   💪 Skill: vs_naive={metrics.get('skill_vs_naive', 0):.4f}, "
                f"vs_mean={metrics.get('skill_vs_mean', 0):.4f}")
    
    # RMSE metrics
    logger.info(f"{prefix}   📐 RMSE: {metrics.get('rmse_model', 0):.6f}, "
                f"NRMSE={metrics.get('nrmse', 0):.4f}")
    
    # Information Coefficient
    ic_mean = metrics.get('ic_mean', 0)
    ic_std = metrics.get('ic_std', 0)
    ic_p_value = metrics.get('ic_p_value', 1)
    logger.info(f"{prefix}   🔗 IC: {ic_mean:.4f}±{ic_std:.4f} "
                f"(t={metrics.get('ic_t_stat', 0):.2f}, p={ic_p_value:.4f})")
    
    # Statistical significance
    significance = "***" if ic_p_value < 0.01 else "**" if ic_p_value < 0.05 else "*" if ic_p_value < 0.1 else ""
    if significance:
        logger.info(f"{prefix}   ⭐ Statistical Significance: {significance}")
    
    # Diebold-Mariano
    dm_stat = metrics.get('dm_statistic', 0)
    dm_p = metrics.get('dm_p_value', 1)
    logger.info(f"{prefix}   🔬 Diebold-Mariano: stat={dm_stat:.3f}, p={dm_p:.4f}")
    
    # Directional accuracy
    logger.info(f"{prefix}   🎲 Directional Accuracy: {metrics.get('directional_accuracy', 0.5):.4f}")
    
    # Vol-adjusted metrics (if available)
    if 'skill_vol_adj' in metrics:
        logger.info(f"{prefix}   🌪️ Vol-Adjusted: skill={metrics.get('skill_vol_adj', 0):.4f}, "
                    f"IC={metrics.get('ic_vol_adj', 0):.4f}, "
                    f"NRMSE={metrics.get('nrmse_vol_adj', 0):.4f}")
    
    # Classification metrics (if available)
    if 'auc_roc' in metrics:
        logger.info(f"{prefix}   🎯 Classification: AUC-ROC={metrics.get('auc_roc', 0.5):.4f}, "
                    f"AUC-PR={metrics.get('auc_pr', 0):.4f}")
        
        # Precision at thresholds
        precision_metrics = [k for k in metrics.keys() if k.startswith('precision_at_')]
        if precision_metrics:
            precision_str = ", ".join([f"{k.split('_')[-1]}={metrics[k]:.3f}" for k in precision_metrics])
            logger.info(f"{prefix}   📊 Precision@: {precision_str}")
