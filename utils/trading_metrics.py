#!/usr/bin/env python3
"""
Trading-specific metrics for model evaluation
Implements proper baselines, skill scores, and statistical tests
"""

import numpy as np
import pandas as pd
import cupy as cp
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Optional, Union
import warnings

class TradingMetrics:
    """
    Comprehensive trading metrics with proper baselines and statistical tests
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize trading metrics calculator"""
        self.use_gpu = use_gpu and cp is not None
        
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
            
        # GPU support for heavy computations
        use_gpu_here = (self.use_gpu and isinstance(y_true_clean, np.ndarray) and y_true_clean.size > 1e6)
        xp = cp if use_gpu_here else np
        
        # Convert to GPU arrays if needed
        if use_gpu_here:
            yt = xp.asarray(y_true_clean)
            yp = xp.asarray(y_pred_clean)
        else:
            yt = y_true_clean
            yp = y_pred_clean
        
        # Basic statistics
        y_std = float(xp.std(yt))
        y_mean = float(xp.mean(yt))
        
        # MSE calculations
        mse_model = float(xp.mean((yt - yp) ** 2))
        mse_naive = float(xp.mean(yt ** 2))  # Baseline: predict 0
        mse_mean = float(xp.mean((yt - y_mean) ** 2))  # Baseline: predict mean
        
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
        dm_stats = self._diebold_mariano_test(y_true_clean, y_pred_clean)
        
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
        block_size: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, float]:
        """
        IC com circular block bootstrap e block size adaptativo
        """
        rng = np.random.default_rng(random_state)

        ic_original = stats.spearmanr(y_true, y_pred)[0]
        if np.isnan(ic_original):
            ic_original = 0.0

        n = len(y_true)
        if block_size is None:
            block_size = max(20, int(round(1.5 * n ** (1/3))))  # regra de bolso
        n_blocks = max(1, n // block_size)
        if n_blocks < 10:
            return {'ic_mean': ic_original, 'ic_std': 0.0, 'ic_t_stat': 0.0,
                    'ic_p_value': 1.0, 'ic_confidence_95': (ic_original, ic_original)}

        # circular bootstrap: duplica a série para evitar bordas
        y_true2 = np.concatenate([y_true, y_true])
        y_pred2 = np.concatenate([y_pred, y_pred])

        ic_bs = []
        for _ in range(n_bootstrap):
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
            idx = idx[:n]  # trim
            ytb = y_true2[idx]
            ypb = y_pred2[idx]
            r = stats.spearmanr(ytb, ypb)[0]
            if np.isfinite(r):
                ic_bs.append(r)

        if not ic_bs:
            ic_bs = [0.0]

        ic_bs = np.asarray(ic_bs)
        ic_mean, ic_std = float(ic_bs.mean()), float(ic_bs.std(ddof=0))
        if ic_std > 0:
            ic_t = ic_mean / (ic_std / np.sqrt(len(ic_bs)))
            ic_p = 2 * (1 - stats.t.cdf(abs(ic_t), len(ic_bs) - 1))
        else:
            ic_t, ic_p = 0.0, 1.0
        ic_ci = tuple(np.percentile(ic_bs, [2.5, 97.5]).astype(float))
        return {'ic_mean': ic_mean, 'ic_std': ic_std, 'ic_t_stat': float(ic_t),
                'ic_p_value': float(ic_p), 'ic_confidence_95': ic_ci}
    
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
    
    def _diebold_mariano_test(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        DM test: modelo vs baseline ingênuo (y_hat=0) com HAC (Newey–West, Bartlett)
        """
        # Losses
        e_model = (y_true - y_pred) ** 2
        e_base  = (y_true - 0.0) ** 2
        d_t = e_model - e_base  # loss differential

        n = len(d_t)
        if n < 50:  # pouca amostra: DM instável
            return {'dm_statistic': 0.0, 'dm_p_value': 1.0}

        d_mean = np.mean(d_t)

        # Bandwidth (Parzen/Newey-West plug-in simples)
        h = int(np.floor(4 * (n / 100.0) ** (2.0/9.0)))
        h = max(1, min(h, n - 1))

        # HAC variance with Bartlett weights
        gamma_0 = np.var(d_t, ddof=0)
        gamma_sum = 0.0
        for j in range(1, h + 1):
            cov_j = np.cov(d_t[:-j], d_t[j:], ddof=0)[0, 1]
            w_j = 1.0 - j / (h + 1.0)
            gamma_sum += 2.0 * w_j * cov_j

        var_d = (gamma_0 + gamma_sum) / n
        if var_d <= 0 or not np.isfinite(var_d):
            return {'dm_statistic': 0.0, 'dm_p_value': 1.0}

        dm_stat = d_mean / np.sqrt(var_d)
        dm_p = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
        return {'dm_statistic': float(dm_stat), 'dm_p_value': float(dm_p)}
    
    def _compute_vol_adjusted_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        volatility: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute vol-adjusted metrics: ỹ = y/σ̂_t with winsorization
        """
        
        # Avoid division by zero
        vol_safe = np.maximum(volatility, 1e-6)
        
        # Vol-adjusted targets and predictions
        y_true_adj = y_true / vol_safe
        y_pred_adj = y_pred / vol_safe
        
        # Winsorization to reduce outliers (0.1% and 99.9% quantiles)
        q_low, q_high = np.percentile(y_true_adj, [0.1, 99.9])
        y_true_adj = np.clip(y_true_adj, q_low, q_high)
        y_pred_adj = np.clip(y_pred_adj, q_low, q_high)
        
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
        top_fracs: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict[str, float]:
        """
        Métricas de classificação para triple-barrier.
        y_true: {-1,0,1}, positivo = 1
        top_fracs: frações do TOP-K (não limiar de prob).
        """
        y_true_bin = (y_true == 1).astype(int)
        p_pos = y_pred_proba[:, -1] if y_pred_proba.ndim == 2 else y_pred_proba

        try:
            auc_roc = roc_auc_score(y_true_bin, p_pos)
        except Exception:
            auc_roc = 0.5

        try:
            auc_pr = average_precision_score(y_true_bin, p_pos)
        except Exception:
            auc_pr = float(y_true_bin.mean())

        # Brier score (mean squared error of probabilities)
        try:
            brier_score = np.mean((y_true_bin - p_pos) ** 2)
        except Exception:
            brier_score = 0.25  # Worst case for binary classification

        # Expected Calibration Error (ECE)
        try:
            ece = self._compute_ece(y_true_bin, p_pos)
        except Exception:
            ece = 0.0

        out = {
            'auc_roc': float(auc_roc), 
            'auc_pr': float(auc_pr),
            'brier_score': float(brier_score),
            'ece': float(ece)
        }
        for f in top_fracs:
            out[f'precision_at_top_{int(f*100)}pct'] = float(self._precision_at_threshold(y_true_bin, p_pos, f))
        return out
    
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

    def _compute_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine which samples are in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def _fmt(x, fmt="{:.6f}"):
    """Format value, handling nan/inf"""
    return "nan" if not np.isfinite(x) else fmt.format(x)

def log_comprehensive_metrics(
    metrics: Dict[str, float],
    logger,
    prefix: str = "[TradingMetrics]"
) -> None:
    """Log comprehensive metrics in a structured way"""
    
    logger.info(f"{prefix} 📊 COMPREHENSIVE EVALUATION METRICS:")
    
    # Basic statistics
    logger.info(f"{prefix}   📈 Dataset: n={metrics.get('n_samples', 0):,}, "
                f"std(y)={_fmt(metrics.get('y_std', 0))}, "
                f"mean(y)={_fmt(metrics.get('y_mean', 0))}")
    
    # MSE baselines
    logger.info(f"{prefix}   🎯 MSE: model={_fmt(metrics.get('mse_model', 0), '{:.8f}')}, "
                f"naive={_fmt(metrics.get('mse_naive', 0), '{:.8f}')}, "
                f"mean={_fmt(metrics.get('mse_mean', 0), '{:.8f}')}")
    
    # Skill scores
    logger.info(f"{prefix}   💪 Skill: vs_naive={_fmt(metrics.get('skill_vs_naive', 0), '{:.4f}')}, "
                f"vs_mean={_fmt(metrics.get('skill_vs_mean', 0), '{:.4f}')}")
    
    # RMSE metrics
    logger.info(f"{prefix}   📐 RMSE: {_fmt(metrics.get('rmse_model', 0))}, "
                f"NRMSE={_fmt(metrics.get('nrmse', 0), '{:.4f}')}")
    
    # Information Coefficient
    ic_mean = metrics.get('ic_mean', 0)
    ic_std = metrics.get('ic_std', 0)
    ic_p_value = metrics.get('ic_p_value', 1)
    logger.info(f"{prefix}   🔗 IC: {_fmt(ic_mean, '{:.4f}')}±{_fmt(ic_std, '{:.4f}')} "
                f"(t={_fmt(metrics.get('ic_t_stat', 0), '{:.2f}')}, p={_fmt(ic_p_value, '{:.4f}')})")
    
    # Statistical significance
    significance = "***" if ic_p_value < 0.01 else "**" if ic_p_value < 0.05 else "*" if ic_p_value < 0.1 else ""
    if significance:
        logger.info(f"{prefix}   ⭐ Statistical Significance: {significance}")
    
    # Diebold-Mariano
    dm_stat = metrics.get('dm_statistic', 0)
    dm_p = metrics.get('dm_p_value', 1)
    logger.info(f"{prefix}   🔬 Diebold-Mariano: stat={_fmt(dm_stat, '{:.3f}')}, p={_fmt(dm_p, '{:.4f}')}")
    
    # Directional accuracy
    logger.info(f"{prefix}   🎲 Directional Accuracy: {_fmt(metrics.get('directional_accuracy', 0.5), '{:.4f}')}")
    
    # Vol-adjusted metrics (if available)
    if 'skill_vol_adj' in metrics:
        logger.info(f"{prefix}   🌪️ Vol-Adjusted: skill={_fmt(metrics.get('skill_vol_adj', 0), '{:.4f}')}, "
                    f"IC={_fmt(metrics.get('ic_vol_adj', 0), '{:.4f}')}, "
                    f"NRMSE={_fmt(metrics.get('nrmse_vol_adj', 0), '{:.4f}')}")
    
    # Classification metrics (if available)
    if 'auc_roc' in metrics:
        logger.info(f"{prefix}   🎯 Classification: AUC-ROC={_fmt(metrics.get('auc_roc', 0.5), '{:.4f}')}, "
                    f"AUC-PR={_fmt(metrics.get('auc_pr', 0), '{:.4f}')}")
        
        # Calibration metrics
        if 'brier_score' in metrics or 'ece' in metrics:
            logger.info(f"{prefix}   🎲 Calibration: Brier={_fmt(metrics.get('brier_score', 0.25), '{:.4f}')}, "
                        f"ECE={_fmt(metrics.get('ece', 0), '{:.4f}')}")
        
        # Precision at thresholds
        precision_metrics = [k for k in metrics.keys() if k.startswith('precision_at_')]
        if precision_metrics:
            precision_str = ", ".join([f"{k.split('_')[-1]}={_fmt(metrics[k], '{:.3f}')}" for k in precision_metrics])
            logger.info(f"{prefix}   📊 Precision@: {precision_str}")
