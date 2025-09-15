#!/usr/bin/env python3
"""
Trading-specific metrics for model evaluation
Implements proper baselines, skill scores, and statistical tests
"""

import numpy as np
import pandas as pd
import cupy as cp
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score
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
