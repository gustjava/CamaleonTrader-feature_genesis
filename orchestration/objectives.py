"""Optuna objectives for Study A (preprocessing search) and Study B (model tuning).

[IMPLEMENTAÇÃO FINAL] Pipeline de Otimização Optuna com Template Avançado

Esta implementação fornece um pipeline robusto para seleção de features em trading
que inclui:
- Simulação de PnL líquido com custos de transação
- Validação temporal rigorosa (TimeSeriesSplit)
- Métricas de trading especializadas (Sharpe, Information Coefficient, Max Drawdown)
- Otimização multi-objetivo (Fronteira de Pareto)
- Análise de estabilidade de features
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Suppress GPU memory warnings globally for this module
warnings.filterwarnings('ignore', message='.*less than 75% GPU memory available.*')
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from omegaconf import DictConfig

try:  # Optional dependency used when Hydra is installed.
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DictConfig = Any  # type: ignore

    class _OmegaConfShim:  # pragma: no cover - fallback utility
        @staticmethod
        def to_container(cfg: Any, **_: Any) -> Dict[str, Any]:
            if isinstance(cfg, dict):
                return cfg
            if hasattr(cfg, '__dict__'):
                return dict(cfg.__dict__)
            return {}

    OmegaConf = _OmegaConfShim()  # type: ignore


try:  # Scikit-learn is required for the surrogate pipeline.
    from sklearn.base import BaseEstimator, TransformerMixin, clone
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SKLEARN_AVAILABLE = False

try:  # LightGBM for advanced modeling
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    LIGHTGBM_AVAILABLE = False

# Constants for trading-specific calculations
ANN_FACTOR = np.sqrt(252 * 78)  # Para barras de 5 minutos (252 dias * 78 barras/dia)
# ANN_FACTOR = np.sqrt(252)     # Para dados diários
# ANN_FACTOR = np.sqrt(252 * 24) # Para dados horários

RANDOM_SEED = 42


if TYPE_CHECKING:  # pragma: no cover
    import optuna


def _safe_json_convert(obj: Any) -> Any:
    """
    Recursively and safely convert an object to be JSON serializable.
    Handles numpy, pandas, and other non-native Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _safe_json_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json_convert(i) for i in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _safe_json_convert(obj.tolist())
    if isinstance(obj, (pd.Series, pd.Index)):
        return _safe_json_convert(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    # Handle other numpy types
    if hasattr(obj, 'item'):  # numpy scalars
        try:
            return obj.item()
        except (ValueError, OverflowError):
            return str(obj)
    # Handle any other non-serializable types
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _set_trial_user_attr_safely(trial: "optuna.trial.Trial", key: str, value: Any):
    """
    Safely set a user attribute for a trial, ensuring JSON serializability.
    """
    try:
        trial.set_user_attr(key, _safe_json_convert(value))
    except Exception:
        # Fallback for extreme cases: just convert to string
        try:
            trial.set_user_attr(key, str(value))
        except Exception as e:
            print(f"[Trial {trial.number}] ⚠️ Could not save user attribute '{key}': {e}")
            pass


@dataclass(frozen=True)
class DatasetKey:
    """Cache key for loaded datasets."""

    path: str
    fmt: str
    sample_rows: Optional[int]
    target: str

    def to_string(self) -> str:
        return json.dumps(
            {
                'path': self.path,
                'format': self.fmt,
                'sample_rows': _safe_json_convert(self.sample_rows),
                'target': self.target,
            },
            sort_keys=True,
        )


DATASET_CACHE = {}
PREPROCESS_CACHE = {}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    try:
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    except Exception:
        if isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, '__dict__'):
            return dict(cfg.__dict__)
        return {}


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _get_dataset_key(dataset_cfg: Dict[str, Any]) -> DatasetKey:
    return DatasetKey(
        path=str(dataset_cfg.get('path')),
        fmt=str(dataset_cfg.get('format', 'parquet')).lower(),
        sample_rows=dataset_cfg.get('sample_rows'),
        target=str(dataset_cfg.get('target')),
    )


def _auto_discover_dataset_path(dataset_cfg: Dict[str, Any]) -> Optional[Path]:
    """Try to find a reasonable dataset parquet under /data when path is missing."""
    try:
        data_dir = Path('/data')
        if not data_dir.exists():
            return None
        symbol = str(dataset_cfg.get('symbol') or dataset_cfg.get('pair') or '').upper()
        patterns = []
        if symbol:
            patterns.extend([
                f"*{symbol}*master_features*.parquet",
                f"*{symbol}*.parquet",
            ])
        patterns.extend([
            "*master_features*.parquet",
            "*.parquet",
        ])
        for pat in patterns:
            matches = sorted(data_dir.glob(pat))
            if matches:
                return matches[0]
    except Exception:
        return None
    return None


def _fast_parquet_rowcount(dataset_path: Path) -> Optional[int]:
    """Quickly estimate total number of rows in a Parquet file or directory.

    Uses pyarrow metadata to sum row groups without materializing data. Returns
    None if pyarrow is unavailable or on error.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
        total = 0
        if dataset_path.is_dir():
            any_file = False
            for fp in dataset_path.rglob('*.parquet'):
                try:
                    pf = pq.ParquetFile(str(fp))
                    meta = pf.metadata
                    if meta is not None:
                        total += meta.num_rows
                        any_file = True
                except Exception:
                    continue
            return total if any_file else None
        else:
            pf = pq.ParquetFile(str(dataset_path))
            meta = pf.metadata
            return meta.num_rows if meta is not None else None
    except Exception:
        return None


def _load_dataset(dataset_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    if not dataset_cfg.get('path'):
        discovered = _auto_discover_dataset_path(dataset_cfg)
        if discovered is not None:
            dataset_cfg['path'] = str(discovered)
        else:
            raise ValueError("study.dataset.path must be defined to run the objectives")

    if not SKLEARN_AVAILABLE:
        raise ModuleNotFoundError(
            "scikit-learn is required for the surrogate optimisation pipeline."
        )

    dataset_path = _resolve_path(dataset_cfg['path'])
    fmt = str(dataset_cfg.get('format', 'parquet')).lower()

    # Read raw frame (with optional sampling) to resolve target and reduce memory pressure
    max_rows = int(dataset_cfg.get('sample_rows', 0) or 0)
    if fmt == 'parquet':
        if max_rows > 0:
            try:
                import dask.dataframe as dd  # type: ignore
                # Lazily read and pull only up to available rows to avoid head() warning
                ddf = dd.read_parquet(str(dataset_path), engine='pyarrow', gather_statistics=False)
                # Try a fast rowcount using pyarrow metadata first
                total_rows = _fast_parquet_rowcount(dataset_path)
                n = max_rows
                if total_rows is None:
                    # Cheap-ish fallback: sum partition lengths
                    try:
                        total_rows = int(ddf.map_partitions(len).sum().compute())
                    except Exception:
                        total_rows = None
                if isinstance(total_rows, int) and total_rows >= 0:
                    n = min(max_rows, total_rows)
                df = ddf.head(n, compute=True)
            except Exception:
                # Fallback to full read if dask path fails
                df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_parquet(dataset_path)
    elif fmt in {'csv', 'txt'}:
        if max_rows > 0:
            df = pd.read_csv(dataset_path, nrows=max_rows)
        else:
            df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")

    # Resolve target column robustly
    target_col = str(dataset_cfg.get('target') or '').strip()
    if (not target_col) or (target_col not in df.columns):
        # Try target_candidates in priority order
        candidates = dataset_cfg.get('target_candidates') or []
        if isinstance(candidates, (list, tuple)):
            for c in candidates:
                c = str(c)
                if c in df.columns:
                    target_col = c
                    break
        # If still not found, scan common forward-return labels
        if (not target_col) or (target_col not in df.columns):
            prefer = [
                'y_ret_fwd_60m',
                'y_ret_fwd_30m',
                'y_ret_fwd_15m',
                'y_ret_fwd_10m',
                'y_ret_fwd_5m',
                'y_ret_fwd_3m',
                'y_ret_fwd_1m',
                'y_ret_fwd_120m',
                'y_ret_fwd_240m',
                'y_ret_fwd_20m',
            ]
            present = [c for c in prefer if c in df.columns]
            if present:
                target_col = present[0]
        # Last resort: any column starting with pattern
        if (not target_col) or (target_col not in df.columns):
            auto = [c for c in df.columns if isinstance(c, str) and c.startswith('y_ret_fwd_')]
            if auto:
                target_col = auto[0]
        # If still not resolved, raise with details
        if (not target_col) or (target_col not in df.columns):
            raise ValueError(
                f"Target column '{dataset_cfg.get('target')}' not found and could not auto-resolve. "
                f"Available forward-return candidates: {[c for c in df.columns if isinstance(c, str) and c.startswith('y_ret_fwd_')]}"
            )
        # Persist resolved target back into cfg for caching key
        dataset_cfg['target'] = target_col
        warnings.warn(f"Resolved target column to '{target_col}' automatically.")

    # Now that target is known, compute cache key and try cache
    key = _get_dataset_key(dataset_cfg).to_string()
    if key in DATASET_CACHE:
        features, target = DATASET_CACHE[key]
        return features.copy(), target.copy()

    # Keep only numeric features to simplify the surrogate pipeline.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)
    df = df[numeric_cols]

    df = df.dropna(subset=[target_col]).copy()
    # If sampling requested, apply deterministic head to control memory
    if max_rows > 0 and len(df) > max_rows:
        df = df.head(max_rows).copy()

    # PRESERVE TARGET BEFORE DENYLIST FILTERING
    # Extract target values before removing features - target is NOT a feature for training
    y_target = df[target_col].astype(float).copy()
    print(f"    🎯 Target '{target_col}' preserved: {len(y_target)} samples")

    # --- COMPREHENSIVE DENYLIST ENFORCEMENT - CRITICAL FOR DATA LEAKAGE PREVENTION ---
    # This is a HARD BARRIER against data leakage - removes ALL problematic features
    
    # HARDCODED SECURITY DENYLISTS (cannot be overridden)
    # 1. Model features (m1_* to m9_*) - these are future-looking model outputs
    model_prefixes = [f'm{i}_' for i in range(1, 10)]
    
    # 2. Forward return targets (y_ret_fwd_*) - these are literally the future we're trying to predict
    forward_prefixes = ['y_ret_fwd_']
    
    # 3. Other critical leakage patterns
    critical_prefixes = [
        'is_',           # Boolean flags that might contain future info
        'y_is_',         # Target-related flags
        'bk_filter_',    # Baxter-King outputs
        'adf_stat_',     # Raw rolling ADF statistics
    ]
    
    # 4. Critical exact matches
    critical_exact = [
        'y_tick_volume',
        'y_total_volume', 
        'y_minutes_since_open',
        'y_fvg_age_bars',
        'y_fvg_confidence_score',
        'y_fvg_distance_pips',
        'y_fvg_width_pips',
        'y_jarque_bera_30m',
        'dxy_ret_1m_var_240',
        'y_fracdiff_d'
    ]
    
    # Get additional deny prefixes from config (if available) 
    cfg_dict = _cfg_to_dict(dataset_cfg) if hasattr(dataset_cfg, '__dict__') else dataset_cfg
    features_config = cfg_dict.get('features', {})
    config_deny_prefixes = features_config.get('feature_deny_prefixes', [])
    config_deny_exact = features_config.get('feature_denylist', [])
    
    # COMBINE ALL DENY RULES
    all_deny_prefixes = model_prefixes + forward_prefixes + critical_prefixes + config_deny_prefixes
    all_deny_exact = critical_exact + config_deny_exact
    
    # APPLY DENYLIST
    cols_to_remove = []
    for col in df.columns:
        col_str = str(col)
        # Check prefixes
        if any(col_str.startswith(prefix) for prefix in all_deny_prefixes):
            cols_to_remove.append(col)
        # Check exact matches
        elif col_str in all_deny_exact:
            cols_to_remove.append(col)
        # Check regex patterns ending with _mask or _flag
        elif col_str.endswith('_mask') or col_str.endswith('_flag'):
            cols_to_remove.append(col)
    
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        print(f"    🛡️ COMPREHENSIVE DENYLIST: Removed {len(cols_to_remove)} features to prevent data leakage")
        print(f"    🛡️ Examples: {', '.join(cols_to_remove[:8])}{'...' if len(cols_to_remove) > 8 else ''}")
        
        # Detailed breakdown
        model_removed = [c for c in cols_to_remove if any(str(c).startswith(p) for p in model_prefixes)]
        forward_removed = [c for c in cols_to_remove if any(str(c).startswith(p) for p in forward_prefixes)]
        
        if model_removed:
            print(f"    🛡️ Model features (m1_*-m9_*): {len(model_removed)} removed")
        if forward_removed:
            print(f"    🛡️ Forward targets (y_ret_fwd_*): {len(forward_removed)} removed")
    
    print(f"    ✅ Final dataset: {len(df)} samples, {len(df.columns)} features (after denylist)")
    # --- END COMPREHENSIVE DENYLIST ---

    sample_rows = dataset_cfg.get('sample_rows')
    if sample_rows and sample_rows > 0 and len(df) > sample_rows:
        df = df.tail(sample_rows)
        # Also apply sampling to target
        y_target = y_target.tail(sample_rows)

    # Features are everything except the target (which was already extracted)
    X = df.drop(columns=[target_col], errors='ignore').astype(float)  # errors='ignore' in case target was already removed by denylist
    y = y_target  # Use the preserved target

    DATASET_CACHE[key] = (X.copy(), y.copy())
    return X, y


def _param_key(params: Dict[str, Any]) -> str:
    return json.dumps({k: _safe_json_convert(params[k]) for k in sorted(params)}, sort_keys=True)


def _preprocess_dataset(
    raw_features: pd.DataFrame,
    raw_target: pd.Series,
    params: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"    🔧 _preprocess_dataset called with {len(raw_features)} samples, {len(raw_features.columns)} features")
    
    cache_key = (
        _param_key(params),
        dataset_cfg.get('target'),
        dataset_cfg.get('path'),
        dataset_cfg.get('sample_rows'),
    )
    if cache_key in PREPROCESS_CACHE:
        cached_X, cached_y = PREPROCESS_CACHE[cache_key]
        print(f"    ⚡ Using cached preprocessing result")
        return cached_X.copy(), cached_y.copy()

    print(f"    ⚙️ Starting fresh preprocessing...")

    X = raw_features.copy()
    y = raw_target.copy()
    print(f"    📊 Initial: {len(X)} samples, {len(X.columns)} features")

    sample_override = params.get('sampling.selection_max_rows')
    try:
        sample_override = int(sample_override) if sample_override else None
    except Exception:
        sample_override = None
    if sample_override and sample_override > 0 and len(X) > sample_override:
        X = X.tail(sample_override)
        y = y.tail(sample_override)
        print(f"    📊 After sampling: {len(X)} samples, {len(X.columns)} features")

    # ------------------------------------------------------------------
    # Fractional differencing surrogate: blend raw values with 1-step diff.
    # ------------------------------------------------------------------
    diff_alpha = float(params.get('frac_diff.d', 0.0) or 0.0)
    diff_alpha = min(max(diff_alpha, 0.0), 1.0)
    if diff_alpha > 0.0 and len(X) > 1:
        diff_df = X.diff().fillna(0.0)
        X = diff_alpha * diff_df + (1.0 - diff_alpha) * X

    variance_threshold = float(params.get('frac_diff.threshold', 0.0) or 0.0)
    if variance_threshold > 0.0:
        std = X.std()
        keep = std[std > variance_threshold].index.tolist()
        if keep:
            X = X[keep]
        print(f"    📊 After variance filter (threshold={variance_threshold:.2e}): {len(X.columns)} features")

    # ------------------------------------------------------------------
    # Distance correlation surrogate: absolute Pearson correlation filter.
    # ------------------------------------------------------------------
    if not X.empty:
        corr = X.corrwith(y).abs().fillna(0.0)
        threshold = float(params.get('dcor.threshold', 0.0) or 0.0)
        top_k = params.get('dcor.top_k')
        if top_k is None:
            # fall back to dataset defaults when available
            top_k = dataset_cfg.get('dcor_top_k', len(corr))
        try:
            top_k = int(top_k) if top_k else len(corr)
        except Exception:
            top_k = len(corr)

        selected = corr[corr >= threshold].index.tolist() if threshold > 0 else []
        if not selected:
            selected = corr.sort_values(ascending=False).head(max(1, min(len(corr), top_k))).index.tolist()
        elif top_k > 0 and len(selected) > top_k:
            selected = corr[selected].sort_values(ascending=False).head(top_k).index.tolist()

        if selected:
            X = X[selected]
        print(f"    📊 After dCor filter (threshold={threshold:.3f}, top_k={top_k}): {len(X.columns)} features")

    # ------------------------------------------------------------------
    # VIF surrogate: greedy pairwise correlation pruning.
    # ------------------------------------------------------------------
    vif_threshold = float(params.get('vif.threshold', 0.0) or 0.0)
    if vif_threshold > 1.0 and X.shape[1] > 1:
        corr_matrix = X.corr().abs().fillna(0.0)
        max_corr_allowed = max(0.0, min(0.999, 1.0 - 1.0 / vif_threshold))
        corr_with_target = X.corrwith(y).abs().fillna(0.0)
        ordered = list(corr_with_target.sort_values(ascending=False).index)
        keep: List[str] = []
        for col in ordered:
            if all(corr_matrix.loc[col, existing] <= max_corr_allowed for existing in keep):
                keep.append(col)
        if keep:
            X = X[keep]
        print(f"    📊 After VIF filter (threshold={vif_threshold:.1f}): {len(X.columns)} features")

    # ------------------------------------------------------------------
    # Mutual-information based pruning (best-effort, requires sklearn).
    # ------------------------------------------------------------------
    mi_threshold = float(params.get('mi.threshold', 0.0) or 0.0)
    if mi_threshold > 0.0 and SKLEARN_AVAILABLE and X.shape[1] > 0:
        try:
            rng = np.random.default_rng(random_state)
            # mutual_info_regression expects finite values
            X_filled = X.fillna(0.0)
            mi_scores = mutual_info_regression(X_filled.values, y.values, random_state=rng.integers(0, 2**32 - 1))
            mi_series = pd.Series(mi_scores, index=X.columns)
            keep = mi_series[mi_series >= mi_threshold].index.tolist()
            if not keep:
                keep = mi_series.sort_values(ascending=False).head(max(1, min(len(mi_series), 10))).index.tolist()
            X = X[keep]
        except Exception:
            # If MI computation fails, simply continue with existing columns.
            pass
    print(f"    📊 After MI filter (threshold={mi_threshold:.3f}): {len(X.columns)} features")

    # ------------------------------------------------------------------
    # Stage 3 surrogate: limit to top-N features by correlation.
    # ------------------------------------------------------------------
    top_n = params.get('stage3.top_n')
    try:
        top_n = int(top_n) if top_n else 0
    except Exception:
        top_n = 0
    if top_n and top_n > 0 and X.shape[1] > top_n:
        corr = X.corrwith(y).abs().fillna(0.0)
        keep = corr.sort_values(ascending=False).head(top_n).index.tolist()
        X = X[keep]
    print(f"    📊 After Stage 3 filter (top_n={top_n}): {len(X.columns)} features")

    if X.empty:
        # Ensure at least one feature is available to the model.
        print(f"    ⚠️ All features filtered out, using first feature as fallback")
        X = raw_features.iloc[:, :1].fillna(0.0).copy()

    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    PREPROCESS_CACHE[cache_key] = (X_scaled.copy(), y.copy())
    print(f"    ✅ Preprocessing complete: {len(X_scaled)} samples, {len(X_scaled.columns)} features")
    return X_scaled, y.copy()


def _suggest_from_search_space(
    trial: "optuna.trial.Trial",
    search_space: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key, raw_node in search_space.items():
        node = _cfg_to_dict(raw_node)
        name = f"{prefix}.{key}" if prefix else str(key)

        if any(k in node for k in ("low", "high", "choices")):
            if 'choices' in node:
                params[name] = trial.suggest_categorical(name, node['choices'])
                continue

            low = node.get('low')
            high = node.get('high')
            step = node.get('step')
            log = bool(node.get('log', False))
            dtype = node.get('dtype')

            if step is not None:
                values = []
                current = float(low)
                while current <= float(high) + 1e-12:
                    values.append(round(current, 10))
                    current += float(step)
                params[name] = trial.suggest_categorical(name, values)
            elif dtype == 'int' or all(float(v).is_integer() for v in (low, high)):
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high), log=log)
        elif isinstance(node, dict):
            params.update(_suggest_from_search_space(trial, node, name))
        else:
            # Literal value (e.g., None/null)
            params[name] = node

    return params


def _build_proxy_model(proxy_cfg: Dict[str, Any], random_state: int) -> Pipeline:
    estimators = []
    iterations = int(proxy_cfg.get('iterations', 200))
    learning_rate = float(proxy_cfg.get('learning_rate', 0.05))
    depth = int(proxy_cfg.get('depth', 6))

    model = GradientBoostingRegressor(
        n_estimators=max(10, iterations),
        learning_rate=max(1e-4, learning_rate),
        max_depth=max(1, depth),
        random_state=random_state,
    )

    estimators.append(('model', model))
    return Pipeline(estimators)


def _build_stage_b_model(model_name: str, params: Dict[str, Any], random_state: int):
    if model_name == 'catboost':
        iterations = int(params.get('iterations', params.get('n_estimators', 500)))
        depth = int(params.get('depth', 6))
        learning_rate = float(params.get('learning_rate', 0.05))
        l2_leaf_reg = float(params.get('l2_leaf_reg', 3.0))

        try:
            from catboost import CatBoostRegressor

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*less than 75% GPU memory available.*')
                return CatBoostRegressor(
                iterations=max(10, iterations),
                depth=max(1, depth),
                learning_rate=max(1e-4, learning_rate),
                l2_leaf_reg=max(0.0, l2_leaf_reg),
                loss_function=params.get('loss_function', 'RMSE'),
                random_seed=random_state,
                verbose=False,
                task_type=params.get('task_type', 'CPU'),
            )
        except Exception:
            # Fall back to HistGradientBoostingRegressor when CatBoost is unavailable.
            pass

        return HistGradientBoostingRegressor(
            max_iter=max(10, iterations),
            learning_rate=max(1e-4, learning_rate),
            max_depth=depth if depth > 0 else None,
            l2_regularization=max(0.0, l2_leaf_reg),
            random_state=random_state,
        )

    # Default to a HistGradientBoostingRegressor approximation for LightGBM.
    learning_rate = float(params.get('learning_rate', 0.05))
    num_leaves = int(params.get('num_leaves', 31))
    max_depth = int(params.get('max_depth', -1))

    return HistGradientBoostingRegressor(
        max_iter=max(10, int(params.get('iterations', 500))),
        learning_rate=max(1e-4, learning_rate),
        max_depth=None if max_depth <= 0 else max_depth,
        max_leaf_nodes=max(15, num_leaves),
        l2_regularization=float(params.get('lambda_l2', 0.0)),
        random_state=random_state,
    )


def _calculate_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    entry_threshold: float,
    cost_per_trade: float,
    ann_factor: float = np.sqrt(252 * 24)  # For hourly data
) -> Dict[str, float]:
    """
    Calculate comprehensive trading metrics for a single fold.
    
    Args:
        y_true: True returns
        y_pred: Predicted returns
        entry_threshold: Minimum prediction magnitude to enter position
        cost_per_trade: Transaction cost in basis points
        ann_factor: Annualization factor for Sharpe ratio
    
    Returns:
        Dictionary with trading metrics
    """
    # Calculate positions: +1 (long), -1 (short), 0 (neutral)
    positions = np.sign(y_pred) * (np.abs(y_pred) > entry_threshold)
    
    # Debug info for position taking
    n_predictions = len(y_pred)
    n_above_threshold = np.sum(np.abs(y_pred) > entry_threshold)
    pred_mean = np.mean(np.abs(y_pred))
    pred_max = np.max(np.abs(y_pred))
    
    # Calculate trades (position changes)
    trades = np.abs(np.diff(np.r_[0, positions]))
    
    # Calculate net returns (strategy returns minus transaction costs)
    strategy_returns = positions * y_true
    transaction_costs = trades * cost_per_trade
    net_returns = strategy_returns - transaction_costs
    
    # Sharpe Ratio (annualized)
    returns_mean = net_returns.mean()
    returns_std = net_returns.std()
    
    # Debug: log key metrics for first few trials
    n_positions_taken = int(np.sum(positions != 0))
    n_trades_made = int(trades.sum())
    
    # Use a more conservative annualization factor for intraday data
    # Assume 5-minute bars, ~288 bars per day, 252 trading days
    conservative_ann_factor = np.sqrt(252 * 288)  # More appropriate for 5-min data
    
    sharpe = (returns_mean / (returns_std + 1e-9)) * conservative_ann_factor if returns_std > 0 else 0.0
    
    # Fallback: if no positions taken, return 0
    if n_positions_taken == 0:
        sharpe = 0.0
    
    # Debug info for poor performance
    if sharpe == 0.0 and n_positions_taken > 0:
        # Model is taking positions but Sharpe is still 0
        if returns_std < 1e-10:
            pass  # Very low volatility, normal for some periods
        elif abs(returns_mean) < 1e-10:
            pass  # Very low mean return, normal for some periods
    
    # Information Coefficient (Spearman rank correlation)
    try:
        from scipy.stats import spearmanr
        ic, ic_pvalue = spearmanr(y_pred, y_true)
        if np.isnan(ic):
            ic = 0.0
    except Exception:
        ic = 0.0
    
    # Turnover (average number of trades)
    turnover = trades.mean()
    
    # Max Drawdown calculation
    cumulative_returns = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = np.abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    # Hit ratio (percentage of profitable periods)
    hit_ratio = np.mean(net_returns > 0) if len(net_returns) > 0 else 0.0
    
    # Total return
    total_return = net_returns.sum()
    
    # Calmar ratio (total return / max drawdown)
    calmar_ratio = total_return / (max_drawdown + 1e-9) if max_drawdown > 0 else 0.0
    
    return {
        'sharpe_ratio': float(sharpe),
        'information_coefficient': float(ic),
        'turnover': float(turnover),
        'max_drawdown': float(max_drawdown),
        'hit_ratio': float(hit_ratio),
        'total_return': float(total_return),
        'calmar_ratio': float(calmar_ratio),
        'net_returns_mean': float(returns_mean),
        'net_returns_std': float(returns_std),
        'n_trades': n_trades_made,
        'n_positions': n_positions_taken,
        # Debug info
        'n_predictions': n_predictions,
        'n_above_threshold': n_above_threshold,
        'pred_mean_abs': float(pred_mean),
        'pred_max_abs': float(pred_max),
        'entry_threshold': float(entry_threshold)
    }


def _evaluate_model_robust(
    trial: "optuna.trial.Trial",
    model,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    entry_threshold: float,
    cost_per_trade: float,
    embargo_gap: int,
    n_splits: int = 5,
    feature_importance_method: str = 'native'
) -> Tuple[float, float, float]:
    """
    Robust model evaluation with walk-forward validation and trading metrics.
    
    Returns:
        Tuple of (mean_sharpe, mean_turnover, mean_max_drawdown)
    """
    if X.empty or len(X) < 50:
        return 0.0, float('inf'), float('inf')

    # TimeSeriesSplit with embargo gap
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo_gap)
    
    # Storage for metrics across folds
    fold_metrics = {
        'sharpe_ratio': [],
        'information_coefficient': [],
        'turnover': [],
        'max_drawdown': [],
        'hit_ratio': [],
        'total_return': [],
        'calmar_ratio': [],
        'pred_max_abs': [],
        'n_above_threshold': [],
        'n_positions': [],
        'n_trades': []
    }
    
    # Storage for feature importance (optional SHAP analysis)
    feature_importance_scores = []
    
    X_values = X.values
    y_values = y.values
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_values)):
        try:
            model_instance = clone(model)
            X_train, X_test = X_values[train_idx], X_values[test_idx]
            y_train, y_test = y_values[train_idx], y_values[test_idx]
            
            # Fit model
            model_instance.fit(X_train, y_train)
            
            # Generate predictions
            y_pred = model_instance.predict(X_test)
            
            # Calculate trading metrics for this fold
            metrics = _calculate_trading_metrics(
                y_true=y_test,
                y_pred=y_pred,
                entry_threshold=entry_threshold,
                cost_per_trade=cost_per_trade
            )
            
            # Log fold results occasionally
            if fold_idx == 0 or len(fold_metrics['sharpe_ratio']) % 2 == 0:
                print(f"    Fold {fold_idx}: Sharpe={metrics['sharpe_ratio']:.4f}, Positions={metrics['n_positions']}, Trades={metrics['n_trades']}")
                print(f"              Pred_max={metrics['pred_max_abs']:.6f}, Threshold={metrics['entry_threshold']:.6f}, Above_thresh={metrics['n_above_threshold']}")
                if metrics['n_positions'] == 0:
                    print(f"              ⚠️ No positions taken. Consider lowering entry_threshold.")
            
            # Store metrics
            for key, value in metrics.items():
                if key in fold_metrics:
                    fold_metrics[key].append(value)
            
            # Feature importance (if supported by model)
            if hasattr(model_instance, 'feature_importances_'):
                importance_scores = model_instance.feature_importances_
                feature_importance_scores.append({
                    'fold': fold_idx,
                    'importances': dict(zip(X.columns, importance_scores))
                })
            
        except Exception as e:
            # Handle fold failures gracefully
            for key in fold_metrics:
                fold_metrics[key].append(0.0 if key != 'turnover' else float('inf'))
            print(f"    Fold {fold_idx}: ❌ Exception during evaluation: {e}")
    
    # Calculate mean metrics across folds
    mean_sharpe = float(np.mean(fold_metrics['sharpe_ratio']))
    # Replace NaN with 0 and inf with a large finite sentinel before averaging
    _turn = np.array(fold_metrics['turnover'], dtype=float)
    _turn = np.nan_to_num(_turn, nan=0.0, posinf=1e6, neginf=1e6)
    mean_turnover = float(np.mean(_turn))
    mean_max_drawdown = float(np.mean(fold_metrics['max_drawdown']))
    
    # Store additional metrics as trial attributes
    _set_trial_user_attr_safely(trial, 'mean_ic', float(np.mean(fold_metrics['information_coefficient'])))
    _set_trial_user_attr_safely(trial, 'mean_hit_ratio', float(np.mean(fold_metrics['hit_ratio'])))
    _set_trial_user_attr_safely(trial, 'mean_total_return', float(np.mean(fold_metrics['total_return'])))
    _set_trial_user_attr_safely(trial, 'mean_calmar_ratio', float(np.mean(fold_metrics['calmar_ratio'])))
    _set_trial_user_attr_safely(trial, 'std_sharpe', float(np.std(fold_metrics['sharpe_ratio'])))
    _set_trial_user_attr_safely(trial, 'std_ic', float(np.std(fold_metrics['information_coefficient'])))
    
    # Store feature importance if available
    if feature_importance_scores:
        _set_trial_user_attr_safely(trial, 'feature_importance_folds', feature_importance_scores)
    
    # Store detailed fold metrics for analysis
    _set_trial_user_attr_safely(trial, 'fold_metrics', fold_metrics)
    
    return mean_sharpe, mean_turnover, mean_max_drawdown


def _extract_proxy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    study_cfg = cfg.get('study', {})
    return _cfg_to_dict(study_cfg.get('proxy_model', {}))


def _extract_search_space(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _cfg_to_dict(cfg.get('study', {}).get('search_space', {}))


def _get_dataset_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve dataset configuration robustly across possible nesting variants
    dataset_cfg = _cfg_to_dict(cfg.get('study', {}).get('dataset', {}))
    if not dataset_cfg:
        # Fallbacks in case packaging placed dataset at root or under an alternate key
        dataset_cfg = _cfg_to_dict(cfg.get('dataset', {}))
    if not dataset_cfg:
        dataset_cfg = _cfg_to_dict(cfg.get('data', {}))
    if 'target' not in dataset_cfg:
        dataset_cfg['target'] = cfg.get('features', {}).get('selection_target_column', 'target')
    if 'dcor_top_k' not in dataset_cfg:
        dataset_cfg['dcor_top_k'] = cfg.get('features', {}).get('dcor_top_k', 50)
    return dataset_cfg


def _get_random_state(cfg: Dict[str, Any]) -> int:
    try:
        return int(cfg.get('study', {}).get('optuna', {}).get('seed', 42))
    except Exception:
        return 42


# ---------------------------------------------------------------------------
# Study objectives
# ---------------------------------------------------------------------------


def objective_study_a(trial: "optuna.trial.Trial", cfg: DictConfig) -> Tuple[float, float, float]:
    """
    Robust Study A objective for feature selection optimization.
    
    Implements walk-forward validation with trading-specific metrics.
    Returns multi-objective optimization targets:
    - Sharpe ratio (maximize)
    - Turnover (minimize) 
    - Max drawdown (minimize)
    """
    
    print(f"\n🚀 [Trial {trial.number}] STARTING OBJECTIVE FUNCTION")
    
    # Clear caches periodically to avoid stale data
    if trial.number % 10 == 0:
        DATASET_CACHE.clear()
        PREPROCESS_CACHE.clear()
        print(f"[Trial {trial.number}] 🧹 Cleared caches")
    
    cfg_dict = _cfg_to_dict(cfg)
    # Detect if this run is multi-objective (expects a tuple) or single-objective
    optuna_cfg = _cfg_to_dict(cfg_dict.get('study', {}).get('optuna', {}))
    _dirs = optuna_cfg.get('directions')
    multi_objective = isinstance(_dirs, (list, tuple)) and len(_dirs) >= 2
    dataset_cfg = _get_dataset_cfg(cfg_dict)
    
    print(f"[Trial {trial.number}] 🔧 Multi-objective: {multi_objective}")
    print(f"[Trial {trial.number}] 📁 Dataset path: {dataset_cfg.get('path', 'NOT_SET')}")

    # If user provided target candidates, make target a hyperparameter
    try:
        candidates = dataset_cfg.get('target_candidates') or []
        if isinstance(candidates, (list, tuple)) and len(candidates) > 0:
            candidates = [str(c) for c in candidates]
            chosen_target = trial.suggest_categorical("target_column", candidates)
            # Store chosen target back into cfg for caching key and downstream use
            dataset_cfg['target'] = chosen_target
            # Record for analysis/visualization
            _set_trial_user_attr_safely(trial, "target_column", chosen_target)
            print(f"[Trial {trial.number}] 🎯 Target selected: {chosen_target} (from {len(candidates)} candidates)")
        else:
            # ensure a concrete string for caching
            dataset_cfg['target'] = str(dataset_cfg.get('target') or cfg_dict.get('features', {}).get('selection_target_column', 'target'))
            print(f"[Trial {trial.number}] 🎯 Target fixed: {dataset_cfg['target']}")
    except Exception:
        dataset_cfg['target'] = str(dataset_cfg.get('target') or cfg_dict.get('features', {}).get('selection_target_column', 'target'))
        print(f"[Trial {trial.number}] 🎯 Target fallback: {dataset_cfg['target']}")

    search_space = _extract_search_space(cfg_dict)
    try:
        root_keys = list(search_space.keys())
        print(f"[Trial {trial.number}] 🧭 Search space roots: {len(root_keys)} keys -> {root_keys[:6]}{'...' if len(root_keys) > 6 else ''}")
    except Exception:
        pass
    proxy_cfg = _extract_proxy_cfg(cfg_dict)
    random_state = _get_random_state(cfg_dict)

    # Load dataset with the selected target
    print(f"[Trial {trial.number}] 📥 Loading dataset...")
    features, target = _load_dataset(dataset_cfg)
    print(f"[Trial {trial.number}] ✅ Dataset loaded: {len(features)} samples, {len(features.columns)} raw features")
    
    # === NEW HYPERPARAMETERS ===
    
    print(f"[Trial {trial.number}] 🎲 Suggesting hyperparameters...")
    
    # Trading execution parameters
    entry_threshold = trial.suggest_float("entry_threshold", 0.0001, 0.002, log=True)  # Much lower range
    cost_per_trade = trial.suggest_float("cost_per_trade", 0.00005, 0.0005, log=True)
    embargo_gap = trial.suggest_int("embargo_gap", 5, 30)
    
    # Feature selection parameters (more reasonable ranges)
    top_k_features = trial.suggest_int("top_k_features", 5, 25)  # Lower range since we have fewer features now
    feature_selection_threshold = trial.suggest_float("feature_selection_threshold", 0.001, 0.02, log=True)  # Lower threshold
    
    # Model hyperparameters
    model_n_estimators = trial.suggest_int("model_n_estimators", 100, 300)
    model_learning_rate = trial.suggest_float("model_learning_rate", 0.01, 0.2, log=True)
    model_max_depth = trial.suggest_int("model_max_depth", 4, 8)
    model_l2_leaf_reg = trial.suggest_float("model_l2_leaf_reg", 0.5, 5.0, log=True)
    model_subsample = trial.suggest_float("model_subsample", 0.7, 0.95)
    
    # Validation parameters
    cv_splits = trial.suggest_int("cv_splits", 3, 6)
    
    print(f"[Trial {trial.number}] ✅ Hyperparameters: entry_thresh={entry_threshold:.6f}, features={top_k_features}, threshold={feature_selection_threshold:.4f}")
    
    # === PREPROCESSING WITH HYPERPARAMETERS ===
    
    # Apply basic preprocessing (simplified version of original)
    stage_a_space = search_space.copy()
    preprocess_params = _suggest_from_search_space(trial, stage_a_space)
    
    print(f"[Trial {trial.number}] 🔄 Starting preprocessing with {len(preprocess_params)} parameters...")
    if len(preprocess_params) == 0:
        try:
            print(f"[Trial {trial.number}] ⚠️ Empty preprocess params. Verify study.search_space in config.")
        except Exception:
            pass
    
    # Apply preprocessing to get initial feature set
    X_processed, y_processed = _preprocess_dataset(features, target, preprocess_params, dataset_cfg, random_state)

    # Leakage guard: drop any other forward-return columns accidentally surviving as features
    try:
        if len(X_processed.columns) > 0:
            forward_cols = [c for c in X_processed.columns if isinstance(c, str) and c.startswith('y_ret_fwd_')]
            keep_cols = [c for c in X_processed.columns if c not in forward_cols]
            if len(keep_cols) == 0:
                # ensure at least one column remains
                keep_cols = X_processed.columns[:1].tolist()
            X_processed = X_processed[keep_cols]
            if forward_cols:
                print(f"[Trial {trial.number}] 🛡️ Removed {len(forward_cols)} forward-return features to prevent leakage")
    except Exception:
        pass
    
    print(f"[Trial {trial.number}] 📊 Dataset: {len(X_processed)} samples, {len(X_processed.columns)} features after preprocessing")
    
    if X_processed.empty or len(X_processed) < 100:
        print(f"[Trial {trial.number}] ⚠️ Dataset too small ({len(X_processed)} samples), returning 0.0")
        return (0.0, float('inf'), float('inf')) if multi_objective else 0.0
    
    # === FEATURE SELECTION ===
    
    # Calculate feature importance/correlation for selection
    try:
        # Filter out features with zero or very low variance first to avoid division warnings
        feature_std = X_processed.std()
        valid_features = feature_std[feature_std > 1e-8].index.tolist()
        
        if len(valid_features) == 0:
            # All features have zero variance, use first feature as fallback
            valid_features = X_processed.columns[:1].tolist()
        
        X_valid = X_processed[valid_features]
        
        # Use correlation as initial feature ranking (with valid features only)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            feature_correlations = X_valid.corrwith(y_processed).abs().fillna(0.0)
        
        # Apply threshold-based selection
        threshold_selected = feature_correlations[feature_correlations >= feature_selection_threshold].index.tolist()
        
        # Apply top-k selection
        if not threshold_selected:
            threshold_selected = feature_correlations.sort_values(ascending=False).head(max(1, min(top_k_features, len(valid_features)))).index.tolist()
        elif len(threshold_selected) > top_k_features:
            threshold_selected = feature_correlations[threshold_selected].sort_values(ascending=False).head(top_k_features).index.tolist()
        
        X_selected = X_processed[threshold_selected]
        
        print(f"[Trial {trial.number}] 🎯 Feature selection: {len(X_processed.columns)} → {len(X_selected.columns)} features (threshold: {feature_selection_threshold:.4f})")
        
    except Exception as e:
        # Fallback to top features by variance if correlation fails
        feature_variances = X_processed.var().fillna(0.0)
        top_features = feature_variances.sort_values(ascending=False).head(min(top_k_features, len(X_processed.columns))).index.tolist()
        if not top_features:
            top_features = X_processed.columns[:1].tolist()
        X_selected = X_processed[top_features]
        print(f"[Trial {trial.number}] ⚠️ Feature selection fallback: {len(X_selected.columns)} features selected by variance")
    
    # === MODEL BUILDING ===
    
    # Build model with suggested hyperparameters
    try:
        from catboost import CatBoostRegressor
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*less than 75% GPU memory available.*')
            model = CatBoostRegressor(
            iterations=model_n_estimators,
            learning_rate=model_learning_rate,
            depth=model_max_depth,
            l2_leaf_reg=model_l2_leaf_reg,
            subsample=model_subsample,
            loss_function=f'Huber:delta={proxy_cfg.get("huber_delta", 1.0)}',  # Configurable delta parameter
            eval_metric=f'Huber:delta={proxy_cfg.get("huber_delta", 1.0)}',
            random_seed=random_state,
            verbose=False,
            task_type='GPU' if proxy_cfg.get('task_type') == 'GPU' else 'CPU',
        )
    except Exception:
        # Fallback to sklearn if CatBoost fails
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=model_n_estimators,
            learning_rate=model_learning_rate,
            max_depth=model_max_depth,
            subsample=model_subsample,
            random_state=random_state,
        )
    
    # === WALK-FORWARD VALIDATION ===
    
    print(f"[Trial {trial.number}] 🔄 Starting walk-forward validation with {cv_splits} splits, embargo_gap={embargo_gap}")
    
    mean_sharpe, mean_turnover, mean_max_drawdown = _evaluate_model_robust(
        trial=trial,
        model=model,
        X=X_selected,
        y=y_processed,
        random_state=random_state,
        entry_threshold=entry_threshold,
        cost_per_trade=cost_per_trade,
        embargo_gap=embargo_gap,
        n_splits=cv_splits
    )
    
    print(f"[Trial {trial.number}] 📈 Results: Sharpe={mean_sharpe:.4f}, Turnover={mean_turnover:.4f}, MaxDD={mean_max_drawdown:.4f}")
    print(f"[Trial {trial.number}] 🎯 Target used: {dataset_cfg.get('target', 'unknown')}")
    
    # Log debug info if no positions taken
    if mean_turnover == float('inf') or mean_sharpe == 0.0:
        try:
            fm = trial.user_attrs.get('fold_metrics', {}) if hasattr(trial, 'user_attrs') else {}
            fold_debug_data = [m for m in fm.get('pred_max_abs', []) if m is not None]
            if fold_debug_data:
                avg_pred_max = np.mean(fold_debug_data)
                fold_thresh_data = [m for m in fm.get('n_above_threshold', []) if m is not None]
                avg_above_thresh = np.mean(fold_thresh_data) if fold_thresh_data else 0
                print(f"[Trial {trial.number}] 🔍 Debug: Avg_pred_max={avg_pred_max:.6f}, Avg_above_thresh={avg_above_thresh:.0f}, Entry_thresh={entry_threshold:.6f}")
            else:
                print(f"[Trial {trial.number}] 🔍 Debug: No fold metrics recorded; Entry_thresh={entry_threshold:.6f}")
        except Exception as _e:
            print(f"[Trial {trial.number}] 🔍 Debug: Entry_thresh={entry_threshold:.6f} (prediction analysis failed: {_e})")
    
    # === STORE TRIAL METADATA ===
    
    # Simplified user attributes - only save essential info as basic Python types
    _set_trial_user_attr_safely(trial, 'n_features_selected', len(X_selected.columns))
    _set_trial_user_attr_safely(trial, 'n_features_raw', len(X_processed.columns))
    _set_trial_user_attr_safely(trial, 'entry_threshold', entry_threshold)
    _set_trial_user_attr_safely(trial, 'feature_selection_threshold', feature_selection_threshold)
    _set_trial_user_attr_safely(trial, 'target_column', dataset_cfg.get('target', 'unknown'))
    _set_trial_user_attr_safely(trial, 'sharpe_ratio', mean_sharpe)
    _set_trial_user_attr_safely(trial, 'turnover', mean_turnover)
    _set_trial_user_attr_safely(trial, 'max_drawdown', mean_max_drawdown)
    
    # Ensure return values are native Python floats, not numpy
    mean_sharpe_float = float(mean_sharpe) if not (np.isnan(mean_sharpe) or np.isinf(mean_sharpe)) else 0.0
    mean_turnover_float = float(mean_turnover) if not (np.isnan(mean_turnover) or np.isinf(mean_turnover)) else float('inf')
    mean_max_drawdown_float = float(mean_max_drawdown) if not (np.isnan(mean_max_drawdown) or np.isinf(mean_max_drawdown)) else float('inf')
    
    # Return targets according to study mode
    # Multi: (maximize Sharpe, minimize turnover, minimize max_drawdown)
    # Single: maximize Sharpe
    return (mean_sharpe_float, mean_turnover_float, mean_max_drawdown_float) if multi_objective else mean_sharpe_float


def objective_study_b(
    trial: "optuna.trial.Trial",
    cfg: DictConfig,
    fixed_params: Dict[str, Any],
) -> Tuple[float, float, float]:
    """Study B objective: optimise model hyperparameters with frozen preprocessing."""

    cfg_dict = _cfg_to_dict(cfg)
    dataset_cfg = _get_dataset_cfg(cfg_dict)
    search_space = _extract_search_space(cfg_dict)
    random_state = _get_random_state(cfg_dict)

    features, target = _load_dataset(dataset_cfg)
    X_processed, y_processed = _preprocess_dataset(features, target, fixed_params, dataset_cfg, random_state)

    model_space = _cfg_to_dict(search_space.get('model', {}))
    name_cfg = _cfg_to_dict(model_space.get('name', {}))
    model_choices: Iterable[str] = name_cfg.get('choices', ['catboost'])
    model_name = trial.suggest_categorical('model.name', list(model_choices))

    model_params_cfg = _cfg_to_dict(model_space.get(model_name, {}))
    model_params = _suggest_from_search_space(trial, model_params_cfg, prefix=f"model.{model_name}")

    model = _build_stage_b_model(model_name, model_params, random_state)
    score, directional, turnover = _evaluate_model(trial, model, X_processed, y_processed, random_state)

    _set_trial_user_attr_safely(trial, 'model_name', model_name)
    _set_trial_user_attr_safely(trial, 'model_params', model_params)
    _set_trial_user_attr_safely(trial, 'feature_count', int(X_processed.shape[1]))

    # Directions: maximise score, maximise directional_accuracy, minimise turnover.
    return score, directional, turnover


# =============================================================================
# [IMPLEMENTAÇÃO FINAL] PIPELINE AVANÇADO COM TEMPLATE
# =============================================================================

def max_drawdown(pnl: np.ndarray) -> float:
    """Calcular Maximum Drawdown de uma série de PnL."""
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    return float(-np.min(drawdown))


class InformationCoefficientSelector(BaseEstimator, TransformerMixin):
    """
    Seletor de features baseado no Information Coefficient (IC)
    Compatible com sklearn.pipeline.Pipeline
    """
    
    def __init__(self, k: int = 50, min_ic: float = 0.01):
        self.k = k
        self.min_ic = min_ic
        self.selected_features_ = None
        self.support_ = None
        self.ic_scores_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calcular IC para cada feature e selecionar as top k"""
        n_features = X.shape[1]
        ic_scores = np.zeros(n_features)
        
        # Calcular IC para cada feature
        for i in range(n_features):
            try:
                # Spearman correlation é mais robusta para dados financeiros
                ic, p_value = spearmanr(X[:, i], y)
                if not np.isnan(ic) and p_value < 0.05:  # Significância estatística
                    ic_scores[i] = abs(ic)
                else:
                    ic_scores[i] = 0.0
            except:
                ic_scores[i] = 0.0
        
        self.ic_scores_ = ic_scores
        
        # Selecionar top k features com IC acima do mínimo
        valid_features = ic_scores >= self.min_ic
        if np.sum(valid_features) < self.k:
            # Se não há features suficientes, usar as melhores disponíveis
            selected_indices = np.argsort(ic_scores)[-self.k:]
        else:
            # Selecionar top k entre as válidas
            valid_indices = np.where(valid_features)[0]
            valid_scores = ic_scores[valid_indices]
            top_valid = np.argsort(valid_scores)[-self.k:]
            selected_indices = valid_indices[top_valid]
        
        # Criar mask de seleção
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected_indices] = True
        self.selected_features_ = selected_indices
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transformar dados selecionando apenas as features escolhidas"""
        if self.support_ is None:
            raise ValueError("Selector must be fitted before transform")
        return X[:, self.support_]
    
    def get_support(self, indices: bool = False):
        """Retornar mask ou índices das features selecionadas"""
        if self.support_ is None:
            raise ValueError("Selector must be fitted before get_support")
        return self.selected_features_ if indices else self.support_


def build_selector(trial: "optuna.trial.Trial", feature_names: List[str]) -> InformationCoefficientSelector:
    """Construir seletor de features com hiperparâmetros do Optuna"""
    k = trial.suggest_int("top_k_features", 30, min(80, len(feature_names)))
    min_ic = trial.suggest_float("min_ic_threshold", 0.005, 0.05, log=True)
    
    return InformationCoefficientSelector(k=k, min_ic=min_ic)


def build_model(trial: "optuna.trial.Trial"):
    """Construir modelo LightGBM com hiperparâmetros do Optuna"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required for advanced modeling")
    
    # Hiperparâmetros específicos para dados financeiros
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': RANDOM_SEED,
        'deterministic': True,
        
        # Hiperparâmetros a otimizar
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        
        # Fixos para estabilidade
        'n_estimators': 200,  # Será ajustado por early stopping
        'max_depth': -1,
    }
    
    return lgb.LGBMRegressor(**params)


def calculate_trading_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    cost_per_trade: float = 0.0001, 
    entry_threshold: float = 0.001
) -> Dict[str, float]:
    """
    Calcular métricas de trading baseadas em predições e targets
    
    Args:
        predictions: Array de predições do modelo
        targets: Array de retornos reais
        cost_per_trade: Custo por trade (spread + comissão)
        entry_threshold: Threshold mínimo para entrada em posição
    
    Returns:
        Dict com métricas de trading
    """
    # Gerar sinais de trading
    signals = np.where(predictions > entry_threshold, 1, 
                      np.where(predictions < -entry_threshold, -1, 0))
    
    # Calcular retornos brutos
    gross_returns = signals * targets
    
    # Calcular número de trades (mudanças de posição)
    position_changes = np.diff(signals, prepend=0)
    n_trades = np.sum(np.abs(position_changes))
    
    # Calcular custos de transação
    transaction_costs = np.abs(position_changes) * cost_per_trade
    
    # PnL líquido
    net_returns = gross_returns - transaction_costs
    
    # Métricas básicas
    total_return = np.sum(net_returns)
    volatility = np.std(net_returns) if len(net_returns) > 1 else 0.0
    sharpe_ratio = (total_return / volatility * ANN_FACTOR) if volatility > 0 else 0.0
    
    # Turnover (frequência de trading)
    turnover = n_trades / len(predictions) if len(predictions) > 0 else 0.0
    
    # Maximum Drawdown
    mdd = max_drawdown(net_returns)
    
    # Information Coefficient
    ic, _ = spearmanr(predictions, targets) if len(predictions) > 10 else (0.0, 1.0)
    ic = ic if not np.isnan(ic) else 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'volatility': volatility,
        'max_drawdown': mdd,
        'turnover': turnover,
        'n_trades': n_trades,
        'information_coefficient': ic,
        'hit_ratio': np.mean((predictions * targets) > 0) if len(predictions) > 0 else 0.0
    }


def objective_study_a_advanced(
    trial: "optuna.trial.Trial",
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> Tuple[float, float, float]:
    """
    Função objetivo avançada para otimização multi-objetivo com validação temporal
    
    Returns:
        Tuple[float, float, float]: (sharpe_ratio, turnover, max_drawdown)
    """
    try:
        # Hiperparâmetros de trading
        cost_per_trade = trial.suggest_float('cost_per_trade', 0.00005, 0.0005, log=True)
        entry_threshold = trial.suggest_float('entry_threshold', 0.0005, 0.005, log=True)
        
        # Configuração de validação temporal
        n_splits = trial.suggest_int('n_splits', 3, 8)
        test_size_ratio = trial.suggest_float('test_size_ratio', 0.15, 0.3)
        
        # Calcular tamanho do teste
        test_size = int(len(X) * test_size_ratio)
        
        # Time Series Split com gap para evitar data leakage
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=10)
        
        # Armazenar resultados de cada fold
        fold_metrics = []
        selected_features_per_fold = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Dividir dados
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Construir pipeline
            selector = build_selector(trial, feature_names)
            model = build_model(trial)
            
            # Pipeline com scaler + seletor + modelo
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', selector),
                ('model', model)
            ])
            
            # Treinar pipeline
            pipeline.fit(X_fold_train, y_fold_train)
            
            # Predições
            val_predictions = pipeline.predict(X_fold_val)
            
            # Calcular métricas de trading
            metrics = calculate_trading_metrics(
                val_predictions, y_fold_val, 
                cost_per_trade=cost_per_trade,
                entry_threshold=entry_threshold
            )
            
            fold_metrics.append(metrics)
            
            # Armazenar features selecionadas
            selected_features = pipeline.named_steps['selector'].get_support(indices=True)
            selected_features_per_fold.append(selected_features)
            
            # Pruning baseado no Sharpe ratio do fold atual
            trial.report(metrics['sharpe_ratio'], fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Agregar métricas de todos os folds
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in fold_metrics])
        avg_turnover = np.mean([m['turnover'] for m in fold_metrics])
        avg_mdd = np.mean([m['max_drawdown'] for m in fold_metrics])
        avg_ic = np.mean([m['information_coefficient'] for m in fold_metrics])
        
        # Calcular estabilidade das features selecionadas
        all_features = set()
        for features in selected_features_per_fold:
            all_features.update(features)
        
        feature_stability = 0.0
        if len(all_features) > 0:
            stability_scores = []
            for feature in all_features:
                appearances = sum(1 for features in selected_features_per_fold if feature in features)
                stability_scores.append(appearances / len(selected_features_per_fold))
            feature_stability = np.mean(stability_scores)
        
        # Armazenar métricas adicionais no trial
        _set_trial_user_attr_safely(trial, 'avg_ic', avg_ic)
        _set_trial_user_attr_safely(trial, 'feature_stability', feature_stability)
        _set_trial_user_attr_safely(trial, 'n_features_avg', np.mean([len(f) for f in selected_features_per_fold]))
        _set_trial_user_attr_safely(trial, 'selected_features', selected_features_per_fold)
        
        # Retornar objetivos para otimização multi-objetivo
        # Maximizar Sharpe, Minimizar Turnover, Minimizar Max Drawdown
        return avg_sharpe, avg_turnover, avg_mdd
        
    except Exception as e:
        # Log do erro para debugging
        print(f"Erro no trial {trial.number}: {str(e)}")
        # Retornar valores ruins para que o trial seja rejeitado
        return -999.0, 999.0, 999.0
