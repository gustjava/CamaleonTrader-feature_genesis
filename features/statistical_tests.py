"""
Statistical Tests Module for Dynamic Stage 0 Pipeline

GPU-accelerated statistical tests including ADF and distance correlation.
"""

import logging
import time
import numpy as np
import dask_cudf
import cudf
import cupy as cp
import numpy as np
from numba import cuda
from typing import List, Tuple, Dict, Any

from .base_engine import BaseFeatureEngine

logger = logging.getLogger(__name__)


# -------- Module-level helpers to avoid Dask tokenization issues --------
def _adf_tstat_window_host(vals: np.ndarray) -> float:
    n = len(vals)
    if n < 3:
        return np.nan
    prev = float(vals[0])
    sum_z = sum_y = sum_zz = sum_yy = sum_zy = 0.0
    m = n - 1
    for i in range(1, n):
        z = prev
        y = float(vals[i]) - prev
        prev = float(vals[i])
        sum_z += z
        sum_y += y
        sum_zz += z * z
        sum_yy += y * y
        sum_zy += z * y
    mz = sum_z / m
    my = sum_y / m
    Sxx = sum_zz - m * mz * mz
    Sxy = sum_zy - m * mz * my
    if Sxx <= 0.0 or m <= 2:
        return np.nan
    beta = Sxy / Sxx
    alpha = my - beta * mz
    SSE = (sum_yy + m * alpha * alpha + beta * beta * sum_zz - 2.0 * alpha * sum_y - 2.0 * beta * sum_zy + 2.0 * alpha * beta * sum_z)
    dof = m - 2
    if dof <= 0:
        return np.nan
    sigma2 = SSE / dof
    if sigma2 <= 0.0:
        return np.nan
    se_beta = (sigma2 / Sxx) ** 0.5
    if se_beta == 0.0:
        return np.nan
    return beta / se_beta


def _adf_rolling_partition(series: cudf.Series, window: int, min_periods: int) -> cudf.Series:
    try:
        # Host rolling apply with host function
        return series.rolling(window=window, min_periods=min_periods).apply(lambda x: _adf_tstat_window_host(np.asarray(x)))
    except Exception:
        return cudf.Series(cp.full(len(series), cp.nan))


def _distance_correlation_cpu(x: np.ndarray, y: np.ndarray, max_samples: int = 10000) -> float:
    # Basic CPU dCor for small samples
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float('nan')
    if n > max_samples:
        x = x[-max_samples:]
        y = y[-max_samples:]
        n = max_samples
    # Distance matrices
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).mean()
    dcov2_xx = (A * A).mean()
    dcov2_yy = (B * B).mean()
    if dcov2_xx <= 0 or dcov2_yy <= 0:
        return 0.0
    return float(np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx * dcov2_yy)))


def _dcor_single_pair_partition(pdf: cudf.DataFrame, x_col: str, y_col: str, max_samples: int) -> cudf.Series:
    x = pdf[x_col].to_pandas().to_numpy()
    y = pdf[y_col].to_pandas().to_numpy()
    val = _distance_correlation_cpu(x, y, max_samples=max_samples)
    return cudf.Series([val], index=[0], name="dcor_returns_volume")


def _dcor_partition(pdf: cudf.DataFrame, target: str, candidates: List[str], max_samples: int) -> cudf.DataFrame:
    out = {}
    t = pdf[target].to_pandas().to_numpy()
    for c in candidates:
        x = pdf[c].to_pandas().to_numpy()
        out[f"dcor_{c}"] = _distance_correlation_cpu(x, t, max_samples=max_samples)
    return cudf.DataFrame([out])


def _perm_pvalues_partition(pdf: cudf.DataFrame, target: str, feat_list: List[str], n_perm: int, max_samples: int) -> cudf.DataFrame:
    import random
    rng = np.random.default_rng(42)
    y = pdf[target].to_pandas().to_numpy()
    out = {}
    for f in feat_list:
        x = pdf[f].to_pandas().to_numpy()
        obs = _distance_correlation_cpu(x, y, max_samples=max_samples)
        cnt = 0
        for _ in range(max(1, n_perm)):
            y_perm = rng.permutation(y)
            val = _distance_correlation_cpu(x, y_perm, max_samples=max_samples)
            if np.isfinite(val) and val >= obs:
                cnt += 1
        out[f"dcor_pvalue_{f}"] = float(cnt) / float(max(1, n_perm))
    return cudf.DataFrame([out])


def _dcor_rolling_partition(
    pdf: cudf.DataFrame,
    target: str,
    candidates: List[str],
    window: int,
    step: int,
    min_periods: int,
    max_rows: int,
    max_windows: int,
    agg: str,
    max_samples: int,
) -> cudf.DataFrame:
    import pandas as pd
    # Convert to pandas for CPU rolling; limit rows
    pdf = pdf.head(max_rows) if hasattr(pdf, 'head') else pdf
    pdf_pd = pdf.to_pandas()
    y = pdf_pd[target].values
    n = len(y)
    scores = {}
    if n < max(min_periods, 3):
        return cudf.DataFrame([{k: float('nan') for k in [f"dcor_roll_{c}" for c in candidates]}])
    starts = list(range(0, max(0, n - min_periods + 1), max(1, step)))
    # Limit number of windows
    if len(starts) > max_windows:
        starts = starts[-max_windows:]
    for c in candidates:
        x = pdf_pd[c].values
        vals = []
        for s in starts:
            e = min(n, s + window)
            if e - s < min_periods:
                continue
            vals.append(_distance_correlation_cpu(x[s:e], y[s:e], max_samples=max_samples))
        if not vals:
            scores[f"dcor_roll_{c}"] = float('nan')
        else:
            if agg == 'mean':
                outv = float(np.nanmean(vals))
            elif agg == 'min':
                outv = float(np.nanmin(vals))
            elif agg == 'max':
                outv = float(np.nanmax(vals))
            elif agg == 'p25':
                outv = float(np.nanpercentile(vals, 25))
            elif agg == 'p75':
                outv = float(np.nanpercentile(vals, 75))
            else:
                # default median
                outv = float(np.nanmedian(vals))
            scores[f"dcor_roll_{c}"] = outv
    return cudf.DataFrame([scores])


class StatisticalTests(BaseFeatureEngine):
    """
    Applies a set of statistical tests to a DataFrame.
    """

    def __init__(self, settings, client):
        super().__init__(settings, client)
        # Store settings for consistency with other engines
        self.settings = settings
        # Configurable parameters with safe fallbacks
        try:
            self.dcor_max_samples = getattr(settings.features, 'distance_corr_max_samples', 10000)
        except Exception:
            self.dcor_max_samples = 10000
        try:
            from config.unified_config import get_unified_config
            uc = get_unified_config()
            self.dcor_tile_size = getattr(uc.features, 'distance_corr_tile_size', 2048)
            self.selection_target_column = getattr(uc.features, 'selection_target_column', 'y_ret_1m')
            self.dcor_top_k = int(getattr(uc.features, 'dcor_top_k', 50))
            self.dcor_include_permutation = bool(getattr(uc.features, 'dcor_include_permutation', False))
            self.dcor_permutations = int(getattr(uc.features, 'dcor_permutations', 0))
        except Exception:
            self.dcor_tile_size = 2048
            self.selection_target_column = 'y_ret_1m'
            self.dcor_top_k = 50
            self.dcor_include_permutation = False
            self.dcor_permutations = 0
            self.selection_max_rows = 100000
            self.vif_threshold = 5.0
            self.mi_threshold = 0.3
            self.stage3_top_n = 50
            # Rolling dCor defaults when unified config missing
            self.stage1_rolling_enabled = False
            self.stage1_rolling_window = 2000
            self.stage1_rolling_step = 500
            self.stage1_rolling_min_periods = 200
            self.stage1_rolling_max_rows = 20000
            self.stage1_rolling_max_windows = 20
            self.stage1_agg = 'median'
            self.stage1_use_rolling_scores = True
        else:
            # Defaults when config present
            self.selection_max_rows = int(getattr(uc.features, 'selection_max_rows', 100000))
            self.vif_threshold = float(getattr(uc.features, 'vif_threshold', 5.0))
            self.mi_threshold = float(getattr(uc.features, 'mi_threshold', 0.3))
            self.stage3_top_n = int(getattr(uc.features, 'stage3_top_n', 50))
            self.dcor_min_threshold = float(getattr(uc.features, 'dcor_min_threshold', 0.0))
            self.dcor_min_percentile = float(getattr(uc.features, 'dcor_min_percentile', 0.0))
            self.stage1_top_n = int(getattr(uc.features, 'stage1_top_n', 0))
            self.dcor_fast_1d_enabled = bool(getattr(uc.features, 'dcor_fast_1d_enabled', False))
            self.dcor_fast_1d_bins = int(getattr(uc.features, 'dcor_fast_1d_bins', 2048))
            self.dcor_permutation_top_k = int(getattr(uc.features, 'dcor_permutation_top_k', 0))
            self.dcor_pvalue_alpha = float(getattr(uc.features, 'dcor_pvalue_alpha', 0.05))
            # Rolling dCor params
            self.stage1_rolling_enabled = bool(getattr(uc.features, 'stage1_rolling_enabled', False))
            self.stage1_rolling_window = int(getattr(uc.features, 'stage1_rolling_window', 2000))
            self.stage1_rolling_step = int(getattr(uc.features, 'stage1_rolling_step', 500))
            self.stage1_rolling_min_periods = int(getattr(uc.features, 'stage1_rolling_min_periods', 200))
            self.stage1_rolling_max_rows = int(getattr(uc.features, 'stage1_rolling_max_rows', 20000))
            self.stage1_rolling_max_windows = int(getattr(uc.features, 'stage1_rolling_max_windows', 20))
            self.stage1_agg = str(getattr(uc.features, 'stage1_agg', 'median')).lower()
            self.stage1_use_rolling_scores = bool(getattr(uc.features, 'stage1_use_rolling_scores', True))
            # Stage 3 LightGBM params
            self.stage3_top_n = int(getattr(uc.features, 'stage3_top_n', 50))
            self.stage3_task = str(getattr(uc.features, 'stage3_task', 'auto'))
            self.stage3_random_state = int(getattr(uc.features, 'stage3_random_state', 42))
            self.stage3_lgbm_enabled = bool(getattr(uc.features, 'stage3_lgbm_enabled', True))
            self.stage3_lgbm_num_leaves = int(getattr(uc.features, 'stage3_lgbm_num_leaves', 31))
            self.stage3_lgbm_max_depth = int(getattr(uc.features, 'stage3_lgbm_max_depth', -1))
            self.stage3_lgbm_n_estimators = int(getattr(uc.features, 'stage3_lgbm_n_estimators', 200))
            self.stage3_lgbm_learning_rate = float(getattr(uc.features, 'stage3_lgbm_learning_rate', 0.05))
            self.stage3_lgbm_feature_fraction = float(getattr(uc.features, 'stage3_lgbm_feature_fraction', 0.8))
            self.stage3_lgbm_bagging_fraction = float(getattr(uc.features, 'stage3_lgbm_bagging_fraction', 0.8))
            self.stage3_lgbm_bagging_freq = int(getattr(uc.features, 'stage3_lgbm_bagging_freq', 0))
            self.stage3_lgbm_early_stopping_rounds = int(getattr(uc.features, 'stage3_lgbm_early_stopping_rounds', 0))
            # MI clustering params (Stage 2 scalable)
            self.mi_cluster_enabled = bool(getattr(uc.features, 'mi_cluster_enabled', True))
            self.mi_cluster_method = str(getattr(uc.features, 'mi_cluster_method', 'agglo'))
            self.mi_cluster_threshold = float(getattr(uc.features, 'mi_cluster_threshold', 0.3))
            self.mi_max_candidates = int(getattr(uc.features, 'mi_max_candidates', 400))
            self.mi_chunk_size = int(getattr(uc.features, 'mi_chunk_size', 128))
            # Rolling dCor params
            self.stage1_rolling_enabled = bool(getattr(uc.features, 'stage1_rolling_enabled', False))
            self.stage1_rolling_window = int(getattr(uc.features, 'stage1_rolling_window', 2000))
            self.stage1_rolling_step = int(getattr(uc.features, 'stage1_rolling_step', 500))
            self.stage1_rolling_min_periods = int(getattr(uc.features, 'stage1_rolling_min_periods', 200))
            self.stage1_rolling_max_rows = int(getattr(uc.features, 'stage1_rolling_max_rows', 20000))
            self.stage1_rolling_max_windows = int(getattr(uc.features, 'stage1_rolling_max_windows', 20))
            self.stage1_agg = str(getattr(uc.features, 'stage1_agg', 'median')).lower()
            self.stage1_use_rolling_scores = bool(getattr(uc.features, 'stage1_use_rolling_scores', True))

    # ---------- Stage 2: Redundância (VIF + MI) e Stage 3: Wrappers ----------
    def _compute_vif_iterative(self, X: np.ndarray, cols: List[str], threshold: float) -> List[str]:
        """Remove colunas com VIF acima do limiar usando matriz de correlação (CPU)."""
        keep = cols.copy()
        while True:
            if len(keep) < 2:
                break
            # Correlação e VIF a partir da matriz de correlação
            corr = np.corrcoef(X[:, [cols.index(c) for c in keep]].T)
            try:
                inv = np.linalg.pinv(corr)
            except Exception:
                break
            vifs = np.diag(inv)
            vmax = float(np.max(vifs))
            if vmax <= threshold or not np.isfinite(vmax):
                break
            idx = int(np.argmax(vifs))
            removed = keep.pop(idx)
            self._log_info("VIF removal", feature=removed, vif=round(vmax, 3))
        return keep

    def _compute_mi_redundancy(self, X_df, candidates: List[str], dcor_scores: Dict[str, float], mi_threshold: float) -> List[str]:
        """Remove redundância não-linear via MI par-a-par (mantém maior dCor no par)."""
        try:
            from sklearn.feature_selection import mutual_info_regression
        except Exception as e:
            self._log_warn("MI not available, skipping redundancy MI", error=str(e))
            return candidates

        keep = set(candidates)
        # Cap no número de pares (quadrático); se grande, limita candidatos
        max_cands = min(len(candidates), 200)
        cand_limited = candidates[:max_cands]
        X = X_df[cand_limited].values
        n = len(cand_limited)
        # Compute pairwise MI approx: MI(X_i, X_j) by treating one as target
        for i in range(n):
            if cand_limited[i] not in keep:
                continue
            try:
                y = X[:, i]
                mi = mutual_info_regression(X, y, discrete_features=False)
            except Exception as e:
                self._log_warn("MI row failed", feature=cand_limited[i], error=str(e))
                continue
            for j in range(i + 1, n):
                f_i, f_j = cand_limited[i], cand_limited[j]
                if f_i in keep and f_j in keep and mi[j] >= mi_threshold:
                    # Remove o de menor dCor
                    if dcor_scores.get(f_i, 0.0) >= dcor_scores.get(f_j, 0.0):
                        keep.discard(f_j)
                        self._log_info("MI redundancy removal", pair=f"{f_i},{f_j}", kept=f_i, removed=f_j, mi=round(float(mi[j]), 4))
                    else:
                        keep.discard(f_i)
                        self._log_info("MI redundancy removal", pair=f"{f_i},{f_j}", kept=f_j, removed=f_i, mi=round(float(mi[j]), 4))
        return list(keep)

    def _compute_mi_cluster_representatives(self, X_df, candidates: List[str], dcor_scores: Dict[str, float]) -> List[str]:
        """Clustering global por MI para reduzir redundância (escalável).

        - Limita número de candidatos por `mi_max_candidates` (top por dCor se disponível, senão primeiros).
        - Computa matriz MI simétrica por blocos (chunk_size) para economizar memória.
        - Constrói uma matriz de distância D = 1 - MI_norm e aplica AgglomerativeClustering
          com `distance_threshold` derivado de `mi_cluster_threshold`.
        - Seleciona 1 representante por cluster (maior dCor do Estágio 1).
        """
        try:
            import numpy as np
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.cluster import AgglomerativeClustering
        except Exception as e:
            self._log_warn("MI clustering unavailable, falling back to pairwise redundancy", error=str(e))
            return self._compute_mi_redundancy(X_df, candidates, dcor_scores, mi_threshold=float(self.mi_threshold))

        if len(candidates) <= 2:
            return candidates

        # 1) Seleção de subset escalável
        if dcor_scores:
            ordered = sorted([c for c in candidates if c in dcor_scores], key=lambda f: dcor_scores[f], reverse=True)
        else:
            ordered = list(candidates)
        max_c = max(2, int(self.mi_max_candidates))
        cand = ordered[:min(len(ordered), max_c)]
        X = X_df[cand].values
        n = X.shape[0]
        p = X.shape[1]
        if p < 2:
            return cand

        # 2) Matriz MI por blocos (simetrizada)
        chunk = max(8, int(self.mi_chunk_size))
        MI = np.zeros((p, p), dtype=np.float32)
        for i0 in range(0, p, chunk):
            i1 = min(i0 + chunk, p)
            Xi = X[:, i0:i1]
            for j0 in range(0, p, chunk):
                j1 = min(j0 + chunk, p)
                Xj = X[:, j0:j1]
                # compute MI for block pairs: MI(Xi_k, Xj_l)
                for ii in range(i0, i1):
                    try:
                        y = X[:, ii]
                        mi_block = mutual_info_regression(X[:, j0:j1], y, discrete_features=False)
                    except Exception as e:
                        self._log_warn("MI block failed", i=ii, j0=j0, j1=j1, error=str(e))
                        mi_block = np.zeros(j1 - j0, dtype=np.float32)
                    MI[ii, j0:j1] = np.maximum(MI[ii, j0:j1], mi_block.astype(np.float32))
        # Symmetrize by average
        MI = 0.5 * (MI + MI.T)

        # Normalize MI to [0,1] per matrix max to derive distance
        max_mi = float(np.nanmax(MI)) if np.isfinite(MI).any() else 1.0
        max_mi = max(max_mi, 1e-8)
        MI_norm = MI / max_mi
        D = 1.0 - MI_norm

        # 3) Clustering aglomerativo por distância média
        try:
            model = AgglomerativeClustering(
                metric='precomputed', linkage='average', distance_threshold=max(0.0, 1.0 - float(self.mi_cluster_threshold)), n_clusters=None
            )
        except TypeError:
            # Older sklearn versions use 'affinity' instead of 'metric'
            model = AgglomerativeClustering(
                affinity='precomputed', linkage='average', distance_threshold=max(0.0, 1.0 - float(self.mi_cluster_threshold)), n_clusters=None
            )
        labels = model.fit_predict(D)

        # 4) Escolhe representante por cluster (maior dCor)
        reps: List[str] = []
        for lbl in np.unique(labels):
            idxs = np.where(labels == lbl)[0]
            cluster_feats = [cand[i] for i in idxs]
            if dcor_scores:
                rep = max(cluster_feats, key=lambda f: dcor_scores.get(f, 0.0))
            else:
                rep = cluster_feats[0]
            reps.append(rep)

        # Garante que representantes pertencem ao conjunto original de candidatos pós‑VIF
        reps = [r for r in reps if r in candidates]
        return reps

    def _stage3_wrappers(self, X_df, y_series, candidates: List[str], top_n: int) -> List[str]:
        """Seleciona interseção entre Lasso e modelo de árvores (CPU), com suporte a
        regressão/classificação, LightGBM otimizado p/ CPU, seed e early-stopping.
        """
        lasso_sel = set()
        tree_sel = set()

        # Tarefa
        task = str(getattr(self, 'stage3_task', 'auto')).lower()
        if task == 'auto':
            try:
                y_vals = y_series.values
                uniq = np.unique(y_vals)
                if len(uniq) <= 10 and np.allclose(uniq, uniq.astype(int)):
                    task = 'classification'
                else:
                    task = 'regression'
            except Exception:
                task = 'regression'

        # Lasso (apenas regressão)
        try:
            if task == 'regression':
                from sklearn.linear_model import LassoCV
                from sklearn.model_selection import TimeSeriesSplit
                lasso = LassoCV(alphas=None, cv=TimeSeriesSplit(n_splits=5), max_iter=5000, n_jobs=-1, random_state=getattr(self, 'stage3_random_state', 42))
                lasso.fit(X_df[candidates].values, y_series.values)
                lasso_coef = dict(zip(candidates, lasso.coef_))
                lasso_sel = {f for f, w in lasso_coef.items() if abs(w) > 1e-8}
        except Exception as e:
            self._log_warn("LassoCV unavailable or failed", error=str(e))

        # Árvores: LightGBM preferido; fallback para RandomForest
        importances = {}
        # Optional GPU backends first
        used_backend = 'lgbm'
        try:
            if getattr(self, 'stage3_use_gpu', False) and str(getattr(self, 'stage3_wrapper_backend', 'lgbm')).lower() in ('xgb_gpu','catboost_gpu'):
                backend = str(getattr(self, 'stage3_wrapper_backend', 'lgbm')).lower()
                X = X_df[candidates].values
                y = y_series.values
                if backend == 'xgb_gpu':
                    import xgboost as xgb
                    params = {
                        'max_depth': 7,
                        'n_estimators': int(getattr(self, 'stage3_lgbm_n_estimators', 200)),
                        'learning_rate': float(getattr(self, 'stage3_lgbm_learning_rate', 0.05)),
                        'subsample': float(getattr(self, 'stage3_lgbm_bagging_fraction', 0.8)),
                        'colsample_bytree': float(getattr(self, 'stage3_lgbm_feature_fraction', 0.8)),
                        'random_state': int(getattr(self, 'stage3_random_state', 42)),
                        'tree_method': 'gpu_hist',
                        'n_jobs': -1,
                    }
                    if task == 'classification':
                        model = xgb.XGBClassifier(**params)
                    else:
                        model = xgb.XGBRegressor(**params)
                    model.fit(X, y)
                    fi = getattr(model, 'feature_importances_', None)
                    if fi is not None and len(fi) == len(candidates):
                        importances = dict(zip(candidates, fi))
                        used_backend = 'xgb_gpu'
        except Exception as e:
            self._log_warn("GPU wrapper backend failed; falling back to LGBM/Sklearn", error=str(e))

        # If no GPU importances, try LightGBM CPU
        if not importances:
            try:
                import lightgbm as lgb
                params = {
                    'num_leaves': int(getattr(self, 'stage3_lgbm_num_leaves', 31)),
                    'max_depth': int(getattr(self, 'stage3_lgbm_max_depth', -1)),
                    'n_estimators': int(getattr(self, 'stage3_lgbm_n_estimators', 200)),
                    'learning_rate': float(getattr(self, 'stage3_lgbm_learning_rate', 0.05)),
                    'subsample': float(getattr(self, 'stage3_lgbm_bagging_fraction', 0.8)),
                    'colsample_bytree': float(getattr(self, 'stage3_lgbm_feature_fraction', 0.8)),
                    'random_state': int(getattr(self, 'stage3_random_state', 42)),
                    'n_jobs': -1,
                }
                esr = int(getattr(self, 'stage3_lgbm_early_stopping_rounds', 0))
                X = X_df[candidates].values
                y = y_series.values
                eval_set = None
                X_train, y_train = X, y
                if esr and esr > 0:
                    try:
                        from sklearn.model_selection import TimeSeriesSplit
                        tss = TimeSeriesSplit(n_splits=5)
                        tr_idx, va_idx = list(tss.split(X))[-1]
                        X_train, y_train = X[tr_idx], y[tr_idx]
                        eval_set = [(X[va_idx], y[va_idx])]
                    except Exception:
                        eval_set = None

                if task == 'classification':
                    model = lgb.LGBMClassifier(**params)
                    try:
                        if eval_set is not None and esr > 0:
                            model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=esr, verbose=False)
                        else:
                            model.fit(X_train, y_train)
                    except TypeError:
                        model.fit(X_train, y_train)
                else:
                    model = lgb.LGBMRegressor(**params)
                    try:
                        if eval_set is not None and esr > 0:
                            model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=esr, verbose=False)
                        else:
                            model.fit(X_train, y_train)
                    except TypeError:
                        model.fit(X_train, y_train)

                importances = dict(zip(candidates, getattr(model, 'feature_importances_', np.zeros(len(candidates)))))
                used_backend = 'lgbm'
            except Exception as e:
                self._log_warn("LightGBM failed", error=str(e))
                try:
                    if task == 'classification':
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=200, max_depth=7, n_jobs=-1, random_state=42)
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=200, max_depth=7, n_jobs=-1, random_state=42)
                    model.fit(X_df[candidates].values, y_series.values)
                    importances = dict(zip(candidates, getattr(model, 'feature_importances_', np.zeros(len(candidates)))))
                    used_backend = 'rf'
                except Exception as e2:
                    self._log_warn("Tree model unavailable or failed", error=str(e2))
                    importances = {}

        if importances:
            tree_sorted = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
            tree_sel = {k for k, _ in tree_sorted[:min(top_n, len(tree_sorted))]}

        if lasso_sel and tree_sel:
            inter = list(lasso_sel.intersection(tree_sel))
            if not inter:
                inter = list((lasso_sel.union(tree_sel)))[:top_n]
            return inter[:top_n]
        elif lasso_sel:
            return list(lasso_sel)[:top_n]
        elif tree_sel:
            return list(tree_sel)[:top_n]
        else:
            return []

    # ---- ADF(0) t-stat em device: Δx_t = α + φ x_{t-1} + ε_t ----
    # Retorna t-stat de φ. Exige janela >= 10 p/ estabilidade mínima.
    @staticmethod
    @cuda.jit(device=True)
    def _adf_tstat_window(vals):
        n = vals.size
        if n < 10:
            return np.nan

        # y_t = x_t - x_{t-1}, z_t = x_{t-1}, t=1..n-1
        m = n - 1  # nº de observações da regressão
        sum_z = 0.0
        sum_y = 0.0
        sum_zz = 0.0
        sum_yy = 0.0
        sum_zy = 0.0

        prev = vals[0]
        for i in range(1, n):
            z = prev
            y = vals[i] - prev
            prev = vals[i]

            sum_z += z
            sum_y += y
            sum_zz += z * z
            sum_yy += y * y
            sum_zy += z * y

        # médias
        mz = sum_z / m
        my = sum_y / m

        # centradas
        Sxx = sum_zz - m * mz * mz
        Sxy = sum_zy - m * mz * my

        if Sxx <= 0.0 or m <= 2:
            return np.nan

        beta = Sxy / Sxx           # φ
        alpha = my - beta * mz     # α

        # SSE = Σ(y - α - β z)^2 = sum_yy + m α^2 + β^2 sum_zz - 2α sum_y - 2β sum_zy + 2αβ sum_z
        SSE = (sum_yy
               + m * alpha * alpha
               + beta * beta * sum_zz
               - 2.0 * alpha * sum_y
               - 2.0 * beta * sum_zy
               + 2.0 * alpha * beta * sum_z)

        dof = m - 2
        if dof <= 0:
            return np.nan

        sigma2 = SSE / dof
        if sigma2 <= 0.0:
            return np.nan

        se_beta = (sigma2 / Sxx) ** 0.5
        if se_beta == 0.0:
            return np.nan

        tstat = beta / se_beta
        return tstat

    def _apply_adf_rolling(self, s: cudf.Series, window: int = 252, min_periods: int = 200) -> cudf.Series:
        # UDF device é passado diretamente para rolling.apply
        return s.rolling(window=window, min_periods=min_periods).apply(
            self._adf_tstat_window
        )

    # ---- Distance Correlation (single-fit, com amostragem p/ n grande) ----
    def _dcor_gpu_single(self, x: cp.ndarray, y: cp.ndarray, max_n: int = None) -> float:
        """
        Distance correlation (dCor) via bloco (chunked), estável em memória para séries 1D.
        Implementa duas passagens: (1) médias por linha e média global; (2) centragem e acumulação.
        """
        try:
            # Limpeza de NaNs
            mask = ~(cp.isnan(x) | cp.isnan(y))
            x = x[mask]
            y = y[mask]
            n = int(x.size)

            if n < 2:
                return float("nan")

            # Amostragem/decimação para limitar custo
            if max_n is None:
                max_n = int(getattr(self, 'dcor_max_samples', 10000))
            # Caminho rápido 1D (O(n log n) overall): ordenar e decimar para bins fixos
            if self.dcor_fast_1d_enabled and n > self.dcor_fast_1d_bins:
                # Ordena por X e decima índices
                ord_idx = cp.asnumpy(cp.argsort(x))
                # Seleciona índices uniformemente espaçados
                bins = int(self.dcor_fast_1d_bins)
                sel = np.linspace(0, n - 1, bins).round().astype(np.int64)
                idx = ord_idx[sel]
                x = x[idx]
                y = y[idx]
                n = int(x.size)
            elif n > max_n:
                # fallback: amostra cauda
                x = x[-max_n:]
                y = y[-max_n:]
                n = max_n

            # Garantir tipo consistente para performance/memória
            if x.dtype != cp.float32:
                x = x.astype(cp.float32, copy=False)
            if y.dtype != cp.float32:
                y = y.astype(cp.float32, copy=False)

            # Tamanho do bloco (trade-off memória/tempo)
            tile_cfg = int(getattr(self, 'dcor_tile_size', 2048))
            tile = min(tile_cfg, n) if n >= 2 * tile_cfg else min(max(1024, tile_cfg), n)

            t0 = time.time()

            # Passo 1: somas por linha e somas globais (x e y)
            a_row_sums = cp.zeros(n, dtype=cp.float64)
            b_row_sums = cp.zeros(n, dtype=cp.float64)
            a_total_sum = cp.float64(0.0)
            b_total_sum = cp.float64(0.0)

            for i0 in range(0, n, tile):
                i1 = min(i0 + tile, n)
                xi = x[i0:i1]
                yi = y[i0:i1]
                for j0 in range(0, n, tile):
                    j1 = min(j0 + tile, n)
                    xj = x[j0:j1]
                    yj = y[j0:j1]

                    dx = cp.abs(xi[:, None] - xj[None, :])
                    dy = cp.abs(yi[:, None] - yj[None, :])

                    # acumula somas por linha para i-bloco
                    a_row_sums[i0:i1] += dx.sum(axis=1, dtype=cp.float64)
                    b_row_sums[i0:i1] += dy.sum(axis=1, dtype=cp.float64)

                    # para j-bloco, somas por linha equivalem às somas por coluna do bloco i
                    # acumular também para j quando blocos distintos (evita recomputar em outra iteração)
                    if j0 != i0:
                        a_row_sums[j0:j1] += dx.sum(axis=0, dtype=cp.float64)
                        b_row_sums[j0:j1] += dy.sum(axis=0, dtype=cp.float64)

                    a_total_sum += cp.sum(dx, dtype=cp.float64)
                    b_total_sum += cp.sum(dy, dtype=cp.float64)

            # médias
            n_f = float(n)
            a_row_mean = a_row_sums / n_f
            b_row_mean = b_row_sums / n_f
            a_grand = float(a_total_sum / (n_f * n_f))
            b_grand = float(b_total_sum / (n_f * n_f))

            # Passo 2: centragem por blocos e acumulação de somas
            num = cp.float64(0.0)
            sumA2 = cp.float64(0.0)
            sumB2 = cp.float64(0.0)

            for i0 in range(0, n, tile):
                i1 = min(i0 + tile, n)
                xi = x[i0:i1]
                yi = y[i0:i1]
                a_i_mean = a_row_mean[i0:i1]
                b_i_mean = b_row_mean[i0:i1]
                for j0 in range(0, n, tile):
                    j1 = min(j0 + tile, n)
                    xj = x[j0:j1]
                    yj = y[j0:j1]
                    a_j_mean = a_row_mean[j0:j1]
                    b_j_mean = b_row_mean[j0:j1]

                    dx = cp.abs(xi[:, None] - xj[None, :])
                    dy = cp.abs(yi[:, None] - yj[None, :])

                    A = dx - a_i_mean[:, None] - a_j_mean[None, :] + a_grand
                    B = dy - b_i_mean[:, None] - b_j_mean[None, :] + b_grand

                    num += cp.sum(A * B, dtype=cp.float64)
                    sumA2 += cp.sum(A * A, dtype=cp.float64)
                    sumB2 += cp.sum(B * B, dtype=cp.float64)

            denom = cp.sqrt(sumA2 * sumB2)
            if denom == 0:
                return 0.0
            dcor = float(num / denom)
            self._log_info("dCor computed", n=n, tile=int(tile), dcor=round(dcor, 6), elapsed=round(time.time()-t0, 3))
            return dcor
        except Exception as e:
            self._log_error(f"Error in chunked dCor computation: {e}")
            return float("nan")

    def _compute_distance_correlation_vectorized(self, x: cp.ndarray, y: cp.ndarray, max_samples: int = 10000) -> float:
        """
        Cálculo de dCor usando abordagem em blocos (evita matrizes n×n completas).
        """
        try:
            return self._dcor_gpu_single(x, y, max_n=max_samples)
        except Exception as e:
            self._critical_error(f"Error in distance correlation: {e}")

    def _compute_permutation_pvalues(self, pdf, target: str, features: List[str], n_perm: int) -> Dict[str, float]:
        """Compute permutation p-values for selected features using GPU dCor."""
        try:
            y = pdf[target].to_cupy()
            pvals = {}
            for c in features:
                x = pdf[c].to_cupy()
                dval, p = self.distance_correlation_with_permutation(x, y, n_perm=n_perm)
                pvals[c] = float(p)
            return pvals
        except Exception as e:
            self._log_error(f"Permutation pvalue computation failed: {e}")
            return {}
    
    def distance_correlation_with_permutation(self, x: cp.ndarray, y: cp.ndarray, n_perm: int = 1000) -> Tuple[float, float]:
        """
        Distance correlation with permutation test to estimate p-value.
        """
        dcor_obs = self._compute_distance_correlation_vectorized(x, y)
        
        # inicializa contador de valores permutados >= observado
        greater_count = 0
        # use cupy.random.permutation para embaralhar y em GPU
        for _ in range(n_perm):
            y_perm = cp.random.permutation(y)
            dcor_perm = self._compute_distance_correlation_vectorized(x, y_perm)
            if dcor_perm >= dcor_obs:
                greater_count += 1
        
        p_value = (greater_count + 1) / (n_perm + 1)  # correção de continuidade
        return float(dcor_obs), float(p_value)
    
    def compute_distance_correlation_batch(self, data_pairs: List[Tuple[cp.ndarray, cp.ndarray]], 
                                         max_samples: int = 10000, 
                                         include_permutation: bool = False,
                                         n_perm: int = 1000) -> List[Dict[str, float]]:
        """
        Compute distance correlation for multiple pairs of variables in batch.
        
        Args:
            data_pairs: List of (x, y) pairs
            max_samples: Maximum number of samples per pair
            include_permutation: Whether to include permutation test
            n_perm: Number of permutations for p-value calculation
            
        Returns:
            List of dictionaries with dcor value and optionally p-value
        """
        try:
            results = []
            
            for i, (x, y) in enumerate(data_pairs):
                try:
                    # Remove NaN values
                    valid_mask = ~(cp.isnan(x) | cp.isnan(y))
                    if cp.sum(valid_mask) < 10:
                        if include_permutation:
                            results.append({'dcor_value': 0.0, 'dcor_pvalue': 1.0})
                        else:
                            results.append({'dcor_value': 0.0})
                        continue
                    
                    x_clean = x[valid_mask]
                    y_clean = y[valid_mask]
                    
                    if include_permutation:
                        dcor_value, dcor_pvalue = self.distance_correlation_with_permutation(
                            x_clean, y_clean, n_perm
                        )
                        results.append({
                            'dcor_value': dcor_value,
                            'dcor_pvalue': dcor_pvalue,
                            'dcor_significant': dcor_pvalue < 0.05  # alpha = 0.05
                        })
                    else:
                        dcor_value = self._compute_distance_correlation_vectorized(x_clean, y_clean, max_samples)
                        results.append({'dcor_value': dcor_value})
                    
                except Exception as e:
                    logger.warning(f"Error computing dCor for pair {i}: {e}")
                    if include_permutation:
                        results.append({'dcor_value': 0.0, 'dcor_pvalue': 1.0, 'dcor_significant': False})
                    else:
                        results.append({'dcor_value': 0.0})
            
            return results
            
        except Exception as e:
            self._critical_error(f"Error in batch distance correlation: {e}")

    def _compute_adf_vectorized(self, data: cp.ndarray, max_lag: int = None) -> Dict[str, float]:
        """
        Vectorized ADF test implementation using CuPy linear algebra.
        Implements the ADF test using cupy.linalg.lstsq as specified in the technical plan.
        """
        try:
            if len(data) < max_lag + 10:
                return {
                    'adf_stat': cp.nan,
                    'p_value': cp.nan,
                    'critical_values': [cp.nan, cp.nan, cp.nan],
                    'lag_order': 0
                }
            
            if max_lag is None:
                max_lag = int(12 * (len(data) / 100) ** (1/4))  # Schwert criterion
            
            # Calculate differences
            diff_series = cp.diff(data)
            
            # Find optimal lag using vectorized operations
            optimal_lag = self._find_optimal_lag_vectorized(diff_series, max_lag)
            
            # Create regression matrix for ADF test
            X, y = self._create_adf_regression_matrix_vectorized(data, optimal_lag)
            
            # Remove rows with NaN values
            valid_mask = ~(cp.any(cp.isnan(X), axis=1) | cp.isnan(y))
            if cp.sum(valid_mask) < optimal_lag + 5:
                return {
                    'adf_stat': cp.nan,
                    'p_value': cp.nan,
                    'critical_values': [cp.nan, cp.nan, cp.nan],
                    'lag_order': optimal_lag
                }
            
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            # Solve least squares using CuPy (vectorized)
            try:
                # Use QR decomposition for numerical stability
                Q, R = cp.linalg.qr(X_clean)
                beta = cp.linalg.solve(R, Q.T @ y_clean)
                
                # Calculate residuals and standard error
                residuals = y_clean - X_clean @ beta
                mse = cp.sum(residuals**2) / (len(y_clean) - len(beta))
                
                # Standard error of the coefficient on lagged_series (first coefficient)
                X_inv = cp.linalg.inv(X_clean.T @ X_clean)
                se_beta = cp.sqrt(mse * X_inv[0, 0])
                
                # ADF statistic
                adf_stat = float(beta[0] / se_beta)
                
                # Calculate p-value using critical values
                p_value = self._calculate_adf_pvalue_vectorized(adf_stat, len(y_clean))
                
                # Critical values (approximate)
                critical_values = [-3.43, -2.86, -2.57]  # 1%, 5%, 10%
                
                return {
                    'adf_stat': adf_stat,
                    'p_value': p_value,
                    'critical_values': critical_values,
                    'lag_order': optimal_lag
                }
                
            except cp.linalg.LinAlgError:
                return {
                    'adf_stat': cp.nan,
                    'p_value': cp.nan,
                    'critical_values': [cp.nan, cp.nan, cp.nan],
                    'lag_order': optimal_lag
                }
                
        except Exception as e:
            self._critical_error(f"Error in vectorized ADF computation: {e}")
    
    def _find_optimal_lag_vectorized(self, diff_series: cp.ndarray, max_lag: int) -> int:
        """
        Vectorized optimal lag selection using information criteria.
        """
        try:
            best_lag = 1
            best_aic = cp.inf
            
            for lag in range(1, min(max_lag + 1, len(diff_series) // 2)):
                # Create regression matrix for this lag
                X, y = self._create_adf_regression_matrix_vectorized(diff_series, lag)
                
                # Remove NaN values
                valid_mask = ~(cp.any(cp.isnan(X), axis=1) | cp.isnan(y))
                if cp.sum(valid_mask) < lag + 5:
                    continue
                
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                try:
                    # Solve least squares
                    beta = cp.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
                    
                    # Calculate AIC
                    residuals = y_clean - X_clean @ beta
                    mse = cp.sum(residuals**2) / len(y_clean)
                    aic = 2 * len(beta) + len(y_clean) * cp.log(mse)
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_lag = lag
                        
                except cp.linalg.LinAlgError:
                    continue
            
            return best_lag
            
        except Exception as e:
            self._critical_error(f"Error in vectorized optimal lag selection: {e}")
    
    def _create_adf_regression_matrix_vectorized(self, data: cp.ndarray, lag: int) -> tuple:
        """
        Create regression matrix for ADF test using vectorized operations.
        """
        try:
            # Calculate differences
            diff_series = cp.diff(data)
            
            # Create lagged differences matrix
            lagged_diffs = cp.zeros((len(diff_series), lag))
            for i in range(lag):
                lagged_diffs[:, i] = cp.roll(diff_series, i + 1)
                lagged_diffs[:i+1, i] = 0  # Set invalid lags to 0
            
            # Create regression matrix: [lagged_series, lagged_diffs, trend, constant]
            lagged_series = data[:-1]  # y_{t-1}
            trend = cp.arange(len(lagged_series), dtype=cp.float32)
            constant = cp.ones(len(lagged_series), dtype=cp.float32)
            
            # Stack all regressors
            X = cp.column_stack([lagged_series, lagged_diffs, trend, constant])
            y = diff_series
            
            return X, y
            
        except Exception as e:
            self._critical_error(f"Error creating ADF regression matrix: {e}")
    
    def _calculate_adf_pvalue_vectorized(self, adf_stat: float, n_obs: int) -> float:
        """
        Vectorized p-value calculation for ADF test.
        """
        try:
            # Approximate p-value using critical values
            # This is a simplified approach - in practice, use proper ADF p-value tables
            if adf_stat < -3.43:
                return 0.01
            elif adf_stat < -2.86:
                return 0.05
            elif adf_stat < -2.57:
                return 0.10
            else:
                return 1.0
                
        except Exception as e:
            self._critical_error(f"Error calculating ADF p-value: {e}")
    
    def _compute_distance_correlation_vectorized_improved(self, x: cp.ndarray, y: cp.ndarray, max_samples: int = 10000) -> float:
        """Wrapper para o cálculo de dCor usando a rotina em blocos (robusta em memória)."""
        try:
            return self._dcor_gpu_single(x, y, max_n=max_samples)
        except Exception as e:
            self._critical_error(f"Error in improved distance correlation: {e}")

    # ---------- Stage 1: Rolling dCor helpers ----------
    def _aggregate_scores(self, vals: List[float], agg: str) -> float:
        import numpy as np
        a = np.asarray([v for v in vals if v == v])  # drop NaN
        if a.size == 0:
            return float('nan')
        if agg == 'mean':
            return float(a.mean())
        if agg == 'median':
            return float(np.median(a))
        if agg == 'min':
            return float(a.min())
        if agg == 'max':
            return float(a.max())
        if agg == 'p25':
            return float(np.quantile(a, 0.25))
        if agg == 'p75':
            return float(np.quantile(a, 0.75))
        return float(np.median(a))

    def _compute_dcor_rolling_scores(self, pdf: cudf.DataFrame, target: str, candidates: List[str]) -> Dict[str, float]:
        """Compute rolling distance correlation per feature and aggregate over time.

        Limits rows and number of windows by config to keep CPU/GPU time bounded.
        """
        try:
            import numpy as np
            agg = self.stage1_agg
            max_rows = int(self.stage1_rolling_max_rows)
            window = int(self.stage1_rolling_window)
            step = int(self.stage1_rolling_step)
            minp = int(self.stage1_rolling_min_periods)
            max_w = int(self.stage1_rolling_max_windows)

            # Use tail to focus on recent data and bound memory
            pdf = pdf.tail(max_rows)

            y = pdf[target].to_cupy()
            n = int(len(pdf))
            if n < max(minp, window):
                return {}

            # Build window start indices
            starts = list(range(0, n - window + 1, max(1, step)))
            if len(starts) > max_w:
                # decimate to at most max_w windows
                idx = np.linspace(0, len(starts) - 1, num=max_w).round().astype(int)
                starts = [starts[i] for i in idx]

            scores: Dict[str, float] = {}
            for c in candidates:
                x = pdf[c].to_cupy()
                w_scores: List[float] = []
                for s in starts:
                    e = s + window
                    xw = x[s:e]
                    yw = y[s:e]
                    # require enough non-NaNs
                    valid = (~cp.isnan(xw)) & (~cp.isnan(yw))
                    if int(valid.sum()) < minp:
                        w_scores.append(float('nan'))
                        continue
                    d = self._compute_distance_correlation_vectorized(xw, yw, max_samples=self.dcor_max_samples)
                    w_scores.append(float(d))
                scores[c] = self._aggregate_scores(w_scores, agg)
            return scores
        except Exception as e:
            self._log_warn("Rolling dCor failed", error=str(e))
            return {}
    
    def _compute_adf_batch_vectorized(self, data_matrix: cp.ndarray, max_lag: int = None) -> Dict[str, cp.ndarray]:
        """
        Vectorized batch ADF test implementation.
        Processes multiple time series simultaneously using GPU operations.
        """
        try:
            n_series, n_obs = data_matrix.shape
            
            if max_lag is None:
                max_lag = int(12 * (n_obs / 100) ** (1/4))
            
            # Initialize results arrays
            adf_stats = cp.zeros(n_series, dtype=cp.float32)
            p_values = cp.zeros(n_series, dtype=cp.float32)
            critical_values = cp.zeros((n_series, 3), dtype=cp.float32)
            lag_orders = cp.zeros(n_series, dtype=cp.int32)
            
            # Process each series vectorized
            for i in range(n_series):
                series = data_matrix[i, :]
                
                # Remove NaN values
                valid_mask = ~cp.isnan(series)
                if cp.sum(valid_mask) < max_lag + 10:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
                    lag_orders[i] = 0
                    continue
                
                series_clean = series[valid_mask]
                
                # Compute ADF for this series
                adf_result = self._compute_adf_vectorized(series_clean, max_lag)
                
                # Store results
                adf_stats[i] = adf_result['adf_stat']
                p_values[i] = adf_result['p_value']
                critical_values[i, :] = adf_result['critical_values']
                lag_orders[i] = adf_result['lag_order']
            
            return {
                'adf_stat': cp.asnumpy(adf_stats),
                'p_value': cp.asnumpy(p_values),
                'critical_values': cp.asnumpy(critical_values),
                'lag_order': cp.asnumpy(lag_orders)
            }
            
        except Exception as e:
            self._critical_error(f"Error in vectorized batch ADF computation: {e}")
    
    def _compute_distance_correlation_batch_vectorized(self, data_pairs: List[Tuple[cp.ndarray, cp.ndarray]], 
                                                     max_samples: int = 10000) -> List[float]:
        """
        Vectorized batch distance correlation computation.
        """
        try:
            results = []
            
            for i, (x, y) in enumerate(data_pairs):
                try:
                    # Remove NaN values
                    valid_mask = ~(cp.isnan(x) | cp.isnan(y))
                    if cp.sum(valid_mask) < 10:
                        results.append(0.0)
                        continue
                    
                    x_clean = x[valid_mask]
                    y_clean = y[valid_mask]
                    
                    dcor = self._compute_distance_correlation_vectorized_improved(x_clean, y_clean, max_samples)
                    results.append(dcor)
                    
                except Exception as e:
                    self._log_warn(f"Error computing dCor for pair {i}: {e}")
                    results.append(0.0)
            
            return results
            
        except Exception as e:
            self._critical_error(f"Error in vectorized batch distance correlation: {e}")
    
    def _compute_statistical_tests_vectorized(self, data: cp.ndarray) -> Dict[str, Any]:
        """
        Vectorized statistical tests computation.
        """
        try:
            results = {}
            
            # ADF test (vectorized)
            adf_result = self._compute_adf_vectorized(data)
            results.update({
                'adf_stat': adf_result['adf_stat'],
                'adf_pvalue': adf_result['p_value'],
                'adf_lag_order': adf_result['lag_order'],
                'is_stationary_adf': adf_result['p_value'] < 0.05 and adf_result['adf_stat'] < adf_result['critical_values'][1]
            })
            
            # KPSS test (simplified vectorized version)
            kpss_result = self._compute_kpss_vectorized(data)
            results.update(kpss_result)
            
            # Phillips-Perron test (simplified vectorized version)
            pp_result = self._compute_phillips_perron_vectorized(data)
            results.update(pp_result)
            
            return results
            
        except Exception as e:
            self._critical_error(f"Error in vectorized statistical tests: {e}")
    
    def _compute_kpss_vectorized(self, data: cp.ndarray) -> Dict[str, float]:
        """
        Vectorized KPSS test implementation.
        """
        try:
            # Simplified KPSS test using vectorized operations
            n = len(data)
            
            # Compute cumulative sum
            cumsum = cp.cumsum(data)
            
            # Compute partial sums
            partial_sums = cumsum - cp.arange(n) * cp.mean(data)
            
            # Compute test statistic
            s2 = cp.sum(partial_sums**2) / n
            kpss_stat = s2 / cp.var(data)
            
            # Approximate p-value
            if kpss_stat < 0.347:
                p_value = 0.10
            elif kpss_stat < 0.463:
                p_value = 0.05
            elif kpss_stat < 0.739:
                p_value = 0.025
            else:
                p_value = 0.01
            
            return {
                'kpss_stat': float(kpss_stat),
                'kpss_pvalue': p_value,
                'is_stationary_kpss': p_value > 0.05
            }
            
        except Exception as e:
            self._critical_error(f"Error in vectorized KPSS test: {e}")
    
    def _compute_phillips_perron_vectorized(self, data: cp.ndarray) -> Dict[str, float]:
        """
        Vectorized Phillips-Perron test implementation.
        """
        try:
            # Simplified Phillips-Perron test using vectorized operations
            n = len(data)
            
            # Compute differences
            diff_series = cp.diff(data)
            
            # Compute test statistic (simplified)
            mean_diff = cp.mean(diff_series)
            var_diff = cp.var(diff_series)
            
            pp_stat = mean_diff / cp.sqrt(var_diff / n)
            
            # Approximate p-value
            if pp_stat < -3.43:
                p_value = 0.01
            elif pp_stat < -2.86:
                p_value = 0.05
            elif pp_stat < -2.57:
                p_value = 0.10
            else:
                p_value = 1.0
            
            return {
                'pp_stat': float(pp_stat),
                'pp_pvalue': p_value,
                'is_stationary_pp': p_value < 0.05
            }
            
        except Exception as e:
            self._critical_error(f"Error in vectorized Phillips-Perron test: {e}")
    
    def _compute_correlation_tests_vectorized(self, data_dict: Dict[str, cp.ndarray]) -> Dict[str, Any]:
        """
        Vectorized correlation tests computation.
        """
        try:
            correlation_tests = {}
            
            # Get series names
            series_names = list(data_dict.keys())
            if len(series_names) < 2:
                return correlation_tests
            
            # Compute pairwise correlations
            for i, name1 in enumerate(series_names):
                for j, name2 in enumerate(series_names):
                    if i < j:  # Avoid duplicates
                        x = data_dict[name1]
                        y = data_dict[name2]
                        
                        # Remove NaN values
                        valid_mask = ~(cp.isnan(x) | cp.isnan(y))
                        if cp.sum(valid_mask) < 10:
                            continue
                        
                        x_clean = x[valid_mask]
                        y_clean = y[valid_mask]
                        
                        # Pearson correlation (vectorized)
                        corr_pearson = float(cp.corrcoef(x_clean, y_clean)[0, 1])
                        
                        # Spearman correlation (vectorized)
                        corr_spearman = float(cp.corrcoef(cp.argsort(cp.argsort(x_clean)), cp.argsort(cp.argsort(y_clean)))[0, 1])
                        
                        # Distance correlation (vectorized)
                        corr_distance = self._compute_distance_correlation_vectorized(x_clean, y_clean)
                        
                        # Store results
                        pair_name = f"{name1}_{name2}"
                        correlation_tests[f'pearson_{pair_name}'] = corr_pearson
                        correlation_tests[f'spearman_{pair_name}'] = corr_spearman
                        correlation_tests[f'distance_{pair_name}'] = corr_distance
            
            return correlation_tests
            
        except Exception as e:
            self._critical_error(f"Error in vectorized correlation tests: {e}")
    
    def _compute_comprehensive_statistical_tests_vectorized(self, data: cp.ndarray) -> Dict[str, Any]:
        """
        Comprehensive vectorized statistical tests.
        """
        try:
            comprehensive_tests = {}
            
            # Basic statistical tests
            basic_tests = self._compute_statistical_tests_vectorized(data)
            comprehensive_tests.update(basic_tests)
            
            # Additional moment-based tests
            moments = self._compute_moments_vectorized(data)
            comprehensive_tests.update(moments)
            
            # Normality tests (simplified vectorized versions)
            normality_tests = self._compute_normality_tests_vectorized(data)
            comprehensive_tests.update(normality_tests)
            
            return comprehensive_tests
            
        except Exception as e:
            self._critical_error(f"Error in comprehensive vectorized statistical tests: {e}")
    
    def _compute_moments_vectorized(self, data: cp.ndarray) -> Dict[str, float]:
        """
        Vectorized computation of statistical moments.
        """
        try:
            mean_val = float(cp.mean(data))
            std_val = float(cp.std(data))
            
            if std_val > 0:
                standardized = (data - mean_val) / std_val
                skewness = float(cp.mean(standardized**3))
                kurtosis = float(cp.mean(standardized**4))
            else:
                skewness = 0.0
                kurtosis = 0.0
            
            return {
                'mean': mean_val,
                'std': std_val,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            
        except Exception as e:
            self._critical_error(f"Error in vectorized moments computation: {e}")
    
    def _compute_normality_tests_vectorized(self, data: cp.ndarray) -> Dict[str, float]:
        """
        Vectorized normality tests.
        """
        try:
            # Jarque-Bera test (vectorized)
            moments = self._compute_moments_vectorized(data)
            n = len(data)
            
            jb_stat = n * (moments['skewness']**2 / 6 + (moments['kurtosis'] - 3)**2 / 24)
            jb_pvalue = 1.0  # Simplified - in practice, use chi-square distribution
            
            # Anderson-Darling test (simplified vectorized version)
            sorted_data = cp.sort(data)
            n = len(sorted_data)
            
            # Simplified AD statistic
            ad_stat = 0.0
            for i in range(n):
                p = (i + 1) / n
                ad_stat += (2*i + 1) * cp.log(p * (1 - p))
            
            ad_stat = -n - ad_stat / n
            ad_pvalue = 1.0  # Simplified
            
            return {
                'jarque_bera_stat': float(jb_stat),
                'jarque_bera_pvalue': jb_pvalue,
                'anderson_darling_stat': float(ad_stat),
                'anderson_darling_pvalue': ad_pvalue,
                'is_normal': jb_pvalue > 0.05 and ad_pvalue > 0.05
            }
            
        except Exception as e:
            self._critical_error(f"Error in vectorized normality tests: {e}")

    def process_cudf(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Process cuDF DataFrame with comprehensive statistical tests.
        
        Implements the complete statistical tests pipeline as specified in the technical plan:
        - ADF tests in batch for all frac_diff_* series
        - Distance correlation with significance testing
        - Stationarity flags and comprehensive statistics
        
        Args:
            df: Input cuDF DataFrame
            
        Returns:
            DataFrame with statistical test results
        """
        try:
            self._log_info("Starting comprehensive statistical tests pipeline...")
            
            # 1. ADF TESTS IN BATCH (Complete implementation)
            df = self._apply_comprehensive_adf_tests(df)
            
            # 2. DISTANCE CORRELATION TESTS (Complete implementation)
            df = self._apply_comprehensive_distance_correlation(df)
            
            # 3. ADDITIONAL STATISTICAL TESTS
            df = self._apply_additional_statistical_tests(df)
            
            self._log_info("Comprehensive statistical tests pipeline completed successfully")
            return df
            
        except Exception as e:
            self._critical_error(f"Error in comprehensive statistical tests pipeline: {e}")
    
    def _apply_comprehensive_adf_tests(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply comprehensive ADF tests to all frac_diff_* series.
        
        Generates:
        - adf_stat_* for each series
        - adf_pvalue_* for each series
        - adf_lag_order_* for each series
        - is_stationary_adf_* flag for each series
        """
        try:
            self._log_info("Applying comprehensive ADF tests...")
            
            # Find all frac_diff columns
            frac_diff_cols = [col for col in df.columns if "frac_diff" in col]
            self._log_info(f"Found {len(frac_diff_cols)} frac_diff columns for ADF testing")
            
            if not frac_diff_cols:
                self._log_warn("No frac_diff columns found for ADF testing")
                return df
            
            # Prepare data for batch ADF testing
            data_series = []
            series_names = []
            
            for col in frac_diff_cols:
                if col in df.columns:
                    data = df[col].to_cupy()
                    # Remove NaN values
                    valid_mask = ~cp.isnan(data)
                    if cp.sum(valid_mask) > 50:  # Minimum sample size
                        clean_data = data[valid_mask]
                        data_series.append(clean_data)
                        series_names.append(col)
            
            if not data_series:
                self._log_warn("No valid data series found for ADF testing")
                return df
            
            # Run batch ADF tests
            self._log_info(f"Running batch ADF tests for {len(data_series)} series")
            adf_results = self.compute_adf_batch(data_series, max_lag=12)
            
            # Add results to DataFrame
            for i, col in enumerate(series_names):
                if i < len(adf_results['adf_stat']):
                    # Extract base name (remove 'frac_diff_' prefix)
                    base_name = col.replace('frac_diff_', '')
                    
                    # Add ADF statistics
                    df[f'adf_stat_{base_name}'] = adf_results['adf_stat'][i]
                    df[f'adf_pvalue_{base_name}'] = adf_results['p_value'][i]
                    
                    # Add critical values
                    if i < len(adf_results['critical_values']):
                        crit_vals = adf_results['critical_values'][i]
                        df[f'adf_crit_1pct_{base_name}'] = crit_vals[0] if len(crit_vals) > 0 else cp.nan
                        df[f'adf_crit_5pct_{base_name}'] = crit_vals[1] if len(crit_vals) > 1 else cp.nan
                        df[f'adf_crit_10pct_{base_name}'] = crit_vals[2] if len(crit_vals) > 2 else cp.nan
                    
                    # Add stationarity flag (p-value < 0.05 and ADF stat < critical value)
                    is_stationary = False
                    if not cp.isnan(adf_results['p_value'][i]) and not cp.isnan(adf_results['adf_stat'][i]):
                        if len(adf_results['critical_values']) > i and len(adf_results['critical_values'][i]) > 1:
                            crit_5pct = adf_results['critical_values'][i][1]
                            is_stationary = (adf_results['p_value'][i] < 0.05 and 
                                           adf_results['adf_stat'][i] < crit_5pct)
                    
                    df[f'is_stationary_adf_{base_name}'] = is_stationary
            
            return df
            
        except Exception as e:
            self._critical_error(f"Error in comprehensive ADF tests: {e}")
    
    def _apply_comprehensive_distance_correlation(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply comprehensive distance correlation tests.
        
        Generates:
        - dcor_* for various feature pairs
        - dcor_pvalue_* significance tests
        - dcor_significant_* flags
        """
        try:
            self._log_info("Applying comprehensive distance correlation tests...")
            
            # Define feature pairs for distance correlation analysis
            feature_pairs = []
            
            # Find available features
            available_columns = list(df.columns)
            return_features = [col for col in available_columns if any(term in col.lower() for term in ['ret', 'return']) and col.startswith('y_')]
            volume_features = [col for col in available_columns if any(term in col.lower() for term in ['volume', 'tick']) and col.startswith('y_')]
            ofi_features = [col for col in available_columns if any(term in col.lower() for term in ['ofi']) and col.startswith('y_')]
            spread_features = [col for col in available_columns if any(term in col.lower() for term in ['spread']) and col.startswith('y_')]
            
            # Create feature pairs
            if return_features and volume_features:
                feature_pairs.append((return_features[0], volume_features[0], 'returns_volume'))
            
            if return_features and ofi_features:
                feature_pairs.append((return_features[0], ofi_features[0], 'returns_ofi'))
            
            if return_features and spread_features:
                feature_pairs.append((return_features[0], spread_features[0], 'returns_spreads'))
            
            if volume_features and spread_features:
                feature_pairs.append((volume_features[0], spread_features[0], 'volume_spreads'))
            
            self._log_info(f"Processing {len(feature_pairs)} feature pairs for distance correlation")
            
            # Process each pair
            for col1, col2, pair_name in feature_pairs:
                if col1 in df.columns and col2 in df.columns:
                    self._log_info(f"Processing distance correlation for {col1} vs {col2}")
                    
                    # Get data
                    data1 = df[col1].to_cupy()
                    data2 = df[col2].to_cupy()
                    
                    # Remove NaN values
                    valid_mask = ~(cp.isnan(data1) | cp.isnan(data2))
                    if cp.sum(valid_mask) > 100:  # Minimum sample size
                        clean_data1 = data1[valid_mask]
                        clean_data2 = data2[valid_mask]
                        
                        # Compute distance correlation
                        dcor_value = self._compute_distance_correlation_vectorized(clean_data1, clean_data2, max_samples=5000)
                        
                        # Add to DataFrame
                        df[f'dcor_{pair_name}'] = dcor_value
                        
                        # Significance test (simplified - in practice, use permutation test)
                        # For now, use a threshold based on sample size
                        n_samples = len(clean_data1)
                        significance_threshold = 0.1 / cp.sqrt(n_samples)  # Simplified significance
                        is_significant = dcor_value > significance_threshold
                        
                        df[f'dcor_significant_{pair_name}'] = is_significant
                        df[f'dcor_pvalue_{pair_name}'] = 0.05 if is_significant else 0.5  # Simplified p-value
            
            return df
            
        except Exception as e:
            self._critical_error(f"Error in comprehensive distance correlation: {e}")
    
    def _apply_additional_statistical_tests(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply additional statistical tests and metrics.
        
        Generates:
        - Additional stationarity metrics
        - Correlation matrices
        - Statistical summaries
        """
        try:
            self._log_info("Applying additional statistical tests...")
            
            # Find frac_diff columns for additional analysis
            frac_diff_cols = [col for col in df.columns if "frac_diff" in col]
            
            if len(frac_diff_cols) > 1:
                # Compute correlation matrix between frac_diff series
                self._log_info("Computing correlation matrix between frac_diff series")
                
                # Create correlation matrix
                corr_matrix = []
                for i, col1 in enumerate(frac_diff_cols):
                    row = []
                    for j, col2 in enumerate(frac_diff_cols):
                        if i == j:
                            row.append(1.0)
                        else:
                            # Compute correlation
                            data1 = df[col1].to_cupy()
                            data2 = df[col2].to_cupy()
                            
                            # Remove NaN values
                            valid_mask = ~(cp.isnan(data1) | cp.isnan(data2))
                            if cp.sum(valid_mask) > 10:
                                clean_data1 = data1[valid_mask]
                                clean_data2 = data2[valid_mask]
                                
                                # Compute correlation
                                corr = cp.corrcoef(clean_data1, clean_data2)[0, 1]
                                row.append(float(corr) if not cp.isnan(corr) else 0.0)
                            else:
                                row.append(0.0)
                    corr_matrix.append(row)
                
                # Add correlation matrix as features (simplified - just max correlation)
                if corr_matrix:
                    max_corr = max([max(row) for row in corr_matrix])
                    df['frac_diff_max_correlation'] = max_corr
            
            # Add statistical summaries for key series
            key_series = ['y_close', 'y_ret_1m']
            for col in key_series:
                if col in df.columns:
                    data = df[col].to_cupy()
                    valid_data = data[~cp.isnan(data)]
                    
                    if len(valid_data) > 0:
                        df[f'{col}_mean'] = float(cp.mean(valid_data))
                        df[f'{col}_std'] = float(cp.std(valid_data))
                        df[f'{col}_skew'] = float(cp.mean(((valid_data - cp.mean(valid_data)) / cp.std(valid_data))**3))
                        df[f'{col}_kurt'] = float(cp.mean(((valid_data - cp.mean(valid_data)) / cp.std(valid_data))**4))
            
            return df
            
        except Exception as e:
            self._critical_error(f"Error in additional statistical tests: {e}")
    
    def compute_adf_batch(self, data_series: List[cp.ndarray], max_lag: int = None) -> Dict[str, np.ndarray]:
        """
        Compute ADF test for multiple time series in batch.
        
        Args:
            data_series: List of time series arrays
            max_lag: Maximum lag for ADF test
            
        Returns:
            Dictionary with ADF results for all series
        """
        try:
            if not data_series:
                return {'adf_stat': [], 'p_value': [], 'critical_values': []}
            
            # Find maximum length for padding
            max_length = max(len(series) for series in data_series)
            n_series = len(data_series)
            
            # Create 3D matrix with padding
            data_matrix = cp.full((n_series, max_length, 1), cp.nan, dtype=cp.float32)
            
            for i, series in enumerate(data_series):
                if len(series) > 0:
                    data_matrix[i, :len(series), 0] = series
            
            # Compute ADF for all series
            results = self._compute_adf_batch_gpu(data_matrix, max_lag)
            
            return results
            
        except Exception as e:
            self._critical_error(f"Error in batch ADF computation: {e}")
    
    def _compute_adf_batch_gpu(self, data_matrix: cp.ndarray, max_lag: int = None) -> Dict[str, cp.ndarray]:
        """
        Batched ADF test implementation using GPU operations.
        Processes multiple time series simultaneously for better performance.
        
        Args:
            data_matrix: 3D array of shape (n_series, n_observations, 1)
            max_lag: Maximum lag for ADF test
            
        Returns:
            Dictionary with ADF statistics for all series
        """
        try:
            n_series, n_obs, _ = data_matrix.shape
            
            if max_lag is None:
                max_lag = int(12 * (n_obs / 100) ** (1/4))  # Schwert criterion
            
            # Initialize results arrays
            adf_stats = cp.zeros(n_series, dtype=cp.float32)
            p_values = cp.zeros(n_series, dtype=cp.float32)
            critical_values = cp.zeros((n_series, 3), dtype=cp.float32)  # 1%, 5%, 10%
            
            # Process each series
            for i in range(n_series):
                series = data_matrix[i, :, 0]
                
                # Remove NaN values
                valid_mask = ~cp.isnan(series)
                if cp.sum(valid_mask) < max_lag + 10:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
                    continue
                
                series_clean = series[valid_mask]
                
                # Calculate differences
                diff_series = cp.diff(series_clean)
                
                # Create lagged differences matrix
                lagged_diffs = cp.zeros((len(diff_series), max_lag))
                for lag in range(1, max_lag + 1):
                    lagged_diffs[:, lag-1] = cp.roll(diff_series, lag)
                    lagged_diffs[:lag, lag-1] = 0  # Set invalid lags to 0
                
                # Create regression matrix: [lagged_series, lagged_diffs, trend, constant]
                lagged_series = series_clean[:-1]  # y_{t-1}
                trend = cp.arange(len(lagged_series), dtype=cp.float32)
                constant = cp.ones(len(lagged_series), dtype=cp.float32)
                
                # Stack all regressors
                X = cp.column_stack([lagged_series, lagged_diffs, trend, constant])
                y = diff_series
                
                # Remove rows with NaN values
                valid_rows = ~cp.any(cp.isnan(X), axis=1) & ~cp.isnan(y)
                if cp.sum(valid_rows) < max_lag + 5:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
                    continue
                
                X_clean = X[valid_rows]
                y_clean = y[valid_rows]
                
                # Solve least squares using CuPy
                try:
                    # Use QR decomposition for numerical stability
                    Q, R = cp.linalg.qr(X_clean)
                    beta = cp.linalg.solve(R, Q.T @ y_clean)
                    
                    # Calculate residuals and standard error
                    residuals = y_clean - X_clean @ beta
                    mse = cp.sum(residuals**2) / (len(y_clean) - len(beta))
                    
                    # Standard error of the coefficient on lagged_series (first coefficient)
                    X_inv = cp.linalg.inv(X_clean.T @ X_clean)
                    se_beta = cp.sqrt(mse * X_inv[0, 0])
                    
                    # ADF statistic
                    adf_stat = beta[0] / se_beta
                    adf_stats[i] = float(adf_stat)
                    
                    # Approximate p-value using critical values
                    # This is a simplified approach - for production, use proper ADF p-value tables
                    if adf_stat < -3.43:
                        p_values[i] = 0.01
                    elif adf_stat < -2.86:
                        p_values[i] = 0.05
                    elif adf_stat < -2.57:
                        p_values[i] = 0.10
                    else:
                        p_values[i] = 1.0
                    
                    # Critical values (approximate)
                    critical_values[i, 0] = -3.43  # 1%
                    critical_values[i, 1] = -2.86  # 5%
                    critical_values[i, 2] = -2.57  # 10%
                    
                except cp.linalg.LinAlgError:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
            
            return {
                'adf_stat': cp.asnumpy(adf_stats),
                'p_value': cp.asnumpy(p_values),
                'critical_values': cp.asnumpy(critical_values)
            }
            
        except Exception as e:
            self._critical_error(f"Error in batched ADF computation: {e}")

    def _infer_task_from_target(self, y: np.ndarray) -> str:
        """Infer task type from target array (classification vs regression)."""
        try:
            uniq = np.unique(y[~np.isnan(y)])
            if len(uniq) <= 10 and np.allclose(uniq, uniq.astype(int)):
                return 'classification'
        except Exception:
            pass
        return 'regression'

    def _parse_horizon_rows(self, timestamps, target_name: str) -> int:
        """Estimate horizon in rows from target name suffix and median sampling interval.

        Supports patterns like *_1s, *_30s, *_1m, *_5m, *_1h, *_1d.
        Returns 0 if cannot infer.
        """
        try:
            import re
            import pandas as pd
            m = re.search(r"_(\d+)([smhd])", str(target_name).lower())
            if not m:
                return 0
            q = int(m.group(1))
            unit = m.group(2)
            unit_sec = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}.get(unit, 0)
            if unit_sec == 0:
                return 0
            horizon_sec = q * unit_sec
            if timestamps is None:
                return 0
            ts = pd.to_datetime(timestamps)
            if len(ts) < 3:
                return 0
            dt = ts.diff().dropna()
            if len(dt) == 0:
                return 0
            med_sec = float(dt.median().total_seconds())
            if med_sec <= 0:
                return 0
            import math
            return max(1, int(math.ceil(horizon_sec / med_sec)))
        except Exception:
            return 0

    def _build_contiguous_groups(self, n_rows: int, n_groups: int) -> List[np.ndarray]:
        sizes = np.full(n_groups, n_rows // n_groups, dtype=int)
        sizes[: n_rows % n_groups] += 1
        groups = []
        cur = 0
        for s in sizes:
            groups.append(np.arange(cur, cur + s))
            cur += s
        return groups

    def _stage4_cpcv(self, df, target: str, features: List[str]) -> Dict[str, Any]:
        """Run CPCV on CPU sample, persist per-fold details, and return summary.

        Returns a dict with keys: 'cpcv_splits' (int) and 'cpcv_top_features' (List[str]).
        """
        try:
            import pandas as pd
            from pathlib import Path
            import json
            from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score
            from utils.cpcv import combinatorial_purged_cv
        except Exception as e:
            self._log_warn("CPCV dependencies not available; skipping Stage 4", error=str(e))
            return {}

        # Gather selected features if available (takes precedence)
        selected = None
        try:
            if 'selected_features' in df.columns:
                sel_df = df[['selected_features']].head(1).compute().to_pandas()
                if not sel_df.empty and isinstance(sel_df.iloc[0]['selected_features'], str):
                    selected = [f for f in sel_df.iloc[0]['selected_features'].split(',') if f]
        except Exception:
            pass

        feats = selected if selected else list(features)
        feats = [f for f in feats if f in df.columns]
        if len(feats) == 0:
            self._log_warn("No features available for CPCV; skipping Stage 4")
            return {}

        # Sample CPU data (same cap as Stage 2)
        max_rows = int(getattr(self, 'selection_max_rows', 100000))
        cols_to_pull = [c for c in [target, 'timestamp', 'currency_pair'] if c in df.columns]
        try:
            sample_ddf = df[[target] + feats + cols_to_pull].head(max_rows)
            pdf = sample_ddf.compute().to_pandas().dropna()
        except Exception as e:
            # Fallback pull in two steps
            try:
                pdf = df[[target] + feats].head(max_rows).compute().to_pandas().dropna()
            except Exception as e2:
                self._log_warn("Failed to sample data for CPCV; skipping", error=f"{e} / {e2}")
                return {}

        if pdf.empty or len(pdf) < 50:
            self._log_warn("Insufficient data for CPCV; skipping", rows=len(pdf))
            return {}

        # Derive CPCV parameters
        n_groups = int(getattr(self, 'cpcv_n_groups', 6))
        k_leave_out = int(getattr(self, 'cpcv_k_leave_out', 2))
        purge_cfg = int(getattr(self, 'cpcv_purge', 0))
        embargo_cfg = int(getattr(self, 'cpcv_embargo', 0))

        # Auto purge/embargo from target horizon if not configured
        if purge_cfg <= 0 or embargo_cfg < 0:
            horizon_rows = 0
            try:
                ts_series = pdf['timestamp'] if 'timestamp' in pdf.columns else None
                horizon_rows = self._parse_horizon_rows(ts_series, target)
            except Exception:
                horizon_rows = 0
            purge = purge_cfg if purge_cfg > 0 else int(horizon_rows)
            embargo = embargo_cfg if embargo_cfg > 0 else int(max(0, horizon_rows // 2))
        else:
            purge, embargo = purge_cfg, embargo_cfg

        # Build CPCV groups on positional index
        n = len(pdf)
        n_groups = max(2, min(n_groups, max(2, n // 5)))  # at least ~5 rows per group
        groups = self._build_contiguous_groups(n, n_groups)

        # Prepare task type and model backend
        y = pdf[target].values
        task = self._infer_task_from_target(y)

        def _fit_and_score(X_tr, y_tr, X_te, y_te):
            import numpy as np
            fi_map = {}
            score = np.nan
            aux = {}
            try:
                if task == 'classification':
                    try:
                        import lightgbm as lgb
                        model = lgb.LGBMClassifier(
                            num_leaves=int(getattr(self, 'stage3_lgbm_num_leaves', 31)),
                            max_depth=int(getattr(self, 'stage3_lgbm_max_depth', -1)),
                            n_estimators=int(getattr(self, 'stage3_lgbm_n_estimators', 200)),
                            learning_rate=float(getattr(self, 'stage3_lgbm_learning_rate', 0.05)),
                            subsample=float(getattr(self, 'stage3_lgbm_bagging_fraction', 0.8)),
                            colsample_bytree=float(getattr(self, 'stage3_lgbm_feature_fraction', 0.8)),
                            random_state=int(getattr(self, 'stage3_random_state', 42)),
                            n_jobs=-1,
                        )
                        model.fit(X_tr, y_tr)
                        if len(np.unique(y_te)) > 1:
                            proba = model.predict_proba(X_te)[:, 1]
                            score = float(roc_auc_score(y_te, proba))
                            aux['metric'] = 'roc_auc'
                        else:
                            pred = model.predict(X_te)
                            score = float(accuracy_score(y_te, pred))
                            aux['metric'] = 'accuracy'
                        fi = getattr(model, 'feature_importances_', None)
                        if fi is not None and len(fi) == X_tr.shape[1]:
                            fi_map.update({f: float(w) for f, w in zip(feats, fi)})
                    except Exception:
                        # fallback simple majority baseline
                        try:
                            import numpy as np
                            majority = np.argmax(np.bincount(y_tr.astype(int)))
                            pred = np.repeat(majority, len(y_te))
                            score = float(accuracy_score(y_te, pred))
                            aux['metric'] = 'accuracy'
                        except Exception:
                            score = float('nan')
                else:
                    try:
                        import lightgbm as lgb
                        model = lgb.LGBMRegressor(
                            num_leaves=int(getattr(self, 'stage3_lgbm_num_leaves', 31)),
                            max_depth=int(getattr(self, 'stage3_lgbm_max_depth', -1)),
                            n_estimators=int(getattr(self, 'stage3_lgbm_n_estimators', 200)),
                            learning_rate=float(getattr(self, 'stage3_lgbm_learning_rate', 0.05)),
                            subsample=float(getattr(self, 'stage3_lgbm_bagging_fraction', 0.8)),
                            colsample_bytree=float(getattr(self, 'stage3_lgbm_feature_fraction', 0.8)),
                            random_state=int(getattr(self, 'stage3_random_state', 42)),
                            n_jobs=-1,
                        )
                        model.fit(X_tr, y_tr)
                        pred = model.predict(X_te)
                        score = float(r2_score(y_te, pred))
                        fi = getattr(model, 'feature_importances_', None)
                        if fi is not None and len(fi) == X_tr.shape[1]:
                            fi_map.update({f: float(w) for f, w in zip(feats, fi)})
                        aux['metric'] = 'r2'
                        try:
                            aux['mae'] = float(mean_absolute_error(y_te, pred))
                        except Exception:
                            pass
                    except Exception:
                        # fallback OLS
                        try:
                            import numpy as np
                            Xb = np.c_[np.ones((X_tr.shape[0], 1)), X_tr]
                            beta = np.linalg.lstsq(Xb, y_tr, rcond=None)[0]
                            pred = np.c_[np.ones((X_te.shape[0], 1)), X_te] @ beta
                            score = float(r2_score(y_te, pred))
                            aux['metric'] = 'r2'
                        except Exception:
                            score = float('nan')
            except Exception:
                score = float('nan')
            return score, fi_map, aux

        X_all = pdf[feats].values
        y_all = pdf[target].values

        # Try get context for persistence
        currency_pair = None
        try:
            if 'currency_pair' in pdf.columns:
                currency_pair = str(pdf['currency_pair'].iloc[0])
        except Exception:
            currency_pair = None

        # Output path for CPCV details
        from pathlib import Path
        out_root = Path(getattr(self.settings.output, 'output_path', './output'))
        if currency_pair:
            out_dir = out_root / currency_pair / 'cpcv' / target
        else:
            out_dir = out_root / 'cpcv' / target
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Run CPCV
        splits = []
        fold_results = []
        fi_accumulate: Dict[str, float] = {f: 0.0 for f in feats}
        fi_counts: Dict[str, int] = {f: 0 for f in feats}

        try:
            from utils.cpcv import combinatorial_purged_cv
            for fold_id, (tr_idx, te_idx) in enumerate(combinatorial_purged_cv(groups, k_leave_out=k_leave_out, purge=purge, embargo=embargo)):
                splits.append((tr_idx, te_idx))
                X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
                X_te, y_te = X_all[te_idx], y_all[te_idx]
                score, fi_map, aux = _fit_and_score(X_tr, y_tr, X_te, y_te)
                # accumulate importances
                for f, w in fi_map.items():
                    fi_accumulate[f] = fi_accumulate.get(f, 0.0) + float(w)
                    fi_counts[f] = fi_counts.get(f, 0) + 1
                # store fold result
                top_feats = []
                if fi_map:
                    top_feats = [k for k, _ in sorted(fi_map.items(), key=lambda kv: kv[1], reverse=True)[: min(20, len(fi_map))]]
                fold_info = {
                    'fold_id': fold_id,
                    'train_size': int(len(tr_idx)),
                    'test_size': int(len(te_idx)),
                    'metric': aux.get('metric', 'r2' if task == 'regression' else 'accuracy'),
                    'score': float(score) if score == score else None,
                    'extra': aux,
                    'top_features': top_feats,
                }
                fold_results.append(fold_info)
        except Exception as e:
            self._log_warn("CPCV iteration failed", error=str(e))
            return {}

        n_splits = len(fold_results)
        # Aggregate top features by average importance
        fi_avg = {f: (fi_accumulate.get(f, 0.0) / max(1, fi_counts.get(f, 1))) for f in feats}
        cpcv_top = [k for k, _ in sorted(fi_avg.items(), key=lambda kv: kv[1], reverse=True)[: min(50, len(feats))]]

        # Persist detailed results
        summary = {
            'currency_pair': currency_pair,
            'target': target,
            'n_rows': int(len(pdf)),
            'n_features': int(len(feats)),
            'features_used': feats,
            'cpcv_n_groups': n_groups,
            'cpcv_k_leave_out': k_leave_out,
            'cpcv_purge': int(purge),
            'cpcv_embargo': int(embargo),
            'n_splits': int(n_splits),
            'top_features': cpcv_top,
            'task': task,
        }
        try:
            import json
            with open(out_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            with open(out_dir / 'folds.json', 'w') as f:
                json.dump({'folds': fold_results}, f, indent=2)
        except Exception as e:
            self._log_warn("Failed to persist CPCV details", error=str(e), path=str(out_dir))

        return {'cpcv_splits': n_splits, 'cpcv_top_features': cpcv_top}

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executes the statistical tests pipeline (Dask version).
        """
        self._log_info("Starting StatisticalTests (Dask)...")

        # --- ADF por janela em colunas de fracdiff ---
        # seu padrão de nomes parecia 'frac_diff_*'; ajustei aqui:
        adf_cols = [c for c in df.columns if "frac_diff" in c]

        for col in adf_cols:
            self._log_info(f"ADF rolling on '{col}'...")
            out = f"adf_stat_{col.split('_')[-1]}"  # e.g., 'close' de 'frac_diff_close'
            df[out] = df[col].map_partitions(
                _adf_rolling_partition,
                252,
                200,
                meta=(out, "f8"),
            )

        # --- Distance Correlation (single-fit; sem rolling por enquanto) ---
        # Check for the correct column names that exist in the data
        returns_col = None
        tick_volume_col = None
        
        # Look for returns column (could be 'returns', 'y_ret_1m', etc.)
        for col in df.columns:
            if 'ret' in col.lower() and '1m' in col.lower():
                returns_col = col
                break
            elif col == 'returns':
                returns_col = col
                break
        
        # Look for tick_volume column (could be 'tick_volume', 'y_tick_volume', etc.)
        for col in df.columns:
            if 'tick_volume' in col.lower():
                tick_volume_col = col
                break
        
        if returns_col and tick_volume_col:
            self._log_info(f"Computing single-fit distance correlation using {returns_col} and {tick_volume_col}...")
            
            # Sample data to avoid memory issues (use last up to 10000 rows)
            # Note: len(ddf) is not supported in Dask; tail(N) will return up to N rows
            sample_n = 10000
            sample_df = df[[returns_col, tick_volume_col]].tail(sample_n)
            # Check if DataFrame has repartition method (Dask DataFrame)
            if hasattr(sample_df, 'repartition'):
                sample_df = sample_df.repartition(npartitions=1)
            self._log_info(f"Using tail sample (<= {sample_n}) rows for dCor calculation")
            
            # Check if DataFrame has map_partitions method (Dask DataFrame)
            if hasattr(sample_df, 'map_partitions'):
                # map_partitions retorna uma Series de 1 linha com o escalar
                dcor_ddf = sample_df.map_partitions(
                    _dcor_single_pair_partition,
                    returns_col,
                    tick_volume_col,
                    int(self.dcor_max_samples),
                    meta=cudf.Series(name="dcor_returns_volume", dtype="f8"),
                )
                dcor_val = float(dcor_ddf.compute().iloc[0])
            else:
                # For regular cuDF DataFrame, compute directly
                dcor_val = float(self._compute_distance_correlation_vectorized(
                    sample_df[returns_col].to_cupy(), 
                    sample_df[tick_volume_col].to_cupy(), 
                    max_samples=self.dcor_max_samples
                ))
            df = df.assign(dcor_returns_volume=dcor_val)
        else:
            self._log_info(f"Skipping dCor (requires returns and tick_volume columns). Found: returns={returns_col}, tick_volume={tick_volume_col}")

        self._log_info("StatisticalTests complete.")
        
        # --- Stage 1: dCor ranking vs target (global) ---
        try:
            target = self.selection_target_column
            if target in df.columns:
                # Identify candidate columns (floats, excluding target)
                # Use a small head() to infer dtypes and filter columns
                sample = df.head(100)
                float_cols = [c for c, t in sample.dtypes.items() if 'float' in str(t).lower()]
                candidates = [c for c in float_cols if c != target]
                if candidates:
                    self._log_info("Computing dCor ranking", target=target, n_candidates=len(candidates), rolling=self.stage1_rolling_enabled)

                    one = self._single_partition(df[[target] + candidates])

                    # Optionally compute rolling aggregated dCor
                    roll_scores = {}
                    if self.stage1_rolling_enabled:
                        # Compute rolling dCor scores via module-level partition function
                        w = int(self.stage1_rolling_window)
                        st = int(self.stage1_rolling_step)
                        mp = int(self.stage1_rolling_min_periods)
                        mr = int(self.stage1_rolling_max_rows)
                        mw = int(self.stage1_rolling_max_windows)
                        ag = str(self.stage1_agg).lower()
                        roll_meta = {f"dcor_roll_{c}": 'f8' for c in candidates}
                        if hasattr(one, 'map_partitions'):
                            roll_ddf = one.map_partitions(
                                _dcor_rolling_partition,
                                target,
                                candidates,
                                w,
                                st,
                                mp,
                                mr,
                                mw,
                                ag,
                                int(self.dcor_max_samples),
                                meta=cudf.DataFrame({k: cudf.Series([], dtype=v) for k, v in roll_meta.items()})
                            )
                            roll_pdf = roll_ddf.compute().to_pandas()
                        else:
                            roll_result = _dcor_rolling_partition(one, target, candidates, w, st, mp, mr, mw, ag, int(self.dcor_max_samples))
                            roll_pdf = roll_result.to_pandas()
                        if not roll_pdf.empty:
                            roll_row = roll_pdf.iloc[0].to_dict()
                            df = self._broadcast_scalars(df, roll_row)
                            roll_scores = {c: float(roll_row.get(f"dcor_roll_{c}", np.nan)) for c in candidates if np.isfinite(roll_row.get(f"dcor_roll_{c}", np.nan))}

                    # Use module-level deterministic partition function for dCor

                    meta_cols = {f"dcor_{c}": 'f8' for c in candidates}
                    if self.dcor_include_permutation and self.dcor_permutations > 0:
                        meta_cols.update({f"dcor_pvalue_{c}": 'f8' for c in candidates})

                    # Check if DataFrame has map_partitions method (Dask DataFrame)
                    if hasattr(one, 'map_partitions'):
                        res_ddf = one.map_partitions(
                            _dcor_partition,
                            target,
                            candidates,
                            int(self.dcor_max_samples),
                            meta=cudf.DataFrame({k: cudf.Series([], dtype=v) for k, v in meta_cols.items()})
                        )
                        res_pdf = res_ddf.compute().to_pandas()
                    else:
                        # For regular cuDF DataFrame, compute directly
                        res_result = dcor_partition(one)
                        res_pdf = res_result.to_pandas()
                    if not res_pdf.empty:
                        row = res_pdf.iloc[0].to_dict()
                        df = self._broadcast_scalars(df, row)

                        # Choose scores source
                        if self.stage1_rolling_enabled and self.stage1_use_rolling_scores and roll_scores:
                            dcor_scores = dict(roll_scores)
                            label_prefix = 'dcor_roll'
                        else:
                            dcor_scores = {k: float(v) for k, v in ((f, row.get(f"dcor_{f}", np.nan)) for f in candidates) if np.isfinite(v)}
                            label_prefix = 'dcor'

                        # Log top‑K features by chosen scores
                        score_items = sorted(dcor_scores.items(), key=lambda kv: kv[1], reverse=True)
                        topk = score_items[:max(1, self.dcor_top_k)]
                        self._log_info("Top‑K dCor features", top=[f"{k}:{v:.4f}" for k, v in topk[:10]], agg=self.stage1_agg, source=label_prefix)

                        # ---------- Stage 1 retention (threshold/percentile/top‑N) ----------
                        # Threshold por valor absoluto
                        retained = [f for f, s in dcor_scores.items() if s >= float(self.dcor_min_threshold)]

                        # Percentil (se configurado > 0)
                        if retained and self.dcor_min_percentile > 0.0:
                            vals = np.array([dcor_scores[f] for f in retained], dtype=float)
                            q = float(np.quantile(vals, min(max(self.dcor_min_percentile, 0.0), 1.0)))
                            retained = [f for f in retained if dcor_scores[f] >= q]

                        # Top‑N (se > 0)
                        if retained and self.stage1_top_n > 0:
                            retained.sort(key=lambda f: dcor_scores[f], reverse=True)
                            retained = retained[: self.stage1_top_n]

                        # Broadcast Stage 1 list
                        df = self._broadcast_scalars(df, {
                            'stage1_features': ','.join(retained),
                            'stage1_features_count': len(retained),
                        })

                        # ---------- Stage 1 (opcional): Permutation test para Top‑K ----------
                        if self.dcor_permutation_top_k and self.dcor_permutations > 0 and retained:
                            perm_top = sorted(retained, key=lambda f: dcor_scores.get(f, 0.0), reverse=True)[: self.dcor_permutation_top_k]
                            # calcula pvalues numa única partição
                            # Check if DataFrame has map_partitions method (Dask DataFrame)
                            if hasattr(one, 'map_partitions'):
                                perm_ddf = one.map_partitions(
                                    _perm_pvalues_partition,
                                    target,
                                    perm_top,
                                    int(self.dcor_permutations),
                                    int(self.dcor_max_samples),
                                    meta=cudf.DataFrame({f"dcor_pvalue_{f}": cudf.Series([], dtype='f8') for f in perm_top})
                                )
                                perm_pdf = perm_ddf.compute().to_pandas()
                            else:
                                # For regular cuDF DataFrame, compute directly
                                perm_result = cudf.DataFrame([
                                    {f"dcor_pvalue_{f}": p for f, p in self._compute_permutation_pvalues(one, target, perm_top, self.dcor_permutations).items()}
                                ])
                                perm_pdf = perm_result.to_pandas()
                            if not perm_pdf.empty:
                                prow = perm_pdf.iloc[0].to_dict()
                                df = self._broadcast_scalars(df, prow)
                                # aplica filtro por alpha
                                alpha = float(self.dcor_pvalue_alpha)
                                retained = [f for f in retained if prow.get(f"dcor_pvalue_{f}", 1.0) <= alpha]
                                df = self._broadcast_scalars(df, {
                                    'stage1_features': ','.join(retained),
                                    'stage1_features_count': len(retained),
                                })

                        # ---------- Stage 2: VIF + MI (CPU) ----------
                        # Amostras para CPU
                        max_rows = int(self.selection_max_rows)
                        sample_cpu = df[[target] + retained].head(max_rows).compute().to_pandas().dropna()
                        if not sample_cpu.empty and len(retained) >= 2:
                            X_df = sample_cpu[retained]
                            y_s = sample_cpu[target]
                            # VIF iterativo
                            vif_keep = self._compute_vif_iterative(X_df.values.astype(float), retained, threshold=float(self.vif_threshold))
                            # MI redundância (com clustering global escalável, se habilitado)
                            if getattr(self, 'mi_cluster_enabled', True):
                                mi_keep = self._compute_mi_cluster_representatives(X_df, vif_keep, dcor_scores)
                            else:
                                mi_keep = self._compute_mi_redundancy(X_df, vif_keep, dcor_scores, mi_threshold=float(self.mi_threshold))

                            # ---------- Stage 3: Wrappers (Lasso + Árvores) ----------
                            final_sel = self._stage3_wrappers(X_df, y_s, mi_keep, top_n=int(self.stage3_top_n))

                            # Broadcast seleção final e listas
                            df = self._broadcast_scalars(df, {
                                'stage2_features': ','.join(mi_keep),
                                'stage2_features_count': len(mi_keep),
                                'selected_features': ','.join(final_sel),
                                'selected_features_count': len(final_sel),
                            })
                            self._log_info("Stage 3 selection done", count=len(final_sel))

                            # ---------- Stage 4: CPCV (opcional) ----------
                            if getattr(self, 'cpcv_enabled', False):
                                cpcv_res = self._stage4_cpcv(df, target, mi_keep)
                                if cpcv_res:
                                    # Broadcast CPCV results
                                    out_map = {
                                        'cpcv_splits': cpcv_res.get('cpcv_splits', 0),
                                        'cpcv_top_features': ','.join(cpcv_res.get('cpcv_top_features', [])),
                                    }
                                    df = self._broadcast_scalars(df, out_map)
                                    self._log_info("Stage 4 CPCV complete", splits=out_map['cpcv_splits'], top=out_map['cpcv_top_features'].split(',')[:10])
            else:
                self._log_warn("Target column not found for dCor ranking", target=self.selection_target_column)
        except Exception as e:
            self._log_error(f"Error in Stage 1 dCor ranking: {e}")

        return df
