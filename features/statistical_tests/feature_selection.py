"""
Feature Selection Module

This module contains functionality for feature selection including VIF (Variance Inflation Factor),
mutual information redundancy removal, clustering, and embedded feature selection methods.
"""

import logging
import numpy as np
import cupy as cp
import cudf
from typing import List, Dict, Any, Tuple
from .utils import _hermitian_pinv_gpu

logger = logging.getLogger(__name__)


class FeatureSelection:
    """Class for feature selection and ranking operations."""
    
    def __init__(self, logger_instance=None, vif_threshold=5.0, mi_threshold=0.3, 
                 mi_max_candidates=400, mi_chunk_size=128, mi_cluster_threshold=0.3):
        """Initialize feature selection with configuration parameters."""
        self.logger = logger_instance or logger
        self.vif_threshold = vif_threshold
        self.mi_threshold = mi_threshold
        self.mi_max_candidates = mi_max_candidates
        self.mi_chunk_size = mi_chunk_size
        self.mi_cluster_threshold = mi_cluster_threshold
    
    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context."""
        if self.logger:
            self.logger.info(f"FeatureSelection: {message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context."""
        if self.logger:
            self.logger.warning(f"FeatureSelection: {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context."""
        if self.logger:
            self.logger.error(f"FeatureSelection: {message}", extra=kwargs)
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"FeatureSelection Critical Error: {message}")

    def _compute_vif_iterative(self, X: np.ndarray, cols: List[str], threshold: float) -> List[str]:
        """Remove columns with VIF above threshold using correlation matrix (CPU)."""
        keep = cols.copy()  # Start with all columns
        it = 0  # Iteration counter
        self._log_info("VIF iterative start", features=len(keep), threshold=round(float(threshold), 3))
        while True:
            if len(keep) < 2:  # Need at least 2 features for VIF calculation
                break
            # Compute correlation matrix for remaining features
            corr = np.corrcoef(X[:, [cols.index(c) for c in keep]].T)
            try:
                inv = np.linalg.pinv(corr)  # Pseudo-inverse of correlation matrix
            except Exception:
                break  # Stop if matrix inversion fails
            vifs = np.diag(inv)  # VIF values are diagonal elements of inverse correlation matrix
            vmax = float(np.max(vifs))  # Find maximum VIF
            if vmax <= threshold or not np.isfinite(vmax):  # Stop if VIF is below threshold or invalid
                break
            idx = int(np.argmax(vifs))  # Find index of feature with highest VIF
            removed = keep.pop(idx)  # Remove feature with highest VIF
            it += 1
            self._log_info("VIF removal", iter=it, feature=removed, vif=round(vmax, 3), remaining=len(keep))
        self._log_info("VIF iterative done", kept=len(keep))
        return keep  # Return remaining features

    def _compute_mi_redundancy(self, X_df, candidates: List[str], dcor_scores: Dict[str, float], mi_threshold: float) -> List[str]:
        """Remove non-linear redundancy via pairwise MI (keeps higher dCor in pair)."""
        try:
            from sklearn.feature_selection import mutual_info_regression  # Import MI computation
        except Exception as e:
            self._critical_error("MI not available for redundancy computation", error=str(e))
            return candidates  # unreachable

        keep = set(candidates)  # Start with all candidates
        # Cap number of pairs (quadratic); if large, limit candidates
        max_cands = min(len(candidates), 200)  # Limit to 200 candidates for performance
        cand_limited = candidates[:max_cands]  # Take first max_cands candidates
        # Drop rows with NaNs across selected columns to satisfy sklearn
        try:
            X_sub = X_df[cand_limited].dropna()
        except Exception as e:
            self._critical_error("Failed to drop NaNs for MI redundancy", error=str(e))
            X_sub = X_df[cand_limited]
        # Ensure CPU NumPy array for sklearn
        try:
            import cudf as _cudf
            if isinstance(X_sub, _cudf.DataFrame):
                X = X_sub.to_pandas().values
            else:
                X = X_sub.values  # pandas/NumPy path
        except Exception:
            # Fallback try without cudf import
            try:
                X = X_sub.to_pandas().values
            except Exception as e:
                self._critical_error("Failed to build CPU matrix for MI redundancy", error=str(e))
        n = len(cand_limited)
        # Compute pairwise MI approx: MI(X_i, X_j) by treating one as target
        for i in range(n):  # For each feature as target
            if cand_limited[i] not in keep:  # Skip if already removed
                continue
            try:
                y = X[:, i]  # Target feature
                mi = mutual_info_regression(X, y, discrete_features=False)  # Compute MI with all features
            except Exception as e:
                self._critical_error("MI row failed", feature=cand_limited[i], error=str(e))
                # unreachable
            for j in range(i + 1, n):  # Compare with remaining features
                f_i, f_j = cand_limited[i], cand_limited[j]
                if f_i in keep and f_j in keep and mi[j] >= mi_threshold:  # If both features still exist and MI above threshold
                    # Remove the one with lower dCor
                    if dcor_scores.get(f_i, 0.0) >= dcor_scores.get(f_j, 0.0):
                        keep.discard(f_j)  # Remove feature with lower dCor
                        self._log_info("MI redundancy removal", pair=f"{f_i},{f_j}", kept=f_i, removed=f_j, mi=round(float(mi[j]), 4))
                    else:
                        keep.discard(f_i)  # Remove feature with lower dCor
                        self._log_info("MI redundancy removal", pair=f"{f_i},{f_j}", kept=f_j, removed=f_i, mi=round(float(mi[j]), 4))
        return list(keep)  # Return remaining features

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
            self._critical_error("MI clustering unavailable", error=str(e))
            return self._compute_mi_redundancy(X_df, candidates, dcor_scores, mi_threshold=float(self.mi_threshold))  # unreachable

        if len(candidates) <= 2:
            return candidates

        # 1) Seleção de subset escalável
        if dcor_scores:
            ordered = sorted([c for c in candidates if c in dcor_scores], key=lambda f: dcor_scores[f], reverse=True)
        else:
            ordered = list(candidates)
        max_c = max(2, int(self.mi_max_candidates))
        cand = ordered[:min(len(ordered), max_c)]
        # Drop rows with NaNs across selected columns to satisfy sklearn
        try:
            X_sub = X_df[cand].dropna()
        except Exception as e:
            self._critical_error("Failed to drop NaNs for MI clustering", error=str(e))
            X_sub = X_df[cand]
        # Ensure CPU NumPy array for sklearn
        try:
            import cudf as _cudf
            if isinstance(X_sub, _cudf.DataFrame):
                X = X_sub.to_pandas().values
            else:
                X = X_sub.values
        except Exception:
            try:
                X = X_sub.to_pandas().values
            except Exception as e:
                self._critical_error("Failed to build CPU matrix for MI clustering", error=str(e))
        n = X.shape[0]
        p = X.shape[1]
        if p < 2:
            return cand

        # 2) Matriz MI por blocos (simetrizada)
        chunk = max(8, int(self.mi_chunk_size))
        MI = np.zeros((p, p), dtype=np.float32)
        # progress bookkeeping
        nb = int(np.ceil(p / chunk))
        total_blocks = nb * nb
        done_blocks = 0
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
                        self._critical_error("MI block failed", i=int(ii), j0=int(j0), j1=int(j1), error=str(e))
                        # unreachable
                    MI[ii, j0:j1] = np.maximum(MI[ii, j0:j1], mi_block.astype(np.float32))
                done_blocks += 1
                # Log progress roughly every 10% of blocks
                if total_blocks >= 10 and done_blocks % max(1, total_blocks // 10) == 0:
                    self._log_info("MI blocks progress", done=done_blocks, total=total_blocks)
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
        # quick clustering summary
        try:
            import numpy as _np
            unique, counts = _np.unique(labels, return_counts=True)
            sizes = sorted(list(map(int, counts)), reverse=True)
            self._log_info("MI clustering summary", clusters=int(len(unique)), largest=sizes[:5])
        except Exception as e:
            logger.error(f"Failed to log MI clustering summary: {e}")
            pass

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

    def _to_cupy_matrix(self, gdf: cudf.DataFrame, cols: List[str], dtype: str = 'f4') -> Tuple[cp.ndarray, List[str]]:
        """Build a CuPy matrix (n_rows x n_cols) from cuDF columns without CPU copies.

        Drops columns that fail conversion; raises if none usable. Returns (X, used_cols).
        """
        # Normalize input to cuDF DataFrame
        try:
            if not isinstance(gdf, cudf.DataFrame):
                import pandas as _pd  # noqa: F401
                gdf = cudf.from_pandas(gdf)
        except Exception as e:
            self._critical_error("GPU matrix expects cuDF DataFrame", error=str(e))
        used: List[str] = []
        arrays = []
        invalid: List[str] = []
        last_err = None
        for c in cols:
            try:
                s = gdf[c]
                sc = s.astype(dtype, copy=False)
                arrays.append(sc.to_cupy())
                used.append(c)
            except Exception as e:
                last_err = e
                invalid.append(str(c))
                # do not log per-column; escalate after scan
                continue
        if invalid:
            # Log and continue with the usable subset
            preview = ",".join(invalid[:10])
            more = max(0, len(invalid) - 10)
            details = f"{preview}{'... (+%d more)' % more if more > 0 else ''}"
            self._log_info("Dropping non-numeric/invalid columns for GPU matrix", count=len(invalid), examples=details)
        if not arrays:
            # No usable columns – escalate as critical
            msg = "to_cupy matrix failed: no usable numeric columns"
            if last_err is not None:
                msg = f"{msg} (last error: {last_err})"
            self._critical_error(msg, requested=len(cols))
        try:
            X = cp.stack(arrays, axis=1)
        except Exception as e:
            self._critical_error("Failed to stack GPU matrix", error=str(e), used=len(used))
            X = cp.empty((0, 0), dtype=dtype)
        return X, used

    def _compute_vif_iterative_gpu(self, X: cp.ndarray, features: List[str], threshold: float = 5.0) -> List[str]:
        """Iteratively remove features with VIF above threshold using GPU ops.

        VIF_i is taken as the ith diagonal of inv(corr(X)). Uses pinvh for stability.
        """
        keep = list(features)  # Start with all features
        idx = cp.arange(X.shape[1])  # Feature indices
        # Pre-standardize to unit variance (robust corr computation)
        try:
            Xw = X
            # Remove rows with NaNs
            if cp.isnan(Xw).any():
                mask = cp.all(~cp.isnan(Xw), axis=1)  # Keep rows without any NaN values
                Xw = Xw[mask]
            if Xw.shape[0] < 5 or Xw.shape[1] < 2:  # Need sufficient data for VIF calculation
                return keep
            # Standardize columns to unit variance
            mu = cp.nanmean(Xw, axis=0)  # Column means
            sd = cp.nanstd(Xw, axis=0)  # Column standard deviations
            sd = cp.where(sd == 0, 1.0, sd)  # Avoid division by zero
            Z = (Xw - mu) / sd  # Standardized data
        except Exception:
            return keep  # Return all features if standardization fails

        while True:  # Iterative VIF removal loop
            try:
                # Correlation matrix via dot product (faster than corrcoef for standardized Z)
                n = Z.shape[0]  # Number of samples
                R = (Z.T @ Z) / cp.float32(max(1, n - 1))  # Correlation matrix computation
                # Numerical guard for stability
                eps = cp.float32(1e-6)
                R = (R + R.T) * 0.5  # Ensure symmetry
                R += eps * cp.eye(R.shape[0], dtype=R.dtype)  # Add small diagonal for numerical stability
                # Use Hermitian pseudo-inverse for stability (eigh-based)
                Rinv = _hermitian_pinv_gpu(R)  # Compute pseudo-inverse
                vif_diag = cp.diag(Rinv)  # Extract VIF values from diagonal
                vmax = float(cp.max(vif_diag).item())  # Find maximum VIF
                if vmax <= float(threshold) or len(keep) <= 2:  # Stop if VIF below threshold or too few features
                    break
                imax = int(cp.argmax(vif_diag).item())  # Find index of feature with highest VIF
                removed = keep.pop(imax)  # Remove feature with highest VIF
                # Drop column from Z to update for next iteration
                cols = [i for i in range(Z.shape[1]) if i != imax]
                Z = Z[:, cols]  # Remove corresponding column from standardized data
                self._log_info("VIF removal (GPU)", feature=removed, vif=round(vmax, 3), remaining=len(keep))
            except Exception as e:
                self._log_warn("VIF GPU failed; keeping current set", error=str(e))
                break  # Stop if computation fails
        self._log_info("VIF iterative done (GPU)", kept=len(keep))
        return keep  # Return remaining features

    def _mi_nmi_gpu(self, x: cp.ndarray, y: cp.ndarray, bins: int = 64) -> float:
        """Compute normalized mutual information on GPU via 2D histograms.

        NMI = I(X;Y) / max(H(X), H(Y)) in [0,1].
        """
        try:
            # Remove NaNs
            m = ~(cp.isnan(x) | cp.isnan(y))
            x = x[m]
            y = y[m]
            if x.size < 10:
                return 0.0
            # Histogram edges (uniform)
            x_min, x_max = float(cp.min(x)), float(cp.max(x))
            y_min, y_max = float(cp.min(y)), float(cp.max(y))
            if not (cp.isfinite(x_min) and cp.isfinite(x_max) and cp.isfinite(y_min) and cp.isfinite(y_max)):
                return 0.0
            if x_min == x_max or y_min == y_max:
                return 0.0
            H, _, _ = cp.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
            Pxy = H / cp.sum(H)
            Px = cp.sum(Pxy, axis=1)
            Py = cp.sum(Pxy, axis=0)
            # Entropies (base e)
            def _H(p):
                p = p[p > 0]
                return float(-cp.sum(p * cp.log(p)).item()) if p.size else 0.0
            Hx = _H(Px)
            Hy = _H(Py)
            # MI = sum Pxy log( Pxy / (Px Py) )
            denom = Px[:, None] * Py[None, :]
            mask = (Pxy > 0) & (denom > 0)
            I = float(cp.sum(Pxy[mask] * cp.log(Pxy[mask] / denom[mask])).item())
            nmi = I / max(Hx, Hy, 1e-12)
            return float(max(0.0, min(1.0, nmi)))
        except Exception:
            return 0.0

    def _compute_mi_redundancy_gpu(self, X: cp.ndarray, features: List[str], dcor_scores: Dict[str, float], threshold: float, bins: int = 64, chunk: int = 64) -> List[str]:
        """GPU pairwise redundancy removal using normalized MI threshold.

        Keeps the feature with higher Stage 1 dCor within each redundant pair.
        """
        p = len(features)
        if p < 2:  # Need at least 2 features for redundancy analysis
            return list(features)
        keep = set(features)  # Start with all features
        order = list(range(p))  # Feature order
        # Pre-extract columns to speed up access
        cols = [X[:, i] for i in range(p)]
        # Iterate in blocks to bound memory usage
        for i0 in range(0, p, chunk):  # Process features in chunks
            i1 = min(i0 + chunk, p)
            for i in range(i0, i1):  # For each feature in current chunk
                if features[i] not in keep:  # Skip if already removed
                    continue
                xi = cols[i]  # Current feature
                for j in range(i + 1, p):  # Compare with remaining features
                    if features[j] not in keep:  # Skip if already removed
                        continue
                    xj = cols[j]  # Comparison feature
                    nmi = self._mi_nmi_gpu(xi, xj, bins=bins)  # Compute normalized mutual information
                    if nmi >= float(threshold):  # If MI above threshold, features are redundant
                        fi = features[i]
                        fj = features[j]
                        # Choose by Stage 1 dCor, fallback to keep first
                        if dcor_scores.get(fi, 0.0) >= dcor_scores.get(fj, 0.0):
                            if fj in keep:
                                keep.remove(fj)  # Remove feature with lower dCor
                                self._log_info("MI redundancy (GPU)", pair=f"{fi},{fj}", kept=fi, removed=fj, nmi=round(float(nmi), 4))
                        else:
                            if fi in keep:
                                keep.remove(fi)  # Remove feature with lower dCor
                                self._log_info("MI redundancy (GPU)", pair=f"{fi},{fj}", kept=fj, removed=fi, nmi=round(float(nmi), 4))
        # Preserve original order
        kept_ordered = [f for f in features if f in keep]  # Return features in original order
        return kept_ordered

    def _parse_importance_threshold(self, threshold_cfg: Any, importances: List[float]) -> float:
        try:
            if isinstance(threshold_cfg, str):
                thr_s = threshold_cfg.strip().lower()
                if thr_s == 'median':
                    arr = np.array([float(v) for v in importances if np.isfinite(v)], dtype=float)
                    if arr.size == 0:
                        return 0.0
                    # Use median of strictly positive importances if available
                    pos = arr[arr > 0]
                    return float(np.median(pos)) if pos.size > 0 else float(np.median(arr))
                # try to parse as float string
                return float(threshold_cfg)
            return float(threshold_cfg)
        except Exception:
            return 0.0

    def _stage3_selectfrommodel(self, X_df, y_series, candidates: List[str]) -> Tuple[List[str], dict, str]:
        """Select features using an embedded model with importance threshold.

        Returns: (selected_features, importances_map, backend_used)
        """
        # Determine task type (classification vs regression)
        task = str(getattr(self, 'stage3_task', 'auto')).lower()
        if task == 'auto':
            try:
                y_vals = y_series.values
                uniq = np.unique(y_vals)
                # Classify as classification if <= 10 unique integer values, otherwise regression
                task = 'classification' if (len(uniq) <= 10 and np.allclose(uniq, uniq.astype(int))) else 'regression'
            except Exception:
                task = 'regression'  # Default to regression if auto-detection fails

        backend_used = 'lgbm'  # Use LightGBM as the embedded model
        importances: dict = {}  # Store feature importances
        model = None
        
        # Optimize data types for performance - convert to float32 for GPU acceleration
        X = X_df[candidates].values.astype(np.float32)
        y = y_series.values.astype(np.float32)
        
        # Apply sampling for large datasets to control computational cost
        max_rows = int(getattr(self, 'selection_max_rows', 100000))
        if X.shape[0] > max_rows:
            self._log_info("Applying systematic sampling for feature selection", 
                          original_rows=X.shape[0], sampled_rows=max_rows, 
                          sampling_ratio=round(max_rows/X.shape[0], 3))
            # Use systematic sampling to preserve time-series structure
            indices = np.linspace(0, X.shape[0] - 1, max_rows, dtype=int)
            X = X[indices]
            y = y[indices]
        
        esr = int(getattr(self, 'stage3_lgbm_early_stopping_rounds', 0))  # Early stopping rounds
        eval_set = None
        # Build TimeSeriesSplit for validation and early stopping if requested
        if esr and esr > 0 and X.shape[0] >= 10:
            try:
                from sklearn.model_selection import TimeSeriesSplit
                # Use more splits for better validation, minimum data size per split
                min_samples_per_split = max(50, X.shape[0] // 10)  # At least 50 samples per split
                n_splits = max(3, min(5, X.shape[0] // min_samples_per_split - 1))  # 3-5 splits
                tss = TimeSeriesSplit(n_splits=n_splits)
                tr_idx, va_idx = list(tss.split(X))[-1]  # Use last split for validation
                X_tr, y_tr = X[tr_idx], y[tr_idx]  # Training data
                X_va, y_va = X[va_idx], y[va_idx]  # Validation data
                eval_set = (X_tr, y_tr, X_va, y_va)  # Evaluation set for early stopping
                self._log_info("TimeSeriesSplit configured for feature selection",
                              splits=n_splits, train_size=len(X_tr), val_size=len(X_va))
            except Exception as e_ts:
                eval_set = None
                self._log_warn("TimeSeriesSplit setup failed", error=str(e_ts))

        # Try preferred backend: LightGBM
        try:
            import lightgbm as lgb
            params = {
                'num_leaves': int(getattr(self, 'stage3_lgbm_num_leaves', 31)),  # Number of leaves
                'max_depth': int(getattr(self, 'stage3_lgbm_max_depth', -1)),  # Maximum depth
                'n_estimators': int(getattr(self, 'stage3_lgbm_n_estimators', 200)),  # Number of trees
                'learning_rate': float(getattr(self, 'stage3_lgbm_learning_rate', 0.05)),  # Learning rate
                'subsample': float(getattr(self, 'stage3_lgbm_bagging_fraction', 0.8)),  # Subsample ratio
                'colsample_bytree': float(getattr(self, 'stage3_lgbm_feature_fraction', 0.8)),  # Feature fraction
                'random_state': int(getattr(self, 'stage3_random_state', 42)),  # Random seed
                'n_jobs': -1,  # Use all available cores
            }
            
            if task == 'classification':
                params['objective'] = 'binary' if len(np.unique(y)) == 2 else 'multiclass'
                params['metric'] = 'binary_logloss' if len(np.unique(y)) == 2 else 'multi_logloss'
            else:
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
            
            # Train model
            if eval_set is not None:
                X_tr, y_tr, X_va, y_va = eval_set
                model = lgb.LGBMClassifier(**params) if task == 'classification' else lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(esr)])
            else:
                model = lgb.LGBMClassifier(**params) if task == 'classification' else lgb.LGBMRegressor(**params)
                model.fit(X, y)
            
            # Extract feature importances
            importances = dict(zip(candidates, model.feature_importances_))
            
            # Apply importance threshold
            threshold = self._parse_importance_threshold('median', list(importances.values()))
            selected = [f for f, imp in importances.items() if imp >= threshold]
            
            self._log_info("LightGBM feature selection completed", 
                          task=task, selected=len(selected), threshold=round(threshold, 6))
            
        except Exception as e:
            self._log_warn("LightGBM feature selection failed", error=str(e))
            # Fallback: return all candidates
            selected = candidates
            importances = {f: 1.0 for f in candidates}
            backend_used = 'fallback'
        
        return selected, importances, backend_used

    def apply_feature_selection_pipeline(self, df: cudf.DataFrame, target_col: str, 
                                       candidates: List[str], dcor_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Apply the complete feature selection pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            candidates: List of candidate feature names
            dcor_scores: Dictionary of distance correlation scores
            
        Returns:
            Dictionary with selection results and statistics
        """
        try:
            self._log_info("Starting feature selection pipeline", 
                          n_candidates=len(candidates), target=target_col)
            
            results = {
                'stage1_candidates': candidates,
                'stage2_vif_selected': [],
                'stage2_mi_selected': [],
                'stage3_final_selected': [],
                'importances': {},
                'selection_stats': {}
            }
            
            if not candidates:
                self._log_warn("No candidates provided for feature selection")
                return results
            
            # Stage 2: VIF-based multicollinearity removal
            X_matrix, used_cols = self._to_cupy_matrix(df, candidates)
            if X_matrix.shape[1] == 0 or len(used_cols) == 0:
                self._critical_error("to_cupy matrix failed; no usable numeric features for VIF", candidates=len(candidates))
            vif_selected = self._compute_vif_iterative_gpu(X_matrix, used_cols, self.vif_threshold)
            results['stage2_vif_selected'] = vif_selected
            self._log_info("VIF selection completed", original=len(candidates), selected=len(vif_selected))
            
            # Stage 2: MI-based redundancy removal
            if len(results['stage2_vif_selected']) > 1:
                try:
                    mi_selected = self._compute_mi_cluster_representatives(
                        df, results['stage2_vif_selected'], dcor_scores or {}
                    )
                    results['stage2_mi_selected'] = mi_selected
                    self._log_info("MI redundancy removal completed", original=len(results['stage2_vif_selected']), selected=len(mi_selected))
                except Exception as e:
                    self._critical_error("MI redundancy removal failed", error=str(e))
            else:
                results['stage2_mi_selected'] = results['stage2_vif_selected']
            
            # Stage 3: Embedded feature selection
            if len(results['stage2_mi_selected']) > 1:
                try:
                    final_selected, importances, backend = self._stage3_selectfrommodel(
                        df, df[target_col], results['stage2_mi_selected']
                    )
                    results['stage3_final_selected'] = final_selected
                    results['importances'] = importances
                    results['selection_stats']['backend_used'] = backend
                    self._log_info("Embedded feature selection completed", original=len(results['stage2_mi_selected']), selected=len(final_selected), backend=backend)
                except Exception as e:
                    self._critical_error("Embedded feature selection failed", error=str(e))
            else:
                results['stage3_final_selected'] = results['stage2_mi_selected']
                results['importances'] = {f: 1.0 for f in results['stage2_mi_selected']}
                results['selection_stats']['backend_used'] = 'single_feature'
            
            # Compute selection statistics
            results['selection_stats'].update({
                'stage1_count': len(candidates),
                'stage2_vif_count': len(results['stage2_vif_selected']),
                'stage2_mi_count': len(results['stage2_mi_selected']),
                'stage3_final_count': len(results['stage3_final_selected']),
                'vif_removed': len(candidates) - len(results['stage2_vif_selected']),
                'mi_removed': len(results['stage2_vif_selected']) - len(results['stage2_mi_selected']),
                'embedded_removed': len(results['stage2_mi_selected']) - len(results['stage3_final_selected']),
                'total_removed': len(candidates) - len(results['stage3_final_selected']),
                'reduction_ratio': round(1.0 - len(results['stage3_final_selected']) / len(candidates), 3)
            })
            
            self._log_info("Feature selection pipeline completed", 
                          final_count=len(results['stage3_final_selected']),
                          reduction_ratio=results['selection_stats']['reduction_ratio'])
            
            return results
            
        except Exception as e:
            self._critical_error(f"Error in feature selection pipeline: {e}")
