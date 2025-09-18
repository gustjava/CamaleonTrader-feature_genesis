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
from utils.logging_utils import get_logger

logger = get_logger(__name__, "features.statistical_tests.feature_selection")


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
        cluster_details = []
        removed_by_clustering = []
        
        for lbl in np.unique(labels):
            idxs = np.where(labels == lbl)[0]
            cluster_feats = [cand[i] for i in idxs]
            if dcor_scores:
                rep = max(cluster_feats, key=lambda f: dcor_scores.get(f, 0.0))
            else:
                rep = cluster_feats[0]
            reps.append(rep)
            
            # Track cluster details
            cluster_removed = [f for f in cluster_feats if f != rep]
            removed_by_clustering.extend(cluster_removed)
            cluster_details.append({
                'cluster_id': int(lbl),
                'representative': rep,
                'removed': cluster_removed,
                'size': len(cluster_feats)
            })

        # Garante que representantes pertencem ao conjunto original de candidatos pós‑VIF
        reps = [r for r in reps if r in candidates]
        
        # Log detailed MI clustering results
        self._log_info("MI clustering completed", 
                      original_count=len(candidates),
                      clustered_count=len(cand),
                      final_representatives=len(reps),
                      removed_by_clustering=len(removed_by_clustering),
                      cluster_details=cluster_details,
                      removed_features=removed_by_clustering,
                      kept_features=reps)
        
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
            self._log_info("Dropping non-numeric/invalid columns for GPU matrix", 
                          count=len(invalid), 
                          examples=details,
                          invalid_columns=invalid,
                          valid_columns=used)
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
        removed_features = []  # Track removed features
        idx = cp.arange(X.shape[1])  # Feature indices
        
        self._log_info(f"VIF: Starting with {len(features)} features", 
                      threshold=threshold, 
                      features=features[:10])
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
                removed_features.append((removed, round(vmax, 3)))  # Track removed feature with VIF value
                # Drop column from Z to update for next iteration
                cols = [i for i in range(Z.shape[1]) if i != imax]
                Z = Z[:, cols]  # Remove corresponding column from standardized data
                self._log_info("VIF removal (GPU)", feature=removed, vif=round(vmax, 3), remaining=len(keep))
            except Exception as e:
                self._log_warn("VIF GPU failed; keeping current set", error=str(e))
                break  # Stop if computation fails
        # Log detailed VIF results
        self._log_info("VIF iterative done (GPU)", 
                      original_count=len(features),
                      removed_count=len(removed_features),
                      kept_count=len(keep),
                      removed_features=removed_features,
                      kept_features=keep)
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

    def _stage3_selectfrommodel(self, X_df, y_series, candidates: List[str]) -> Tuple[List[str], dict, str, float, Dict[str, Any]]:
        """Embedded CatBoost selector with CPCV/TSS and GPU-only training.

        Returns: (selected_features, importances_map, backend_used, model_score, detailed_metrics)
        """
        
        # FORÇA LOGGING PARA DEBUG - SEMPRE APARECER
        print(f"🚀 [FORCE DEBUG] _stage3_selectfrommodel START - candidates: {len(candidates)}, X_df shape: {X_df.shape}")
        print(f"🔧 [FORCE DEBUG] y_series shape: {y_series.shape}, dtype: {y_series.dtype}")
        import sys
        sys.stdout.flush()
        
        backend_used = 'catboost'

        # Resolve task type
        task = str(getattr(self, 'stage3_task', 'auto')).lower()
        if task == 'auto':
            try:
                y_vals = y_series.values if hasattr(y_series, 'values') else np.asarray(y_series)
                uniq = np.unique(y_vals)
                max_classes = int(getattr(self, 'stage3_classification_max_classes', 10))
                task = 'classification' if (len(uniq) <= max(2, max_classes) and np.allclose(uniq, np.round(uniq))) else 'regression'
            except Exception:
                task = 'regression'

        # Pre-filter: ensure candidates exist and are numeric; drop zero-variance columns
        present = [c for c in candidates if c in X_df.columns]
        if not present:
            self._critical_error('No valid candidate features present in DataFrame')
            # unreachable

        # Build a clean pandas DataFrame and numpy arrays; drop NaN/Inf jointly
        try:
            import cudf as _cudf
            Xp_all = X_df[present].to_pandas() if isinstance(X_df, _cudf.DataFrame) else X_df[present].copy()
            yp = y_series.to_pandas() if hasattr(y_series, 'to_pandas') else y_series
            # Keep only numeric columns
            import pandas as _pd
            num_cols = [c for c in Xp_all.columns if _pd.api.types.is_numeric_dtype(Xp_all[c])]
            dropped_non_numeric = [c for c in Xp_all.columns if c not in num_cols]
            if dropped_non_numeric:
                self._log_warn('Dropping non-numeric features for CatBoost', dropped=len(dropped_non_numeric), examples=dropped_non_numeric[:10])
            Xp = Xp_all[num_cols]
            # Append target and clean
            data = Xp.copy()
            data['__y__'] = yp
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data = data.dropna()
            # Drop constant columns (zero variance)
            try:
                nunq = data.drop(columns=['__y__']).nunique()
                const_cols = nunq[nunq <= 1].index.tolist()
                if const_cols:
                    data = data.drop(columns=const_cols, errors='ignore')
                    self._log_warn('Dropping constant-variance features for CatBoost', dropped=len(const_cols), examples=const_cols[:10])
            except Exception:
                pass
            # Final safety: ensure we still have features
            if data.shape[1] <= 1:  # only __y__ remains
                self._critical_error('No usable numeric features after cleaning for CatBoost')
                # unreachable
            y = data.pop('__y__').to_numpy(dtype=np.float32, copy=False)
            X = data.to_numpy(dtype=np.float32, copy=False)
            # Update candidates list to the used order
            candidates = list(data.columns)
        except Exception as e:
            self._critical_error('Failed to materialize training arrays for CatBoost', error=str(e))
            return candidates, {f: 1.0 for f in candidates}, backend_used, 0.0, {'error': str(e)}

        # Optional downsampling
        use_full = bool(getattr(self, 'stage3_catboost_use_full_dataset', False))
        if not use_full and X.shape[0] > int(getattr(self, 'selection_max_rows', 100000)):
            max_rows = int(getattr(self, 'selection_max_rows', 100000))
            if task == 'classification' and bool(getattr(self, 'stage3_stratified_sampling', True)):
                try:
                    classes, counts = np.unique(y, return_counts=True)
                    quotas = {c: max(1, int(round(max_rows * cnt / float(len(y))))) for c, cnt in zip(classes, counts)}
                    idx_parts = []
                    for c in classes:
                        cls_idx = np.nonzero(y == c)[0]
                        k = min(len(cls_idx), quotas[c])
                        if k > 0:
                            sel = np.linspace(0, len(cls_idx) - 1, k, dtype=int)
                            idx_parts.append(cls_idx[sel])
                    if idx_parts:
                        idx = np.sort(np.concatenate(idx_parts))
                        if idx.shape[0] > max_rows:
                            idx = idx[:max_rows]
                        self._log_info('Stratified sampling applied', original_rows=X.shape[0], sampled_rows=int(idx.shape[0]))
                        X, y = X[idx], y[idx]
                    else:
                        ids = np.linspace(0, X.shape[0] - 1, max_rows, dtype=int)
                        X, y = X[ids], y[ids]
                except Exception as se:
                    self._log_warn('Stratified sampling failed; using systematic', error=str(se))
                    ids = np.linspace(0, X.shape[0] - 1, max_rows, dtype=int)
                    X, y = X[ids], y[ids]
            else:
                ids = np.linspace(0, X.shape[0] - 1, max_rows, dtype=int)
                X, y = X[ids], y[ids]

        # Model params
        iterations = int(getattr(self, 'stage3_catboost_iterations', 200))
        learning_rate = float(getattr(self, 'stage3_catboost_learning_rate', 0.05))
        depth = int(getattr(self, 'stage3_catboost_depth', 6))
        task_type = str(getattr(self, 'stage3_catboost_task_type', 'GPU'))
        devices_cfg = str(getattr(self, 'stage3_catboost_devices', '0'))
        thread_count = int(getattr(self, 'stage3_catboost_thread_count', 1))
        random_state = int(getattr(self, 'stage3_random_state', 42))

        # GPU validation and device mapping
        if task_type.upper() == 'GPU':
            try:
                import cupy as _cp
                if int(_cp.cuda.runtime.getDeviceCount()) <= 0:
                    raise RuntimeError('No CUDA devices available for CatBoost GPU task')
            except Exception as de:
                self._critical_error('GPU validation failed', error=str(de))
            import os as _os
            devices = '0' if _os.environ.get('CUDA_VISIBLE_DEVICES') else devices_cfg
        else:
            self._log_error('CPU CatBoost not allowed; GPU-only policy enforced')
            raise RuntimeError('CPU CatBoost not allowed')

        # Build model
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool as _Pool
        if task == 'classification':
            unique_y = np.unique(y)
            is_binary = len(unique_y) == 2
            loss_function = 'Logloss' if is_binary else 'MultiClass'
            eval_metric = 'AUC' if is_binary else 'Accuracy'
            model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth if depth > 0 else 6,
                random_seed=random_state,
                task_type=task_type,
                devices=devices,
                loss_function=loss_function,
                eval_metric=eval_metric,
                verbose=0,
                thread_count=thread_count,
            )
        else:
            loss_fn = str(getattr(self, 'stage3_catboost_loss_regression', 'RMSE'))
            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth if depth > 0 else 6,
                random_seed=random_state,
                task_type=task_type,
                devices=devices,
                loss_function=loss_fn,
                eval_metric=loss_fn,
                verbose=0,
                thread_count=thread_count,
            )

        # Early stopping + CV setup
        esr = int(getattr(self, 'stage3_catboost_early_stopping_rounds', 0))
        n_splits_cfg = int(getattr(self, 'stage3_cv_splits', 0))
        min_train = int(getattr(self, 'stage3_cv_min_train', 200))

        def _fit_eval(X_fit, y_fit, tr_idx=None, va_idx=None):
            feat_names = list(candidates)
            try:
                if tr_idx is not None and va_idx is not None:
                    train_pool = _Pool(X_fit[tr_idx], y_fit[tr_idx], feature_names=feat_names)
                    valid_pool = _Pool(X_fit[va_idx], y_fit[va_idx], feature_names=feat_names)
                    fit_kwargs = {'eval_set': [valid_pool]}
                    if esr > 0:
                        fit_kwargs.update({'use_best_model': True, 'early_stopping_rounds': esr})
                    try:
                        self._log_info('CatBoost fit start (with validation)', train_size=len(tr_idx), valid_size=len(va_idx))
                        model.fit(train_pool, **fit_kwargs)
                        self._log_info('CatBoost fit successful (with validation)')
                    except Exception as fit_err:
                        # Log detailed traceback and re-raise
                        tb = _tb.format_exc()
                        self._log_error('CatBoost fit failed (with validation)', error=str(fit_err), traceback=tb)
                        # Return safe fallback values to prevent pipeline crash
                        fi = np.ones(len(candidates), dtype=float)  # Uniform importances indicate failure
                        return fi, 0.0, {'error': str(fit_err), 'traceback': tb}
                    
                    try:
                        pred_tr = model.predict(train_pool)
                        pred_va = model.predict(valid_pool)
                        ytr, yva = y_fit[tr_idx], y_fit[va_idx]
                    except Exception as pred_err:
                        self._log_error('CatBoost predict failed', error=str(pred_err))
                        fi = np.ones(len(candidates), dtype=float)
                        return fi, 0.0, {'error': f'predict failed: {pred_err}'}
                else:
                    train_pool = _Pool(X_fit, y_fit, feature_names=feat_names)
                    try:
                        self._log_info('CatBoost fit start (no validation)', train_size=len(X_fit))
                        model.fit(train_pool)
                        self._log_info('CatBoost fit successful (no validation)')
                    except Exception as fit_err:
                        tb = _tb.format_exc()
                        self._log_error('CatBoost fit failed (no validation)', error=str(fit_err), traceback=tb)
                        fi = np.ones(len(candidates), dtype=float)
                        return fi, 0.0, {'error': str(fit_err), 'traceback': tb}
                    
                    try:
                        pred_tr = model.predict(train_pool)
                        ytr = y_fit
                        pred_va, yva = None, None
                    except Exception as pred_err:
                        self._log_error('CatBoost predict failed (no validation)', error=str(pred_err))
                        fi = np.ones(len(candidates), dtype=float)
                        return fi, 0.0, {'error': f'predict failed: {pred_err}'}
            except Exception as pool_err:
                self._log_error('CatBoost Pool creation failed', error=str(pool_err), traceback=_tb.format_exc())
                fi = np.ones(len(candidates), dtype=float)
                return fi, 0.0, {'error': f'pool creation failed: {pool_err}'}

            metrics = {}
            if task == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics['training_accuracy'] = float(accuracy_score(ytr, pred_tr))
                metrics['training_precision'] = float(precision_score(ytr, pred_tr, average='weighted', zero_division=0))
                metrics['training_recall'] = float(recall_score(ytr, pred_tr, average='weighted', zero_division=0))
                metrics['training_f1'] = float(f1_score(ytr, pred_tr, average='weighted', zero_division=0))
                if pred_va is not None:
                    metrics['validation_accuracy'] = float(accuracy_score(yva, pred_va))
                    metrics['validation_precision'] = float(precision_score(yva, pred_va, average='weighted', zero_division=0))
                    metrics['validation_recall'] = float(recall_score(yva, pred_va, average='weighted', zero_division=0))
                    metrics['validation_f1'] = float(f1_score(yva, pred_va, average='weighted', zero_division=0))
                    score = metrics['validation_f1']
                else:
                    score = metrics['training_f1']
            else:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                metrics['training_r2'] = float(r2_score(ytr, pred_tr))
                metrics['training_mse'] = float(mean_squared_error(ytr, pred_tr))
                metrics['training_mae'] = float(mean_absolute_error(ytr, pred_tr))
                metrics['training_rmse'] = float(np.sqrt(metrics['training_mse']))
                if pred_va is not None:
                    metrics['validation_r2'] = float(r2_score(yva, pred_va))
                    metrics['validation_mse'] = float(mean_squared_error(yva, pred_va))
                    metrics['validation_mae'] = float(mean_absolute_error(yva, pred_va))
                    metrics['validation_rmse'] = float(np.sqrt(metrics['validation_mse']))
                    score = metrics['validation_r2']
                else:
                    score = metrics['training_r2']

            # Feature importance: always specify a valid type and provide data when possible
            fi = None
            _fi_errors = []
            try:
                fi = model.get_feature_importance(data=train_pool, type='PredictionValuesChange')
            except Exception as _e1:
                _fi_errors.append(str(_e1))
                try:
                    fi = model.get_feature_importance(data=train_pool, type='FeatureImportance')
                except Exception as _e2:
                    _fi_errors.append(str(_e2))
                    try:
                        fi = model.get_feature_importance(type='FeatureImportance')
                    except Exception as _e3:
                        _fi_errors.append(str(_e3))
                        try:
                            fi = model.get_feature_importance(type='PredictionValuesChange')
                        except Exception as _e4:
                            _fi_errors.append(str(_e4))
                            fi = None
            if fi is None:
                # As a last resort, produce zeros and log
                self._log_warn('CatBoost get_feature_importance failed; returning zeros', errors=_fi_errors[:3])
                fi = np.zeros(len(candidates), dtype=float)
            else:
                pass  # Feature importance obtained successfully
                # Ensure correct length; if mismatch, fallback to zeros to keep pipeline stable
                try:
                    fi = np.asarray(fi, dtype=float)
                    if fi.shape[0] != len(candidates):
                        self._log_warn('Feature importance length mismatch; normalizing', fi_len=int(fi.shape[0]), n_features=int(len(candidates)))
                        if fi.shape[0] > 0:
                            # Try to trim or pad
                            if fi.shape[0] > len(candidates):
                                fi = fi[:len(candidates)]
                            else:
                                pad = np.zeros(len(candidates) - fi.shape[0], dtype=float)
                                fi = np.concatenate([fi, pad])
                        else:
                            fi = np.zeros(len(candidates), dtype=float)
                except Exception as _e5:
                    self._log_warn('Failed to process feature importances; returning zeros', error=str(_e5))
                    fi = np.zeros(len(candidates), dtype=float)

            # Model info (log only)
            try:
                it_used = int(model.get_best_iteration())
            except Exception:
                it_used = iterations
            metrics['model_info'] = {
                'iterations_used': it_used,
                'learning_rate': learning_rate,
                'task_type': task_type,
                'depth': int(depth if depth > 0 else 6),
            }
            return fi, float(score), metrics

        # Build CV plan: CPCV preferred
        final_metrics = {}
        used_splits = 0
        model_score = 0.0
        agg_metrics = []
        agg_fi = None

        use_cpcv = bool(getattr(self, 'cpcv_enabled', False))
        cpcv_n = int(getattr(self, 'cpcv_n_groups', 6))
        cpcv_k = int(getattr(self, 'cpcv_k_leave_out', 2))
        cpcv_purge = int(getattr(self, 'cpcv_purge', 0))
        cpcv_embargo = int(getattr(self, 'cpcv_embargo', 0))

        try:
            if use_cpcv and cpcv_n >= 3 and X.shape[0] >= cpcv_n:
                from utils.cpcv import combinatorial_purged_cv
                n_samples = X.shape[0]
                sizes = (n_samples // cpcv_n) * np.ones(cpcv_n, dtype=int)
                sizes[: n_samples % cpcv_n] += 1
                idx = np.arange(n_samples)
                groups = []
                pos = 0
                for s in sizes:
                    groups.append(idx[pos:pos + int(s)])
                    pos += int(s)
                # Log CPCV plan details for transparency
                self._log_info('CPCV plan', enabled=bool(use_cpcv), n_groups=int(cpcv_n), k_leave_out=int(cpcv_k), purge=int(cpcv_purge), embargo=int(cpcv_embargo), total_samples=int(n_samples))
                for tr_idx, te_idx in combinatorial_purged_cv(groups, k_leave_out=cpcv_k, purge=cpcv_purge, embargo=cpcv_embargo):
                    if len(tr_idx) < max(10, min_train) or len(te_idx) < max(5, max(1, min_train // 5)):
                        continue
                    fi, score, m = _fit_eval(X, y, tr_idx, te_idx)
                    agg_fi = fi if agg_fi is None else (agg_fi + fi)
                    agg_metrics.append(m)
                    used_splits += 1
                    model_score += score
                if used_splits > 0:
                    agg_fi = agg_fi / float(used_splits)
                    model_score = model_score / float(used_splits)
                    final_metrics['cv_scheme'] = 'cpcv'
                    final_metrics['cv_folds_used'] = used_splits
        except Exception as e:
            self._log_warn('CPCV failed; will try TimeSeriesSplit', error=str(e))

        # Fallback to TimeSeriesSplit if no CPCV splits used
        if used_splits == 0 and n_splits_cfg >= 2 and X.shape[0] >= max(10, 2 * min_train):
            from sklearn.model_selection import TimeSeriesSplit
            tss_n = max(2, n_splits_cfg)
            self._log_info('TSS plan', n_splits=int(tss_n), min_train=int(min_train), total_samples=int(X.shape[0]))
            tss = TimeSeriesSplit(n_splits=tss_n)
            for tr_idx, va_idx in tss.split(X):
                if len(tr_idx) < max(10, min_train) or len(va_idx) < max(5, max(1, min_train // 5)):
                    continue
                fi, score, m = _fit_eval(X, y, tr_idx, va_idx)
                agg_fi = fi if agg_fi is None else (agg_fi + fi)
                agg_metrics.append(m)
                used_splits += 1
                model_score += score
            if used_splits > 0:
                agg_fi = agg_fi / float(used_splits)
                model_score = model_score / float(used_splits)
                final_metrics['cv_scheme'] = 'tss'
                final_metrics['cv_folds_used'] = used_splits

        # No CV? Fit once (optionally with early stopping)
        if used_splits == 0:
            fi, model_score, m = _fit_eval(X, y, None, None)
            agg_fi = fi
            final_metrics.update(m)
            final_metrics['cv_scheme'] = 'none'
            final_metrics['cv_folds_used'] = 0

        # Aggregate per-fold metrics (mean) if available
        if agg_metrics:
            sums: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            model_infos = []
            for m in agg_metrics:
                if not isinstance(m, dict):
                    continue
                if 'model_info' in m:
                    model_infos.append(m['model_info'])
                for k, v in m.items():
                    if k == 'model_info':
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    sums[k] = sums.get(k, 0.0) + fv
                    counts[k] = counts.get(k, 0) + 1
            for k in sums:
                if counts.get(k, 0) > 0:
                    final_metrics[k] = float(sums[k] / counts[k])
            # Summarize model_info if present
            if model_infos:
                try:
                    iters = [int(info.get('iterations_used', 0)) for info in model_infos if isinstance(info, dict)]
                    it_mean = int(round(sum(iters) / max(1, len(iters)))) if iters else iterations
                except Exception:
                    it_mean = iterations
                final_metrics['model_info'] = {
                    'iterations_used': it_mean,
                    'learning_rate': learning_rate,
                    'task_type': task_type,
                    'depth': int(depth if depth > 0 else 6),
                }
        # Check if any errors occurred during training
        has_errors = False
        error_messages = []
        
        # Check for errors in aggregated metrics
        if agg_metrics:
            for m in agg_metrics:
                if isinstance(m, dict) and 'error' in m:
                    has_errors = True
                    error_messages.append(m.get('error', 'unknown error'))
        
        # Check if all importances are uniform (indicates failure)
        if agg_fi is not None:
            fi_array = np.asarray(agg_fi, dtype=float)
            if fi_array.size > 1:
                # Check if all values are identical (uniform)
                if np.allclose(fi_array, fi_array[0], rtol=1e-10):
                    has_errors = True
                    error_messages.append('uniform feature importances detected')
        
        # Update backend status based on error detection
        if has_errors:
            backend_used = 'error'
            final_metrics['catboost_errors'] = error_messages
            self._log_error('CatBoost training had errors', errors=error_messages)
        
        # Record FI type used for transparency
        final_metrics.setdefault('feature_importance_types', ['PredictionValuesChange'])

        # Map importances and select
        importances = dict(zip(candidates, np.asarray(agg_fi, dtype=float).tolist()))
        thr_cfg = getattr(self, 'stage3_importance_threshold', 'median')
        threshold = self._parse_importance_threshold(thr_cfg, list(importances.values()))
        selected = [f for f, v in importances.items() if float(v) >= float(threshold)]

        # Log summary
        self._log_info('CatBoost feature selection completed',
                       task=task, selected=len(selected), threshold=float(round(threshold, 6)),
                       model_score=float(round(model_score, 6)), cv_splits_used=int(used_splits),
                       cv_scheme=final_metrics.get('cv_scheme', 'unknown'), backend=backend_used)
        if final_metrics:
            self._log_info('CatBoost detailed metrics', **final_metrics)

        return selected, importances, backend_used, float(model_score), final_metrics

    def apply_feature_selection_pipeline(self, df: cudf.DataFrame, target_col: str, candidates: List[str], dcor_scores: Dict[str, float], full_ddf=None) -> Dict[str, Any]:
        """Run VIF -> MI -> Embedded (CatBoost) pipeline on provided candidates.

        Returns a dict with keys: stage2_vif_selected, stage2_mi_selected, stage3_final_selected, importances, selection_stats
        """
        try:
            # Ensure target exists
            if target_col not in df.columns:
                self._critical_error('Target column missing for selection pipeline', target=target_col)

            # Keep only candidates present in df
            valid = [c for c in candidates if c in df.columns]
            if not valid:
                self._log_warn('No valid candidates after presence check; skipping pipeline')
                return {
                    'stage2_vif_selected': [],
                    'stage2_mi_selected': [],
                    'stage3_final_selected': [],
                    'importances': {},
                    'selection_stats': {'backend_used': 'none'}
                }

            # Stage 2: VIF on GPU
            try:
                X_gpu, used_cols = self._to_cupy_matrix(df, valid, dtype='f4')
                vif_selected = self._compute_vif_iterative_gpu(X_gpu, used_cols, threshold=float(getattr(self, 'vif_threshold', 5.0)))
            except Exception as e:
                self._log_warn('VIF stage failed; passing all valid candidates', error=str(e))
                vif_selected = list(valid)

            # Stage 2b: MI redundancy or clustering
            try:
                if bool(getattr(self, 'mi_cluster_enabled', True)):
                    mi_selected = self._compute_mi_cluster_representatives(df, vif_selected, dcor_scores or {})
                else:
                    # Try GPU MI redundancy first
                    try:
                        X_gpu2, used_cols2 = self._to_cupy_matrix(df, vif_selected, dtype='f4')
                        mi_selected = self._compute_mi_redundancy_gpu(X_gpu2, used_cols2, dcor_scores or {}, threshold=float(getattr(self, 'mi_threshold', 0.3)))
                    except Exception as eg:
                        self._log_warn('GPU MI redundancy failed; using CPU fallback', error=str(eg))
                        mi_selected = self._compute_mi_redundancy(df, vif_selected, dcor_scores or {}, mi_threshold=float(getattr(self, 'mi_threshold', 0.3)))
            except Exception as e:
                self._log_warn('MI stage failed; using VIF-selected set', error=str(e))
                mi_selected = list(vif_selected)

            # Stage 3: Embedded CatBoost
            final_selected, importances, backend, model_score, detailed_metrics = self._stage3_selectfrommodel(
                df, df[target_col], mi_selected
            )

            # Apply optional top-N cap
            try:
                top_n = int(getattr(self, 'stage3_top_n', 0))
            except Exception:
                top_n = 0
            if top_n and top_n > 0 and final_selected:
                # Sort by importance desc and keep top_n
                ranked = sorted(final_selected, key=lambda f: float(importances.get(f, 0.0)), reverse=True)
                final_selected = ranked[:top_n]

            self._log_info('Selection pipeline completed',
                           candidates=len(candidates), vif_selected=len(vif_selected), mi_selected=len(mi_selected), final_selected=len(final_selected))

            # Merge selection stats with detailed metrics
            sel_stats = {'backend_used': backend, 'model_score': float(model_score)}
            try:
                if isinstance(detailed_metrics, dict):
                    sel_stats.update(detailed_metrics)
            except Exception:
                pass

            return {
                'stage2_vif_selected': vif_selected,
                'stage2_mi_selected': mi_selected,
                'stage3_final_selected': final_selected,
                'importances': importances,
                'selection_stats': sel_stats,
            }
        except Exception as e:
            self._critical_error('Selection pipeline failed', error=str(e))
            return {
                'stage2_vif_selected': [],
                'stage2_mi_selected': [],
                'stage3_final_selected': [],
                'importances': {},
                'selection_stats': {'backend_used': 'error', 'error': str(e)}
            }
