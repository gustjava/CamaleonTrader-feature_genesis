"""
Feature Engineering Engine (Stage 0)

Applies early feature transformations before selection stages.
Initial scope: Generalized Baxter–King (BK) filter over configured source columns.
"""

import logging  # For logging functionality
from typing import List, Dict, Any, Optional  # For type hints

import dask_cudf  # For distributed GPU DataFrames
import cudf  # For GPU DataFrames

from .base_engine import BaseFeatureEngine  # Base class for feature engines

# Local BK helpers (decoupled from legacy SignalProcessor)
import numpy as _np  # For numerical computations on CPU
import cupy as _cp  # For GPU array operations
from functools import lru_cache as _lru_cache  # For caching BK weights computation

try:
    from cusignal import fftconvolve as _fftconv  # Try GPU-accelerated FFT convolution
except Exception:
    try:
        from cusignal.filtering import fftconvolve as _fftconv  # Try alternative GPU FFT convolution import
    except Exception:
        from scipy.signal import fftconvolve as _scipy_fft  # Fallback to CPU FFT convolution
        def _fftconv(x, w, mode="same"):  # Wrapper to convert between CPU and GPU arrays
            return _cp.asarray(_scipy_fft(_cp.asnumpy(x), _cp.asnumpy(w), mode=mode))


@_lru_cache(maxsize=16)  # Cache up to 16 different BK weight configurations
def _bk_weights_gpu(k: int, low_period: float, high_period: float) -> _cp.ndarray:
    """GPU Baxter–King weights (float32), cached per worker.

    low_period > high_period (e.g., 32 > 6) ⇒ pass-band between [w_high, w_low].
    """
    w_low = 2 * _cp.pi / float(low_period)  # Convert low period to angular frequency
    w_high = 2 * _cp.pi / float(high_period)  # Convert high period to angular frequency
    weights = _cp.zeros(2 * k + 1, dtype=_cp.float32)  # Initialize weight array (symmetric around center)
    weights[k] = (w_high - w_low) / _cp.pi  # Center weight (DC component)
    j = _cp.arange(1, k + 1, dtype=_cp.float32)  # Lag indices for positive lags
    weights[k + 1:] = (_cp.sin(w_high * j) - _cp.sin(w_low * j)) / (_cp.pi * j)  # Positive lag weights
    weights[:k] = weights[k + 1:][::-1]  # Mirror weights for negative lags (symmetric filter)
    wsum = weights.sum(dtype=_cp.float64)  # Calculate sum for normalization
    weights[k] -= (wsum - weights[k]).astype(_cp.float32)  # Adjust center weight to ensure sum = 0
    return weights


def _apply_bk_filter_gpu_partition(series: cudf.Series, k: int, low_period: float, high_period: float, causal: bool = True, fill_borders: bool = True) -> cudf.Series:
    """Deterministic partition function: apply BK on a single partition (GPU).

    Handles nulls by filling with 0 for the convolution and restoring NaNs
    at original null positions in the output.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Fast-path: if dtype is obviously non-numeric, return all-NaN
    try:
        dt = str(series.dtype).lower()
        if not (dt.startswith('float') or dt.startswith('int') or dt.startswith('uint')):  # Check if numeric
            return cudf.Series(_cp.full(len(series), _cp.nan, dtype=_cp.float32), index=series.index)
    except Exception:
        pass
    
    # Ensure float32 and capture null mask before conversion (coerce failure → all-NaN)
    try:
        s = series.astype('f4')  # Convert to float32
    except Exception:
        return cudf.Series(_cp.full(len(series), _cp.nan, dtype=_cp.float32), index=series.index)
    
    null_mask = s.isna()  # Capture null positions before filling
    # Fill nulls to allow cudf->cupy conversion
    s_filled = s.fillna(0.0)  # Fill nulls with zeros for convolution
    x = s_filled.to_cupy()  # Convert to CuPy array for GPU processing
    
    w = _bk_weights_gpu(int(k), float(low_period), float(high_period))  # Get BK weights (GPU)
    n_kernel = 2 * int(k) + 1  # Kernel size
    
    if n_kernel <= 129:  # Use direct convolution for small kernels
        y = _cp.convolve(x, w, mode="same")
    else:  # Use FFT convolution for large kernels (more efficient)
        y = _fftconv(x, w, mode="same")
    k = int(k)
    # Restore NaNs where input had nulls
    try:
        m = null_mask.to_cupy()  # Convert null mask to CuPy
        if m.any():  # If there were any nulls
            y[m] = _cp.nan  # Restore NaNs at original null positions
    except Exception:
        # Best-effort; if mask conversion fails, proceed without restoration
        pass
    # BK borders are NaN by definition (insufficient data for convolution)
    # IMPROVEMENT: Replace NaN borders with appropriate values instead of leaving them as NaN
    if fill_borders and k > 0 and len(y) > 2 * k:
        # Forward fill for the first k values (use the first valid value after the border)
        first_valid_idx = k
        if first_valid_idx < len(y) and not _cp.isnan(y[first_valid_idx]):
            y[:k] = y[first_valid_idx]
        else:
            # If the first valid value is also NaN, find the next non-NaN value
            for i in range(k, len(y)):
                if not _cp.isnan(y[i]):
                    y[:k] = y[i]
                    break
        
        # Backward fill for the last k values (use the last valid value before the border)
        last_valid_idx = len(y) - k - 1
        if last_valid_idx >= 0 and not _cp.isnan(y[last_valid_idx]):
            y[-k:] = y[last_valid_idx]
        else:
            # If the last valid value is also NaN, find the previous non-NaN value
            for i in range(len(y) - k - 1, -1, -1):
                if not _cp.isnan(y[i]):
                    y[-k:] = y[i]
                    break
    elif fill_borders and k > 0:
        # Edge case: series is too short, use the only available value
        if len(y) > 0:
            valid_val = None
            for i in range(len(y)):
                if not _cp.isnan(y[i]):
                    valid_val = y[i]
                    break
            if valid_val is not None:
                y[:] = valid_val

    # Build output series and apply causal shift if requested
    out = cudf.Series(y, index=series.index)  # Create output series
    if causal and k > 0:  # Apply causal shift to avoid look-ahead bias
        # Shift by +k to ensure each value at t depends only on x[<= t]
        out = out.shift(k)
    
    return out

logger = logging.getLogger(__name__)


class FeatureEngineeringEngine(BaseFeatureEngine):
    """
    Stage 0 feature engineering engine.

    Currently supports:
    - Baxter–King band‑pass filter over multiple source columns.
    """

    def _bk_params(self) -> Dict[str, Any]:
        """Resolve BK parameters from config with backward compatibility.

        Priority:
        1) features.feature_engineering.baxter_king.{k,low_freq,high_freq,source_columns}
        2) features.baxter_king.{k,low_freq,high_freq}
        3) legacy flat: features.baxter_king_k, _low_freq, _high_freq
        """
        feats = getattr(self.settings, 'features', None)  # Get features configuration
        k = 12  # Default kernel size (2k+1 = 25 taps)
        low = 32.0  # Default low frequency period (long-term trend)
        high = 6.0  # Default high frequency period (short-term noise)
        sources: List[str] = []  # Source columns to apply BK filter to
        apply_all = False  # Whether to apply to all numeric columns
        causal = True  # Whether to apply causal shift
        fill_borders = True  # Whether to fill NaN borders with appropriate values
        if feats is not None:
            try:
                fe = getattr(feats, 'feature_engineering', {}) or {}  # Get feature engineering config
                bk = fe.get('baxter_king', {}) if isinstance(fe, dict) else {}  # Get BK config
                if isinstance(bk, dict):
                    k = int(bk.get('k', k))  # Kernel size
                    low = float(bk.get('low_freq', low))  # Low frequency period
                    high = float(bk.get('high_freq', high))  # High frequency period
                    sc = bk.get('source_columns', [])  # Source columns
                    if isinstance(sc, list):
                        sources = [str(c) for c in sc]  # Convert to strings
                    apply_all = bool(bk.get('apply_to_all', False))  # Apply to all flag
                    causal = bool(bk.get('causal', causal))  # Causal flag
                    fill_borders = bool(bk.get('fill_borders', fill_borders))  # Fill borders flag
            except Exception:
                pass
            # fallback to features.baxter_king dict
            try:
                if not sources:  # Only use fallback if no sources specified
                    bk2 = getattr(feats, 'baxter_king', {}) or {}
                    if isinstance(bk2, dict):
                        k = int(bk2.get('k', k))  # Kernel size
                        low = float(bk2.get('low_freq', low))  # Low frequency period
                        high = float(bk2.get('high_freq', high))  # High frequency period
                        causal = bool(bk2.get('causal', causal))  # Causal flag
            except Exception:
                pass
            # legacy flat keys
            try:
                k = int(getattr(feats, 'baxter_king_k', k))  # Legacy kernel size
                low = float(getattr(feats, 'baxter_king_low_freq', low))  # Legacy low frequency
                high = float(getattr(feats, 'baxter_king_high_freq', high))  # Legacy high frequency
            except Exception:
                pass
        return {'k': k, 'low': low, 'high': high, 'source_columns': sources, 'apply_to_all': apply_all, 'causal': causal, 'fill_borders': fill_borders}

    def _eligible_all_numeric(self, df, exclude: List[str]) -> List[str]:
        """Return all numeric columns suitable for BK, excluding prefixes/names.

        - Uses DataFrame dtypes (cuDF or Dask-cuDF meta) to pick floats only.
        - Excludes targets/deny/metrics prefixes and already-derived bk_filter_*.
        """
        try:
            # Use meta dtypes for Dask-cuDF to avoid compute
            dtypes = getattr(getattr(df, '_meta', None), 'dtypes', None) or getattr(df, 'dtypes', None)
        except Exception:
            dtypes = None
        cols = list(map(str, getattr(df, 'columns', [])))  # Get all column names
        numeric = []
        if dtypes is not None:
            try:
                for c in cols:  # Check each column's data type
                    dt = dtypes.get(c, None) if hasattr(dtypes, 'get') else None
                    sdt = str(dt).lower() if dt is not None else ''
                    if sdt.startswith('float') or sdt in ('f4', 'f8'):  # Check if float type
                        numeric.append(c)  # Add to numeric columns
            except Exception:
                # Fallback: include all, will cast later
                numeric = cols[:]
        else:
            numeric = cols[:]  # Include all columns if dtypes unavailable

        # Build exclusion predicates from settings
        feats = getattr(self.settings, 'features', None)
        deny_exact = set((getattr(feats, 'feature_denylist', []) or [])) if feats else set()  # Exact deny list
        deny_prefixes = list((getattr(feats, 'feature_deny_prefixes', []) or [])) if feats else []  # Deny prefixes
        dataset_target_columns = list((getattr(feats, 'dataset_target_columns', []) or [])) if feats else []  # Dataset targets
        dataset_target_prefixes = list((getattr(feats, 'dataset_target_prefixes', []) or [])) if feats else []  # Dataset target prefixes
        metrics_prefixes = list((getattr(feats, 'metrics_prefixes', []) or [])) if feats else []  # Metrics prefixes
        sel_target = str(getattr(feats, 'selection_target_column', '')) if feats else ''  # Selection target
        sel_targets = list((getattr(feats, 'selection_target_columns', []) or [])) if feats else []  # Selection targets

        def _excluded(name: str) -> bool:
            if name in exclude:  # Check explicit exclude list
                return True
            if name.startswith('bk_filter_'):  # Exclude already processed BK columns
                return True
            if name == sel_target or name in sel_targets:  # Exclude selection targets
                return True
            if name in deny_exact or name in dataset_target_columns:  # Exclude denied columns
                return True
            if any(name.startswith(p) for p in (deny_prefixes + dataset_target_prefixes + metrics_prefixes)):  # Exclude by prefix
                return True
            return False

        return [c for c in numeric if not _excluded(c)]  # Return eligible columns

    def _bk_sources_present(self, df_cols: List[str], configured: List[str]) -> List[str]:
        """Choose source columns: prefer configured; else heuristics.

        Heuristics when not configured:
        - Include 'y_close' if present
        - Include 'log_stabilized_y_close' if present
        - Include first column containing 'close' (to avoid explosion)
        """
        cols = list(map(str, df_cols))  # Convert to strings
        if configured:  # If columns are explicitly configured
            return [c for c in configured if c in cols]  # Return only those present in DataFrame
        picks: List[str] = []  # Selected columns
        if 'y_close' in cols:  # Prefer y_close if available
            picks.append('y_close')
        if 'log_stabilized_y_close' in cols:  # Prefer log-stabilized close if available
            picks.append('log_stabilized_y_close')
        # one generic close if none selected yet
        if not picks:  # If no preferred columns found
            for c in cols:  # Look for any column containing 'close'
                if 'close' in c.lower():
                    picks.append(c)  # Add first match
                    break  # Stop after first match to avoid explosion
        return picks

    # -------------------- cuDF path --------------------
    def process_cudf(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        self._log_info("🚀 Starting FeatureEngineering (cuDF)…")
        self._log_info(f"📊 Input DataFrame: {len(gdf)} rows, {len(gdf.columns)} columns")
        params = self._bk_params()  # Get BK filter parameters
        apply_all = bool(params.get('apply_to_all', False))  # Check if apply to all numeric columns
        if apply_all:
            src = self._eligible_all_numeric(gdf, exclude=[])  # Apply BK to all eligible numeric columns
        else:
            src = self._bk_sources_present(list(gdf.columns), params.get('source_columns', []))  # Apply BK only to configured columns
        if not src:  # If no source columns found
            self._log_warn("FeatureEngineering: no BK source columns detected; skipping BK")
            return gdf
        k = int(params['k'])  # Filter kernel size
        low = float(params['low'])  # Low frequency (long period)
        high = float(params['high'])  # High frequency (short period)
        causal = bool(params.get('causal', True))  # Temporal shift to avoid look-ahead bias
        fill_borders = bool(params.get('fill_borders', True))  # Fill NaN borders with appropriate values
        new_cols: List[str] = []  # Track newly created columns
        for i, col in enumerate(src, 1):  # Apply BK filter to each source column
            out = f"bk_filter_{col}"  # New column name with 'bk_filter_' prefix
            
            # Log progress every 10 columns or for the first/last column
            if i == 1 or i == len(src) or i % 10 == 0:
                self._log_info(f"🔄 Processing BK filter {i}/{len(src)}: {col} → {out}")
            
            try:
                # Apply BK filter on GPU
                gdf[out] = _apply_bk_filter_gpu_partition(gdf[col], k, low, high, causal, fill_borders)
                
                # Log completion every 10 columns or for the first/last column
                if i == 1 or i == len(src) or i % 10 == 0:
                    result_data = gdf[out]
                    self._log_info(f"✅ BK filter completed for {col}: dtype={result_data.dtype}, nulls={result_data.isna().sum()}")
                
                new_cols.append(out)  # Add to list of created columns
            except Exception as e:
                self._log_warn(f"❌ BK application failed for {col}", source=col, error=str(e), exc_info=True)

        # Record metrics/artifact
        try:
            metrics = {  # Create metrics dictionary
                'new_columns': new_cols,  # List of new columns created
                'new_columns_count': len(new_cols),  # Count of new columns
                'bk_k': k,  # BK kernel size
                'bk_low_period': low,  # BK low frequency period
                'bk_high_period': high,  # BK high frequency period
            }
            self._record_metrics('feature_engineering', metrics)  # Record metrics
            if bool(getattr(self.settings.features, 'debug_write_artifacts', True)):  # If artifact writing enabled
                from pathlib import Path
                import json as _json
                out_root = Path(getattr(self.settings.output, 'output_path', './output'))  # Get output path
                subdir = str(getattr(self.settings.features, 'artifacts_dir', 'artifacts'))  # Get artifacts subdirectory
                out_dir = out_root / subdir / 'signal'  # Create signal artifacts directory
                out_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
                summary_path = out_dir / 'summary_stage0_bk.json'  # Create summary file path
                with open(summary_path, 'w') as f:
                    _json.dump(metrics, f, indent=2)  # Write metrics to JSON file
                self._record_artifact('feature_engineering', str(summary_path), kind='json')  # Record artifact
        except Exception:
            pass  # Ignore errors in metrics/artifact recording

        self._log_info(f"🎯 FeatureEngineering complete (cuDF): {len(new_cols)} new columns created")
        self._log_info(f"📊 Output DataFrame: {len(gdf)} rows, {len(gdf.columns)} columns")
        return gdf

    # -------------------- Dask-cuDF path --------------------
    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        self._log_info("🚀 Starting FeatureEngineering (Dask)…")
        self._log_info(f"📊 Input DataFrame: {len(df)} rows, {len(df.columns)} columns, {df.npartitions} partitions")
        params = self._bk_params()  # Get BK filter parameters
        try:
            cols = list(df.columns)  # Get column names
        except Exception:
            cols = []
        apply_all = bool(params.get('apply_to_all', False))  # Check if apply to all numeric columns
        if apply_all:
            src = self._eligible_all_numeric(df, exclude=[])  # Apply BK to all eligible numeric columns
        else:
            src = self._bk_sources_present(cols, params.get('source_columns', []))  # Apply BK only to configured columns
        if not src:  # If no source columns found
            self._log_warn("FeatureEngineering: no BK source columns detected (Dask); skipping BK")
            return df
        k = int(params['k'])  # Filter kernel size
        low = float(params['low'])  # Low frequency (long period)
        high = float(params['high'])  # High frequency (short period)
        causal = bool(params.get('causal', True))  # Temporal shift to avoid look-ahead bias
        fill_borders = bool(params.get('fill_borders', True))  # Fill NaN borders with appropriate values
        new_cols: List[str] = []  # Track newly created columns
        # Batch persist to reduce graph size
        try:
            batch_size = int(getattr(self.settings.features, 'feature_engineering_batch_size', 16))
        except Exception:
            batch_size = 16
        since_last = 0
        for i, col in enumerate(src, 1):  # Apply BK filter to each source column
            out = f"bk_filter_{col}"  # New column name with 'bk_filter_' prefix
            
            # Log progress every 10 columns or for the first/last column
            if i == 1 or i == len(src) or i % 10 == 0:
                self._log_info(f"🔄 Processing BK filter (Dask) {i}/{len(src)}: {col} → {out}")
            
            try:
                # Apply BK filter on GPU using Dask
                df[out] = df[col].map_partitions(  # Apply BK filter to each partition
                    _apply_bk_filter_gpu_partition, k, low, high, bool(causal), bool(fill_borders), meta=(out, 'f4')  # GPU partition function with metadata
                )
                
                # Log completion every 10 columns or for the first/last column
                if i == 1 or i == len(src) or i % 10 == 0:
                    result_data = df[out]
                    self._log_info(f"✅ BK filter completed (Dask) for {col}: dtype={result_data.dtype}, npartitions={result_data.npartitions}")
                
                new_cols.append(out)  # Add to list of created columns
                since_last += 1
                if since_last >= max(1, batch_size):
                    try:
                        import dask as _dask
                        # In debug mode, avoid graph optimization to keep task names visible
                        debug_dash = bool(getattr(self.settings, 'development', {}).get('debug_dashboard', False)) if hasattr(self, 'settings') else False
                        if debug_dash:
                            df, = _dask.persist(df, optimize_graph=False)
                        else:
                            df = df.persist()
                        since_last = 0
                    except Exception:
                        pass
            except Exception as e:
                self._log_warn(f"❌ BK application failed (Dask) for {col}", source=col, error=str(e), exc_info=True)

        # Record metrics/artifact (Dask)
        try:
            metrics = {  # Create metrics dictionary
                'new_columns': new_cols,  # List of new columns created
                'new_columns_count': len(new_cols),  # Count of new columns
                'bk_k': k,  # BK kernel size
                'bk_low_period': low,  # BK low frequency period
                'bk_high_period': high,  # BK high frequency period
            }
            self._record_metrics('feature_engineering', metrics)  # Record metrics
            if bool(getattr(self.settings.features, 'debug_write_artifacts', True)):  # If artifact writing enabled
                from pathlib import Path
                import json as _json
                out_root = Path(getattr(self.settings.output, 'output_path', './output'))  # Get output path
                subdir = str(getattr(self.settings.features, 'artifacts_dir', 'artifacts'))  # Get artifacts subdirectory
                out_dir = out_root / subdir / 'signal'  # Create signal artifacts directory
                out_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
                summary_path = out_dir / 'summary_stage0_bk.json'  # Create summary file path
                with open(summary_path, 'w') as f:
                    _json.dump(metrics, f, indent=2)  # Write metrics to JSON file
                self._record_artifact('feature_engineering', str(summary_path), kind='json')  # Record artifact
        except Exception:
            pass  # Ignore errors in metrics/artifact recording
        # Final checkpoint to flush pending tasks
        try:
            import dask as _dask
            debug_dash = bool(getattr(self.settings, 'development', {}).get('debug_dashboard', False)) if hasattr(self, 'settings') else False
            if debug_dash:
                df, = _dask.persist(df, optimize_graph=False)
            else:
                df = df.persist()
        except Exception:
            pass
        self._log_info(f"🎯 FeatureEngineering complete (Dask): {len(new_cols)} new columns created")
        self._log_info(f"📊 Output DataFrame: {len(df)} rows, {len(df.columns)} columns, {df.npartitions} partitions")
        return df
