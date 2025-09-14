import logging
from typing import Dict, Optional

import cudf
import cupy as cp  # noqa: F401  # Reserved for potential GPU ops/extensions
import emd

from utils.logging_utils import get_logger

logger = get_logger(__name__, "features.signal_processing")


def apply_emd_to_series(series: cudf.Series, max_imfs: int = 10) -> cudf.DataFrame:
    """
    Apply Empirical Mode Decomposition (EMD) to a 1D time series and return IMFs.

    Args:
        series: Input cuDF Series (e.g., close prices).
        max_imfs: Maximum number of Intrinsic Mode Functions (IMFs) to extract.

    Returns:
        cuDF DataFrame with IMF columns (imf_1..imf_k), indexed like the input series.
    """
    try:
        series_name = series.name or "series"
    except Exception:
        series_name = "series"

    logger.info(f"Aplicando EMD à série '{series_name}' para extrair até {max_imfs} IMFs.")

    # Move data to CPU NumPy for the EMD library
    try:
        series_cpu = series.to_numpy()
    except Exception as e:
        logger.error(f"Falha ao converter série para NumPy: {e}")
        raise

    # Run EMD sift to get IMFs (shape: [n_samples, n_imfs])
    imfs_cpu = emd.sift.sift(series_cpu, max_imfs=max_imfs)

    if imfs_cpu.ndim != 2 or imfs_cpu.shape[0] != len(series_cpu):
        logger.warning(
            f"Formato inesperado de IMFs: shape={getattr(imfs_cpu, 'shape', None)}, esperado: (n, k)"
        )

    # Build cuDF DataFrame from IMFs and align index
    imf_dict: Dict[str, object] = {
        f"imf_{i+1}": imfs_cpu[:, i] for i in range(imfs_cpu.shape[1])
    }
    imf_df = cudf.DataFrame(imf_dict)
    try:
        imf_df.index = series.index
    except Exception:
        # Fallback: reset to RangeIndex if direct index assignment not supported
        logger.debug("Não foi possível aplicar o índice original aos IMFs; usando RangeIndex")
        pass

    logger.info(f"EMD concluído. {imf_df.shape[1]} IMFs foram extraídos.")
    return imf_df


def apply_rolling_emd(series: cudf.Series, max_imfs: int = 10,
                      window_size: int = 500, step_size: int = 50,
                      embargo: int = 0) -> cudf.DataFrame:
    """
    Rolling EMD that emits IMFs only for out-of-sample portions to prevent leakage.

    Contract:
    - Fit EMD on past window [t-window_size, t) and assign IMFs for
      [t+embargo, t+step_size) positions only.
    - Returns cuDF DataFrame with columns imf_1..imf_k aligned to series index.
    """
    try:
        import pandas as pd
        # Move to pandas for emd, which expects numpy.
        s = series.to_pandas().astype(float)
        s = s.ffill().dropna()
        n = len(s)
        if n < max(window_size + step_size + embargo + 1, 200):
            # Not enough data
            return cudf.DataFrame({f"imf_{i+1}": np.nan for i in range(max_imfs)}, index=series.index)

        # Prepare outputs
        out_cols = {f"imf_{i+1}": pd.Series(np.nan, index=s.index, dtype='float64') for i in range(max_imfs)}

        # Iterate rolling anchors
        end = n
        start_anchor = window_size
        for anchor in range(start_anchor, end, step_size):
            win_start = anchor - window_size
            win_stop = anchor
            x = s.iloc[win_start:win_stop].to_numpy()
            try:
                imfs_cpu = emd.sift.sift(x, max_imfs=max_imfs)
                # Assign only to future slice
                assign_start = min(n, anchor + embargo)
                assign_stop = min(n, anchor + step_size)
                if assign_start < assign_stop:
                    # Simple strategy: assign the last sample IMF value from the fitted window
                    last_vals = imfs_cpu[-1, :]
                    idx_slice = s.index[assign_start:assign_stop]
                    for i in range(min(max_imfs, imfs_cpu.shape[1])):
                        out_cols[f"imf_{i+1}"].loc[idx_slice] = float(last_vals[i])
            except Exception as e:
                logger.debug(f"Rolling EMD window failed at anchor {anchor}: {e}")

        # Build cuDF and reindex to original series index
        out_pd = pd.DataFrame(out_cols)
        out_pd = out_pd.reindex(series.index)
        return cudf.from_pandas(out_pd)
    except Exception as e:
        logger.error(f"Error in apply_rolling_emd: {e}")
        return cudf.DataFrame({f"imf_{i+1}": np.nan for i in range(max_imfs)}, index=series.index)
