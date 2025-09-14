"""
Statistical Tests - ADF Stage (Dask)

Runs rolling ADF tests on frac_diff* features using the existing ADFTests helper
from the StatisticalTests controller. Designed to be invoked from orchestration
with clear start/end logs and optional persist/wait after scheduling.
"""

from typing import Optional
import re

import dask_cudf


def run(stats_engine, ddf: dask_cudf.DataFrame, window: int = 252, min_periods: int = 200) -> dask_cudf.DataFrame:
    """Apply ADF rolling tests on frac_diff* features.

    Args:
        stats_engine: Instance of StatisticalTests controller (already initialized)
        ddf: Input Dask-cuDF DataFrame
        window: Rolling window size
        min_periods: Minimum periods for ADF window

    Returns:
        dask_cudf.DataFrame with ADF feature columns added
    """
    # Discover frac_diff columns
    try:
        cols = list(ddf.columns)
    except Exception:
        cols = []
    adf_cols = [c for c in cols if "frac_diff" in str(c)]

    if not adf_cols:
        stats_engine._log_info("ADF: No frac_diff features found to process")
        return ddf

    stats_engine._log_info(f"ADF: Found {len(adf_cols)} frac_diff features to process", total=len(adf_cols))

    processed = 0
    for col in adf_cols:
        try:
            safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col))
        except Exception:
            safe = str(col)
        out = f"adf_stat_{safe}"

        try:
            if out in ddf.columns:
                continue
        except Exception:
            pass

        ddf[out] = ddf[col].map_partitions(
            stats_engine.adf_tests._apply_adf_rolling,  # reuse existing implementation
            int(window),
            int(min_periods),
            meta=(out, "f8"),
        )
        processed += 1
        if processed % 10 == 0:
            stats_engine._log_info(f"ADF: Processed {processed}/{len(adf_cols)} features...")

    stats_engine._log_info(f"ADF: Scheduling complete for {processed} features")
    return ddf
