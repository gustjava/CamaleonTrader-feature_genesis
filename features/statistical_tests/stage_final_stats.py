"""
Statistical Tests - Final Stats Stage (Dask)

Applies comprehensive statistical analysis on each partition using the existing
StatisticalAnalysis helper. Designed to be followed by a persist/wait checkpoint.
"""

import dask_cudf


def run(stats_engine, ddf: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
    return ddf.map_partitions(stats_engine.statistical_analysis.apply_comprehensive_statistical_analysis)
