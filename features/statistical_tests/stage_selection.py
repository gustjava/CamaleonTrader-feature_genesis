"""
Statistical Tests - Selection Stage (Dask)

Runs the feature selection pipeline using the controller's FeatureSelection
implementation and previously computed dCor scores stored in memory.
"""

from typing import Dict, Any, List
import dask_cudf


def run(stats_engine, ddf: dask_cudf.DataFrame, target: str) -> Dict[str, Any]:
    # Build candidates via controller logic to ensure consistent gating
    candidates: List[str] = stats_engine._find_candidate_features(ddf)
    # Pull dCor scores from memory; fallback handled inside selection pipeline
    dcor_scores = {}
    try:
        dcor_scores = dict(getattr(stats_engine, '_last_dcor_scores', {}) or {})
    except Exception:
        dcor_scores = {}
    # Use the controller's pipeline method (works off a sample internally)
    return stats_engine._apply_feature_selection_pipeline(ddf, target)
