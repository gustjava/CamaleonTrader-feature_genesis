"""
Statistical Tests - Selection Stage 3: Embedded (CatBoost) (Dask)

Applies embedded model-based selection (CatBoost) on MI-selected features.
Logs explicit inputs/outputs and returns a dict with stage results for orchestration.
"""

from typing import Dict, Any, List, Optional
import dask_cudf


def run(stats_engine, ddf: dask_cudf.DataFrame, target: str, mi_selected: Optional[List[str]] = None) -> Dict[str, Any]:
    # Input comes from MI stage
    if mi_selected is None:
        try:
            mi_selected = list(getattr(stats_engine, '_last_mi_selected', []) or [])
        except Exception:
            mi_selected = []
    stats_engine._log_info("[Embedded] Stage start (CatBoost)", input_count=len(mi_selected))
    if not mi_selected:
        stats_engine._log_warn("[Embedded] No MI-selected features; skipping embedded selection")
        return {
            'stage': 'embedded',
            'stage3_final_selected': [],
            'importances': {},
            'selection_stats': {'backend_used': 'none'},
        }

    # Build sample
    sample_df = stats_engine._sample_head_across_partitions(ddf, stats_engine.selection_max_rows, max_parts=16)

    # Run embedded selection via FeatureSelection
    try:
        final_selected, importances, backend = stats_engine.feature_selection._stage3_selectfrommodel(
            sample_df, sample_df[target], mi_selected
        )
    except Exception as e:
        stats_engine._log_error("[Embedded] Selection failed; passing through MI set", error=str(e))
        final_selected = list(mi_selected)
        importances = {f: 1.0 for f in mi_selected}
        backend = 'error'

    stats_engine._log_info(
        "[Embedded] Stage end",
        input_count=len(mi_selected), output_count=len(final_selected), backend=backend, output_preview=final_selected[:15]
    )

    # Persist results
    try:
        stats_engine._last_embedded_selected = list(final_selected)
        stats_engine._last_importances = dict(importances)
    except Exception:
        pass

    return {
        'stage': 'embedded',
        'stage2_mi_selected': mi_selected,
        'stage3_final_selected': final_selected,
        'importances': importances,
        'selection_stats': {'backend_used': backend},
    }
