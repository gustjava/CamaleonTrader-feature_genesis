"""
Statistical Tests - Selection Stage 2B: MI (Dask)

Performs MI-based redundancy reduction or clustering among VIF-selected features.
Logs explicit inputs/outputs and returns a dict with stage results for orchestration.
"""

from typing import Dict, Any, List, Optional
import dask_cudf


def run(stats_engine, ddf: dask_cudf.DataFrame, target: str, vif_selected: Optional[List[str]] = None) -> Dict[str, Any]:
    # Input set comes from VIF stage
    if vif_selected is None:
        try:
            vif_selected = list(getattr(stats_engine, '_last_vif_selected', []) or [])
        except Exception:
            vif_selected = []
    stats_engine._log_info("[MI] Stage start", input_count=len(vif_selected))
    if not vif_selected:
        stats_engine._log_warn("[MI] No VIF-selected features available; skipping MI stage")
        return {
            'stage': 'mi',
            'stage2_mi_selected': [],
            'stage2_vif_selected': [],
        }

    # Build sample
    sample_df = stats_engine._sample_head_across_partitions(ddf, stats_engine.selection_max_rows, max_parts=16)

    # Leakage check
    forbidden = [c for c in vif_selected if (c.startswith(('y_ret_fwd_', 'dcor_', 'adf_', 'stage1_', 'cpcv_')))]
    if forbidden:
        stats_engine._log_error("[MI] DATA LEAKAGE DETECTED in MI input", count=len(forbidden), examples=forbidden[:10])
        vif_selected = [c for c in vif_selected if c not in forbidden]
    else:
        stats_engine._log_info(f"[MI] Leakage check passed: {len(vif_selected)} candidates")

    # Pull dCor scores from previous stage
    try:
        dcor_scores = dict(getattr(stats_engine, '_last_dcor_scores', {}) or {})
    except Exception:
        dcor_scores = {}

    # GPU MI redundancy only - no CPU fallback
    try:
        # Get configurable MI parameters
        mi_bins = getattr(stats_engine, 'mi_bins', 64)
        mi_chunk_size = getattr(stats_engine, 'mi_chunk_size', 64)
        mi_min_samples = getattr(stats_engine, 'mi_min_samples', 10)
        
        stats_engine._log_info("[MI] Starting GPU MI redundancy", 
                              vif_features=len(vif_selected),
                              mi_bins=mi_bins,
                              mi_chunk_size=mi_chunk_size,
                              mi_min_samples=mi_min_samples,
                              mi_threshold=stats_engine.mi_threshold)
        mi_selected = stats_engine.feature_selection._compute_mi_redundancy_gpu(
            sample_df, vif_selected, dcor_scores, 
            float(stats_engine.mi_threshold),
            bins=mi_bins,
            chunk=mi_chunk_size,
            min_samples=mi_min_samples
        )
        stats_engine._log_info("[MI] GPU MI redundancy completed", selected_features=len(mi_selected))
    except Exception as e:
        stats_engine._log_error("[MI] GPU MI redundancy failed; passing through VIF set", error=str(e))
        mi_selected = list(vif_selected)

    stats_engine._log_info(
        "[MI] Stage end",
        input_count=len(vif_selected), output_count=len(mi_selected), removed=len(vif_selected) - len(mi_selected),
        output_preview=mi_selected[:15]
    )

    # Persist results
    try:
        stats_engine._last_mi_selected = list(mi_selected)
    except Exception:
        pass

    return {
        'stage': 'mi',
        'stage2_vif_selected': vif_selected,
        'stage2_mi_selected': mi_selected,
    }
