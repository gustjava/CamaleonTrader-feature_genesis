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

    # Build sample or full dataset based on configuration
    try:
        use_full = bool(getattr(stats_engine, 'stage3_catboost_use_full_dataset', False))
    except Exception:
        use_full = False
    if use_full:
        # Compute only the required columns to reduce memory pressure
        cols = [c for c in mi_selected if c in ddf.columns]
        if target not in cols:
            cols = cols + [target]
        try:
            sample_df = ddf[cols].compute()
            stats_engine._log_info("[Embedded] Using full dataset for CatBoost selection", rows=len(sample_df), cols=len(sample_df.columns))
        except Exception as e:
            stats_engine._log_warn("[Embedded] Full dataset compute failed; falling back to sampling", error=str(e))
            sample_df = stats_engine._sample_head_across_partitions(ddf, stats_engine.selection_max_rows, max_parts=16)
    else:
        sample_df = stats_engine._sample_head_across_partitions(ddf, stats_engine.selection_max_rows, max_parts=16)

    # Run embedded selection via FeatureSelection
    try:
        stats_engine._log_info("[Embedded] Starting CatBoost feature selection", 
                              input_features=len(mi_selected), 
                              dataset_rows=len(sample_df),
                              target=target)
        
        result = stats_engine.feature_selection._stage3_selectfrommodel(
            sample_df, sample_df[target], mi_selected
        )
        if len(result) == 4:
            final_selected, importances, backend, model_score = result
            detailed_metrics = {}
        else:
            final_selected, importances, backend, model_score, detailed_metrics = result
        
        # Log detailed results
        stats_engine._log_info("[Embedded] CatBoost selection completed",
                              backend=backend,
                              model_score=model_score,
                              input_count=len(mi_selected), 
                              output_count=len(final_selected),
                              threshold_used=stats_engine.feature_selection._parse_importance_threshold('median', list(importances.values())))
        
        # Log all winning features with their importance scores
        if final_selected and importances:
            winning_features = [(f, importances.get(f, 0.0)) for f in final_selected]
            winning_features.sort(key=lambda x: x[1], reverse=True)  # Sort by importance
            stats_engine._log_info("[Embedded] All winning features with CatBoost importance scores:",
                                  winning_features=[f"{name}:{score:.6f}" for name, score in winning_features])
        
        # Log all features that were considered (for comparison)
        if importances:
            all_features = [(f, importances.get(f, 0.0)) for f in mi_selected if f in importances]
            all_features.sort(key=lambda x: x[1], reverse=True)
            stats_engine._log_info("[Embedded] All considered features with CatBoost importance scores:",
                                  all_features=[f"{name}:{score:.6f}" for name, score in all_features])
        
    except Exception as e:
        stats_engine._log_error("[Embedded] Selection failed; passing through MI set", error=str(e))
        final_selected = list(mi_selected)
        importances = {f: 1.0 for f in mi_selected}
        backend = 'error'
        model_score = 0.0

    stats_engine._log_info(
        "[Embedded] Stage end",
        input_count=len(mi_selected), output_count=len(final_selected), backend=backend, model_score=model_score
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
        'selection_stats': {'backend_used': backend, 'model_score': model_score},
        'detailed_metrics': detailed_metrics,
    }
