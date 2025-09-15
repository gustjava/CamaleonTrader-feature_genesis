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
        stats_engine._log_info(f"[Embedded] Dataset configuration: stage3_catboost_use_full_dataset={use_full}")
    except Exception:
        use_full = False
        stats_engine._log_warn("[Embedded] Could not read stage3_catboost_use_full_dataset config; defaulting to False")
    
    if use_full:
        # Compute only the required columns to reduce memory pressure
        cols = [c for c in mi_selected if c in ddf.columns]
        if target not in cols:
            cols = cols + [target]
        
        # Log dataset size before compute
        try:
            total_rows = len(ddf)
            stats_engine._log_info(f"[Embedded] 🎯 FULL DATASET MODE: Computing {total_rows:,} rows for CatBoost")
        except Exception:
            stats_engine._log_info("[Embedded] 🎯 FULL DATASET MODE: Computing complete dataset for CatBoost")
        
        try:
            sample_df = ddf[cols].compute()
            stats_engine._log_info(f"[Embedded] ✅ FULL DATASET LOADED: {len(sample_df):,} rows × {len(sample_df.columns)} columns")
            stats_engine._log_info(f"[Embedded] 📊 Dataset size: {len(sample_df):,} rows (FULL DATASET - NO SAMPLING)")
        except Exception as e:
            stats_engine._log_warn(f"[Embedded] ❌ Full dataset compute failed; falling back to sampling", error=str(e))
            sample_df = stats_engine._sample_head_across_partitions(ddf, stats_engine.selection_max_rows, max_parts=16)
            stats_engine._log_warn(f"[Embedded] ⚠️  FALLBACK TO SAMPLING: {len(sample_df):,} rows (SAMPLED - NOT FULL DATASET)")
    else:
        sample_df = stats_engine._sample_head_across_partitions(ddf, stats_engine.selection_max_rows, max_parts=16)
        stats_engine._log_info(f"[Embedded] 📊 SAMPLING MODE: {len(sample_df):,} rows (SAMPLED - NOT FULL DATASET)")
        stats_engine._log_info(f"[Embedded] ⚠️  WARNING: Using sampled data instead of full dataset")

    # Run embedded selection via FeatureSelection
    detailed_metrics = {}
    model_score = 0.0
    try:
        stats_engine._log_info("[Embedded] Starting CatBoost feature selection", 
                              input_features=len(mi_selected), 
                              dataset_rows=len(sample_df),
                              target=target)
        
        result = stats_engine.feature_selection._stage3_selectfrommodel(
            sample_df, sample_df[target], mi_selected
        )
                # Unpack the selection result (handles various tuple lengths for compatibility)
        if isinstance(result, (list, tuple)):
            if len(result) == 5:
                final_selected, importances, backend, model_score, detailed_metrics = result
            elif len(result) == 4:
                final_selected, importances, backend, model_score = result
                detailed_metrics = {}
            elif len(result) == 3:
                final_selected, importances, backend = result
                model_score = 0.0
                detailed_metrics = {}
            else:
                # Unexpected shape
                final_selected, importances, backend = list(mi_selected), {f: 1.0 for f in mi_selected}, 'unknown'
                model_score, detailed_metrics = 0.0, {}
        else:
            # Unexpected type
            final_selected, importances, backend = list(mi_selected), {f: 1.0 for f in mi_selected}, 'unknown'
            model_score, detailed_metrics = 0.0, {}
        
        # Log detailed results
        stats_engine._log_info("[Embedded] CatBoost selection completed",
                  backend=backend,
                  model_score=model_score,
                  input_count=len(mi_selected), 
                  output_count=len(final_selected),
                  threshold_used=stats_engine.feature_selection._parse_importance_threshold('median', list(importances.values())))

        # Emit CatBoost detailed metrics if available (flatten small dictionaries for readability)
        if isinstance(detailed_metrics, dict) and detailed_metrics:
            try:
                # Limit payload size to avoid flooding logs
                light_metrics = {}
                for k, v in list(detailed_metrics.items()):
                    if isinstance(v, dict):
                        # keep a shallow copy; truncate long sequences
                        trunc = {}
                        for mk, mv in list(v.items()):
                            if isinstance(mv, (list, tuple)):
                                trunc[mk] = mv[-1] if len(mv) > 0 else None
                            else:
                                trunc[mk] = mv
                        light_metrics[k] = trunc
                    else:
                        light_metrics[k] = v
                stats_engine._log_info("[Embedded] CatBoost detailed metrics", **light_metrics)
            except Exception:
                # Best-effort: ignore logging issues
                pass
        
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
        import traceback as _tb
        stats_engine._log_error("[Embedded] Selection failed; passing through MI set", error=str(e), traceback=_tb.format_exc())
        final_selected = list(mi_selected)
        importances = {f: 1.0 for f in mi_selected}
        backend = 'error'
        model_score = 0.0
        detailed_metrics = {'error': str(e), 'traceback': _tb.format_exc()}

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
