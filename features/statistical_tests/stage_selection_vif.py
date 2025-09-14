"""
Statistical Tests - Selection Stage 2A: VIF (Dask)

Performs VIF-based multicollinearity filtering on validated candidates.
Logs explicit inputs/outputs and returns a dict with stage results for orchestration.
"""

from typing import Dict, Any, List, Optional
import dask_cudf


def run(stats_engine, ddf: dask_cudf.DataFrame, target: str, candidates: Optional[List[str]] = None) -> Dict[str, Any]:
    # 1) Build/receive candidates via controller to ensure consistent gating
    if candidates is None:
        candidates = stats_engine._find_candidate_features(ddf)
    stats_engine._log_info("[VIF] Stage start", total_candidates=len(candidates))

    # 2) Sample head for numeric validation and GPU matrix creation
    sample_df = stats_engine._sample_head_across_partitions(ddf, stats_engine.selection_max_rows, max_parts=16)

    # 3) Sanitize to numeric candidates
    valid_candidates: List[str] = []
    invalid_candidates: List[str] = []
    missing_columns: List[str] = []
    for c in candidates:
        try:
            if c not in sample_df.columns:
                missing_columns.append(c)
                invalid_candidates.append(c)
                continue
            _ = sample_df[c].astype('f4')
            valid_candidates.append(c)
        except Exception:
            invalid_candidates.append(c)

    stats_engine._log_info(
        "[VIF] Sanitization",
        original=len(candidates), valid=len(valid_candidates), missing=len(missing_columns), invalid=len(invalid_candidates)
    )
    if not valid_candidates:
        stats_engine._log_warn("[VIF] No valid numeric candidates; returning empty selection")
        return {
            'stage': 'vif',
            'stage2_vif_selected': [],
            'stage2_vif_input': candidates,
            'stage2_vif_usable': [],
        }

    # 3.5) Apply dCor-based filtering (Stage 1 filters)
    try:
        # Get dCor scores from memory
        dcor_scores = {}
        try:
            mem_scores = getattr(stats_engine, '_last_dcor_scores', {}) or {}
            if mem_scores:
                for col in valid_candidates:
                    dcor_scores[col] = float(mem_scores.get(col, 0.0))
                stats_engine._log_info(f"[VIF] Retrieved {len(dcor_scores)} dCor scores from memory")
            else:
                stats_engine._log_warn("[VIF] No dCor scores found in memory")
        except Exception as e:
            stats_engine._log_warn(f"[VIF] Error getting dCor scores: {e}")
        
        # Apply dCor filters if scores are available
        if dcor_scores:
            original_count = len(valid_candidates)
            filtered_candidates = []
            
            # Apply dcor_min_threshold filter
            threshold_filtered = []
            for candidate in valid_candidates:
                score = dcor_scores.get(candidate, 0.0)
                if score >= stats_engine.dcor_min_threshold:
                    threshold_filtered.append(candidate)
                else:
                    stats_engine._log_debug(f"[VIF] dCor filter: {candidate} removed (score={score:.4f} < threshold={stats_engine.dcor_min_threshold})")
            
            threshold_count = len(threshold_filtered)
            stats_engine._log_info(f"[VIF] dCor threshold filter: {original_count} → {threshold_count} candidates (threshold={stats_engine.dcor_min_threshold})")
            
            # Apply stage1_top_n filter (if enabled)
            if stats_engine.stage1_top_n > 0 and threshold_count > stats_engine.stage1_top_n:
                # Sort by dCor score (descending) and take top N
                scored_candidates = [(candidate, dcor_scores.get(candidate, 0.0)) for candidate in threshold_filtered]
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                top_candidates = [candidate for candidate, score in scored_candidates[:stats_engine.stage1_top_n]]
                filtered_candidates = top_candidates
                
                stats_engine._log_info(f"[VIF] dCor top_n filter: {threshold_count} → {len(filtered_candidates)} candidates (top_n={stats_engine.stage1_top_n})")
                
                # Log the top candidates for visibility
                top_scores = [(candidate, score) for candidate, score in scored_candidates[:min(10, len(scored_candidates))]]
                stats_engine._log_info(f"[VIF] dCor top candidates: {[(c, f'{s:.4f}') for c, s in top_scores]}")
            else:
                filtered_candidates = threshold_filtered
                if stats_engine.stage1_top_n > 0:
                    stats_engine._log_info(f"[VIF] dCor top_n filter: Not applied (candidates={threshold_count} <= top_n={stats_engine.stage1_top_n})")
                else:
                    stats_engine._log_info(f"[VIF] dCor top_n filter: Disabled (top_n={stats_engine.stage1_top_n})")
            
            final_count = len(filtered_candidates)
            stats_engine._log_info(f"[VIF] dCor filtering complete: {original_count} → {final_count} candidates")
            
            # Update valid_candidates with filtered results
            valid_candidates = filtered_candidates
        else:
            stats_engine._log_info("[VIF] No dCor scores available, skipping dCor filtering")
            
    except Exception as e:
        stats_engine._log_warn(f"[VIF] Error in dCor filtering: {e}")
        # Continue with original valid_candidates on error

    # 4) Leakage check
    forbidden = [c for c in valid_candidates if (c.startswith(('y_ret_fwd_', 'dcor_', 'adf_', 'stage1_', 'cpcv_')))]
    if forbidden:
        stats_engine._log_error("[VIF] DATA LEAKAGE DETECTED in candidates", count=len(forbidden), examples=forbidden[:10])
        # Remove forbidden features even if present
        valid_candidates = [c for c in valid_candidates if c not in forbidden]
    else:
        stats_engine._log_info(f"[VIF] Leakage check passed: {len(valid_candidates)} candidates")

    # 4.1) Collect full dCor scores for all candidates entering VIF and log fully
    try:
        dcor_scores = dict(getattr(stats_engine, '_last_dcor_scores', {}) or {})
    except Exception:
        dcor_scores = {}
    try:
        pairs = [f"{f}:{float(dcor_scores.get(f, 0.0)):.6f}" for f in valid_candidates]
        # Inline the full list in the message so it always appears in logs
        stats_engine._log_info(f"[VIF] Input features with dCor (full) [{len(pairs)}]: " + ", ".join(pairs))
        try:
            stats_engine._last_vif_input_dcor = {f: float(dcor_scores.get(f, 0.0)) for f in valid_candidates}
        except Exception:
            pass
    except Exception:
        pass

    # 5) Build GPU matrix and run VIF
    X_matrix, used_cols = stats_engine.feature_selection._to_cupy_matrix(sample_df, valid_candidates)
    stats_engine._log_info("[VIF] Matrix ready", usable=len(used_cols))
    vif_selected = stats_engine.feature_selection._compute_vif_iterative_gpu(X_matrix, used_cols, stats_engine.vif_threshold)

    # 5.1) Log full dCor mapping for VIF-selected outputs
    try:
        sel_pairs = [f"{f}:{float(dcor_scores.get(f, 0.0)):.6f}" for f in vif_selected]
        stats_engine._log_info(f"[VIF] Selected features with dCor (full) [{len(sel_pairs)}]: " + ", ".join(sel_pairs))
        try:
            stats_engine._last_vif_selected_dcor = {f: float(dcor_scores.get(f, 0.0)) for f in vif_selected}
        except Exception:
            pass
    except Exception:
        pass

    stats_engine._log_info(
        "[VIF] Stage end",
        input_count=len(valid_candidates), output_count=len(vif_selected), removed=len(valid_candidates) - len(vif_selected),
        output_preview=vif_selected[:15]
    )

    # 6) Persist in-memory for next stage visibility
    try:
        stats_engine._last_vif_selected = list(vif_selected)
        stats_engine._last_vif_input = list(valid_candidates)
    except Exception:
        pass

    return {
        'stage': 'vif',
        'stage2_vif_selected': vif_selected,
        'stage2_vif_input': valid_candidates,
    'stage2_vif_usable': used_cols,
    'vif_input_with_dcor': getattr(stats_engine, '_last_vif_input_dcor', {f: float(dcor_scores.get(f, 0.0)) for f in valid_candidates}),
    'vif_selected_with_dcor': getattr(stats_engine, '_last_vif_selected_dcor', {f: float(dcor_scores.get(f, 0.0)) for f in vif_selected}),
    }
