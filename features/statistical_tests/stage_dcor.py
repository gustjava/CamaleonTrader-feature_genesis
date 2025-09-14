"""
Statistical Tests - dCor Stage (Dask)

Builds a representative sample (tail/head), prefilters via Pearson correlation,
computes distance correlation scores for candidates against the target, and stores
scores in-memory for downstream selection. Avoids adding dcor_* columns.
"""

from typing import List, Dict
import numpy as np
import cupy as cp
import cudf
import dask_cudf


def run(stats_engine, ddf: dask_cudf.DataFrame, target: str, candidates: List[str]) -> dask_cudf.DataFrame:
    # Sample tail, fallback to head
    sample_n = int(max(1000, min(stats_engine.dcor_max_samples * 2, 200000)))
    sample = stats_engine._sample_tail_across_partitions(ddf, sample_n, max_parts=16)
    if sample is None or len(sample) == 0 or target not in sample.columns:
        stats_engine._log_warn("dCor tail sample empty or missing target; trying head sample")
        sample = stats_engine._sample_head_across_partitions(ddf, sample_n, max_parts=16)
        if sample is None or len(sample) == 0 or target not in sample.columns:
            stats_engine._log_warn("dCor sampling failed; skipping dCor stage")
            return ddf

    # Target to GPU
    try:
        y = sample[target].astype('f8').to_cupy()
        try:
            import os as _os
            dev = int(cp.cuda.runtime.getDevice())
            vis = _os.environ.get('CUDA_VISIBLE_DEVICES', '')
            stats_engine._log_info("dCor GPU context", gpu_device=dev, visible_devices=vis, sample_rows=len(sample))
        except Exception:
            pass
    except Exception as e:
        stats_engine._log_warn("dCor: failed to pull target to GPU", error=str(e))
        return ddf

    # Prefilter via Pearson corr
    dcor_map: Dict[str, float] = {}
    try:
        pre_k = int(min(max(10, getattr(stats_engine, 'dcor_top_k', 50) * 3), 200))
    except Exception:
        pre_k = 150

    pref_candidates = candidates
    try:
        scores = []
        yf = y.astype(cp.float32, copy=False)
        y_mask = cp.isfinite(yf)
        yv = yf[y_mask]
        if int(yv.size) >= 50:
            for c in candidates:
                if c not in sample.columns:
                    continue
                try:
                    xv = sample[c].astype('f4').to_cupy()
                    m = y_mask & cp.isfinite(xv)
                    if int(m.sum().item()) < 50:
                        continue
                    xa = xv[m]
                    ya = yf[m]
                    xm = xa.mean(); ym = ya.mean()
                    xs = xa - xm; ys = ya - ym
                    denom = cp.sqrt((xs * xs).sum() * (ys * ys).sum())
                    if float(denom) == 0.0:
                        continue
                    corr = float((xs * ys).sum() / denom)
                    scores.append((c, abs(corr)))
                except Exception:
                    continue
            if scores:
                scores.sort(key=lambda kv: kv[1], reverse=True)
                pref_candidates = [name for name, _ in scores[:pre_k]]
                stats_engine._log_info("dCor prefilter via Pearson corr applied", total=len(candidates), kept=len(pref_candidates), pre_k=pre_k)
    except Exception as _pf_err:
        stats_engine._log_warn("dCor prefilter failed; proceeding with full set", error=str(_pf_err))

    total = len(pref_candidates)
    last_log = -1
    for i, c in enumerate(pref_candidates, start=1):
        if c not in sample.columns:
            dcor_map[c] = float('nan')
            continue
        try:
            x = sample[c].astype('f8').to_cupy()
            val = stats_engine.distance_correlation._distance_correlation_gpu(
                x, y, tile=int(stats_engine.dcor_tile_size), max_n=int(stats_engine.dcor_max_samples)
            )
            dcor_map[c] = float(val)
        except Exception:
            dcor_map[c] = float('nan')
        try:
            if total > 0:
                step = max(1, min(25, total // 10 or 1))
                if (i == total) or (i // step) > last_log:
                    last_log = i // step
                    stats_engine._log_info("dCor progress", processed=i, total=total)
        except Exception:
            pass

    # Log topK
    try:
        items = [(k, v) for k, v in dcor_map.items() if np.isfinite(v)]
        items.sort(key=lambda kv: kv[1], reverse=True)
        show = items[:max(1, min(int(getattr(stats_engine, 'dcor_top_k', 20)), 10))]
        stats_engine._log_info("dCor top", top=[f"{name}:{val:.4f}" for name, val in show])
        # Persist a compact summary for orchestration-level visibility
        try:
            stats_engine._last_dcor_summary = {
                'candidates_total': len(candidates),
                'prefilter_kept': len(pref_candidates),
                'top': [f"{name}:{val:.4f}" for name, val in show],
                'scores': dict(dcor_map),
            }
        except Exception:
            stats_engine._last_dcor_summary = {
                'candidates_total': len(candidates),
                'prefilter_kept': len(pref_candidates),
                'top': [f"{name}:{val:.4f}" for name, val in show],
            }
    except Exception:
        pass

    # Store in-memory for selection
    try:
        stats_engine._last_dcor_scores = dict(dcor_map)
    except Exception:
        stats_engine._last_dcor_scores = {}

    return ddf
