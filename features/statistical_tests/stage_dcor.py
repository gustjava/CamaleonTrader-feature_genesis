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
    # Sample tail, fallback to head - Fixed 100k sample
    sample_n = 100000
    sample = stats_engine._sample_tail_across_partitions(ddf, sample_n, max_parts=16)
    if sample is None or len(sample) == 0 or target not in sample.columns:
        stats_engine._log_warn("dCor tail sample empty or missing target; trying head sample")
        sample = stats_engine._sample_head_across_partitions(ddf, sample_n, max_parts=16)
        if sample is None or len(sample) == 0 or target not in sample.columns:
            stats_engine._log_warn("dCor sampling failed; skipping dCor stage")
            return ddf

    # Prepare target on GPU and filter invalid rows once
    try:
        y_series = sample[target].astype('f4')
        y_gpu_all = y_series.to_cupy()
        mask_y = cp.isfinite(y_gpu_all)
        valid_y = int(mask_y.sum().item())
        if valid_y < 50:
            stats_engine._log_warn("dCor: insufficient finite target observations; skipping stage", valid_rows=valid_y)
            return ddf
        if valid_y != int(y_gpu_all.size):
            mask_host = cp.asnumpy(mask_y)
            mask_series = cudf.Series(mask_host)
            sample = sample[mask_series]
            y_gpu_all = y_gpu_all[mask_y]
            stats_engine._log_info("dCor target filtered for finite rows", original_rows=len(mask_host), kept_rows=valid_y)
        try:
            import os as _os
            dev = int(cp.cuda.runtime.getDevice())
            vis = _os.environ.get('CUDA_VISIBLE_DEVICES', '')
            stats_engine._log_info("dCor GPU context", gpu_device=dev, visible_devices=vis, sample_rows=int(y_gpu_all.size))
        except Exception:
            stats_engine._log_info("dCor GPU context", sample_rows=int(y_gpu_all.size))
    except Exception as e:
        stats_engine._log_warn("dCor: failed to prepare target on GPU", error=str(e))
        return ddf

    y_gpu_f32 = y_gpu_all.astype(cp.float32, copy=False)
    y_gpu_f64 = y_gpu_all.astype(cp.float64, copy=False)

    # Prefilter via vectorized Pearson correlation
    dcor_map: Dict[str, float] = {}
    try:
        pre_k = int(min(max(10, getattr(stats_engine, 'dcor_top_k', 50) * 3), 200))
    except Exception:
        pre_k = 150

    pref_candidates = candidates
    vectorized_prefilter_ok = False
    try:
        if candidates:
            cols_gpu = sample[candidates].astype('f4').to_cupy()
            mask_x = cp.isfinite(cols_gpu)
            valid_counts = mask_x.sum(axis=0)
            # Require minimum valid observations per column
            min_valid = 50
            valid_mask = valid_counts >= min_valid
            if not bool(cp.any(valid_mask)):
                raise ValueError("no candidates meet minimum valid count")

            # Center target once
            y_mean = cp.mean(y_gpu_f32)
            y_centered = y_gpu_f32 - y_mean

            # Center columns with NaN handling
            sum_x = cp.sum(cp.where(mask_x, cols_gpu, 0.0), axis=0)
            counts = cp.maximum(valid_counts, 1)
            col_means = cp.divide(sum_x, counts, out=cp.zeros_like(sum_x), where=counts > 0)
            X_centered = cp.where(mask_x, cols_gpu - col_means, 0.0)

            # Numerator and denominators with per-column masks
            numerator = cp.sum((y_centered[:, None] * X_centered), axis=0)
            y_energy = cp.sum((y_centered[:, None] ** 2) * mask_x, axis=0)
            x_energy = cp.sum(X_centered ** 2, axis=0)
            denom = cp.sqrt(y_energy * x_energy)
            corr = cp.divide(cp.abs(numerator), denom, out=cp.zeros_like(numerator), where=denom > 0)

            corr_np = cp.asnumpy(corr)
            counts_np = cp.asnumpy(valid_counts)
            names_np = np.asarray(candidates)

            valid_idx = (counts_np >= min_valid) & np.isfinite(corr_np)
            if not valid_idx.any():
                raise ValueError("no candidates retained after vectorized prefilter")

            names_np = names_np[valid_idx]
            corr_np = corr_np[valid_idx]

            order = np.argsort(-corr_np)
            top_indices = order[:pre_k]
            pref_candidates = names_np[top_indices].tolist()
            vectorized_prefilter_ok = True
            stats_engine._log_info(
                "dCor vectorized Pearson prefilter applied",
                total=len(candidates),
                kept=len(pref_candidates),
                pre_k=pre_k,
            )
    except Exception as vec_err:
        stats_engine._log_warn("dCor vectorized prefilter failed; falling back to per-column", error=str(vec_err))

    if not vectorized_prefilter_ok:
        try:
            scores = []
            yf = y_gpu_f32
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
    
    # PARALLEL dCor computation using feature chunks
    chunk_processing_success = False
    if pref_candidates:
        try:
            chunk_size = int(getattr(stats_engine, 'dcor_feature_chunk_size', 16))
            chunk_size = max(1, min(chunk_size, len(pref_candidates)))
            gpu_batch_size = int(getattr(stats_engine, 'dcor_gpu_batch_size', 32))
            gpu_batch_size = max(1, gpu_batch_size)

            X_pref_gpu = sample[pref_candidates].astype('f8').to_cupy()
            y_gpu_dcor = y_gpu_f64

            total_chunks = (len(pref_candidates) + chunk_size - 1) // chunk_size
            stats_engine._log_info(
                "dCor chunked GPU processing start",
                total_features=len(pref_candidates),
                chunk_size=chunk_size,
                gpu_batch=gpu_batch_size,
                chunks=total_chunks,
            )

            for chunk_idx, chunk_start in enumerate(range(0, len(pref_candidates), chunk_size), start=1):
                chunk_end = min(chunk_start + chunk_size, len(pref_candidates))
                chunk_names = pref_candidates[chunk_start:chunk_end]
                chunk_pairs = []
                for offset, name in enumerate(chunk_names):
                    col_idx = chunk_start + offset
                    try:
                        x_vec = cp.ascontiguousarray(X_pref_gpu[:, col_idx])
                        chunk_pairs.append((x_vec, y_gpu_dcor))
                    except Exception:
                        chunk_pairs.append((None, None))
                        dcor_map[name] = float('nan')

                effective_pairs = [(x, y_gpu_dcor) for (x, _) in chunk_pairs if x is not None]
                if effective_pairs:
                    dcor_values = stats_engine.distance_correlation.compute_distance_correlation_parallel_batch(
                        effective_pairs,
                        max_samples=int(stats_engine.dcor_max_samples),
                        batch_size=min(len(effective_pairs), gpu_batch_size)
                    )
                else:
                    dcor_values = []

                value_iter = iter(dcor_values)
                for name, pair in zip(chunk_names, chunk_pairs):
                    if pair[0] is None:
                        continue
                    try:
                        dcor_map[name] = float(next(value_iter))
                    except StopIteration:
                        dcor_map[name] = float('nan')

                stats_engine._log_info(
                    "dCor chunk processed",
                    chunk_index=chunk_idx,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    processed=len(chunk_names),
                )

            chunk_processing_success = True

        except Exception as chunk_err:
            stats_engine._log_warn("dCor chunked processing failed; using fallback", error=str(chunk_err))

    if not chunk_processing_success:
        try:
            data_pairs = []
            valid_candidates = []

            for c in pref_candidates:
                if c not in sample.columns:
                    dcor_map[c] = float('nan')
                    continue
                try:
                    x = sample[c].astype('f8').to_cupy()
                    data_pairs.append((x, y_gpu_f64))
                    valid_candidates.append(c)
                except Exception:
                    dcor_map[c] = float('nan')

            if data_pairs and valid_candidates:
                batch_size = min(8, len(data_pairs))
                stats_engine._log_info("dCor parallel processing", total_features=len(valid_candidates), batch_size=batch_size)

                try:
                    dcor_values = stats_engine.distance_correlation.compute_distance_correlation_parallel_batch(
                        data_pairs,
                        max_samples=int(stats_engine.dcor_max_samples),
                        batch_size=batch_size
                    )

                    for i, c in enumerate(valid_candidates):
                        if i < len(dcor_values):
                            dcor_map[c] = dcor_values[i]
                        else:
                            dcor_map[c] = float('nan')

                    stats_engine._log_info("dCor parallel batch processing completed", processed=len(valid_candidates), total=total)

                except Exception as e:
                    stats_engine._log_warn("dCor parallel batch processing failed, falling back to sequential", error=str(e))
                    for i, (x, _) in enumerate(data_pairs):
                        c = valid_candidates[i]
                        try:
                            val = stats_engine.distance_correlation._distance_correlation_gpu(
                                x, y_gpu_f64, tile=int(stats_engine.dcor_tile_size), max_n=int(stats_engine.dcor_max_samples)
                            )
                            dcor_map[c] = float(val)
                        except Exception:
                            dcor_map[c] = float('nan')

                        if total > 0:
                            step = max(1, min(25, total // 10 or 1))
                            if (i + 1 == total) or ((i + 1) // step) > last_log:
                                last_log = (i + 1) // step
                                stats_engine._log_info("dCor progress", processed=i + 1, total=total)
        except Exception as e:
            stats_engine._log_warn("dCor batch processing failed, using sequential fallback", error=str(e))
            for i, c in enumerate(pref_candidates, start=1):
                if c not in sample.columns:
                    dcor_map[c] = float('nan')
                    continue
                try:
                    x = sample[c].astype('f8').to_cupy()
                    val = stats_engine.distance_correlation._distance_correlation_gpu(
                        x, y_gpu_f64, tile=int(stats_engine.dcor_tile_size), max_n=int(stats_engine.dcor_max_samples)
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
