"""
Distance Correlation Module

Este módulo implementa a correlação de distância ao quadrado (R^2) entre variáveis,
incluindo:
- caminho rápido O(n log n) 1D em CPU,
- algoritmo em blocos (tiling) em GPU com centragem global (2 passes),
- processamento em lote (batch),
- janelas deslizantes (rolling),
- e teste de permutação para significância.

Todas as funções públicas retornam **R^2 ∈ [0, 1]**.
"""

import logging
import numpy as np
import cupy as cp
import cudf
from typing import List, Dict, Tuple
from .utils import _adaptive_tile
from utils.logging_utils import get_logger

logger = get_logger(__name__, "features.statistical_tests.distance_correlation")


class DistanceCorrelation:
    """Computa correlação de distância ao quadrado (R^2) entre variáveis."""

    def __init__(self, logger_instance=None, dcor_max_samples: int = 10000, dcor_tile_size: int = 2048):
        """Parâmetros:
        - dcor_max_samples: limite de amostras por par (controle de custo)
        - dcor_tile_size: tamanho de bloco (tiling) no caminho GPU
        """
        self.logger = logger_instance or logger
        self.dcor_max_samples = int(dcor_max_samples)
        self.dcor_tile_size = int(dcor_tile_size)

    # -------------------- Logging helpers --------------------

    def _log_info(self, message: str, **kwargs):
        if self.logger:
            self.logger.info(f"dCor: {message}", extra={k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                                                        for k, v in kwargs.items()})

    def _log_warn(self, message: str, **kwargs):
        if self.logger:
            self.logger.warning(f"dCor: {message}", extra={k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                                                           for k, v in kwargs.items()})

    def _log_error(self, message: str, **kwargs):
        if self.logger:
            self.logger.error(f"dCor: {message}", extra={k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                                                         for k, v in kwargs.items()})

    def _critical_error(self, message: str, **kwargs):
        self._log_error(message, **kwargs)
        raise RuntimeError(f"dCor Critical Error: {message}")

    # -------------------- Utils: NaN mask row-wise --------------------

    def _rowwise_valid_mask(self, x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        """Retorna máscara booleana 1-D (len n) selecionando linhas sem NaN em x ou y."""
        if x.ndim == 1 and y.ndim == 1:
            return ~(cp.isnan(x) | cp.isnan(y))
        if x.ndim == 1:
            x = x[:, None]
        if y.ndim == 1:
            y = y[:, None]
        return ~(cp.any(cp.isnan(x), axis=1) | cp.any(cp.isnan(y), axis=1))

    # -------------------- 1D CPU Fast (R^2) --------------------

    def _sum_abs_diffs(self, values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind='mergesort')
        sorted_vals = values[order]
        n = len(values)
        prefix = np.cumsum(sorted_vals, dtype=np.float64)
        total = prefix[-1]
        idx = np.arange(n, dtype=np.float64)
        left = idx * sorted_vals - np.concatenate(([0.0], prefix[:-1]))
        right = (total - prefix) - (n - 1 - idx) * sorted_vals
        sums_sorted = left + right
        sums = np.empty_like(sums_sorted)
        sums[order] = sums_sorted
        return sums

    def _pairwise_squared_diff_sum(self, values: np.ndarray) -> float:
        order = np.argsort(values, kind='mergesort')
        sorted_vals = values[order]
        n = len(values)
        idx = np.arange(n, dtype=np.float64)
        prefix = np.cumsum(sorted_vals, dtype=np.float64)
        prefix_sq = np.cumsum(sorted_vals * sorted_vals, dtype=np.float64)
        prefix_prev = np.concatenate(([0.0], prefix[:-1]))
        prefix_sq_prev = np.concatenate(([0.0], prefix_sq[:-1]))
        term = idx * sorted_vals * sorted_vals - 2.0 * sorted_vals * prefix_prev + prefix_sq_prev
        return term.sum()

    @staticmethod
    def _bit_query(tree: np.ndarray, idx: int) -> float:
        idx += 1
        res = 0.0
        while idx > 0:
            res += tree[idx]
            idx -= idx & -idx
        return res

    @staticmethod
    def _bit_update(tree: np.ndarray, idx: int, delta: float) -> None:
        size = tree.size
        idx += 1
        while idx < size:
            tree[idx] += delta
            idx += idx & -idx

    def _pairwise_abs_product_sum(self, x: np.ndarray, y: np.ndarray) -> float:
        order = np.argsort(x, kind='mergesort')
        x_sorted = x[order]
        y_sorted = y[order]
        _, ranks = np.unique(y_sorted, return_inverse=True)
        size = int(ranks.max()) + 1

        tree_count = np.zeros(size + 2, dtype=np.float64)
        tree_sum_y = np.zeros(size + 2, dtype=np.float64)
        tree_sum_x = np.zeros(size + 2, dtype=np.float64)
        tree_sum_xy = np.zeros(size + 2, dtype=np.float64)

        total_count = 0.0
        total_y = 0.0
        total_x = 0.0
        total_xy = 0.0
        S = 0.0

        for x_val, y_val, rank in zip(x_sorted, y_sorted, ranks):
            count_le = self._bit_query(tree_count, rank)
            sum_y_le = self._bit_query(tree_sum_y, rank)
            sum_x_le = self._bit_query(tree_sum_x, rank)
            sum_xy_le = self._bit_query(tree_sum_xy, rank)

            count_gt = total_count - count_le
            sum_y_gt = total_y - sum_y_le
            sum_x_gt = total_x - sum_x_le
            sum_xy_gt = total_xy - sum_xy_le

            term1 = x_val * (count_le * y_val - sum_y_le) - (y_val * sum_x_le - sum_xy_le)
            term2 = x_val * (sum_y_gt - count_gt * y_val) - (sum_xy_gt - y_val * sum_x_gt)
            S += term1 + term2

            self._bit_update(tree_count, rank, 1.0)
            self._bit_update(tree_sum_y, rank, y_val)
            self._bit_update(tree_sum_x, rank, x_val)
            self._bit_update(tree_sum_xy, rank, x_val * y_val)

            total_count += 1.0
            total_y += y_val
            total_x += x_val
            total_xy += x_val * y_val

        return S

    def _distance_correlation_1d_fast(self, x_np: np.ndarray, y_np: np.ndarray) -> float:
        """
        Algoritmo rápido 1D com complexidade O(n log n).
        Retorna R^2 em [0, 1].
        """
        n = x_np.size
        if n < 2:
            return float('nan')

        sum_abs_x = self._sum_abs_diffs(x_np)
        sum_abs_y = self._sum_abs_diffs(y_np)

        a_i = sum_abs_x / n
        b_i = sum_abs_y / n
        a_bar = sum_abs_x.sum() / (n * n)
        b_bar = sum_abs_y.sum() / (n * n)

        cross_sum = self._pairwise_abs_product_sum(x_np, y_np)  # soma i<j de |x_i-x_j||y_i-y_j|
        m_xy = (2.0 * cross_sum) / (n * n)

        dcov2 = m_xy + a_bar * b_bar - (2.0 / n) * np.dot(a_i, b_i)

        Sxx = self._pairwise_squared_diff_sum(x_np)
        Syy = self._pairwise_squared_diff_sum(y_np)
        m_xx = (2.0 * Sxx) / (n * n)
        m_yy = (2.0 * Syy) / (n * n)

        dvar_x2 = m_xx + a_bar * a_bar - (2.0 / n) * np.dot(a_i, a_i)
        dvar_y2 = m_yy + b_bar * b_bar - (2.0 / n) * np.dot(b_i, b_i)

        dcov2 = max(dcov2, 0.0)
        dvar_x2 = max(dvar_x2, 0.0)
        dvar_y2 = max(dvar_y2, 0.0)

        denom = np.sqrt(dvar_x2 * dvar_y2)
        if denom <= 0.0:
            return 0.0

        r2 = dcov2 / denom
        return float(np.clip(r2, 0.0, 1.0))

    # -------------------- GPU tiled (global centering, R^2) --------------------

    def _dcor_gpu_single(self, x: cp.ndarray, y: cp.ndarray, max_n: int = None, tile: int = None) -> float:
        """
        Correlação de distância ao quadrado (R^2) via tiling em GPU com centragem global (2 passes).
        Aceita entradas 1D (n,) ou multi-D (n,d). Retorna R^2 em [0, 1].
        """
        try:
            # Máscara row-wise de NaNs
            mask = self._rowwise_valid_mask(x, y)
            x = x[mask]
            y = y[mask]
            n = int(x.shape[0])
            if n < 2:
                return float("nan")

            # Limite de amostras
            max_n = int(max_n) if max_n is not None else self.dcor_max_samples
            if n > max_n:
                x = x[-max_n:]
                y = y[-max_n:]
                n = max_n

            # Formato (n,d)
            x = x.astype(cp.float32, copy=False)
            y = y.astype(cp.float32, copy=False)
            if x.ndim == 1:
                x = x[:, None]
            if y.ndim == 1:
                y = y[:, None]

            # Tamanho de bloco (adaptativo por memória + limitado por n)
            try:
                req = int(tile) if tile is not None else int(self.dcor_tile_size)
            except Exception:
                req = int(self.dcor_tile_size)
            t = _adaptive_tile(req)
            t = max(256, min(int(t), n))

            # Passo 1: somas por linha e média global
            a_row_sums = cp.zeros(n, dtype=cp.float64)
            b_row_sums = cp.zeros(n, dtype=cp.float64)
            a_total_sum = cp.float64(0.0)
            b_total_sum = cp.float64(0.0)

            for i0 in range(0, n, t):
                i1 = min(i0 + t, n)
                xi = x[i0:i1]
                yi = y[i0:i1]
                for j0 in range(0, n, t):
                    j1 = min(j0 + t, n)
                    xj = x[j0:j1]
                    yj = y[j0:j1]

                    dx = cp.linalg.norm(xi[:, None, :] - xj[None, :, :], axis=-1)
                    dy = cp.linalg.norm(yi[:, None, :] - yj[None, :, :], axis=-1)

                    a_row_sums[i0:i1] += dx.sum(axis=1, dtype=cp.float64)
                    b_row_sums[i0:i1] += dy.sum(axis=1, dtype=cp.float64)
                    a_total_sum += cp.sum(dx, dtype=cp.float64)
                    b_total_sum += cp.sum(dy, dtype=cp.float64)

            n_f = float(n)
            a_row_mean = a_row_sums / n_f
            b_row_mean = b_row_sums / n_f
            a_grand = float(a_total_sum / (n_f * n_f))
            b_grand = float(b_total_sum / (n_f * n_f))

            # Passo 2: centragem global por bloco + acumulação
            num = cp.float64(0.0)   # sum A*B
            sumA2 = cp.float64(0.0) # sum A^2
            sumB2 = cp.float64(0.0) # sum B^2

            for i0 in range(0, n, t):
                i1 = min(i0 + t, n)
                xi = x[i0:i1]
                yi = y[i0:i1]
                a_i = a_row_mean[i0:i1]
                b_i = b_row_mean[i0:i1]

                for j0 in range(0, n, t):
                    j1 = min(j0 + t, n)
                    xj = x[j0:j1]
                    yj = y[j0:j1]
                    a_j = a_row_mean[j0:j1]
                    b_j = b_row_mean[j0:j1]

                    dx = cp.linalg.norm(xi[:, None, :] - xj[None, :, :], axis=-1)
                    dy = cp.linalg.norm(yi[:, None, :] - yj[None, :, :], axis=-1)

                    # Centragem global in-place para reduzir alocações temporárias
                    dx -= a_i[:, None]
                    dx -= a_j[None, :]
                    dx += a_grand
                    dy -= b_i[:, None]
                    dy -= b_j[None, :]
                    dy += b_grand

                    num  += cp.sum(dx * dy, dtype=cp.float64)
                    sumA2 += cp.sum(dx * dx, dtype=cp.float64)
                    sumB2 += cp.sum(dy * dy, dtype=cp.float64)

            denom = cp.sqrt(sumA2 * sumB2)
            if float(denom) <= 0.0:
                return 0.0

            r2 = float(num / denom)
            return float(min(1.0, max(0.0, r2)))

        except Exception as e:
            self._log_error("Error in tiled GPU dCor", traceback=str(e))
            return float("nan")

    def _distance_correlation_gpu(
        self,
        x: cp.ndarray,
        y: cp.ndarray,
        tile: int = 2048,
        max_n: int = None,
        async_execution: bool = False,  # mantido por compatibilidade; não assíncrono aqui
    ) -> float:
        """
        Seleciona automaticamente o caminho:
        - 1D: CPU rápido (R^2)
        - multi-D: GPU em blocos com centragem global (R^2)
        Retorna R^2 em [0, 1].
        """
        try:
            if tile is not None:
                try:
                    self.dcor_tile_size = int(tile)
                except (ValueError, TypeError):
                    pass

            mask = self._rowwise_valid_mask(x, y)
            x = x[mask]
            y = y[mask]
            if x.shape[0] < 2:
                return float('nan')

            # amostragem
            if max_n is not None and int(x.shape[0]) > int(max_n):
                x = x[-int(max_n):]
                y = y[-int(max_n):]

            # 1D → CPU rápido
            if x.ndim == 1 and y.ndim == 1:
                self._log_info("CPU fast 1D (R^2)", samples=int(x.size))
                return self._distance_correlation_1d_fast(cp.asnumpy(x), cp.asnumpy(y))

            # multi-D → GPU em blocos
            self._log_info("GPU tiled (R^2)", samples=int(x.shape[0]))
            return self._dcor_gpu_single(x, y, max_n=max_n, tile=self.dcor_tile_size)

        except cp.cuda.OutOfMemoryError as e:
            self._log_warn("GPU OOM", traceback=str(e))
            return float('nan')
        except Exception as e:
            self._log_error("dCor GPU path failed", traceback=str(e))
            return float('nan')

    # -------------------- Permutation test (R^2) --------------------

    def distance_correlation_with_permutation(
        self,
        x: cp.ndarray,
        y: cp.ndarray,
        n_perm: int = 1000,
        perm_max_samples: int = 4096
    ) -> Tuple[float, float]:
        """
        Calcula R^2 observado e p-valor via teste de permutação:
        - R^2 observado: com max_n = self.dcor_max_samples (ou o tamanho efetivo)
        - Permutações: usam no máx. perm_max_samples para reduzir custo

        Retorna: (r2_obs, p_value)
        """
        # Observado (pode usar mais amostras)
        r2_obs = self._distance_correlation_gpu(x, y, max_n=self.dcor_max_samples)
        if not np.isfinite(r2_obs):
            return 0.0, 1.0

        # Contador
        ge = 0
        # Permutar em GPU (embaralhando y)
        # Usamos limite menor para acelerar
        for _ in range(int(n_perm)):
            # amostrar mesma porção final para consistência de custo
            x_use = x
            y_use = y
            if int(x.shape[0]) > perm_max_samples:
                x_use = x[-perm_max_samples:]
                y_use = y[-perm_max_samples:]
            # máscara row-wise
            mask = self._rowwise_valid_mask(x_use, y_use)
            if int(mask.sum().item()) < 2:
                continue
            xv = x_use[mask]
            yv = y_use[mask]
            # permuta apenas y
            yp = cp.random.permutation(yv)
            r2_perm = self._distance_correlation_gpu(xv, yp, max_n=perm_max_samples)
            if np.isfinite(r2_perm) and (r2_perm >= r2_obs):
                ge += 1

        p_value = (ge + 1.0) / (n_perm + 1.0)
        return float(r2_obs), float(p_value)

    # -------------------- Batch (R^2) --------------------

    def compute_distance_correlation_batch(
        self,
        data_pairs: List[Tuple[cp.ndarray, cp.ndarray]],
        max_samples: int = 10000,
        include_permutation: bool = False,
        n_perm: int = 1000,
        perm_max_samples: int = 4096
    ) -> List[Dict[str, float]]:
        """
        Computa R^2 (e opcionalmente p-valor) para vários pares (x,y).

        Retorna lista de dicts com chaves:
        - 'dcor_r2' e opcionalmente 'dcor_pvalue', 'dcor_significant'
        """
        results = []
        for i, (x, y) in enumerate(data_pairs):
            try:
                mask = self._rowwise_valid_mask(x, y)
                valid_count = int(mask.sum().item())
                if valid_count < 10:
                    if include_permutation:
                        results.append({'dcor_r2': 0.0, 'dcor_pvalue': 1.0, 'dcor_significant': False})
                    else:
                        results.append({'dcor_r2': 0.0})
                    continue

                x_clean = x[mask]
                y_clean = y[mask]

                if include_permutation:
                    r2, p = self.distance_correlation_with_permutation(
                        x_clean, y_clean, n_perm=n_perm, perm_max_samples=perm_max_samples
                    )
                    results.append({
                        'dcor_r2': float(r2),
                        'dcor_pvalue': float(p),
                        'dcor_significant': bool(p < 0.05)
                    })
                else:
                    r2 = self._distance_correlation_gpu(x_clean, y_clean, max_n=max_samples)
                    results.append({'dcor_r2': float(r2)})

            except Exception as e:
                self._log_warn(f"Batch pair {i} failed", error=str(e))
                if include_permutation:
                    results.append({'dcor_r2': 0.0, 'dcor_pvalue': 1.0, 'dcor_significant': False})
                else:
                    results.append({'dcor_r2': 0.0})
        return results

    def compute_distance_correlation_parallel_batch(
        self,
        data_pairs: List[Tuple[cp.ndarray, cp.ndarray]],
        max_samples: int = 10000,
        batch_size: int = 8
    ) -> List[float]:
        """
        Processa pares em lotes (sem streams explícitos aqui) e retorna lista de R^2.
        """
        results: List[float] = []
        for batch_start in range(0, len(data_pairs), batch_size):
            batch_pairs = data_pairs[batch_start:batch_start + batch_size]
            for x, y in batch_pairs:
                try:
                    mask = self._rowwise_valid_mask(x, y)
                    if int(mask.sum().item()) < 10:
                        results.append(0.0)
                        continue
                    r2 = self._distance_correlation_gpu(x[mask], y[mask], max_n=max_samples)
                    results.append(float(r2))
                except Exception as e:
                    self._log_warn("Parallel batch item failed", error=str(e))
                    results.append(0.0)
        return results

    # -------------------- Helpers para DataFrames cuDF --------------------

    def _dcor_partition_gpu(self, pdf: cudf.DataFrame, target: str, candidates: List[str],
                            max_samples: int, tile: int) -> cudf.DataFrame:
        """Computa R^2 entre target e cada feature candidata numa partição (GPU)."""
        out: Dict[str, float] = {}
        try:
            y = pdf[target].astype('f8').to_cupy()
        except Exception:
            return cudf.DataFrame([{f"dcor_{c}": float('nan') for c in candidates}])

        for i, c in enumerate(candidates):
            try:
                x = pdf[c].astype('f8').to_cupy()
                r2 = self._distance_correlation_gpu(x, y, tile=tile, max_n=max_samples)
                out[f"dcor_{c}"] = float(r2)
                if (i + 1) % 20 == 0 or (i + 1) == len(candidates):
                    self._log_info("partition progress", processed=i + 1, total=len(candidates), last_feature=c)
            except Exception as e:
                self._log_warn("partition feature failed", feature=c, error=str(e))
                out[f"dcor_{c}"] = float('nan')

        return cudf.DataFrame([out])

    def _dcor_rolling_partition_gpu(
        self, pdf: cudf.DataFrame, target: str, candidates: List[str], window: int, step: int,
        min_periods: int, min_valid_pairs: int, max_rows: int, max_windows: int, agg: str,
        max_samples: int, tile: int
    ) -> cudf.DataFrame:
        """Computa R^2 rolling por feature e agrega (mean/median/min/max/p25/p75)."""
        if hasattr(pdf, 'head'):
            pdf = pdf.head(int(max_rows))
        try:
            y_all = pdf[target].astype('f8').to_cupy()
        except Exception:
            base = {f"dcor_roll_{c}": float('nan') for c in candidates}
            cnts = {f"dcor_roll_cnt_{c}": np.int64(0) for c in candidates}
            return cudf.DataFrame([{**base, **cnts}])

        n = int(y_all.size)
        if n < max(3, int(min_periods)):
            base = {f"dcor_roll_{c}": float('nan') for c in candidates}
            cnts = {f"dcor_roll_cnt_{c}": np.int64(0) for c in candidates}
            return cudf.DataFrame([{**base, **cnts}])

        starts = list(range(0, max(0, n - int(min_periods) + 1), max(1, int(step))))
        if len(starts) > int(max_windows):
            starts = starts[-int(max_windows):]

        # Pré-carrega colunas
        X_cols: Dict[str, cp.ndarray] = {}
        for c in candidates:
            try:
                X_cols[c] = pdf[c].astype('f8').to_cupy()
            except Exception:
                X_cols[c] = None

        score_map: Dict[str, float] = {}
        cnt_map: Dict[str, int] = {}

        for c in candidates:
            x_all = X_cols.get(c, None)
            if x_all is None:
                score_map[f"dcor_roll_{c}"] = float('nan')
                cnt_map[f"dcor_roll_cnt_{c}"] = np.int64(0)
                continue

            vals: List[float] = []
            for s in starts:
                e = min(n, s + int(window))
                if e - s < int(min_periods):
                    continue
                xv = x_all[s:e]
                yv = y_all[s:e]
                # máscara row-wise
                mask = self._rowwise_valid_mask(xv, yv)
                valid_cnt = int(mask.sum().item())
                if valid_cnt < int(min_valid_pairs):
                    continue
                r2 = self._distance_correlation_gpu(xv[mask], yv[mask], tile=tile, max_n=max_samples)
                if np.isfinite(r2):
                    vals.append(float(r2))

            if not vals:
                score_map[f"dcor_roll_{c}"] = float('nan')
                cnt_map[f"dcor_roll_cnt_{c}"] = np.int64(0)
            else:
                arr = np.asarray(vals, dtype=float)
                cnt_map[f"dcor_roll_cnt_{c}"] = np.int64(arr.size)
                if agg == 'mean':
                    score_map[f"dcor_roll_{c}"] = float(np.mean(arr))
                elif agg == 'min':
                    score_map[f"dcor_roll_{c}"] = float(np.min(arr))
                elif agg == 'max':
                    score_map[f"dcor_roll_{c}"] = float(np.max(arr))
                elif agg == 'p25':
                    score_map[f"dcor_roll_{c}"] = float(np.percentile(arr, 25))
                elif agg == 'p75':
                    score_map[f"dcor_roll_{c}"] = float(np.percentile(arr, 75))
                else:
                    score_map[f"dcor_roll_{c}"] = float(np.median(arr))

        ordered = {}
        for c in candidates:
            ordered[f"dcor_roll_{c}"] = score_map.get(f"dcor_roll_{c}", float('nan'))
        for c in candidates:
            ordered[f"dcor_roll_cnt_{c}"] = cnt_map.get(f"dcor_roll_cnt_{c}", np.int64(0))
        return cudf.DataFrame([ordered])

    # -------------------- (Opcional) Aplicação em DataFrame amplo --------------------

    def _apply_comprehensive_distance_correlation(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Exemplo de aplicação por features 'frac_diff_*' vs coluna alvo ('y_ret' ou 'target').
        Cria colunas dcor_* com **R^2** e, se habilitado, dcor_roll_* (mediana rolling).
        """
        try:
            self._log_info("Applying comprehensive distance correlation analysis...")
            # encontra alvo
            target_col = None
            for col in df.columns:
                if 'y_ret' in col or 'target' in col.lower():
                    target_col = col
                    break
            if target_col is None:
                self._log_warn("No target column found")
                return df

            # features candidatas
            candidate_features = []
            for col in df.columns:
                if (col != target_col and
                    not col.startswith(('dcor_', 'stage1_', 'cpcv_', 'y_ret_fwd_')) and
                    'frac_diff' in col):
                    candidate_features.append(col)
            if not candidate_features:
                self._log_warn("No candidate features found")
                return df

            # computa por feature
            for feature in candidate_features:
                try:
                    x = df[feature].to_cupy()
                    y = df[target_col].to_cupy()
                    mask = self._rowwise_valid_mask(x, y)
                    if int(mask.sum().item()) > 50:
                        r2 = self._distance_correlation_gpu(x[mask], y[mask], max_n=self.dcor_max_samples)
                        base_name = feature.replace('frac_diff_', '')
                        df[f'dcor_{base_name}'] = float(r2)

                        # rolling opcional
                        if hasattr(self, 'stage1_rolling_enabled') and self.stage1_rolling_enabled:
                            roll = self._compute_rolling_distance_correlation(
                                df, target_col, feature,
                                window=getattr(self, 'stage1_rolling_window', 2000),
                                step=getattr(self, 'stage1_rolling_step', 500)
                            )
                            if roll is not None:
                                df[f'dcor_roll_{base_name}'] = float(roll)
                except Exception as e:
                    self._log_warn("feature dCor failed", feature=feature, error=str(e))
                    continue
            self._log_info("Comprehensive distance correlation completed")
            return df
        except Exception as e:
            self._log_error("Error in comprehensive analysis", error=str(e))
            return df

    def _compute_rolling_distance_correlation(self, df: cudf.DataFrame, target: str, feature: str,
                                              window: int = 2000, step: int = 500) -> float:
        """Mediana de R^2 em janelas deslizantes (retorna escalar ou None)."""
        try:
            x = df[feature].to_cupy()
            y = df[target].to_cupy()
            n = int(x.size)
            if n < window:
                return None

            vals: List[float] = []
            for start in range(0, n - window + 1, step):
                end = start + window
                xv = x[start:end]
                yv = y[start:end]
                mask = self._rowwise_valid_mask(xv, yv)
                if int(mask.sum().item()) > 100:
                    r2 = self._distance_correlation_gpu(xv[mask], yv[mask], max_n=self.dcor_max_samples)
                    if np.isfinite(r2):
                        vals.append(float(r2))
            if vals:
                return float(np.median(np.asarray(vals, dtype=float)))
            return None
        except Exception as e:
            self._log_warn("Rolling dCor failed", error=str(e))
            return None
