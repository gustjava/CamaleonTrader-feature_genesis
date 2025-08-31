"""
Statistical Tests Module for Dynamic Stage 0 Pipeline

GPU-accelerated statistical tests including ADF and distance correlation.
"""

import logging
import dask_cudf
import cudf
import cupy as cp
import numpy as np
from numba import cuda

from .base_engine import BaseFeatureEngine

logger = logging.getLogger(__name__)


class StatisticalTests(BaseFeatureEngine):
    """
    Applies a set of statistical tests to a DataFrame.
    """

    # ---- ADF(0) t-stat em device: Δx_t = α + φ x_{t-1} + ε_t ----
    # Retorna t-stat de φ. Exige janela >= 10 p/ estabilidade mínima.
    @staticmethod
    @cuda.jit(device=True)
    def _adf_tstat_window(vals):
        n = vals.size
        if n < 10:
            return np.nan

        # y_t = x_t - x_{t-1}, z_t = x_{t-1}, t=1..n-1
        m = n - 1  # nº de observações da regressão
        sum_z = 0.0
        sum_y = 0.0
        sum_zz = 0.0
        sum_yy = 0.0
        sum_zy = 0.0

        prev = vals[0]
        for i in range(1, n):
            z = prev
            y = vals[i] - prev
            prev = vals[i]

            sum_z += z
            sum_y += y
            sum_zz += z * z
            sum_yy += y * y
            sum_zy += z * y

        # médias
        mz = sum_z / m
        my = sum_y / m

        # centradas
        Sxx = sum_zz - m * mz * mz
        Sxy = sum_zy - m * mz * my

        if Sxx <= 0.0 or m <= 2:
            return np.nan

        beta = Sxy / Sxx           # φ
        alpha = my - beta * mz     # α

        # SSE = Σ(y - α - β z)^2 = sum_yy + m α^2 + β^2 sum_zz - 2α sum_y - 2β sum_zy + 2αβ sum_z
        SSE = (sum_yy
               + m * alpha * alpha
               + beta * beta * sum_zz
               - 2.0 * alpha * sum_y
               - 2.0 * beta * sum_zy
               + 2.0 * alpha * beta * sum_z)

        dof = m - 2
        if dof <= 0:
            return np.nan

        sigma2 = SSE / dof
        if sigma2 <= 0.0:
            return np.nan

        se_beta = (sigma2 / Sxx) ** 0.5
        if se_beta == 0.0:
            return np.nan

        tstat = beta / se_beta
        return tstat

    def _apply_adf_rolling(self, s: cudf.Series, window: int = 252, min_periods: int = 200) -> cudf.Series:
        # UDF device é passado diretamente para rolling.apply
        return s.rolling(window=window, min_periods=min_periods).apply(
            self._adf_tstat_window
        )

    # ---- Distance Correlation (single-fit, com amostragem p/ n grande) ----
    def _dcor_gpu_single(self, x: cp.ndarray, y: cp.ndarray, max_n: int = 20000) -> float:
        n = x.size
        if n == 0:
            return float("nan")

        # amostragem se necessário (uniforme)
        if n > max_n:
            idx = cp.linspace(0, n - 1, max_n).round().astype(cp.int32)
            x = x[idx]
            y = y[idx]
            n = max_n

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Distâncias (ainda O(n^2); por isso limitamos n)
        a = cp.sqrt(cp.sum((x - x.T) ** 2, axis=-1))
        b = cp.sqrt(cp.sum((y - y.T) ** 2, axis=-1))

        A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

        dCovXY = cp.sqrt(cp.sum(A * B)) / n
        dVarX  = cp.sqrt(cp.sum(A * A)) / n
        dVarY  = cp.sqrt(cp.sum(B * B)) / n

        if dVarX > 0 and dVarY > 0:
            return float(dCovXY / cp.sqrt(dVarX * dVarY))
        return 0.0

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executes the statistical tests pipeline (Dask version).
        """
        self._log_info("Starting StatisticalTests (Dask)...")

        # --- ADF por janela em colunas de fracdiff ---
        # seu padrão de nomes parecia 'frac_diff_*'; ajustei aqui:
        adf_cols = [c for c in df.columns if "frac_diff" in c]

        for col in adf_cols:
            self._log_info(f"ADF rolling on '{col}'...")
            out = f"adf_stat_{col.split('_')[-1]}"  # e.g., 'close' de 'frac_diff_close'
            df[out] = df[col].map_partitions(
                lambda s: self._apply_adf_rolling(s, window=252, min_periods=200),
                meta=(out, "f8"),
            )

        # --- Distance Correlation (single-fit; sem rolling por enquanto) ---
        if {"returns", "tick_volume"}.issubset(set(df.columns)):
            self._log_info("Computing single-fit distance correlation (sampled)...")
            # Traga para 1 partição, compute uma vez e faça broadcast escalar
            one = df[["returns", "tick_volume"]].repartition(npartitions=1)
            # map_partitions retorna uma Series de 1 linha com o escalar
            dcor_ddf = one.map_partitions(
                lambda pdf: cudf.Series(
                    [self._dcor_gpu_single(pdf["returns"].to_cupy(), pdf["tick_volume"].to_cupy())],
                    index=[0],
                    name="dcor_returns_volume",
                ),
                meta=cudf.Series(name="dcor_returns_volume", dtype="f8"),
            )
            dcor_val = float(dcor_ddf.compute().iloc[0])
            df = df.assign(dcor_returns_volume=dcor_val)
        else:
            self._log_info("Skipping dCor (requires 'returns' and 'tick_volume').")

        self._log_info("StatisticalTests complete.")
        return df

    def process_cudf(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Executes the statistical tests pipeline (cuDF version).
        """
        self._log_info("Starting StatisticalTests (cuDF)...")

        # --- ADF por janela em colunas de fracdiff ---
        adf_cols = [c for c in gdf.columns if "frac_diff" in c]

        for col in adf_cols:
            self._log_info(f"ADF rolling on '{col}'...")
            out = f"adf_stat_{col.split('_')[-1]}"
            gdf[out] = self._apply_adf_rolling(gdf[col], window=252, min_periods=200)

        # --- Distance Correlation (single-fit) ---
        if {"returns", "tick_volume"}.issubset(set(gdf.columns)):
            self._log_info("Computing single-fit distance correlation (sampled)...")
            dcor_val = self._dcor_gpu_single(
                gdf["returns"].to_cupy(), 
                gdf["tick_volume"].to_cupy()
            )
            gdf["dcor_returns_volume"] = dcor_val
        else:
            self._log_info("Skipping dCor (requires 'returns' and 'tick_volume').")

        self._log_info("StatisticalTests complete.")
        return gdf
