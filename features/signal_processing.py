"""
Signal Processing Module for Dynamic Stage 0 Pipeline

GPU-accelerated Baxter–King (BK) band-pass filter with per-worker weight cache.
"""

import logging
import numpy as np
import dask_cudf
import cudf
import cupy as cp
from functools import lru_cache
from typing import Optional

# Use the import that your version supports:
try:
    from cusignal import fftconvolve
except ImportError:
    try:
        from cusignal.filtering import fftconvolve
    except ImportError:
        # Fallback to scipy if cusignal is not available
        from scipy.signal import fftconvolve
        def fftconvolve(x, w, mode="same"):
            return cp.asarray(fftconvolve(cp.asnumpy(x), cp.asnumpy(w), mode=mode))

from .base_engine import BaseFeatureEngine

logger = logging.getLogger(__name__)


class SignalProcessor(BaseFeatureEngine):
    """
    Applies signal processing techniques to a DataFrame.
    """

    # ---------- pesos BK em CPU com cache por processo ----------
    @staticmethod
    @lru_cache(maxsize=16)
    def _bk_weights_cpu(k: int, low_period: float, high_period: float) -> np.ndarray:
        """
        Pesos do filtro Baxter–King em CPU (numpy), cacheados por worker.
        low_period > high_period (ex.: 32 e 6) → passa-banda entre [w_high, w_low].
        """
        w_low = 2 * np.pi / float(low_period)
        w_high = 2 * np.pi / float(high_period)

        weights = np.zeros(2 * k + 1, dtype=np.float32)
        # centro
        weights[k] = (w_high - w_low) / np.pi

        # lados (simétricos)
        j = np.arange(1, k + 1, dtype=np.float32)
        weights[k + 1:] = (np.sin(w_high * j) - np.sin(w_low * j)) / (np.pi * j)
        weights[:k] = weights[k + 1:][::-1]

        # impor soma zero ajustando apenas o peso central
        wsum = weights.sum(dtype=np.float64)
        weights[k] -= (wsum - weights[k]).astype(np.float32)

        return weights  # float32

    def _apply_bk_filter_gpu(self, series: cudf.Series) -> cudf.Series:
        """
        Aplica Baxter–King em uma partição, usando GPU.
        """
        k = int(self.settings.features.baxter_king.k)
        low = float(self.settings.features.baxter_king.low_freq)
        high = float(self.settings.features.baxter_king.high_freq)

        # dados da série (GPU) — opcional: fazer cast para f32 por performance
        x = series.to_cupy()
        if x.dtype != cp.float32:
            x = x.astype(cp.float32, copy=False)

        # pesos: pegue do cache (CPU) e envie para GPU (1x por worker/tamanho)
        w_cpu = self._bk_weights_cpu(k, low, high)
        w = cp.asarray(w_cpu)  # GPU

        n_kernel = 2 * k + 1

        # escolha do método de convolução
        if n_kernel <= 129:
            # kernels curtos → conv direta costuma ser mais rápida
            y = cp.convolve(x, w, mode="same")
        else:
            # kernels longos → FFT
            y = fftconvolve(x, w, mode="same")

        # bordas inválidas do BK (k de cada lado)
        y[:k] = cp.nan
        y[-k:] = cp.nan

        return cudf.Series(y, index=series.index)

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executa o pipeline de sinal (Dask version).
        """
        self._log_info("Starting SignalProcessor (Dask)...")

        # (optional) ensure temporal order
        # df = self._ensure_sorted(df, by="ts")

        col_name = "bk_filter_close"
        self._log_info("Applying Baxter–King filter to 'close'...")

        # meta em f4 para combinar com os pesos f32; mude para 'f8' se preferir
        df[col_name] = df["close"].map_partitions(
            self._apply_bk_filter_gpu,
            meta=(col_name, "f4"),
        )

        self._log_info("SignalProcessor complete.")
        return df

    def process_cudf(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Executa o pipeline de sinal (cuDF version).
        """
        self._log_info("Starting SignalProcessor (cuDF)...")

        col_name = "bk_filter_close"
        self._log_info("Applying Baxter–King filter to 'close'...")

        gdf[col_name] = self._apply_bk_filter_gpu(gdf["close"])

        self._log_info("SignalProcessor complete.")
        return gdf
