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

# --- Module-level helper to avoid Dask tokenization issues ---
def _apply_bk_filter_gpu_partition(series: cudf.Series, k: int, low_period: float, high_period: float) -> cudf.Series:
    """Pure function for map_partitions: applies BK filter on a single partition.

    Parameters are plain types to keep Dask tokenization deterministic.
    """
    x = series.to_cupy()
    if x.dtype != cp.float32:
        x = x.astype(cp.float32, copy=False)

    # Get weights from the class static cache (deterministic)
    w_cpu = SignalProcessor._bk_weights_cpu(k, low_period, high_period)
    w = cp.asarray(w_cpu)

    n_kernel = 2 * k + 1
    if n_kernel <= 129:
        y = cp.convolve(x, w, mode="same")
    else:
        y = fftconvolve(x, w, mode="same")

    y[:k] = cp.nan
    y[-k:] = cp.nan
    return cudf.Series(y, index=series.index)

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
        k = int(self.settings.features.baxter_king_k)
        low = float(self.settings.features.baxter_king_low_freq)
        high = float(self.settings.features.baxter_king_high_freq)

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

    def process_cudf(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Executa o pipeline de sinal (cuDF version) com métricas secundárias completas.
        
        Implements the complete signal processing pipeline as specified in the technical plan:
        - Baxter-King filter application
        - Volatility of filtered series
        - Zero-crossing rate
        - Peak-to-valley ratio
        - Additional signal metrics
        
        Args:
            gdf: Input cuDF DataFrame
            
        Returns:
            DataFrame with filtered series and secondary metrics
        """
        self._log_info("Starting comprehensive SignalProcessor (cuDF)...")

        # Apply Baxter-King filter
        col_name = "bk_filter_close"
        self._log_info("Applying Baxter–King filter to 'y_close'...")

        gdf[col_name] = self._apply_bk_filter_gpu(gdf["y_close"])

        # Calculate secondary metrics for the filtered series
        gdf = self._add_signal_processing_metrics(gdf, col_name)

        self._log_info("Comprehensive SignalProcessor complete.")
        return gdf
    
    def _add_signal_processing_metrics(self, df: cudf.DataFrame, filtered_col: str) -> cudf.DataFrame:
        """
        Add comprehensive signal processing metrics for the filtered series.
        
        Generates:
        - bk_volatility_*: Volatility of filtered series
        - bk_zero_crossing_rate_*: Rate of zero crossings
        - bk_peak_valley_ratio_*: Peak-to-valley ratio
        - bk_signal_strength_*: Signal strength metrics
        - bk_cyclical_metrics_*: Cyclical behavior metrics
        """
        try:
            self._log_info("Adding comprehensive signal processing metrics...")
            
            if filtered_col not in df.columns:
                self._log_warn(f"Filtered column {filtered_col} not found")
                return df
            
            # Get filtered series
            filtered_series = df[filtered_col].to_cupy()
            
            # Remove NaN values for calculations
            valid_mask = ~cp.isnan(filtered_series)
            if cp.sum(valid_mask) < 10:
                self._log_warn("Insufficient valid data for signal metrics")
                return df
            
            clean_series = filtered_series[valid_mask]
            
            # 1. VOLATILITY OF FILTERED SERIES
            self._log_info("Calculating volatility of filtered series...")
            
            # Rolling volatility with different windows
            windows = [20, 50, 100, 200]
            for window in windows:
                if len(clean_series) >= window:
                    # Calculate rolling volatility
                    rolling_vol = self._calculate_rolling_volatility(clean_series, window)
                    df[f'bk_volatility_{filtered_col}_{window}'] = rolling_vol
            
            # Overall volatility
            overall_vol = float(cp.std(clean_series))
            df[f'bk_volatility_{filtered_col}_overall'] = overall_vol
            
            # 2. ZERO CROSSING RATE
            self._log_info("Calculating zero crossing rate...")
            zero_crossing_rate = self._calculate_zero_crossing_rate(clean_series)
            df[f'bk_zero_crossing_rate_{filtered_col}'] = zero_crossing_rate
            
            # 3. PEAK-TO-VALLEY RATIO
            self._log_info("Calculating peak-to-valley ratio...")
            peak_valley_ratio = self._calculate_peak_valley_ratio(clean_series)
            df[f'bk_peak_valley_ratio_{filtered_col}'] = peak_valley_ratio
            
            # 4. SIGNAL STRENGTH METRICS
            self._log_info("Calculating signal strength metrics...")
            signal_metrics = self._calculate_signal_strength_metrics(clean_series)
            for metric_name, value in signal_metrics.items():
                df[f'bk_{metric_name}_{filtered_col}'] = value
            
            # 5. CYCLICAL BEHAVIOR METRICS
            self._log_info("Calculating cyclical behavior metrics...")
            cyclical_metrics = self._calculate_cyclical_metrics(clean_series)
            for metric_name, value in cyclical_metrics.items():
                df[f'bk_{metric_name}_{filtered_col}'] = value
            
            return df
            
        except Exception as e:
            self._critical_error(f"Error adding signal processing metrics: {e}")
    
    def _calculate_rolling_volatility(self, series: cp.ndarray, window: int) -> cp.ndarray:
        """
        Calculate rolling volatility of a series.
        """
        try:
            # Convert to cuDF Series for rolling operations
            cudf_series = cudf.Series(series)
            rolling_std = cudf_series.rolling(window).std()
            
            # Convert back to CuPy array
            result = rolling_std.to_cupy()
            
            # Pad the beginning with NaN to match original length
            padded_result = cp.full(len(series), cp.nan)
            padded_result[window-1:] = result
            
            return padded_result
            
        except Exception as e:
            self._critical_error(f"Error calculating rolling volatility: {e}")
    
    def _calculate_zero_crossing_rate(self, series: cp.ndarray) -> float:
        """
        Calculate the rate of zero crossings in the series.
        """
        try:
            # Calculate differences
            diff_series = cp.diff(series)
            
            # Find zero crossings (sign changes)
            sign_changes = cp.diff(cp.sign(diff_series))
            zero_crossings = cp.sum(cp.abs(sign_changes) > 0)
            
            # Calculate rate
            rate = float(zero_crossings) / (len(series) - 2) if len(series) > 2 else 0.0
            
            return rate
            
        except Exception as e:
            self._critical_error(f"Error calculating zero crossing rate: {e}")
    
    def _calculate_peak_valley_ratio(self, series: cp.ndarray) -> float:
        """
        Calculate the peak-to-valley ratio of the series.
        """
        try:
            # Find peaks and valleys using simple local extrema
            peaks = []
            valleys = []
            
            for i in range(1, len(series) - 1):
                if series[i] > series[i-1] and series[i] > series[i+1]:
                    peaks.append(series[i])
                elif series[i] < series[i-1] and series[i] < series[i+1]:
                    valleys.append(series[i])
            
            if peaks and valleys:
                max_peak = float(cp.max(cp.array(peaks)))
                min_valley = float(cp.min(cp.array(valleys)))
                
                # Avoid division by zero
                if abs(min_valley) > 1e-9:
                    ratio = max_peak / abs(min_valley)
                else:
                    ratio = max_peak if max_peak > 0 else 1.0
            else:
                ratio = 1.0
            
            return ratio
            
        except Exception as e:
            self._critical_error(f"Error calculating peak-to-valley ratio: {e}")
    
    def _calculate_signal_strength_metrics(self, series: cp.ndarray) -> dict:
        """
        Calculate signal strength metrics.
        """
        try:
            metrics = {}
            
            # Signal-to-noise ratio (simplified)
            signal_power = float(cp.mean(series**2))
            noise_power = float(cp.var(series))
            snr = signal_power / (noise_power + 1e-9)
            metrics['signal_noise_ratio'] = snr
            
            # Signal energy
            energy = float(cp.sum(series**2))
            metrics['signal_energy'] = energy
            
            # Signal power
            power = float(cp.mean(series**2))
            metrics['signal_power'] = power
            
            # Signal amplitude
            amplitude = float(cp.max(cp.abs(series)))
            metrics['signal_amplitude'] = amplitude
            
            # Signal range
            signal_range = float(cp.max(series) - cp.min(series))
            metrics['signal_range'] = signal_range
            
            return metrics
            
        except Exception as e:
            self._critical_error(f"Error calculating signal strength metrics: {e}")
    
    def _calculate_cyclical_metrics(self, series: cp.ndarray) -> dict:
        """
        Calculate cyclical behavior metrics.
        """
        try:
            metrics = {}
            
            # Autocorrelation at lag 1
            if len(series) > 1:
                autocorr_lag1 = self._calculate_autocorrelation(series, lag=1)
                metrics['autocorr_lag1'] = autocorr_lag1
            
            # Autocorrelation at lag 5
            if len(series) > 5:
                autocorr_lag5 = self._calculate_autocorrelation(series, lag=5)
                metrics['autocorr_lag5'] = autocorr_lag5
            
            # Autocorrelation at lag 10
            if len(series) > 10:
                autocorr_lag10 = self._calculate_autocorrelation(series, lag=10)
                metrics['autocorr_lag10'] = autocorr_lag10
            
            # Periodicity measure (simplified)
            # Calculate FFT and find dominant frequency
            if len(series) > 20:
                periodicity = self._calculate_periodicity_measure(series)
                metrics['periodicity_measure'] = periodicity
            
            return metrics
            
        except Exception as e:
            self._critical_error(f"Error calculating cyclical metrics: {e}")
    
    def _calculate_autocorrelation(self, series: cp.ndarray, lag: int) -> float:
        """
        Calculate autocorrelation at a given lag.
        """
        try:
            if lag >= len(series):
                return 0.0
            
            # Calculate autocorrelation
            series_lagged = series[lag:]
            series_original = series[:-lag]
            
            # Remove mean
            mean_original = cp.mean(series_original)
            mean_lagged = cp.mean(series_lagged)
            
            # Calculate correlation
            numerator = cp.sum((series_original - mean_original) * (series_lagged - mean_lagged))
            denominator = cp.sqrt(cp.sum((series_original - mean_original)**2) * cp.sum((series_lagged - mean_lagged)**2))
            
            if denominator > 1e-9:
                autocorr = float(numerator / denominator)
            else:
                autocorr = 0.0
            
            return autocorr
            
        except Exception as e:
            self._critical_error(f"Error calculating autocorrelation: {e}")
    
    def _calculate_periodicity_measure(self, series: cp.ndarray) -> float:
        """
        Calculate a simple periodicity measure using FFT.
        """
        try:
            # Apply FFT
            fft_result = cp.fft.fft(series)
            power_spectrum = cp.abs(fft_result)**2
            
            # Find dominant frequency (excluding DC component)
            dominant_idx = cp.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = dominant_idx / len(series)
            
            # Calculate periodicity measure
            periodicity = float(dominant_freq)
            
            return periodicity
            
        except Exception as e:
            self._critical_error(f"Error calculating periodicity measure: {e}")

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executa o pipeline de sinal (Dask version).
        """
        self._log_info("Starting SignalProcessor (Dask)...")

        # (optional) ensure temporal order
        # df = self._ensure_sorted(df, by="ts")

        col_name = "bk_filter_close"
        self._log_info("Applying Baxter–King filter to 'y_close'...")

        # Parameters as plain values to keep tokenization deterministic
        k = int(self.settings.features.baxter_king_k)
        low = float(self.settings.features.baxter_king_low_freq)
        high = float(self.settings.features.baxter_king_high_freq)

        # meta em f4 para combinar com os pesos f32; mude para 'f8' se preferir
        df[col_name] = df["y_close"].map_partitions(
            _apply_bk_filter_gpu_partition,
            k,
            low,
            high,
            meta=(col_name, "f4"),
        )

        self._log_info("SignalProcessor complete.")
        return df
