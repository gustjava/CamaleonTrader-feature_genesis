"""Feature replay pipeline for backtesting (GPU or CPU)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, List

import pandas as pd

from config.unified_config import get_unified_config as get_settings
from utils.logging_utils import get_logger

from .config import BacktestPipelineConfig, PipelineComponentConfig
from .cpu_features import (
    apply_feature_engineering_cpu,
    apply_garch_cpu,
    apply_signal_processing_cpu,
    apply_stationarization_cpu,
)
from .data_loader import to_pandas

logger = get_logger(__name__, "backtesting.pipeline")

try:  # Optional GPU stack
    import cupy as cp  # type: ignore
    import cudf  # type: ignore

    from features import FeatureEngineeringEngine, GARCHModels, StationarizationEngine
    from features.signal_processing import apply_emd_to_series
    try:
        from features.signal_processing import apply_rolling_emd
    except Exception:  # pragma: no cover - optional
        apply_rolling_emd = None  # type: ignore

    GPU_AVAILABLE = True
except Exception:  # pragma: no cover - GPU dependencies missing
    cp = None  # type: ignore
    cudf = None  # type: ignore
    FeatureEngineeringEngine = None  # type: ignore
    GARCHModels = None  # type: ignore
    StationarizationEngine = None  # type: ignore
    apply_emd_to_series = None  # type: ignore
    apply_rolling_emd = None  # type: ignore
    GPU_AVAILABLE = False


@contextmanager
def _temporary_attribute(obj: Any, name: str, value: Any):
    has_attr = hasattr(obj, name)
    original = getattr(obj, name) if has_attr else None
    setattr(obj, name, value)
    try:
        yield
    finally:
        if has_attr:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


class FeatureReplayPipeline:
    """Replays the feature engineering pipeline for backtesting."""

    def __init__(self, settings=None, prefer_gpu: bool = True):
        self.settings = settings or get_settings()
        self.use_gpu = bool(prefer_gpu and GPU_AVAILABLE)

        if self.use_gpu:
            self.station = StationarizationEngine(self.settings, client=None)
            self.feature_engineering = FeatureEngineeringEngine(self.settings, client=None)
            self.garch = GARCHModels(self.settings, client=None)
        else:
            self.station = None
            self.feature_engineering = None
            self.garch = None

    def run(self, df: Any, pipeline_cfg: BacktestPipelineConfig) -> Any:
        if self.use_gpu and cudf is not None and isinstance(df, cudf.DataFrame):  # type: ignore[attr-defined]
            return self._run_gpu(df, pipeline_cfg)
        # Always operate on pandas for CPU path
        pdf = to_pandas(df)
        return self._run_cpu(pdf, pipeline_cfg)

    # ------------------------------------------------------------------ GPU path
    def _run_gpu(self, gdf, pipeline_cfg: BacktestPipelineConfig):  # type: ignore[no-untyped-def]
        result = gdf
        for name, component in pipeline_cfg.ordered_components():
            if not component.enabled:
                logger.info("Skipping disabled component %s", name)
                continue
            logger.info("Running component %s (GPU)", name)
            if name == "signal_processing":
                result = self._run_signal_processing_gpu(result, component, pipeline_cfg)
            elif name == "stationarization":
                result = self._run_stationarization_gpu(result, component)
            elif name == "feature_engineering":
                result = self._run_feature_engineering_gpu(result, component)
            elif name == "garch_models":
                result = self._run_garch_models_gpu(result)
            else:
                logger.warning("Unknown pipeline component '%s'; skipping", name)
            if cp is not None:  # pragma: no branch - guard memory cleanup
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
        return result

    def _run_signal_processing_gpu(self, gdf, component, pipeline_cfg):  # type: ignore[no-untyped-def]
        columns = self._resolve_signal_columns(list(gdf.columns), component, pipeline_cfg)
        if not columns or apply_emd_to_series is None:
            return gdf
        max_imfs = int(component.params.get("max_imfs", component.extra.get("max_imfs", 5)))
        rolling_cfg = component.params.get("rolling") or component.extra.get("rolling")
        rolling_enabled = bool(rolling_cfg and rolling_cfg.get("enabled"))
        rolling_window = int(rolling_cfg.get("window", 500)) if rolling_enabled else None
        rolling_step = int(rolling_cfg.get("step", 50)) if rolling_enabled else None
        rolling_embargo = int(rolling_cfg.get("embargo", 0)) if rolling_enabled else None

        frames = [gdf]
        for col in columns:
            try:
                if rolling_enabled and apply_rolling_emd is not None:
                    imfs = apply_rolling_emd(
                        gdf[col],
                        max_imfs=max_imfs,
                        window_size=rolling_window or 500,
                        step_size=rolling_step or 50,
                        embargo=rolling_embargo or 0,
                    )
                else:
                    imfs = apply_emd_to_series(gdf[col], max_imfs=max_imfs)
                imfs = imfs.rename(columns={c: f"emd_{c}" for c in imfs.columns})
                frames.append(imfs)
            except Exception as exc:
                logger.warning("Failed to compute EMD for %s: %s", col, exc)
        if len(frames) == 1:
            return gdf
        return cudf.concat(frames, axis=1)

    def _run_stationarization_gpu(self, gdf, component):  # type: ignore[no-untyped-def]
        columns = component.columns or getattr(self.station, "_station_include", [])
        with _temporary_attribute(self.station, "_station_include", list(columns) if columns else self.station._station_include):
            return self.station.process_currency_pair(gdf)

    def _run_feature_engineering_gpu(self, gdf, component):  # type: ignore[no-untyped-def]
        selected_cols = list(component.columns)
        original_bk_params = self.feature_engineering._bk_params

        def _bk_params_override():
            params = original_bk_params()
            if selected_cols:
                params = dict(params)
                params["source_columns"] = selected_cols
                params["apply_to_all"] = False
            for key, value in component.params.items():
                params[key] = value
            return params

        try:
            if selected_cols or component.params:
                self.feature_engineering._bk_params = _bk_params_override  # type: ignore[assignment]
            return self.feature_engineering.process_cudf(gdf)
        finally:
            self.feature_engineering._bk_params = original_bk_params  # type: ignore[assignment]

    def _run_garch_models_gpu(self, gdf):  # type: ignore[no-untyped-def]
        return self.garch.process_cudf(gdf)

    # ------------------------------------------------------------------ CPU path
    def _run_cpu(self, df: pd.DataFrame, pipeline_cfg: BacktestPipelineConfig) -> pd.DataFrame:
        result = df.copy()
        for name, component in pipeline_cfg.ordered_components():
            if not component.enabled:
                logger.info("Skipping disabled component %s", name)
                continue
            logger.info("Running component %s (CPU)", name)
            if name == "signal_processing":
                columns = self._resolve_signal_columns(result.columns.tolist(), component, pipeline_cfg)
                result = apply_signal_processing_cpu(result, columns, component.params)
            elif name == "stationarization":
                columns = component.columns or None
                result = apply_stationarization_cpu(result, columns, component.params)
            elif name == "feature_engineering":
                columns = component.columns or None
                result = apply_feature_engineering_cpu(result, columns, component.params)
            elif name == "garch_models":
                columns = component.columns or None
                result = apply_garch_cpu(result, columns, component.params)
            else:
                logger.warning("Unknown pipeline component '%s'; skipping", name)
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_signal_columns(
        columns: List[str],
        component: PipelineComponentConfig,
        pipeline_cfg: BacktestPipelineConfig,
    ) -> List[str]:
        if component.columns:
            return [col for col in component.columns if col in columns]
        inferred: List[str] = []
        if pipeline_cfg.price_column and pipeline_cfg.price_column in columns:
            inferred.append(pipeline_cfg.price_column)
        else:
            for candidate in ("y_close", "log_stabilized_y_close"):
                if candidate in columns:
                    inferred.append(candidate)
                    break
            if not inferred:
                for col in columns:
                    if "close" in str(col).lower():
                        inferred.append(str(col))
                        break
        return inferred
