"""High-level orchestrator that executes a backtest run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterator, List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # type: ignore
except Exception as exc:  # pragma: no cover - required dependency
    raise RuntimeError(
        "catboost is required to run the backtesting module. Install it in the runtime environment."
    ) from exc

from utils.logging_utils import get_logger

from .config import BacktestRunConfig
from .data_loader import load_parquet_dataset, to_pandas
from .pipeline import FeatureReplayPipeline
from .signals import SignalGenerator

logger = get_logger(__name__, "backtesting.runner")


class BacktestRunner:
    """Execute a single backtest using persisted model metadata."""

    def __init__(self, run_config: BacktestRunConfig):
        self.config = run_config
        self.pipeline = FeatureReplayPipeline(prefer_gpu=False)
        self.signal_generator = SignalGenerator(run_config.thresholds)
        self._model = None
        self._is_classifier = True
        self.trade_log: List[Dict[str, object]] = []
        self.metrics: Dict[str, float] = {}

    def iterate_events(self) -> Iterator[Dict[str, object]]:
        """Yield streaming events for the front-end."""
        df = load_parquet_dataset(self.config.dataset_path)

        processed = self.pipeline.run(df, self.config.pipeline)
        pdf = to_pandas(processed)

        timestamp_col = self.config.pipeline.timestamp_column
        if timestamp_col not in pdf.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' not present after pipeline execution")

        price_col = self._resolve_price_column(pdf)
        if price_col not in pdf.columns:
            raise KeyError(f"Price column '{price_col}' not found for signal generation")

        features_df = self._prepare_feature_frame(pdf)
        if features_df.empty:
            raise RuntimeError("No valid rows remain after feature preparation (check for NaNs)")

        timestamp_series = pd.to_datetime(pdf.loc[features_df.index, timestamp_col])
        price_series = pdf.loc[features_df.index, price_col].astype(float)

        predictions = self._predict(features_df)

        for idx, pred in zip(features_df.index, predictions):
            ts = timestamp_series.loc[idx]
            price = price_series.loc[idx]
            feature_snapshot = self._feature_snapshot(features_df.loc[idx])
            event = self.signal_generator.step(ts, price, float(pred), feature_snapshot)
            yield event

        last_ts = timestamp_series.iloc[-1]
        last_price = float(price_series.iloc[-1])
        trades = self.signal_generator.finalize(last_ts, last_price)
        self.trade_log = [self._trade_to_dict(t) for t in trades]
        self.metrics = self.signal_generator.performance_metrics()

        yield {
            "type": "summary",
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "metrics": self.metrics,
            "trades": self.trade_log,
        }

    # --- helpers ----------------------------------------------------------------

    def _resolve_price_column(self, pdf: pd.DataFrame) -> str:
        if self.config.pipeline.price_column and self.config.pipeline.price_column in pdf.columns:
            return self.config.pipeline.price_column
        for candidate in ("y_close", "close", "price"):
            if candidate in pdf.columns:
                return candidate
        for col in pdf.columns:
            if "close" in str(col).lower():
                return str(col)
        raise KeyError("Could not infer price column")

    def _prepare_feature_frame(self, pdf: pd.DataFrame) -> pd.DataFrame:
        selected = list(self.config.model.selected_features)
        if not selected:
            raise ValueError("Metadata does not provide selected_features for the model")
        available = [col for col in selected if col in pdf.columns]
        missing = sorted(set(selected) - set(available))
        if missing:
            logger.warning("Missing %s features required by the model: %s", len(missing), missing[:10])
        if not available:
            raise RuntimeError("None of the selected features are present in the processed dataset")
        feature_df = pdf[available].copy()

        if self.config.model.scaler and isinstance(self.config.model.scaler, Mapping):
            scaler = self.config.model.scaler
            means = scaler.get("mean")
            stds_candidate = scaler.get("std")
            means = means if isinstance(means, Mapping) else {}
            if isinstance(stds_candidate, Mapping):
                stds = stds_candidate
            else:
                stds = scaler.get("std_dev")
                stds = stds if isinstance(stds, Mapping) else {}
            for col in available:
                if col in means and col in stds:
                    mu = means[col]
                    sigma = stds[col]
                    if sigma not in (0, None):
                        feature_df[col] = (feature_df[col] - mu) / sigma
        feature_df = feature_df.dropna()
        return feature_df

    def _predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        model = self._load_model()
        pool = Pool(feature_df)
        if self._is_classifier:
            raw = model.predict_proba(pool)
            if raw.ndim == 2 and raw.shape[1] > 1:
                return raw[:, 1]
            return raw.flatten()
        return model.predict(pool).flatten()

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_type = (self.config.model.model_type or "catboostclassifier").lower()
        if "class" in model_type or model_type.endswith("classifier"):
            model = CatBoostClassifier()
            self._is_classifier = True
        else:
            model = CatBoostRegressor()
            self._is_classifier = False
        logger.info("Loading CatBoost model from %s", self.config.model_path)
        model.load_model(str(self.config.model_path))
        self._model = model
        return model

    def _feature_snapshot(self, row: pd.Series) -> Dict[str, float]:
        columns = self.config.pipeline.streaming_features or list(row.index)[:5]
        snapshot = {}
        for col in columns:
            if col in row and pd.notna(row[col]):
                snapshot[col] = float(row[col])
        return snapshot

    def _trade_to_dict(self, trade) -> Dict[str, object]:
        return {
            "direction": trade.direction,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "prediction": trade.prediction,
            "exit_reason": trade.exit_reason,
            "return_pct": trade.return_pct,
            "pnl": trade.pnl,
            "duration_minutes": trade.duration_minutes,
        }
