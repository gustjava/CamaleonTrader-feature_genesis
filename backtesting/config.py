"""Configuration primitives for the backtesting module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


def _lower_keys(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of ``data`` with lower-cased keys."""
    return {str(k).lower(): v for k, v in data.items()}


@dataclass
class SignalThresholds:
    """Parameters that control trade generation."""

    buy_probability: float = 0.55
    sell_probability: float = 0.45
    take_profit_pct: float = 0.002
    stop_loss_pct: float = 0.001
    max_holding_minutes: Optional[int] = None
    cooldown_minutes: Optional[int] = None
    position_size: float = 1.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "SignalThresholds":
        if not data:
            return cls()
        lower = _lower_keys(data)
        return cls(
            buy_probability=float(lower.get("buy_probability", lower.get("buy_threshold", cls.buy_probability))),
            sell_probability=float(lower.get("sell_probability", lower.get("sell_threshold", cls.sell_probability))),
            take_profit_pct=float(lower.get("take_profit_pct", lower.get("take_profit", cls.take_profit_pct))),
            stop_loss_pct=float(lower.get("stop_loss_pct", lower.get("stop_loss", cls.stop_loss_pct))),
            max_holding_minutes=(
                int(lower["max_holding_minutes"]) if lower.get("max_holding_minutes") is not None else None
            ),
            cooldown_minutes=(
                int(lower["cooldown_minutes"]) if lower.get("cooldown_minutes") is not None else None
            ),
            position_size=float(lower.get("position_size", cls.position_size)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buy_probability": self.buy_probability,
            "sell_probability": self.sell_probability,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_holding_minutes": self.max_holding_minutes,
            "cooldown_minutes": self.cooldown_minutes,
            "position_size": self.position_size,
        }


@dataclass
class PipelineComponentConfig:
    """Minimal representation of a pipeline component to replay."""

    enabled: bool = True
    columns: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    order: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]], default_order: int) -> "PipelineComponentConfig":
        if not data:
            return cls(enabled=False, order=default_order)
        lower = _lower_keys(data)
        raw_cols = lower.get("columns") or lower.get("features") or []
        if isinstance(raw_cols, str):
            columns = [c.strip() for c in raw_cols.split(',') if c.strip()]
        elif isinstance(raw_cols, Iterable):
            columns = [str(c) for c in raw_cols]
        else:
            columns = []
        params = lower.get("params") or lower.get("parameters") or {}
        if not isinstance(params, MutableMapping):
            params = {}
        enabled = bool(lower.get("enabled", True))
        order = int(lower.get("order", default_order))
        extra = {
            k: v for k, v in lower.items() if k not in {"columns", "features", "params", "parameters", "enabled", "order"}
        }
        return cls(enabled=enabled, columns=columns, params=dict(params), order=order, extra=extra)


@dataclass
class BacktestPipelineConfig:
    """All pipeline stages required to rebuild features."""

    signal_processing: PipelineComponentConfig = field(
        default_factory=lambda: PipelineComponentConfig(enabled=False, order=10)
    )
    stationarization: PipelineComponentConfig = field(
        default_factory=lambda: PipelineComponentConfig(enabled=True, order=20)
    )
    feature_engineering: PipelineComponentConfig = field(
        default_factory=lambda: PipelineComponentConfig(enabled=True, order=30)
    )
    garch_models: PipelineComponentConfig = field(
        default_factory=lambda: PipelineComponentConfig(enabled=False, order=40)
    )
    execution_order: List[str] = field(default_factory=list)
    timestamp_column: str = "timestamp"
    price_column: Optional[str] = None
    streaming_features: List[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "BacktestPipelineConfig":
        if not data:
            return cls()
        lower = _lower_keys(data)
        pipe = cls(
            signal_processing=PipelineComponentConfig.from_mapping(lower.get("signal_processing"), default_order=10),
            stationarization=PipelineComponentConfig.from_mapping(lower.get("stationarization"), default_order=20),
            feature_engineering=PipelineComponentConfig.from_mapping(lower.get("feature_engineering"), default_order=30),
            garch_models=PipelineComponentConfig.from_mapping(lower.get("garch") or lower.get("garch_models"), default_order=40),
            execution_order=[str(x) for x in lower.get("execution_order", [])],
            timestamp_column=str(lower.get("timestamp_column", "timestamp")),
            price_column=str(lower.get("price_column")) if lower.get("price_column") else None,
            streaming_features=[str(f) for f in lower.get("streaming_features", [])],
        )
        return pipe

    def ordered_components(self) -> List[tuple[str, PipelineComponentConfig]]:
        components = {
            "signal_processing": self.signal_processing,
            "stationarization": self.stationarization,
            "feature_engineering": self.feature_engineering,
            "garch_models": self.garch_models,
        }
        if self.execution_order:
            ordered = [(name, components[name]) for name in self.execution_order if name in components]
            remaining = [(name, comp) for name, comp in components.items() if name not in self.execution_order]
            ordered.extend(sorted(remaining, key=lambda item: item[1].order))
            return ordered
        return sorted(components.items(), key=lambda item: item[1].order)


@dataclass
class ModelConfig:
    """Representation of a persisted model to replay."""

    model_type: str = "CatBoostClassifier"
    model_path: Optional[Path] = None
    target_column: str = "target"
    selected_features: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    scaler: Optional[Dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "ModelConfig":
        if not data:
            return cls()
        lower = _lower_keys(data)
        features_block = lower.get("features")
        if isinstance(features_block, Mapping):
            selected = features_block.get("selected_features", [])
        else:
            selected = lower.get("selected_features", [])
        if isinstance(selected, str):
            selected_features = [c.strip() for c in selected.split(',') if c.strip()]
        elif isinstance(selected, Iterable):
            selected_features = [str(c) for c in selected]
        else:
            selected_features = []
        params = lower.get("params") or lower.get("parameters") or {}
        if not isinstance(params, MutableMapping):
            params = {}
        model_path = lower.get("model_path") or lower.get("path")
        return cls(
            model_type=str(lower.get("model_type", lower.get("type", cls.model_type))),
            model_path=Path(model_path) if model_path else None,
            target_column=str(lower.get("target_column", lower.get("target", cls.target_column))),
            selected_features=selected_features,
            params=dict(params),
            scaler=lower.get("scaler"),
        )


@dataclass
class BacktestMetadata:
    """Metadata required to reproduce feature pipeline and model scoring."""

    symbol: str
    timeframe: Optional[str]
    model: ModelConfig
    pipeline: BacktestPipelineConfig
    thresholds: SignalThresholds
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BacktestMetadata":
        lower = _lower_keys(data)
        # Resolve model block
        model_block = lower.get("model") or lower.get("model_info") or lower
        features_block = lower.get("features") or model_block.get("features") if isinstance(model_block, Mapping) else {}
        if isinstance(model_block, Mapping) and "selected_features" not in model_block and features_block:
            model_block = dict(model_block)
            model_block.setdefault("selected_features", features_block.get("selected_features", []))
        model_cfg = ModelConfig.from_mapping(model_block)

        symbol = str(lower.get("symbol") or model_block.get("symbol") or lower.get("pair", "UNKNOWN"))
        timeframe = lower.get("timeframe") or model_block.get("timeframe")

        pipeline_cfg = BacktestPipelineConfig.from_mapping(
            lower.get("pipeline") or lower.get("feature_pipeline") or {}
        )

        thresholds_cfg = SignalThresholds.from_mapping(
            lower.get("signal_thresholds") or lower.get("signals") or lower.get("trading")
        )

        return cls(
            symbol=symbol,
            timeframe=str(timeframe) if timeframe else None,
            model=model_cfg,
            pipeline=pipeline_cfg,
            thresholds=thresholds_cfg,
            raw=dict(data),
        )


@dataclass
class BacktestRunConfig:
    """Concrete configuration for a backtest execution."""

    dataset_path: Path
    metadata_path: Path
    metadata: BacktestMetadata
    model_path: Path
    batch_size: int = 1

    @property
    def symbol(self) -> str:
        return self.metadata.symbol

    @property
    def timeframe(self) -> Optional[str]:
        return self.metadata.timeframe

    @property
    def pipeline(self) -> BacktestPipelineConfig:
        return self.metadata.pipeline

    @property
    def thresholds(self) -> SignalThresholds:
        return self.metadata.thresholds

    @property
    def model(self) -> ModelConfig:
        return self.metadata.model

    @classmethod
    def from_metadata(
        cls,
        dataset_path: Path,
        metadata_path: Path,
        metadata: BacktestMetadata,
        model_path: Optional[Path] = None,
        batch_size: int = 1,
    ) -> "BacktestRunConfig":
        resolved_model_path = model_path or metadata.model.model_path
        if resolved_model_path is None:
            raise ValueError("Model path must be provided either in metadata or explicitly.")
        return cls(
            dataset_path=Path(dataset_path),
            metadata_path=Path(metadata_path),
            metadata=metadata,
            model_path=Path(resolved_model_path),
            batch_size=int(batch_size),
        )
