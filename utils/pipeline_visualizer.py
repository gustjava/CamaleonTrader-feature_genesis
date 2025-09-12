"""
Pipeline Visualizer utilities.

Provides lightweight, non-intrusive helpers to visualize the pipeline structure,
data flow between stages, and summarize feature evolution and config impacts.

This module is intentionally dependency-light and produces ASCII/JSON artifacts
so it can run in headless environments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class PipelineStageInfo:
    name: str
    description: Optional[str] = None
    order: Optional[int] = None
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class PipelineVisualizer:
    """Generates diagrams and summaries for the pipeline."""

    def __init__(self, config: Optional[Any] = None, artifacts_dir: Optional[str] = None):
        self.config = config
        # Default artifacts folder under output if available in config
        out_dir = None
        try:
            if config and hasattr(config, 'output'):
                out_dir = getattr(config.output, 'artifacts_dir', None) or getattr(config.output, 'output_path', None)
        except Exception:
            out_dir = None
        self.artifacts_dir = Path(artifacts_dir or out_dir or './artifacts')
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Public API ----------------------
    def generate_pipeline_diagram(self) -> str:
        """Return an ASCII diagram of the configured pipeline stages."""
        stages = self._collect_stages()
        lines: List[str] = []
        lines.append("PIPELINE FEATURE GENESIS")
        lines.append("=" * 28)
        if not stages:
            lines.append("(no stages configured)")
            return "\n".join(lines)

        for i, st in enumerate(stages, 1):
            flag = "ON " if st.enabled else "OFF"
            lines.append(f"{i:02d}. [{flag}] {st.name} (order={st.order})")
            if st.description:
                lines.append(f"    - {st.description}")
            if st.parameters:
                # Show only top-level keys to keep it compact
                param_keys = ", ".join(sorted(list(st.parameters.keys()))[:6])
                if param_keys:
                    lines.append(f"    - params: {param_keys} ...")
        return "\n".join(lines)

    def show_data_flow(self, stage: str) -> Dict[str, Any]:
        """Return a structured view of inputs/outputs known for a given stage.

        This is a static mapping based on our current engines; it does not inspect dataframes.
        """
        catalog = self._static_io_catalog()
        return catalog.get(stage, {"stage": stage, "inputs": [], "outputs": [], "notes": "unknown stage"})

    def create_feature_evolution_report(self, evolution: Optional[Dict[str, Any]] = None) -> str:
        """Persist a feature evolution report as JSON and return its path."""
        payload = evolution or {
            "stages": [],
            "summary": {},
        }
        path = self.artifacts_dir / "feature_evolution.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return str(path)

    def generate_config_impact_analysis(self) -> Dict[str, Any]:
        """Analyze config knobs known to affect feature counts and performance."""
        result: Dict[str, Any] = {"impacts": []}
        try:
            if not self.config:
                return result
            # Stationarization: frac_diff d_values
            try:
                dvals = list(getattr(self.config.features, 'frac_diff', {}).get('d_values', []))
            except Exception:
                dvals = []
            result["impacts"].append({
                "stage": "stationarization",
                "parameter": "features.frac_diff.d_values",
                "effect": "Multiplies stationary variants per source column",
                "current": dvals,
            })
            # BK filter window sizes
            try:
                bk = getattr(self.config.features, 'baxter_king', {})
            except Exception:
                bk = {}
            result["impacts"].append({
                "stage": "feature_engineering",
                "parameter": "features.baxter_king",
                "effect": "Adds band-pass filtered columns for selected sources",
                "current": bk,
            })
            # Selection thresholds
            result["impacts"].append({
                "stage": "statistical_tests",
                "parameter": "features.dcor_min_threshold / stage1_top_n",
                "effect": "Controls Stage1 retention and downstream workload",
                "current": {
                    "dcor_min_threshold": getattr(self.config.features, 'dcor_min_threshold', None),
                    "stage1_top_n": getattr(self.config.features, 'stage1_top_n', None),
                },
            })
        finally:
            return result

    # ---------------------- Internals ----------------------
    def _collect_stages(self) -> List[PipelineStageInfo]:
        stages: List[PipelineStageInfo] = []
        try:
            engines = getattr(getattr(self.config, 'pipeline', None), 'engines', {}) if self.config else {}
            items = list(engines.items())
            # sort by configured order
            def _order(v):
                try:
                    return int(getattr(v, 'order', 999))
                except Exception:
                    try:
                        return int(v.get('order', 999))
                    except Exception:
                        return 999
            items.sort(key=lambda kv: _order(kv[1]))

            for name, cfg in items:
                try:
                    enabled = bool(getattr(cfg, 'enabled', True))
                    order = getattr(cfg, 'order', None)
                    desc = getattr(cfg, 'description', None)
                    params = self._to_dict(cfg)
                    # remove noise
                    for k in ["enabled", "order", "description"]:
                        params.pop(k, None)
                    stages.append(PipelineStageInfo(name=name, description=desc, order=order, enabled=enabled, parameters=params))
                except Exception:
                    stages.append(PipelineStageInfo(name=name))
        except Exception:
            pass
        return stages

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return dict(obj)
        try:
            return dict(obj.__dict__)
        except Exception:
            return {}

    def _static_io_catalog(self) -> Dict[str, Dict[str, Any]]:
        return {
            "stationarization": {
                "stage": "stationarization",
                "inputs": ["price/level columns", "rolling windows"],
                "outputs": ["fracdiff_*", "log_stabilized_*", "zscore_*", "rolling_corr_*"],
                "notes": "Applies FFD, variance stabilization, rolling metrics",
            },
            "feature_engineering": {
                "stage": "feature_engineering",
                "inputs": ["numeric sources"],
                "outputs": ["bk_* (band-pass filtered)"],
                "notes": "Baxter–King filter via FFT-conv when large kernels",
            },
            "garch_models": {
                "stage": "garch_models",
                "inputs": ["prices/log-prices"],
                "outputs": ["garch_vol_*", "garch_resid_*"],
                "notes": "Univariate GARCH volatility modeling",
            },
            "statistical_tests": {
                "stage": "statistical_tests",
                "inputs": ["feature matrix", "target"],
                "outputs": ["selected_features", "diagnostics"],
                "notes": "dCor/MI/VIF and gating logic",
            },
        }

