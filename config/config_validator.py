"""
Configuration validation helpers.

Validates the unified YAML config for structure, ranges and conflicts.
Also offers lightweight optimization suggestions and resource estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ValidationIssue:
    level: str  # ERROR|WARN|INFO
    message: str
    path: Optional[str] = None


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, level: str, message: str, path: Optional[str] = None):
        self.issues.append(ValidationIssue(level=level, message=message, path=path))
        if level.upper() == 'ERROR':
            self.is_valid = False


class ConfigValidator:
    def validate_pipeline_config(self, config: Any) -> ValidationResult:
        res = ValidationResult(is_valid=True)
        try:
            # Required top-level sections
            for section in ['database', 'r2', 'dask', 'features', 'pipeline', 'output']:
                if not hasattr(config, section):
                    res.add('ERROR', f"Missing section: {section}", path=section)

            # Pipeline engines sanity
            engines = getattr(getattr(config, 'pipeline', None), 'engines', {})
            if not engines:
                res.add('WARN', 'No pipeline.engines configured', path='pipeline.engines')
            else:
                # Check ordering and enabled flags
                seen_orders = set()
                for name, ecfg in engines.items():
                    order = getattr(ecfg, 'order', None)
                    if order is None:
                        res.add('WARN', f"Engine without explicit order: {name}", path=f'pipeline.engines.{name}.order')
                    else:
                        if order in seen_orders:
                            res.add('WARN', f"Duplicate engine order: {order}", path=f'pipeline.engines.{name}.order')
                        seen_orders.add(order)

            # Dask: avoid enabling UCX knobs with protocol tcp
            if getattr(config.dask, 'protocol', 'tcp') == 'tcp':
                for flag in ['enable_tcp_over_ucx', 'enable_infiniband', 'enable_nvlink']:
                    if getattr(config.dask, flag, False):
                        res.add('WARN', f"{flag} ignored when protocol=tcp", path=f'dask.{flag}')

            # Features: sensible thresholds
            dvals = []
            try:
                dvals = list(getattr(config.features, 'frac_diff', {}).get('d_values', []))
            except Exception:
                pass
            if any((d < 0.0 or d > 1.0) for d in dvals):
                res.add('ERROR', 'frac_diff.d_values must be in [0,1]', path='features.frac_diff.d_values')

        except Exception as e:
            res.add('ERROR', f'Validator crashed: {e}')
        return res

    def suggest_optimizations(self, config: Any) -> List[str]:
        suggestions: List[str] = []
        try:
            # Suggest enabling spilling if memory_limit is small
            try:
                if getattr(config.dask, 'spilling_enabled', True) is False:
                    suggestions.append('Enable dask.spilling_enabled to reduce OOM risk')
            except Exception:
                pass

            # Suggest limiting Stage1 top_n to control downstream workload
            topn = getattr(config.features, 'stage1_top_n', 0)
            if not topn or topn > 200:
                suggestions.append('Consider setting features.stage1_top_n <= 200 to limit workload')
        except Exception:
            pass
        return suggestions

    def check_parameter_conflicts(self, config: Any) -> List[str]:
        conflicts: List[str] = []
        try:
            # If force_gpu_usage is True but protocol is tcp, warn about potential CPU fallback paths
            if getattr(config.features, 'force_gpu_usage', False) and getattr(config.dask, 'protocol', 'tcp') == 'tcp':
                conflicts.append('GPU forcing enabled while Dask protocol=tcp; ensure workers run on GPU')
        except Exception:
            pass
        return conflicts

    def estimate_resource_requirements(self, config: Any) -> Dict[str, Any]:
        # Very rough, conservative estimates
        estimates = {
            'gpu_memory_gb_per_worker': 4.0,
            'expected_feature_multiplier': 1.0,
        }
        try:
            dvals = []
            try:
                dvals = list(getattr(config.features, 'frac_diff', {}).get('d_values', []))
            except Exception:
                pass
            if dvals:
                estimates['expected_feature_multiplier'] *= (1 + max(0, len(dvals) - 1))
        except Exception:
            pass
        return estimates

