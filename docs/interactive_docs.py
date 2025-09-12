"""
Interactive documentation helpers.

Generates explanations, examples and troubleshooting snippets based on
known pipeline stages and configurations. Outputs text/JSON for easy embedding.
"""

from __future__ import annotations

from typing import Dict, Any, List


class InteractiveDocumentation:
    def generate_pipeline_explanation(self, stage: str) -> str:
        mapping = {
            'stationarization': 'Transforms non-stationary series to stationary via FFD, z-scores and variance stabilization.',
            'feature_engineering': 'Applies advanced transforms like Baxter–King band-pass filtering.',
            'garch_models': 'Fits univariate GARCH models to model conditional volatility.',
            'statistical_tests': 'Computes dCor/MI/VIF and applies selection thresholds.',
        }
        return mapping.get(stage, f'Unknown stage: {stage}')

    def show_config_examples(self, use_case: str) -> Dict[str, Any]:
        if use_case == 'high_transparency':
            return {
                'features': {
                    'stage1_broadcast_scores': True,
                    'debug_write_artifacts': True,
                }
            }
        return {'note': 'No specific example for this use case'}

    def demonstrate_feature_impact(self, param: str, values: List[Any]) -> Dict[str, Any]:
        return {
            'parameter': param,
            'values': values,
            'expected_effect': 'May change feature counts and selection thresholds',
        }

    def create_troubleshooting_guide(self, error_type: str) -> str:
        guides = {
            'oom': 'Reduce batch sizes, enable spilling, or lower stage1_top_n.',
            'no_features': 'Check deny lists and thresholds; ensure source columns exist.',
            'cluster_fail': 'Verify CUDA drivers, RMM config and dashboard logs.',
        }
        return guides.get(error_type, 'Collect logs and verify configuration / inputs.')

