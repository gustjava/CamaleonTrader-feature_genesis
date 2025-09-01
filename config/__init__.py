"""
Unified configuration exports for the Dynamic Stage 0 pipeline.

This module re-exports the unified configuration APIs to provide a single
entrypoint and remove legacy duplication.
"""

from .unified_config import (
    UnifiedConfig as Config,
    load_config_from_file as load_config,
    get_unified_config as get_config,
    get_unified_config as get_settings,
)

__all__ = ['Config', 'load_config', 'get_config', 'get_settings']
