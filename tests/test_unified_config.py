import os
from pathlib import Path

import pytest

from config.unified_config import load_config_from_file, get_unified_config, UnifiedConfig


def test_load_unified_config_from_yaml():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    assert cfg_path.exists(), "config/config.yaml must exist for tests"

    cfg: UnifiedConfig = load_config_from_file(str(cfg_path))
    assert isinstance(cfg, UnifiedConfig)

    # Basic fields
    assert cfg.database.host is not None
    assert isinstance(cfg.features.rolling_windows, list)
    assert cfg.output.output_path

    # Validation should pass
    assert cfg.validate() is True


def test_get_unified_config_cached_singleton():
    c1 = get_unified_config()
    c2 = get_unified_config()
    assert c1 is c2, "get_unified_config() should return a cached singleton instance"

