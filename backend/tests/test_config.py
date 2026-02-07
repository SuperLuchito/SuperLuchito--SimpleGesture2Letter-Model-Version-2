from __future__ import annotations

from pathlib import Path

from app.config import AppConfig, load_config, merge_config_values


def test_load_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / 'missing.yaml')
    assert isinstance(cfg, AppConfig)
    assert cfg.hold_ms == 700


def test_merge_config_values(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text('hold_ms: 100\nsim_none: 0.2\n', encoding='utf-8')

    merged = merge_config_values(path, {'sim_none': 0.45})
    assert merged.hold_ms == 100
    assert merged.sim_none == 0.45
