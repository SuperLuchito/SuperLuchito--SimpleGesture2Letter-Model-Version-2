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


def test_nested_words_config_parsing(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        """
recognition_mode: words
word_model:
  path: backend/artifacts/slovo_word_model.onnx
  labels_path: backend/artifacts/labels.txt
  input_size: 224
  window_frames: 32
thresholds:
  no_event_label: no_event
  th_no_event: 0.61
  th_unknown: 0.57
  th_margin: 0.12
""".strip(),
        encoding='utf-8',
    )
    cfg = load_config(path)
    assert cfg.recognition_mode == 'words'
    assert cfg.word_input_size == 224
    assert cfg.word_window_frames == 32
    assert cfg.word_th_no_event == 0.61
    assert cfg.word_th_unknown == 0.57
    assert cfg.word_th_margin == 0.12
