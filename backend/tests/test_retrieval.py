from __future__ import annotations

from pathlib import Path

import numpy as np

from app.retrieval import l2_normalize, scan_gallery


def test_l2_normalize_unit_norm() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    y = l2_normalize(x)
    norms = np.linalg.norm(y, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_scan_gallery_filters_none_and_allowlist(tmp_path: Path) -> None:
    (tmp_path / '_none').mkdir()
    (tmp_path / 'А').mkdir()
    (tmp_path / 'Б').mkdir()

    (tmp_path / '_none' / 'n.jpg').write_bytes(b'x')
    (tmp_path / 'А' / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'Б' / 'b.jpg').write_bytes(b'x')

    all_entries = scan_gallery(tmp_path, none_label_dir='_none', letters_allowlist=[])
    assert {e.letter for e in all_entries} == {'А', 'Б'}

    only_a = scan_gallery(tmp_path, none_label_dir='_none', letters_allowlist=['А'])
    assert [e.letter for e in only_a] == ['А']
