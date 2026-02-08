from __future__ import annotations

from app.retrieval import GalleryEntry
from tools.eval_sanity_split import make_sanity_split


def _entries(label: str, count: int) -> list[GalleryEntry]:
    return [GalleryEntry(path=f"/tmp/{label}/{i:03d}.jpg", letter=label) for i in range(count)]


def test_make_sanity_split_balanced() -> None:
    entries = _entries("А", 10) + _entries("Б", 10)
    train, val, skipped = make_sanity_split(
        entries,
        val_ratio=0.2,
        seed=123,
        min_val_per_class=1,
        min_train_per_class=2,
    )

    assert skipped == {}
    assert len(train) == 16
    assert len(val) == 4

    val_counts = {}
    for item in val:
        val_counts[item.letter] = val_counts.get(item.letter, 0) + 1
    assert val_counts == {"А": 2, "Б": 2}


def test_make_sanity_split_skips_too_small_class() -> None:
    entries = _entries("А", 5) + _entries("Б", 1)
    train, val, skipped = make_sanity_split(
        entries,
        val_ratio=0.4,
        seed=7,
        min_val_per_class=1,
        min_train_per_class=2,
    )

    assert "Б" in skipped
    assert len(train) + len(val) == len(entries)
    assert all(item.letter == "А" for item in val)
