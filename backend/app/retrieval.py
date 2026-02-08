from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - tested via fallback path
    faiss = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class RetrievalHit:
    index: int
    letter: str
    score: float
    exemplar_path: str


@dataclass
class GalleryEntry:
    path: str
    letter: str



def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def scan_gallery(
    gallery_dir: Path,
    none_label_dir: str = "_none",
    letters_allowlist: list[str] | None = None,
) -> list[GalleryEntry]:
    if not gallery_dir.exists():
        return []

    allow = set(letters_allowlist or [])
    use_allow = len(allow) > 0
    entries: list[GalleryEntry] = []

    for label_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label == none_label_dir:
            continue
        if use_allow and label not in allow:
            continue

        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                entries.append(GalleryEntry(path=str(path.resolve()), letter=label))

    return entries


class GalleryIndex:
    def __init__(self, index: Any, metadata: list[dict[str, Any]]) -> None:
        self.index = index
        self.metadata = metadata

    @property
    def size(self) -> int:
        return len(self.metadata)

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "GalleryIndex":
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        if faiss is None:
            raise ImportError("faiss-cpu is required to load retrieval index.")

        index = faiss.read_index(str(index_path))
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, list):
            raise ValueError("meta.json must contain a list of entries.")
        return cls(index=index, metadata=metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[RetrievalHit]:
        if self.size == 0:
            return []
        if faiss is None:
            raise ImportError("faiss-cpu is required for retrieval search.")

        query = query_embedding.astype(np.float32)
        query = l2_normalize(query)
        scores, indices = self.index.search(query, min(k, self.size))
        result: list[RetrievalHit] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            result.append(
                RetrievalHit(
                    index=int(idx),
                    letter=str(meta["letter"]),
                    score=float(score),
                    exemplar_path=str(meta["path"]),
                )
            )
        return result



def save_metadata(meta_path: Path, metadata: list[dict[str, Any]]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)



def build_faiss_index(vectors: np.ndarray):
    if faiss is None:
        raise ImportError("faiss-cpu is required to build retrieval index.")
    vectors = vectors.astype(np.float32)
    vectors = l2_normalize(vectors)
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index
