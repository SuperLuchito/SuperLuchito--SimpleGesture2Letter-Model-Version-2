from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LatencySummary:
    count: int
    avg_ms: float
    p95_ms: float


class WordRuntimeMetrics:
    def __init__(self) -> None:
        self._latencies_ms: list[float] = []
        self._started_ms: int | None = None
        self._last_ms: int | None = None
        self._commit_count = 0

    def record_inference(self, latency_ms: float) -> None:
        if latency_ms < 0:
            return
        self._latencies_ms.append(float(latency_ms))

    def record_state(self, *, timestamp_ms: int, state: str, committed_word: str | None = None) -> None:
        ts = int(timestamp_ms)
        if self._started_ms is None:
            self._started_ms = ts
        self._last_ms = ts
        if state == "COMMIT" and committed_word:
            self._commit_count += 1

    def latency_summary(self) -> LatencySummary:
        if not self._latencies_ms:
            return LatencySummary(count=0, avg_ms=0.0, p95_ms=0.0)
        arr = np.asarray(self._latencies_ms, dtype=np.float32)
        return LatencySummary(
            count=int(arr.size),
            avg_ms=float(arr.mean()),
            p95_ms=float(np.percentile(arr, 95)),
        )

    def fp_per_minute(self) -> float:
        # In quiet evaluation mode each COMMIT is treated as false positive.
        if self._started_ms is None or self._last_ms is None:
            return 0.0
        duration_min = max(1e-6, (self._last_ms - self._started_ms) / 60000.0)
        return float(self._commit_count / duration_min)


def compute_fp_per_minute(events: Iterable[dict]) -> float:
    sorted_events = sorted(events, key=lambda x: int(x.get("timestamp_ms", 0)))
    if not sorted_events:
        return 0.0
    start = int(sorted_events[0].get("timestamp_ms", 0))
    end = int(sorted_events[-1].get("timestamp_ms", 0))
    commits = 0
    for event in sorted_events:
        state = str(event.get("state", "")).upper()
        word = str(event.get("word", "")).strip()
        if state == "COMMIT" and word and word not in {"NONE", "UNKNOWN", "_NO_EVENT", "NO_EVENT"}:
            commits += 1
    duration_min = max(1e-6, (end - start) / 60000.0)
    return float(commits / duration_min)
