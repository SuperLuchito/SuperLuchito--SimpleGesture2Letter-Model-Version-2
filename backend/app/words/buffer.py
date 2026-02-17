from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class BufferedFrame:
    timestamp_ms: int
    frame_bgr: np.ndarray


class FrameRingBuffer:
    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError("FrameRingBuffer maxlen must be >= 1")
        self._frames: deque[BufferedFrame] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def maxlen(self) -> int:
        return int(self._frames.maxlen or 0)

    def clear(self) -> None:
        self._frames.clear()

    def append(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        # Keep a copy to avoid accidental mutation outside the buffer.
        self._frames.append(BufferedFrame(timestamp_ms=int(timestamp_ms), frame_bgr=frame_bgr.copy()))

    def is_ready(self, *, window_frames: int, frame_interval: int) -> bool:
        if window_frames < 1:
            return False
        if frame_interval < 1:
            return False
        required = (int(window_frames) - 1) * int(frame_interval) + 1
        return len(self._frames) >= required

    def sample_clip(self, *, window_frames: int, frame_interval: int) -> tuple[list[np.ndarray], list[int]]:
        if not self.is_ready(window_frames=window_frames, frame_interval=frame_interval):
            raise ValueError("Not enough frames in buffer for requested window/interval.")

        frames = list(self._frames)
        end_idx = len(frames) - 1
        start_idx = end_idx - ((window_frames - 1) * frame_interval)
        indices = [start_idx + (i * frame_interval) for i in range(window_frames)]

        clip = [frames[idx].frame_bgr.copy() for idx in indices]
        timestamps = [int(frames[idx].timestamp_ms) for idx in indices]
        return clip, timestamps

    def latest_timestamp_ms(self) -> int | None:
        if not self._frames:
            return None
        return int(self._frames[-1].timestamp_ms)
