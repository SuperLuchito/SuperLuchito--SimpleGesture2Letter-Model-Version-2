from __future__ import annotations

import numpy as np

from app.words.buffer import FrameRingBuffer


def test_ring_buffer_sampling_order() -> None:
    buf = FrameRingBuffer(maxlen=20)
    for i in range(10):
        frame = np.full((2, 2, 3), i, dtype=np.uint8)
        buf.append(frame, timestamp_ms=1000 + i)

    assert buf.is_ready(window_frames=4, frame_interval=2)

    clip, ts = buf.sample_clip(window_frames=4, frame_interval=2)
    values = [int(x[0, 0, 0]) for x in clip]

    assert values == [3, 5, 7, 9]
    assert ts == [1003, 1005, 1007, 1009]


def test_ring_buffer_requires_enough_frames() -> None:
    buf = FrameRingBuffer(maxlen=8)
    for i in range(3):
        buf.append(np.zeros((2, 2, 3), dtype=np.uint8), timestamp_ms=i)

    assert not buf.is_ready(window_frames=4, frame_interval=2)
