from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WordThresholds:
    th_no_event: float = 0.60
    th_unknown: float = 0.55
    th_margin: float = 0.10


@dataclass
class WordDecodeResult:
    state: str
    top1_label: str
    top1_prob: float
    top2_prob: float
    margin: float
    no_event_prob: float
    hold_count: int
    hold_target: int
    hold_progress: float
    cooldown_left: int
    committed: bool
    committed_word: str | None
    topk_indices: list[int]


class WordDecisionDecoder:
    def __init__(
        self,
        *,
        ema_alpha: float,
        thresholds: WordThresholds,
        hold_frames: int,
        cooldown_frames: int,
        dedup_same_word: bool,
    ) -> None:
        self.ema_alpha = float(np.clip(float(ema_alpha), 0.0, 1.0))
        self.thresholds = thresholds
        self.hold_frames = max(1, int(hold_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.dedup_same_word = bool(dedup_same_word)

        self._smooth_probs: np.ndarray | None = None
        self._hold_count = 0
        self._cooldown_left = 0
        self._last_commit_word: str | None = None
        self._text_value = ""

    @property
    def text_value(self) -> str:
        return self._text_value

    def clear_text(self) -> None:
        self._text_value = ""

    def reset(self) -> None:
        self._smooth_probs = None
        self._hold_count = 0
        self._cooldown_left = 0

    def update(
        self,
        *,
        probs: np.ndarray,
        labels: list[str],
        topk: int,
        no_event_index: int | None,
    ) -> WordDecodeResult:
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        if p.shape[0] != len(labels):
            raise ValueError(f"Probability/label size mismatch: probs={p.shape[0]} labels={len(labels)}")

        p = np.clip(p, 0.0, 1.0)
        denom = float(p.sum())
        if denom > 0.0:
            p = p / denom

        if self._smooth_probs is None or self._smooth_probs.shape != p.shape:
            self._smooth_probs = p.copy()
        else:
            self._smooth_probs = ((1.0 - self.ema_alpha) * self._smooth_probs) + (self.ema_alpha * p)

        smooth = self._smooth_probs
        order_all = list(np.argsort(-smooth))

        no_event_prob = float(smooth[no_event_index]) if no_event_index is not None else 0.0

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            top1_idx = int(order_all[0])
            top2_idx = int(order_all[1]) if len(order_all) > 1 else top1_idx
            top1_prob = float(smooth[top1_idx])
            top2_prob = float(smooth[top2_idx])
            return self._result(
                state="COOLDOWN",
                top1_idx=top1_idx,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                margin=float(top1_prob - top2_prob),
                no_event_prob=no_event_prob,
                labels=labels,
                topk_indices=order_all[: max(1, topk)],
                committed=False,
                committed_word=None,
            )

        if no_event_index is not None and no_event_prob >= float(self.thresholds.th_no_event):
            self._hold_count = 0
            top1_idx = int(no_event_index)
            return self._result(
                state="NONE",
                top1_idx=top1_idx,
                top1_prob=float(smooth[top1_idx]),
                top2_prob=0.0,
                margin=0.0,
                no_event_prob=no_event_prob,
                labels=labels,
                topk_indices=order_all[: max(1, topk)],
                committed=False,
                committed_word=None,
            )

        non_none_order = [idx for idx in order_all if idx != no_event_index]
        if not non_none_order:
            self._hold_count = 0
            fallback_idx = int(order_all[0])
            return self._result(
                state="NONE",
                top1_idx=fallback_idx,
                top1_prob=float(smooth[fallback_idx]),
                top2_prob=0.0,
                margin=0.0,
                no_event_prob=no_event_prob,
                labels=labels,
                topk_indices=order_all[: max(1, topk)],
                committed=False,
                committed_word=None,
            )

        top1_idx = int(non_none_order[0])
        top2_idx = int(non_none_order[1]) if len(non_none_order) > 1 else top1_idx

        top1_prob = float(smooth[top1_idx])
        top2_prob = float(smooth[top2_idx])
        margin = float(top1_prob - top2_prob)
        candidate = labels[top1_idx]

        if top1_prob < float(self.thresholds.th_unknown) or margin < float(self.thresholds.th_margin):
            self._hold_count = 0
            return self._result(
                state="UNKNOWN",
                top1_idx=top1_idx,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                margin=margin,
                no_event_prob=no_event_prob,
                labels=labels,
                topk_indices=non_none_order[: max(1, topk)],
                committed=False,
                committed_word=None,
            )

        if self.dedup_same_word and self._last_commit_word == candidate:
            self._hold_count = 0
            return self._result(
                state="HOLD",
                top1_idx=top1_idx,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                margin=margin,
                no_event_prob=no_event_prob,
                labels=labels,
                topk_indices=non_none_order[: max(1, topk)],
                committed=False,
                committed_word=None,
            )

        self._hold_count += 1
        if self._hold_count >= self.hold_frames:
            self._hold_count = 0
            self._cooldown_left = self.cooldown_frames
            self._last_commit_word = candidate
            self._text_value = f"{self._text_value} {candidate}".strip()
            return self._result(
                state="COMMIT",
                top1_idx=top1_idx,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                margin=margin,
                no_event_prob=no_event_prob,
                labels=labels,
                topk_indices=non_none_order[: max(1, topk)],
                committed=True,
                committed_word=candidate,
            )

        return self._result(
            state="HOLD",
            top1_idx=top1_idx,
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            margin=margin,
            no_event_prob=no_event_prob,
            labels=labels,
            topk_indices=non_none_order[: max(1, topk)],
            committed=False,
            committed_word=None,
        )

    def _result(
        self,
        *,
        state: str,
        top1_idx: int,
        top1_prob: float,
        top2_prob: float,
        margin: float,
        no_event_prob: float,
        labels: list[str],
        topk_indices: list[int],
        committed: bool,
        committed_word: str | None,
    ) -> WordDecodeResult:
        hold_progress = min(1.0, self._hold_count / float(self.hold_frames)) if self.hold_frames > 0 else 0.0
        return WordDecodeResult(
            state=state,
            top1_label=labels[top1_idx],
            top1_prob=float(top1_prob),
            top2_prob=float(top2_prob),
            margin=float(margin),
            no_event_prob=float(no_event_prob),
            hold_count=int(self._hold_count),
            hold_target=int(self.hold_frames),
            hold_progress=float(hold_progress),
            cooldown_left=int(self._cooldown_left),
            committed=bool(committed),
            committed_word=committed_word,
            topk_indices=[int(i) for i in topk_indices],
        )
