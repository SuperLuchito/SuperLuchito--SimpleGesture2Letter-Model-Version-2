from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HoldState:
    hold_elapsed_ms: int
    hold_remaining_ms: int
    hold_progress: float


class HoldToCommitStateMachine:
    def __init__(
        self,
        *,
        hold_ms: int,
        cooldown_ms: int,
        precommit_ratio: float,
        uncertain_streak_frames: int,
        switch_min_frames: int = 3,
    ) -> None:
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms
        self.precommit_ratio = precommit_ratio
        self.uncertain_streak_frames = uncertain_streak_frames
        self.switch_min_frames = max(1, switch_min_frames)

        self.text_value = ""
        self.candidate_letter: str | None = None
        self.candidate_started_ms: int | None = None
        self.cooldown_until_ms: int = 0
        self.uncertain_streak: int = 0
        self.pending_letter: str | None = None
        self.pending_count: int = 0

        self._vlm_called_candidate_key: str | None = None

    def clear_text(self) -> None:
        self.text_value = ""

    def in_cooldown(self, now_ms: int) -> bool:
        return now_ms < self.cooldown_until_ms

    def cooldown_left_ms(self, now_ms: int) -> int:
        return max(0, self.cooldown_until_ms - now_ms)

    def clear_candidate(self) -> None:
        self.candidate_letter = None
        self.candidate_started_ms = None
        self.uncertain_streak = 0
        self.pending_letter = None
        self.pending_count = 0
        self._vlm_called_candidate_key = None

    @property
    def candidate_key(self) -> str | None:
        if self.candidate_letter is None or self.candidate_started_ms is None:
            return None
        return f"{self.candidate_letter}:{self.candidate_started_ms}"

    def _activate_candidate(self, letter: str, now_ms: int) -> None:
        self.candidate_letter = letter
        self.candidate_started_ms = now_ms
        self.uncertain_streak = 0
        self.pending_letter = None
        self.pending_count = 0
        self._vlm_called_candidate_key = None

    def update_candidate(self, letter: str, now_ms: int) -> HoldState:
        if self.candidate_letter is None:
            self._activate_candidate(letter, now_ms)
        elif self.candidate_letter == letter:
            self.pending_letter = None
            self.pending_count = 0
        else:
            if self.pending_letter == letter:
                self.pending_count += 1
            else:
                self.pending_letter = letter
                self.pending_count = 1
            if self.pending_count >= self.switch_min_frames:
                self._activate_candidate(letter, now_ms)

        assert self.candidate_started_ms is not None
        elapsed = max(0, now_ms - self.candidate_started_ms)
        remaining = max(0, self.hold_ms - elapsed)
        progress = min(1.0, elapsed / self.hold_ms) if self.hold_ms > 0 else 0.0
        return HoldState(elapsed, remaining, progress)

    def update_uncertain(self, is_uncertain: bool) -> None:
        if is_uncertain:
            self.uncertain_streak += 1
        else:
            self.uncertain_streak = 0

    def mark_vlm_called(self) -> None:
        self._vlm_called_candidate_key = self.candidate_key

    def should_call_vlm(self, *, hold_elapsed_ms: int, is_uncertain: bool) -> tuple[bool, str]:
        if not is_uncertain:
            return False, ""

        if self.candidate_key and self.candidate_key == self._vlm_called_candidate_key:
            return False, ""

        precommit_ms = int(self.hold_ms * self.precommit_ratio)
        if hold_elapsed_ms >= precommit_ms:
            return True, "precommit"
        return False, ""

    def commit(self, letter: str, now_ms: int) -> None:
        self.text_value += letter
        self.cooldown_until_ms = now_ms + self.cooldown_ms
        self.clear_candidate()
