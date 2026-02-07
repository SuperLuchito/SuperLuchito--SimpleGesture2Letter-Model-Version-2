from __future__ import annotations

from app.state_machine import HoldToCommitStateMachine


def test_hold_and_commit_flow() -> None:
    sm = HoldToCommitStateMachine(
        hold_ms=100,
        cooldown_ms=50,
        precommit_ratio=0.8,
        uncertain_streak_frames=3,
    )

    h0 = sm.update_candidate('А', 1000)
    assert h0.hold_elapsed_ms == 0

    h1 = sm.update_candidate('А', 1080)
    assert h1.hold_elapsed_ms == 80

    sm.commit('А', 1100)
    assert sm.text_value == 'А'
    assert sm.in_cooldown(1120)
    assert not sm.in_cooldown(1160)


def test_candidate_switch_resets_timer() -> None:
    sm = HoldToCommitStateMachine(
        hold_ms=200,
        cooldown_ms=50,
        precommit_ratio=0.8,
        uncertain_streak_frames=3,
        switch_min_frames=1,
    )

    sm.update_candidate('А', 1000)
    h = sm.update_candidate('Б', 1050)
    assert h.hold_elapsed_ms == 0


def test_uncertain_trigger_logic() -> None:
    sm = HoldToCommitStateMachine(
        hold_ms=100,
        cooldown_ms=50,
        precommit_ratio=0.8,
        uncertain_streak_frames=2,
    )
    sm.update_candidate('А', 1000)

    sm.update_uncertain(True)
    should, trigger = sm.should_call_vlm(hold_elapsed_ms=20, is_uncertain=True)
    assert not should

    sm.update_uncertain(True)
    should, trigger = sm.should_call_vlm(hold_elapsed_ms=85, is_uncertain=True)
    assert should
    assert trigger == 'precommit'
