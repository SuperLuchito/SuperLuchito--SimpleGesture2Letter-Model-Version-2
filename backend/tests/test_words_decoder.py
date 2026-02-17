from __future__ import annotations

import numpy as np

from app.words.decoder import WordDecisionDecoder, WordThresholds


def test_decoder_hold_commit_cooldown_flow() -> None:
    labels = ["no_event", "HELLO", "THANKS"]
    dec = WordDecisionDecoder(
        ema_alpha=1.0,
        thresholds=WordThresholds(th_no_event=0.6, th_unknown=0.55, th_margin=0.1),
        hold_frames=2,
        cooldown_frames=2,
        dedup_same_word=True,
    )

    probs_none = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    r0 = dec.update(probs=probs_none, labels=labels, topk=3, no_event_index=0)
    assert r0.state == "NONE"

    probs_word = np.array([0.05, 0.8, 0.15], dtype=np.float32)
    r1 = dec.update(probs=probs_word, labels=labels, topk=3, no_event_index=0)
    assert r1.state == "HOLD"
    assert r1.hold_count == 1

    r2 = dec.update(probs=probs_word, labels=labels, topk=3, no_event_index=0)
    assert r2.state == "COMMIT"
    assert r2.committed_word == "HELLO"

    r3 = dec.update(probs=probs_word, labels=labels, topk=3, no_event_index=0)
    assert r3.state == "COOLDOWN"


def test_decoder_unknown_by_margin() -> None:
    labels = ["no_event", "HELLO", "THANKS"]
    dec = WordDecisionDecoder(
        ema_alpha=1.0,
        thresholds=WordThresholds(th_no_event=0.6, th_unknown=0.55, th_margin=0.15),
        hold_frames=2,
        cooldown_frames=0,
        dedup_same_word=False,
    )

    probs = np.array([0.1, 0.52, 0.38], dtype=np.float32)
    r = dec.update(probs=probs, labels=labels, topk=3, no_event_index=0)
    assert r.state == "UNKNOWN"
