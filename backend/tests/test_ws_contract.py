from __future__ import annotations

from app.schemas import TopKItem, VLMDecision, build_inference_message


def test_inference_message_shape() -> None:
    payload = build_inference_message(
        status='CANDIDATE',
        letter='А',
        score=0.77,
        confidence=0.77,
        hand_present=True,
        bbox_norm=[0.1, 0.2, 0.3, 0.4],
        hold_elapsed_ms=350,
        hold_target_ms=700,
        text_value='А',
        committed_now=False,
        topk=[TopKItem(letter='А', score=0.77, exemplar_path='/tmp/a.jpg')],
        vlm=VLMDecision(used=False),
        sim1=0.77,
        sim2=0.61,
        margin=0.16,
        uncertain=False,
        cooldown_left_ms=0,
    )

    assert payload['status'] == 'CANDIDATE'
    assert payload['hold']['remaining_ms'] == 350
    assert payload['text_state']['value'] == 'А'
    assert isinstance(payload['topk'], list)
    assert 'debug' in payload
