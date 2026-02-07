from __future__ import annotations

from app.vlm_judge import parse_judge_response


def test_parse_plain_json() -> None:
    letter, conf, reason = parse_judge_response(
        '{"letter":"А","confidence":0.88,"reason":"ok"}',
        {'А', 'Б'},
    )
    assert letter == 'А'
    assert conf == 0.88
    assert reason == 'ok'


def test_parse_markdown_wrapped_json() -> None:
    letter, conf, _ = parse_judge_response(
        '```json\n{"letter":"NONE","confidence":0.32,"reason":"blur"}\n```',
        {'А', 'Б'},
    )
    assert letter == 'NONE'
    assert conf == 0.32


def test_unknown_label_becomes_none() -> None:
    letter, conf, reason = parse_judge_response(
        '{"letter":"Z","confidence":0.99,"reason":"bad"}',
        {'А', 'Б'},
    )
    assert letter == 'NONE'
    assert 0.0 <= conf <= 1.0
    assert reason
