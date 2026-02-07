from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .collage import build_collage_jpeg
from .retrieval import RetrievalHit


JSON_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


@dataclass
class JudgeResult:
    letter: str
    confidence: float
    reason: str
    raw: str
    used: bool
    trigger: str



def _data_url_from_jpeg(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"



def parse_judge_response(content: str, allowed_labels: set[str]) -> tuple[str, float, str]:
    snippet = content.strip()
    if snippet.startswith("```"):
        snippet = snippet.strip("`")
        if snippet.startswith("json"):
            snippet = snippet[4:].strip()

    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        match = JSON_PATTERN.search(content)
        if not match:
            raise ValueError("Judge response does not contain JSON object.")
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Judge response JSON must be an object.")

    letter = str(payload.get("letter", "NONE")).strip()
    reason = str(payload.get("reason", ""))
    confidence = float(payload.get("confidence", 0.0))

    if letter not in allowed_labels and letter != "NONE":
        letter = "NONE"
        reason = "VLM вернул метку вне разрешённого набора."

    confidence = max(0.0, min(1.0, confidence))
    return letter, confidence, reason


class VLMJudge:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_ms: int = 1500,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_ms = timeout_ms
        self._client = None

    def _client_or_raise(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError("openai package is required for VLM judge") from exc

        # LM Studio OpenAI-compatible API typically ignores API key but requires non-empty token in some SDKs.
        self._client = OpenAI(base_url=self.base_url, api_key="lm-studio")
        return self._client

    def health(self) -> tuple[bool, str]:
        url = f"{self.base_url}/models"
        try:
            with httpx.Client(timeout=max(1.0, self.timeout_ms / 1000.0)) as client:
                res = client.get(url)
                if res.status_code >= 400:
                    return False, f"HTTP {res.status_code}"
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def judge(
        self,
        *,
        query_rgb,
        topk_hits: list[RetrievalHit],
        allowed_labels: set[str],
        trigger: str,
    ) -> tuple[JudgeResult, bytes]:
        collage_jpeg = build_collage_jpeg(query_rgb=query_rgb, topk_hits=topk_hits)
        data_url = _data_url_from_jpeg(collage_jpeg)

        prompt = (
            "Ты судья распознавания статических букв дактиля РЖЯ. "
            "На изображении: слева QUERY, справа эталоны top-K с их labels. "
            "Выбери только один label из эталонов или NONE. "
            "Ответь строго JSON-объектом вида "
            "{\"letter\":\"...\",\"confidence\":0..1,\"reason\":\"коротко\"}. "
            "Без markdown и без лишнего текста."
        )

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "letter": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["letter", "confidence", "reason"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        request = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 120,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }

        raw_text = ""
        client = self._client_or_raise()

        # First try strict structured output. Some LM Studio versions may ignore/deny this field.
        try:
            response = client.chat.completions.create(**request, response_format=schema, timeout=self.timeout_ms / 1000.0)
            raw_text = response.choices[0].message.content or ""
        except Exception:
            response = client.chat.completions.create(**request, timeout=self.timeout_ms / 1000.0)
            raw_text = response.choices[0].message.content or ""

        letter, confidence, reason = parse_judge_response(raw_text, allowed_labels)
        result = JudgeResult(
            letter=letter,
            confidence=confidence,
            reason=reason,
            raw=raw_text,
            used=True,
            trigger=trigger,
        )
        return result, collage_jpeg
