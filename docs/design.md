# Design Notes: РЖЯ Static Dactyl MVP

## Цель
Локальное распознавание статических букв дактиля РЖЯ в браузере в реальном времени c `NONE`-состоянием и `hold-to-commit`.

## Почему retrieval + VLM judge
- Retrieval (DINOv2 + FAISS cosine) работает быстро и стабильно на потоке кадров.
- VLM вызывается только в спорных моментах (`uncertain`/`precommit`), что уменьшает задержку и стоимость inference.
- Такая схема снижает "дребезг" и риск ложных символов в итоговой строке.

## Ключевые решения
- `FastAPI + WebSocket`: браузер отправляет JPEG-кадры, backend отвечает JSON-состоянием.
- `MediaPipe Hand Landmarker`: bbox руки и фильтр `min_bbox_area` для режима `NONE`.
- `DINOv2 embeddings`: извлечение robust-фич без fine-tune.
- `FAISS IndexFlatIP` + L2-нормализация: cosine similarity retrieval.
- `LM Studio /v1/chat/completions`: VLM judge через collage с одним изображением.

## Форматы
- WS input: binary JPEG кадр, либо text JSON control (`clear_text`).
- WS output: `InferenceMessage` с полями состояния, hold-таймером, top-k и VLM verdict.
- VLM prompt output: строго JSON `{letter, confidence, reason}`.

## Источники
- MediaPipe Hand Landmarker Python guide:
  - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- FastAPI WebSockets:
  - https://fastapi.tiangolo.com/advanced/websockets/
- LM Studio OpenAI-compatible API:
  - https://lmstudio.ai/docs/developer/openai-compat
- LM Studio changelog (поддержка image_url):
  - https://lmstudio.ai/docs/developer/changelog
- OpenAI Chat messages/content parts:
  - https://platform.openai.com/docs/api-reference/chat/create
- FAISS cosine (MetricType):
  - https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
- DINOv2:
  - https://huggingface.co/docs/transformers/model_doc/dinov2
  - https://github.com/facebookresearch/dinov2
- Qwen3-VL-4B-Instruct:
  - https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- getUserMedia secure context:
  - https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
