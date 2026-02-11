# РЖЯ Dactyl MVP v1

Первая стабильно рабочая версия локального MVP для распознавания статических букв дактиля РЖЯ в реальном времени с веб-камеры в браузере.

Ключевая идея: быстрый `retrieval` (MediaPipe + DINOv2 + FAISS) и опциональный VLM-судья через LM Studio для спорных случаев.

Важно: **рост качества распознавания напрямую зависит от качества и разнообразия вашего датасета эталонов**.

## Текущий статус v1

- Стабильный режим: `retrieval-only` (VLM выключен по умолчанию).
- `NONE` режим работает (нет руки/низкая уверенность).
- `hold-to-commit` и `cooldown` работают.
- UI показывает текущий статус, текущую букву, hold-прогресс, debug top-k.
- Поддержан сбор эталонов, сборка индекса, калибровка порогов.

## Стек

- Backend: FastAPI + WebSocket
- Hand detection: MediaPipe Hand Landmarker
- Embeddings: DINOv2 (`facebook/dinov2-small`)
- Retrieval: FAISS `IndexFlatIP` + L2-normalize (cosine)
- Optional judge: Qwen3-VL через LM Studio OpenAI-compatible API
- Frontend: Vanilla JS + Canvas

## Структура проекта

```text
backend/
  app/
  tools/
  tests/
  config.yaml
  requirements.txt
  gallery/
  artifacts/
frontend/
  index.html
  app.js
  style.css
docs/
README.md
agents.md
LICENSE
```

## Требования

- macOS
- Python 3.12 (или совместимый 3.10+)
- LM Studio (опционально, если хотите VLM)

## Установка

```bash
cd '/Users/luchito/Desktop/РЖЯ ТЕСТ#2'
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Быстрый старт (рекомендуется)

### 1. Сбор эталонов

Пример для букв:

```bash
python backend/tools/capture_gallery.py --label А --count 10 --auto
python backend/tools/capture_gallery.py --label Б --count 10 --auto
```

Пример съёмки с зеркалом:

```bash
python backend/tools/capture_gallery.py --label А --count 40 --auto --mirror
```

Для фона/none:

```bash
python backend/tools/capture_gallery.py --label _none --count 30 --auto
```

### Как работает дедупликация кадров (capture_gallery)

В проекте используется 3-ступенчатый дедуп:

- `dHash` (быстрый фильтр): отсекает почти идентичные кадры мгновенно и дёшево.
- `DINOv2 cosine` (основной критерий): сравнивает смысловые эмбеддинги руки, чтобы не хранить почти одинаковые позы при небольших пиксельных отличиях.
- `SSIM` (tie-break в пограничной зоне): включается только когда cosine близок к порогу, чтобы не удалить полезное разнообразие слишком агрессивно.

Настройки порогов:

```bash
python backend/tools/capture_gallery.py \
  --label А --count 40 --auto --mirror \
  --dedup-hamming-th 2 \
  --dedup-cosine-th 0.995 \
  --dedup-cosine-margin 0.004 \
  --dedup-ssim-th 0.985
```

### 2. Сборка индекса

```bash
python backend/tools/build_index.py
```

Сборка индекса + sanity-eval одним запуском:

```bash
python backend/tools/build_index.py --batch-size 16 --sanity-split 0.2 --k 5
```

Отдельный sanity-eval:

```bash
python backend/tools/eval_sanity_split.py --val-ratio 0.2 --k 5
```

### 3. Калибровка порогов

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python backend/tools/calibrate_thresholds.py
```

### 4. Запуск сервера

```bash
cd backend
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Открыть:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/gallery`

## Работа с VLM (опционально)

По умолчанию в `backend/config.yaml`:

```yaml
enable_vlm_judge: false
```

Это сделано для стабильного FPS/latency.

Если хотите включить VLM:

1. Поднимите LM Studio Local Server (`http://localhost:1234/v1`).
2. Загрузите модель `qwen/qwen3-vl-4b`.
3. Поставьте `enable_vlm_judge: true` в `backend/config.yaml`.
4. Перезапустите backend.

## Ключевые параметры (`backend/config.yaml`)

- `frontend_fps`, `jpeg_quality`
- `hold_ms`, `cooldown_ms`
- `sim_none`, `sim_vlm_th`, `margin_th`
- `switch_min_frames`, `precommit_ratio`
- `hand_bbox_padding`, `hand_focus_ratio`, `hand_wrist_extension_ratio`
- `hand_bg_suppression`, `hand_bg_darken_factor`
- `hand_mask_dilate_ratio`, `hand_mask_blur_sigma`

## Почему качество может расти/падать

Качество в этой версии определяется прежде всего качеством галереи:

- мало эталонов -> больше ошибок;
- однообразные эталоны -> плохая переносимость на другие углы/свет;
- рассинхрон предобработки между gallery/live -> деградация.

Рекомендация: 30-60 кадров на букву, разные небольшие вариации позы и освещения, но тот же pipeline.

## Тесты

```bash
cd backend
python -m pytest -q
```

## Troubleshooting

### OpenMP ошибка (`libomp already initialized`)

Запускать калибровку/сервер с:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 ...
```

### Камера не запускается

- Дайте браузеру доступ к камере.
- Используйте `localhost`/`127.0.0.1`.

### Низкая точность

- Переснимите эталоны (больше и лучше).
- Пересоберите индекс и перекалибруйте пороги.
- Проверьте `sim_none` и `hand_*` параметры в `config.yaml`.

### Кроп обрезает ладонь/кисть

- Увеличьте `hand_bbox_padding` (например `0.20-0.28`).
- Не занижайте `hand_focus_ratio` (держите `1.0`, чтобы bbox не сжимался).
- Поднимите `hand_wrist_extension_ratio` (`0.18-0.30`), чтобы кроп расширялся в сторону запястья.

## Полезные ссылки

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [LM Studio OpenAI-compat](https://lmstudio.ai/docs/developer/openai-compat)
- [OpenAI chat image content parts](https://platform.openai.com/docs/api-reference/chat/create)
- [FAISS cosine notes](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [DINOv2 in Transformers](https://huggingface.co/docs/transformers/model_doc/dinov2)

## Лицензия

MIT, см. `LICENSE`.
