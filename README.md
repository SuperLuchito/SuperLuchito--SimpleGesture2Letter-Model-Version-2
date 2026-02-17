# РЖЯ Dactyl + Words MVP

Локальный MVP для распознавания:

- `letters` mode: статические буквы дактиля (MediaPipe + DINOv2 + FAISS),
- `words` mode: isolated words по sliding-window ONNX классификатору.

Важно: **рост качества распознавания напрямую зависит от качества и разнообразия вашего датасета эталонов**.

## Текущий статус

- Стабильный режим: `retrieval-only` (VLM выключен по умолчанию).
- `NONE` режим работает (нет руки/низкая уверенность).
- `hold-to-commit` и `cooldown` работают.
- UI показывает текущий статус, текущую букву, hold-прогресс, debug top-k.
- Поддержан сбор эталонов, сборка индекса, калибровка порогов.
- Добавлен `words` mode с:
  - кольцевым буфером кадров,
  - sliding-window inference (ONNX Runtime),
  - EMA сглаживанием вероятностей,
  - статусами `NONE | UNKNOWN | HOLD | COMMIT | COOLDOWN`,
  - логированием latency и FP/minute.

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
cd /path/to/SuperLuchito--SimpleGesture2Letter-Model-Version-2
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

Сейчас используется минимальный дедуп:

- `min_interval`: не сохраняем слишком часто, чтобы успеть поменять позу.
- `pose_changed`: кадр сохраняется только если заметно изменился:
  - либо сама картинка руки (`mean abs gray diff`),
  - либо площадь руки в кадре (`bbox_area`).

Контроль качества (blur/overexposure) не блокирует сохранение: кадры разных условий тоже нужны для датасета.

Рекомендуемый запуск:

```bash
python backend/tools/capture_gallery.py \
  --label А --count 40 --auto --mirror \
  --interval-ms 1000 \
  --min-save-interval-ms 1000 \
  --min-frame-delta 8.0 \
  --min-bbox-delta 0.015
```

### Протокол съёмки датасета

Для каждой буквы снимайте не подряд, а по комбинациям условий:

- свет: `нормальный / чуть темнее / чуть ярче`
- ракурс: `фронт / левый бок / правый бок`
- расстояние: `far / mid / near`

Минимум: `2-3` кадра на каждую комбинацию.  
Это дает около `54-81` кадров на букву с хорошим покрытием реальных условий.

Для `_none` снимите `50-80` кадров в тех же условиях.

```bash
python backend/tools/capture_gallery.py \
  --label _none --count 60 --auto --mirror \
  --interval-ms 1000 \
  --min-save-interval-ms 1000 \
  --min-frame-delta 8.0
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

## Words mode (Slovo / isolated words)

### 1) Конфиг

В `backend/config.yaml`:

```yaml
recognition_mode: words
word_model:
  path: backend/artifacts/slovo_word_model.onnx
  labels_path: backend/artifacts/labels.txt
  input_size: 224
  window_frames: 32
  frame_interval: 2
  step: 4
  topk: 5
  mean: [123.675, 116.28, 103.53]
  std: [58.395, 57.12, 57.375]
  letterbox: true
  pad_value: 114
  use_hand_presence_gate: true
thresholds:
  no_event_label: "---"
  th_no_event: 0.60
  th_unknown: 0.55
  th_margin: 0.10
smoothing:
  ema_alpha: 0.3
commit_logic:
  hold_frames: 6
  cooldown_frames: 10
  dedup_same_word: true
performance:
  max_fps_inference: 0
  ort_num_threads: 1
runtime_log:
  enabled: true
  path: backend/artifacts/words_runtime.jsonl
```

### 2) Подготовка данных Slovo

Инструкции:

- `backend/scripts/download_slovo.md`

Быстрое скачивание baseline-модели (и опционально датасета):

```bash
./backend/scripts/download_slovo_assets.sh
# или с датасетом:
./backend/scripts/download_slovo_assets.sh --with-dataset
```

Экспорт `labels.txt` из официального `constants.py` Slovo:

```bash
python backend/scripts/export_slovo_labels.py \
  --constants backend/data/slovo_repo/constants.py \
  --out backend/artifacts/labels.txt
```

После загрузки `slovo.zip` можно одним запуском распаковать и собрать split:

```bash
./backend/scripts/finalize_slovo_dataset.sh
```

Групповой split без утечки signer:

```bash
python backend/scripts/prepare_slovo_splits.py \
  --annotations backend/data/slovo/annotations.csv \
  --out backend/data/slovo/splits.json \
  --path-prefix backend/data/slovo/videos \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

Опциональный subset слов:

```bash
python backend/scripts/build_subset.py \
  --splits backend/data/slovo/splits.json \
  --out-dir backend/data/slovo/subset_100 \
  --top-k 100
```

### 3) ONNX модель

Положите веса и labels:

- `backend/artifacts/slovo_word_model.onnx`
- `backend/artifacts/labels.txt`

Если нужен экспорт ONNX:

```bash
python backend/train/export_onnx.py \
  --labels backend/artifacts/labels.txt \
  --out backend/artifacts/slovo_word_model.onnx \
  --input-size 224 \
  --window-frames 32
```

Скрипт `backend/train/export_onnx.py` в этом MVP является каркасом: если у вас уже есть baseline Slovo checkpoint, замените `DummyVideoModel` на конкретную архитектуру baseline перед экспортом.

Примечание: preprocess в words mode приведен к baseline Slovo (`letterbox 224 + mean/std из config_example`).

### 4) Проверка realtime логов

```bash
python backend/tools/eval_realtime_log.py --log backend/artifacts/words_runtime.jsonl
```

Метрики:

- `fp_per_min` (для quiet-сценария, где ожидается NONE),
- `infer_latency_avg_ms`,
- `infer_latency_p95_ms`,
- `commit_rate_per_min`.

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
- `recognition_mode`: `letters | words`
- `hold_ms`, `cooldown_ms`
- `sim_none`, `sim_vlm_th`, `margin_th`
- `switch_min_frames`, `precommit_ratio`
- `hand_bbox_padding`, `hand_focus_ratio`, `hand_wrist_extension_ratio`
- `hand_bg_suppression`, `hand_bg_darken_factor`
- `hand_mask_dilate_ratio`, `hand_mask_blur_sigma`
- `hand_min_detection_confidence`, `hand_min_presence_confidence`, `hand_min_tracking_confidence`
- `word_model.*`, `thresholds.*`, `smoothing.*`, `commit_logic.*`, `performance.*`, `runtime_log.*`

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

Для `words` mode и снижения FP:

- увеличьте `thresholds.th_no_event`;
- увеличьте `thresholds.th_unknown`;
- увеличьте `thresholds.th_margin`;
- увеличьте `commit_logic.hold_frames`;
- увеличьте `commit_logic.cooldown_frames`;
- уменьшите `performance.max_fps_inference` (или увеличьте `word_model.step`) для CPU.

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
