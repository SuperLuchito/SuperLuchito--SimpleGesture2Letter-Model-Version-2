# AGENTS

## Назначение

Проект распознаёт статические буквы дактиля РЖЯ в real-time через веб-камеру.

Текущий production-like режим для v1: `retrieval-only` (`enable_vlm_judge: false` по умолчанию).

## Архитектура

- Frontend (`frontend/`):
  - получает видеопоток через `getUserMedia`,
  - отправляет JPEG-кадры по WebSocket,
  - отображает статус, текущую букву, hold-индикатор и debug.

- Backend (`backend/app/`):
  - `main.py`: FastAPI app, WS endpoint, orchestration pipeline;
  - `hand_detector.py`: MediaPipe bbox/crop + подавление фона;
  - `embedding.py`: DINOv2 embedding extractor;
  - `retrieval.py`: FAISS index/load/search;
  - `state_machine.py`: hold-to-commit/cooldown/switch logic;
  - `vlm_judge.py`: опциональный VLM-judge (LM Studio);
  - `schemas.py`: WS payload schema helpers.

- Data/artifacts:
  - `backend/gallery/`: эталоны букв и `_none`;
  - `backend/artifacts/`: `faiss.index`, `meta.json`, debug events.

## Правила изменений

- Не хардкодить пороги в коде: использовать `backend/config.yaml`.
- Если меняется контракт WS, обновлять одновременно backend (`schemas.py`) и frontend (`app.js`).
- Любые изменения decision logic делать через `state_machine.py` и/или orchestrator в `main.py`.
- Поддерживать режим без VLM как основной стабильный путь.

## Как запускать

```bash
cd '/Users/luchito/Desktop/РЖЯ ТЕСТ#2'
source .venv/bin/activate
cd backend
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Тесты

```bash
cd '/Users/luchito/Desktop/РЖЯ ТЕСТ#2'
source .venv/bin/activate
python -m pytest -q backend/tests
```

## Workflow данных

1. Снять эталоны:

```bash
python backend/tools/capture_gallery.py --label <БУКВА> --count 10 --auto
python backend/tools/capture_gallery.py --label _none --count 30 --auto
```

2. Пересобрать индекс:

```bash
python backend/tools/build_index.py
```

3. Перекалибровать пороги:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python backend/tools/calibrate_thresholds.py
```

## Замена моделей

- Embedding:
  - изменить `embedding_model` в `backend/config.yaml`;
  - при необходимости адаптировать `backend/app/embedding.py`;
  - пересобрать индекс.

- VLM:
  - настроить `vlm_base_url`, `vlm_model`, `enable_vlm_judge`;
  - проверить доступность `/health` и реальное поведение latency.

## Важное замечание по качеству

Рост качества распознавания в этой версии в первую очередь зависит от качества датасета:

- больше чистых и разнообразных эталонов;
- консистентная предобработка для gallery/live;
- регулярная перекалибровка `sim_none` после расширения галереи.
