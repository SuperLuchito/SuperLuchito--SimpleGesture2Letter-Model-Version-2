# Slovo Dataset Setup (Local)

Этот проект поддерживает words mode поверх датасета Slovo (isolated words).

## 1) Получение датасета

1. Перейдите в репозиторий Slovo:
   - `hukenovs/slovo` на GitHub
2. Скачайте необходимые файлы датасета (видео/аннотации/landmarks) согласно инструкции Slovo.
   - Либо используйте скрипт:

```bash
./backend/scripts/download_slovo_assets.sh --with-dataset
```

3. Разместите данные локально, например:

```text
backend/data/slovo/
  annotations.csv
  videos/
  landmarks/
```

Либо автоматически распакуйте архив и соберите split:

```bash
./backend/scripts/finalize_slovo_dataset.sh
```

## 2) Построение split без утечки signer

Используйте Group split по `user_id`/`signer_id`:

```bash
python backend/scripts/prepare_slovo_splits.py \
  --annotations backend/data/slovo/annotations.csv \
  --out backend/data/slovo/splits.json \
  --path-prefix backend/data/slovo/videos \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

## 3) (Опционально) Подмножество слов для быстрого MVP

```bash
python backend/scripts/build_subset.py \
  --splits backend/data/slovo/splits.json \
  --out-dir backend/data/slovo/subset_100 \
  --top-k 100
```

## 4) Экспорт модели в ONNX

```bash
python backend/train/export_onnx.py \
  --checkpoint /path/to/model.ckpt \
  --labels backend/artifacts/labels.txt \
  --out backend/artifacts/slovo_word_model.onnx \
  --input-size 224 \
  --window-frames 32
```

## 5) Labels

Файл labels должен быть в формате "одна строка = один класс":

```text
backend/artifacts/labels.txt
```

Экспорт из официального `constants.py` Slovo:

```bash
python backend/scripts/export_slovo_labels.py \
  --constants backend/data/slovo_repo/constants.py \
  --out backend/artifacts/labels.txt
```

Для baseline `mvit32-2.onnx` класс NONE уже есть в конце словаря как `---`
(используйте `thresholds.no_event_label: '---'` в конфиге).
