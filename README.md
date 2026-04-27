# Nox YOLO Detector

Проект для детекции объектов в окне эмулятора Nox с помощью Ultralytics YOLOv8. Скрипты захватывают окно Nox, запускают YOLO-модель `best.pt`, рисуют bounding boxes поверх найденных объектов и сохраняют результат в изображение или видео.

Download weights:

https://drive.google.com/drive/folders/1ToaxsRSJbxXyKCuzHno5R9TGPkG9NlDK?usp=sharing

## Что внутри проекта

- `main.py` — разовая детекция: находит окно Nox, делает один скриншот, прогоняет его через YOLO и сохраняет `detection_output_YYYYmmdd_HHMMSS.png`.
- `videomain.py` — запись видео с детекцией: захватывает окно Nox в течение заданного времени и сохраняет `detection_video_YYYYmmdd_HHMMSS.mp4`.
- `train_yolo.py` — запуск обучения YOLO-модели через Ultralytics.
- `requirements.txt` — зависимости проекта.

## Требования

- Windows.
- Python 3.10+.
- Запущенный Nox/NoxPlayer.
- Файл весов YOLO `best.pt` в корне проекта.
- Для ускорения желательно использовать NVIDIA GPU с CUDA-совместимым PyTorch.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

После установки положите файл модели `best.pt` рядом со скриптами.

## Быстрый запуск

Разовая детекция:

```bash
python main.py
```

После запуска появится файл вида:

```text
detection_output_20260427_153000.png
```

Запись видео с детекцией:

```bash
python videomain.py
```

После запуска появится файл вида:

```text
detection_video_20260427_153000.mp4
```

## Настройка параметров

Основные параметры находятся в начале `main.py` и `videomain.py`:

```python
WEIGHTS_PATH = "best.pt"
CONF_THRESH = 0.25
CLASS_NAMES = ["sphere", "plasma", "rainbow", "warning", "blob"]
SEARCH_WINDOW_KEYWORDS = ["nox", "noxplayer"]
```

В `videomain.py` дополнительно можно менять:

```python
FPS = 50
RECORD_SECONDS = 30
```

## Обучение модели

Создайте `data.yaml` в формате Ultralytics YOLO:

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 5
names: ["sphere", "plasma", "rainbow", "warning", "blob"]
```

Запуск обучения:

```bash
python train_yolo.py --data data.yaml --epochs 50 --batch 16 --model yolov8n.pt
```

Результаты обучения будут сохранены в папке `runs/detect`.

## Возможные проблемы

Если окно Nox не найдено, убедитесь, что эмулятор запущен, а в заголовке окна есть `Nox` или `NoxPlayer`.

Если модель не загружается, проверьте, что `best.pt` лежит в корне проекта и называется именно так, либо измените `WEIGHTS_PATH`.

Если видео записывается медленно, уменьшите `FPS`, снизьте `imgsz` в `videomain.py` или используйте GPU.
