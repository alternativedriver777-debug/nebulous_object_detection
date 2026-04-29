# Nebulous YOLO Detector

<p align="center">
  <img src="https://github.com/user-attachments/assets/22526dea-7aef-4f1b-b477-4fbb6e9e6ce6" width="600">
</p>

Проект для детекции объектов в окне эмулятора Nox/NoxPlayer с помощью Ultralytics YOLOv8. Программа умеет делать скриншот с bounding boxes распознанных объектов и записывать короткое видео с детекцией в реальном времени. 

Модель можно дообучать и расширять для своих целей.

Веса модели можно скачать здесь:

https://drive.google.com/file/d/1a6VgKJIkP52Mtc_cmnrrTojxvobxQ0TT/view?usp=drivesdk

## Структура проекта

- `nebulous_detector/` - основной пакет с общей логикой.
- `main.py` - совместимый entrypoint для разовой детекции.
- `videomain.py` - совместимый entrypoint для записи видео с детекцией.
- `train_yolo.py` - CLI для запуска обучения YOLO.
- `requirements.txt` - зависимости проекта.

Внутри пакета:

- `config.py` - пути, пороги, классы, цвета, FPS и длительность записи.
- `window_capture.py` - поиск окна Nox и захват кадра.
- `detection.py` - загрузка YOLO и извлечение результатов детекции.
- `drawing.py` - отрисовка bounding boxes и подписей.
- `image_app.py` - сценарий разовой детекции.
- `video_app.py` - сценарий записи видео.
- `training.py` - функция обучения модели.

## Требования

- Windows.
- Python 3.10+.
- Запущенный Nox.
- Файл весов YOLO `best.pt` в корне проекта.
- Для ускорения желательно использовать NVIDIA GPU с CUDA-совместимым PyTorch.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

После установки положите файл модели `best.pt` в корень проекта.

## Быстрый запуск

Разовая детекция:

```bash
python main.py
```

Также можно запустить модуль напрямую:

```bash
python -m nebulous_detector.image_app
```

После запуска появится файл вида:

```text
detection_output_20260427_153000.png
```

Запись видео с детекцией:

```bash
python videomain.py
```

Также можно запустить модуль напрямую:

```bash
python -m nebulous_detector.video_app
```

После запуска появится файл вида:

```text
detection_video_20260427_153000.mp4
```

## Настройка параметров

Основные параметры находятся в `nebulous_detector/config.py`:

```python
WEIGHTS_PATH = "best.pt"
CONF_THRESH = 0.25
FPS = 50
RECORD_SECONDS = 30
CLASS_NAMES = ["sphere", "plasma", "rainbow", "warning", "blob"]
SEARCH_WINDOW_KEYWORDS = ["nox", "noxplayer"]
```

Если окно Nox называется иначе, добавьте нужное слово в `SEARCH_WINDOW_KEYWORDS`.

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

Результаты обучения будут сохранены в `runs/detect`.

## Возможные проблемы

Если окно Nox не найдено, убедитесь, что эмулятор запущен, а в заголовке окна есть `Nox` или `NoxPlayer`.

Если модель не загружается, проверьте, что `best.pt` лежит в корне проекта. При необходимости измените `WEIGHTS_PATH` в `nebulous_detector/config.py`.

Если видео записывается медленно, уменьшите `FPS`, уменьшите размер входа `imgsz` в `nebulous_detector/video_app.py` или используйте GPU.

