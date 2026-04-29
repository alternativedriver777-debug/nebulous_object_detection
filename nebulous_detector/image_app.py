import os
from datetime import datetime

import cv2

from nebulous_detector.config import CONF_THRESH, OUTPUT_IMAGE_PREFIX, WEIGHTS_PATH
from nebulous_detector.detection import detect_objects, load_yolo_model
from nebulous_detector.drawing import draw_boxes
from nebulous_detector.window_capture import find_window, grab_window_frame


def main():
    print("Ищем окно Nox...")
    try:
        hwnd, bbox = find_window()
    except ModuleNotFoundError as error:
        _print_missing_dependency(error)
        return

    if hwnd is None:
        print("Окно Nox не найдено. Убедитесь, что Nox запущен и в заголовке есть 'Nox' или 'NoxPlayer'.")
        return

    print(f"Найдено окно HWND={hwnd}, bbox={bbox}")

    try:
        frame = grab_window_frame(bbox)
    except ModuleNotFoundError as error:
        _print_missing_dependency(error)
        return
    except Exception as error:
        print("Ошибка при создании скриншота окна:", error)
        return

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Файл весов {WEIGHTS_PATH} не найден в текущей папке.")
        return

    print("Загружаем модель YOLOv8 из", WEIGHTS_PATH)
    model = load_yolo_model(WEIGHTS_PATH)

    print("Выполняем детекцию...")
    boxes, classes, confs = detect_objects(model, frame, conf_thresh=CONF_THRESH)

    if len(boxes) == 0:
        print("Объекты не найдены. Сохраняем исходный скриншот без боксов.")
        annotated = frame
    else:
        annotated = draw_boxes(frame, boxes, classes, confs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{OUTPUT_IMAGE_PREFIX}_{timestamp}.png"

    cv2.imwrite(output_name, annotated)
    print(f"Сохранено: {output_name}")
    print("Готово. Скрипт завершает работу.")


def _print_missing_dependency(error):
    print(f"Не установлена зависимость: {error.name}. Выполните `pip install -r requirements.txt`.")


if __name__ == "__main__":
    main()
