import os
from datetime import datetime

import cv2
import mss
import numpy as np
import win32gui
from ultralytics import YOLO


WEIGHTS_PATH = "best.pt"
OUTPUT_BASENAME = "detection_output"
CONF_THRESH = 0.25
CLASS_NAMES = ["sphere", "plasma", "rainbow", "warning", "blob"]
SEARCH_WINDOW_KEYWORDS = ["nox", "noxplayer"]


def find_nox_window():
    matches = []

    def enum_window_callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd)
        if not title:
            return

        title_lower = title.lower()
        for keyword in SEARCH_WINDOW_KEYWORDS:
            if keyword in title_lower:
                matches.append((hwnd, title))
                break

    win32gui.EnumWindows(enum_window_callback, None)

    if not matches:
        return None, None

    hwnd, _ = matches[0]
    return hwnd, win32gui.GetWindowRect(hwnd)


def grab_window_bbox(bbox):
    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        image = sct.grab(monitor)
        image_array = np.array(image)
        return cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)


def draw_boxes_on_image(img_bgr, boxes_xyxy, classes, confs, class_names):
    image = img_bgr.copy()

    for (x1, y1, x2, y2), cls, conf in zip(boxes_xyxy, classes, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls_index = int(cls)

        if 0 <= cls_index < len(class_names):
            label = f"{class_names[cls_index]} {conf:.2f}"
        else:
            label = f"{cls_index} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 6), (x1 + text_width + 6, y1), (0, 255, 0), -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def main():
    print("Ищем окно Nox...")
    hwnd, bbox = find_nox_window()

    if hwnd is None:
        print("Окно Nox не найдено. Убедись, что Nox запущен и в заголовке есть 'Nox' или 'NoxPlayer'.")
        return

    print(f"Найдено окно HWND={hwnd}, bbox={bbox}")

    try:
        frame = grab_window_bbox(bbox)
    except Exception as error:
        print("Ошибка при скриншоте окна:", error)
        return

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Файл весов {WEIGHTS_PATH} не найден в текущей папке.")
        return

    print("Загружаем модель YOLOv8 из", WEIGHTS_PATH)
    model = YOLO(WEIGHTS_PATH)

    print("Выполняем детекцию...")
    results = model(frame, conf=CONF_THRESH, verbose=False)
    result = results[0]

    if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        print("Объекты не найдены. Сохраняем исходный скриншот без боксов.")
        annotated = frame
    else:
        try:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
        except Exception:
            boxes_xyxy = np.array(result.boxes.xyxy)
            classes = np.array(result.boxes.cls)
            confs = np.array(result.boxes.conf)

        mask = confs >= CONF_THRESH
        boxes_xyxy = boxes_xyxy[mask]
        classes = classes[mask]
        confs = confs[mask]

        annotated = draw_boxes_on_image(frame, boxes_xyxy, classes, confs, CLASS_NAMES)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{OUTPUT_BASENAME}_{timestamp}.png"

    cv2.imwrite(output_name, annotated)
    print(f"Сохранено: {output_name}")
    print("Готово. Скрипт завершает работу.")


if __name__ == "__main__":
    main()
