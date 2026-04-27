import os
import time
from datetime import datetime

import cv2
import mss
import numpy as np
import torch
import win32gui
from ultralytics import YOLO


WEIGHTS_PATH = "best.pt"
OUTPUT_VIDEO_PREFIX = "detection_video"
CONF_THRESH = 0.25
FPS = 50
RECORD_SECONDS = 30
CLASS_NAMES = ["sphere", "plasma", "rainbow", "warning", "blob"]
CLASS_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (0, 165, 255),
    (0, 0, 255),
    (255, 255, 0),
]
SEARCH_WINDOW_KEYWORDS = ["nox", "noxplayer"]


def find_nox_window():
    matches = []

    def enum_callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd)
        if not title:
            return

        title_lower = title.lower()
        for keyword in SEARCH_WINDOW_KEYWORDS:
            if keyword in title_lower:
                matches.append(hwnd)
                break

    win32gui.EnumWindows(enum_callback, None)

    if not matches:
        return None, None

    hwnd = matches[0]
    return hwnd, win32gui.GetWindowRect(hwnd)


def grab_window(bbox, sct):
    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    monitor = {"left": left, "top": top, "width": width, "height": height}
    image = np.array(sct.grab(monitor))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def draw_boxes(image, boxes, classes, confs):
    for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
        cls_index = int(cls)
        color = CLASS_COLORS[cls_index]

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{CLASS_NAMES[cls_index]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(image, (x1, y1 - text_height - 12), (x1 + text_width + 12, y1), color, -1)

        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(
            image,
            label,
            (x1 + 6, y1 - 6),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return image


def main():
    if torch.cuda.is_available():
        print(f"✅ GPU найден: {torch.cuda.get_device_name(0)}")
        print("   TensorRT будет использован автоматически (первый запуск может занять 1-5 мин)")
    else:
        print("❌ GPU не найден — работа будет на CPU, медленно.")

    print("🔍 Ищем окно Nox...")
    hwnd, bbox = find_nox_window()

    if hwnd is None:
        print("❌ Окно Nox не найдено. Убедитесь, что эмулятор запущен и содержит 'nox' в заголовке.")
        return

    print("✅ Окно найдено:", bbox)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Файл модели {WEIGHTS_PATH} не найден")
        return

    print("🧠 Загружаем модель YOLO (с автоматическим TensorRT на GPU)...")
    model = YOLO(WEIGHTS_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = f"{OUTPUT_VIDEO_PREFIX}_{timestamp}.mp4"

    with mss.mss() as sct:
        frame = grab_window(bbox, sct)
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video, fourcc, FPS, (width, height))

        print(f"🎥 Начало записи: {output_video}")
        print(f"⏱ Длительность: {RECORD_SECONDS} сек при {FPS} FPS")
        print("Нажмите Ctrl+C для досрочной остановки")

        start_time = time.time()
        next_frame_time = start_time
        frame_interval = 1.0 / FPS

        try:
            current_frame = None

            while True:
                current_time = time.time()
                elapsed = current_time - start_time

                if elapsed >= RECORD_SECONDS:
                    break

                frame = grab_window(bbox, sct)
                results = model(frame, conf=CONF_THRESH, verbose=False, imgsz=640)[0]

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    frame = draw_boxes(frame, boxes, classes, confs)

                current_frame = frame.copy()

                while current_time >= next_frame_time:
                    if current_frame is not None:
                        video_writer.write(current_frame)

                    next_frame_time += frame_interval

                time.sleep(max(0, next_frame_time - current_time))

        except KeyboardInterrupt:
            print("\n⏹ Запись остановлена пользователем")
        finally:
            video_writer.release()

    print(f"✅ Видео успешно сохранено: {output_video}")
    print("   Последующие запуски будут значительно быстрее благодаря кэшированному TensorRT!")


if __name__ == "__main__":
    main()
