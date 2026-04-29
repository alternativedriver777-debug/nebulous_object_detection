import os
import time
from datetime import datetime

from nebulous_detector.config import CONF_THRESH, FPS, OUTPUT_VIDEO_PREFIX, RECORD_SECONDS, WEIGHTS_PATH
from nebulous_detector.detection import detect_objects, load_yolo_model
from nebulous_detector.drawing import draw_boxes
from nebulous_detector.window_capture import find_window, grab_window_frame


def main():
    if not _print_device_info():
        return

    print("Ищем окно Nox...")
    try:
        hwnd, bbox = find_window()
    except ModuleNotFoundError as error:
        _print_missing_dependency(error)
        return

    if hwnd is None:
        print("Окно Nox не найдено. Убедитесь, что эмулятор запущен и содержит 'nox' в заголовке.")
        return

    print("Окно найдено:", bbox)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Файл модели {WEIGHTS_PATH} не найден.")
        return

    print("Загружаем модель YOLO...")
    model = load_yolo_model(WEIGHTS_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = f"{OUTPUT_VIDEO_PREFIX}_{timestamp}.mp4"

    try:
        import cv2
        import mss

        with mss.mss() as sct:
            frame = grab_window_frame(bbox, sct)
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video, fourcc, FPS, (width, height))

            try:
                _record_video(model, bbox, sct, video_writer, output_video)
            finally:
                video_writer.release()
    except ModuleNotFoundError as error:
        _print_missing_dependency(error)
        return

    print(f"Видео успешно сохранено: {output_video}")


def _print_device_info():
    try:
        import torch
    except ModuleNotFoundError as error:
        _print_missing_dependency(error)
        return False

    if torch.cuda.is_available():
        print(f"GPU найден: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU не найден. Детекция будет выполняться на CPU и может быть медленной.")

    return True


def _print_missing_dependency(error):
    print(f"Не установлена зависимость: {error.name}. Выполните `pip install -r requirements.txt`.")


def _record_video(model, bbox, sct, video_writer, output_video):
    print(f"Начало записи: {output_video}")
    print(f"Длительность: {RECORD_SECONDS} сек при {FPS} FPS")
    print("Нажмите Ctrl+C для досрочной остановки.")

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

            frame = grab_window_frame(bbox, sct)
            boxes, classes, confs = detect_objects(model, frame, conf_thresh=CONF_THRESH, imgsz=640)

            if len(boxes) > 0:
                frame = draw_boxes(frame, boxes, classes, confs)

            current_frame = frame.copy()

            while current_time >= next_frame_time:
                if current_frame is not None:
                    video_writer.write(current_frame)

                next_frame_time += frame_interval

            time.sleep(max(0, next_frame_time - current_time))

    except KeyboardInterrupt:
        print("\nЗапись остановлена пользователем.")


if __name__ == "__main__":
    main()
