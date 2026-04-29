import os
from datetime import datetime

import cv2

from nebulous_detector.config import CONF_THRESH, OUTPUT_IMAGE_PREFIX, WEIGHTS_PATH
from nebulous_detector.detection import detect_objects, load_yolo_model
from nebulous_detector.drawing import draw_boxes
from nebulous_detector.window_capture import find_window, grab_window_frame


def main():
    print("Searching for the Nox window...")
    try:
        hwnd, bbox = find_window()
    except ModuleNotFoundError as error:
        _print_missing_dependency(error)
        return

    if hwnd is None:
        print("Nox window was not found. Make sure Nox is running and the window title contains 'Nox' or 'NoxPlayer'.")
        return

    print(f"Found window HWND={hwnd}, bbox={bbox}")

    try:
        frame = grab_window_frame(bbox)
    except ModuleNotFoundError as error:
        _print_missing_dependency(error)
        return
    except Exception as error:
        print("Error while creating the window screenshot:", error)
        return

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Weights file {WEIGHTS_PATH} was not found in the current folder.")
        return

    print("Loading YOLOv8 model from", WEIGHTS_PATH)
    model = load_yolo_model(WEIGHTS_PATH)

    print("Running detection...")
    boxes, classes, confs = detect_objects(model, frame, conf_thresh=CONF_THRESH)

    if len(boxes) == 0:
        print("No objects found. Saving the original screenshot without boxes.")
        annotated = frame
    else:
        annotated = draw_boxes(frame, boxes, classes, confs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{OUTPUT_IMAGE_PREFIX}_{timestamp}.png"

    cv2.imwrite(output_name, annotated)
    print(f"Saved: {output_name}")
    print("Done. The script is exiting.")


def _print_missing_dependency(error):
    print(f"Missing dependency: {error.name}. Run `pip install -r requirements.txt`.")


if __name__ == "__main__":
    main()
