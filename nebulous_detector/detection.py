import numpy as np

from nebulous_detector.config import CONF_THRESH


def load_yolo_model(weights_path):
    from ultralytics import YOLO

    return YOLO(weights_path)


def detect_objects(model, frame, conf_thresh=CONF_THRESH, imgsz=None):
    predict_kwargs = {"conf": conf_thresh, "verbose": False}
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    result = model(frame, **predict_kwargs)[0]
    return extract_detections(result, conf_thresh)


def extract_detections(result, conf_thresh=CONF_THRESH):
    if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return _empty_detections()

    boxes_xyxy = _to_numpy(result.boxes.xyxy)
    classes = _to_numpy(result.boxes.cls)
    confs = _to_numpy(result.boxes.conf)

    mask = confs >= conf_thresh
    return boxes_xyxy[mask], classes[mask], confs[mask]


def _to_numpy(value):
    if hasattr(value, "cpu"):
        return value.cpu().numpy()

    return np.array(value)


def _empty_detections():
    return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

