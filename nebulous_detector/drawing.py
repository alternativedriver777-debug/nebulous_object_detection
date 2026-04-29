import cv2

from nebulous_detector.config import CLASS_COLORS, CLASS_NAMES

DEFAULT_COLOR = (0, 255, 0)


def draw_boxes(image, boxes, classes, confs, class_names=None, class_colors=None):
    annotated = image.copy()
    names = class_names or CLASS_NAMES
    colors = class_colors or CLASS_COLORS

    for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
        cls_index = int(cls)
        color = colors[cls_index] if 0 <= cls_index < len(colors) else DEFAULT_COLOR
        label = _format_label(cls_index, conf, names)

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _draw_label(annotated, label, x1, y1, color)

    return annotated


def _format_label(cls_index, conf, class_names):
    if 0 <= cls_index < len(class_names):
        return f"{class_names[cls_index]} {conf:.2f}"

    return f"class_{cls_index} {conf:.2f}"


def _draw_label(image, label, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    top = max(0, y - text_height - 12)
    cv2.rectangle(image, (x, top), (x + text_width + 12, y), color, -1)

    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(
        image,
        label,
        (x + 6, max(text_height + 2, y - 6)),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

