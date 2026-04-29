from nebulous_detector.config import SEARCH_WINDOW_KEYWORDS


def find_window(keywords=None):
    """Find the first visible window whose title contains one of the keywords."""
    import win32gui

    search_keywords = keywords or SEARCH_WINDOW_KEYWORDS
    matches = []

    def enum_window_callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd)
        if not title:
            return

        title_lower = title.lower()
        if any(keyword.lower() in title_lower for keyword in search_keywords):
            matches.append((hwnd, title))

    win32gui.EnumWindows(enum_window_callback, None)

    if not matches:
        return None, None

    hwnd, _ = matches[0]
    return hwnd, win32gui.GetWindowRect(hwnd)


def grab_window_frame(bbox, sct=None):
    """Capture a BGR frame from a window bounding box."""
    import mss

    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    monitor = {"left": left, "top": top, "width": width, "height": height}

    if sct is None:
        with mss.mss() as local_sct:
            return _grab_monitor(local_sct, monitor)

    return _grab_monitor(sct, monitor)


def _grab_monitor(sct, monitor):
    import cv2
    import numpy as np

    image = np.array(sct.grab(monitor))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
