import cv2 as cv
import numpy as np

from effects.hollow_purple import _INNER, _OUTER, _RING


def render(frame: np.ndarray, center: tuple, radius: int, alpha: float) -> np.ndarray:
    if alpha <= 0 or radius <= 0:
        return frame

    cx, cy = int(center[0]), int(center[1])
    blend  = min(1.0, alpha * 0.85)

    overlay = np.zeros_like(frame)
    cv.circle(overlay, (cx, cy), radius + 22, _OUTER, -1)
    cv.circle(overlay, (cx, cy), radius,      _INNER, -1)

    cv.addWeighted(frame, 1.0, overlay, blend, 0, frame)
    cv.circle(frame, (cx, cy), radius, _RING, max(1, int(3 * alpha)), cv.LINE_AA)

    return frame
