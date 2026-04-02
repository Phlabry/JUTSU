import math
import time

import cv2 as cv
import numpy as np

from effects.hollow_purple import _INNER, _OUTER, _RING


def render(frame: np.ndarray, center: tuple, base_radius: int) -> np.ndarray:
    t      = time.time()
    pulse  = 1.0 + 0.12 * math.sin(t * 5.0)
    r      = max(1, int(base_radius * pulse))
    cx, cy = int(center[0]), int(center[1])

    # Black overlay so additive blend only affects the circle area.
    overlay = np.zeros_like(frame)
    cv.circle(overlay, (cx, cy), r + 28, _OUTER, -1)
    cv.circle(overlay, (cx, cy), r + 14, (*_OUTER[:2], 200), -1)
    cv.circle(overlay, (cx, cy), r,      _INNER, -1)

    cv.addWeighted(frame, 1.0, overlay, 0.65, 0, frame)
    cv.circle(frame, (cx, cy), r, _RING, 2, cv.LINE_AA)

    return frame
