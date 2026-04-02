import math

import joblib
import numpy as np

CONF_THRESHOLD = 0.75


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Translation- and scale-invariant landmark features.
    Wrist (0) translated to origin, scaled by wrist→middle-MCP (9) distance.
    Only x/y — z is too noisy on a standard webcam.
    Also imported by the training script to guarantee identical features.
    """
    pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


class CustomGestureDetector:
    gesture_name = None

    def __init__(self, model_path: str):
        payload      = joblib.load(model_path)
        self._clf    = payload["model"]
        self._labels = payload["labels"]

    def process_frame(self, frame, result):
        h, w = frame.shape[:2]

        right_hand_found = False
        for i, handedness in enumerate(result.handedness):
            if handedness and handedness[0].category_name == "Right":
                lms     = result.hand_landmarks[i]
                points  = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in lms]
                top_px  = min(points, key=lambda p: p[1])
                wrist   = points[0]
                mid_tip = points[12]
                base_radius = max(20, int(math.hypot(mid_tip[0] - wrist[0],
                                                      mid_tip[1] - wrist[1])) // 4)

                self.on_hand_visible(top_px, base_radius)

                feat       = normalize_landmarks(lms).reshape(1, -1)
                proba      = self._clf.predict_proba(feat)[0]
                label_idx  = int(proba.argmax())
                confidence = float(proba[label_idx])

                if self._labels[label_idx] == self.gesture_name and confidence >= CONF_THRESHOLD:
                    self.on_detect(confidence, top_px, base_radius)

                right_hand_found = True
                break

        if not right_hand_found:
            self.on_no_detect()

    def on_hand_visible(self, top_px: tuple, base_radius: int):
        pass

    def on_detect(self, confidence: float, top_px: tuple, base_radius: int):
        raise NotImplementedError

    def on_no_detect(self):
        pass
