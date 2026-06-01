"""
Feature vector (83D):
  [0:42]   Normalized x,y positions (21 landmarks × 2)
  [42:63]  Normalized z depth      (21 landmarks)
  [63:68]  Finger extension ratios  (5 fingers)
  [68:78]  Joint bend cosines       (10 joints: 2 per finger)
  [78:83]  Inter-fingertip distances (5 pairs)
"""
import collections
import math

import joblib
import numpy as np

CONF_THRESHOLD = 0.70   # temporal smoothing absorbs noise, so threshold can be lower

_WRIST       = 0
_THUMB_CMC, _THUMB_MCP, _THUMB_IP, _THUMB_TIP       = 1, 2, 3, 4
_INDEX_MCP,  _INDEX_PIP, _INDEX_DIP, _INDEX_TIP      = 5, 6, 7, 8
_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_DIP, _MIDDLE_TIP   = 9, 10, 11, 12
_RING_MCP,   _RING_PIP,   _RING_DIP,   _RING_TIP     = 13, 14, 15, 16
_PINKY_MCP,  _PINKY_PIP,  _PINKY_DIP,  _PINKY_TIP    = 17, 18, 19, 20

# (tip_idx, base_idx) — extension = ||tip|| / ||base|| after wrist-origin normalisation
_EXT_PAIRS = [
    (_THUMB_TIP,  _THUMB_MCP),
    (_INDEX_TIP,  _INDEX_MCP),
    (_MIDDLE_TIP, _MIDDLE_MCP),
    (_RING_TIP,   _RING_MCP),
    (_PINKY_TIP,  _PINKY_MCP),
]

# (prev, joint, next) — cosine of bend angle at each joint
_BEND_TRIPLES = [
    (_THUMB_CMC,  _THUMB_MCP,  _THUMB_IP),
    (_THUMB_MCP,  _THUMB_IP,   _THUMB_TIP),
    (_INDEX_MCP,  _INDEX_PIP,  _INDEX_DIP),
    (_INDEX_PIP,  _INDEX_DIP,  _INDEX_TIP),
    (_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_DIP),
    (_MIDDLE_PIP, _MIDDLE_DIP, _MIDDLE_TIP),
    (_RING_MCP,   _RING_PIP,   _RING_DIP),
    (_RING_PIP,   _RING_DIP,   _RING_TIP),
    (_PINKY_MCP,  _PINKY_PIP,  _PINKY_DIP),
    (_PINKY_PIP,  _PINKY_DIP,  _PINKY_TIP),
]

# (a, b) — normalized inter-tip distances
_TIP_PAIRS = [
    (_INDEX_TIP,  _MIDDLE_TIP),
    (_MIDDLE_TIP, _RING_TIP),
    (_RING_TIP,   _PINKY_TIP),
    (_THUMB_TIP,  _INDEX_TIP),
    (_THUMB_TIP,  _PINKY_TIP),
]


def _landmarks_to_array(landmarks) -> np.ndarray:
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)


def _features_from_array(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts[_WRIST]
    scale = float(np.linalg.norm(pts[_MIDDLE_MCP, :2]))
    if scale > 1e-6:
        pts = pts / scale

    xy = pts[:, :2].flatten()
    z = pts[:, 2]

    ext = np.array([
        np.linalg.norm(pts[tip]) / max(float(np.linalg.norm(pts[base])), 1e-6)
        for tip, base in _EXT_PAIRS
    ], dtype=np.float32)

    bend = np.empty(len(_BEND_TRIPLES), dtype=np.float32)
    for k, (a, b, c) in enumerate(_BEND_TRIPLES):
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        bend[k] = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)) \
                  if (n1 > 1e-6 and n2 > 1e-6) else 1.0

    tip_d = np.array([
        float(np.linalg.norm(pts[a, :2] - pts[b, :2]))
        for a, b in _TIP_PAIRS
    ], dtype=np.float32)

    return np.concatenate([xy, z, ext, bend, tip_d])


def normalize_landmarks(landmarks) -> np.ndarray:
    return _features_from_array(_landmarks_to_array(landmarks))


class CustomGestureDetector:
    gesture_name  = None
    _SMOOTH_WINDOW = 7

    def __init__(self, model_path: str):
        payload      = joblib.load(model_path)
        self._clf    = payload["model"]
        self._labels = payload["labels"]
        self._prob_buf = collections.deque(maxlen=self._SMOOTH_WINDOW)

    def process_frame(self, frame, result):
        h, w = frame.shape[:2]

        right_hand_found = False
        for i, handedness in enumerate(result.handedness):
            if handedness and handedness[0].category_name == "Right":
                lms = result.hand_landmarks[i]

                # landmark x is in [0,1] from the left of the *mirrored* camera image
                points  = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in lms]
                top_px  = min(points, key=lambda p: p[1])
                wrist   = points[_WRIST]
                mid_tip = points[_MIDDLE_TIP]
                base_radius = max(20, int(math.hypot(mid_tip[0] - wrist[0],
                                                     mid_tip[1] - wrist[1])) // 4)

                self.on_hand_visible(top_px, base_radius)

                feat  = normalize_landmarks(lms).reshape(1, -1)
                proba = self._clf.predict_proba(feat)[0]
                self._prob_buf.append(proba)

                avg_proba  = np.mean(self._prob_buf, axis=0)
                label_idx  = int(avg_proba.argmax())
                confidence = float(avg_proba[label_idx])

                if self._labels[label_idx] == self.gesture_name \
                        and confidence >= CONF_THRESHOLD:
                    self.on_detect(confidence, top_px, base_radius)

                right_hand_found = True
                break

        if not right_hand_found:
            self._prob_buf.clear()
            self.on_no_detect()

    def on_hand_visible(self, top_px: tuple, base_radius: int):
        pass

    def on_detect(self, confidence: float, top_px: tuple, base_radius: int):
        raise NotImplementedError

    def on_no_detect(self):
        pass
