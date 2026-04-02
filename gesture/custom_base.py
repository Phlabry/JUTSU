# gesture/custom_base.py
#
# Base class for landmark-based custom gesture detectors.
# Uses a scikit-learn classifier trained on normalized hand landmarks.

import os
import time

import cv2 as cv
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_HAND_MODEL    = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task"))
CONF_THRESHOLD = 0.75   # minimum prediction probability to fire on_detect


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Make landmarks translation- and scale-invariant.

    Steps:
      1. Translate — wrist (landmark 0) moved to origin.
      2. Scale     — divide by wrist→middle-finger-MCP (landmark 9) distance.
      3. Flatten   — 21 landmarks × (x, y) = 42 features.

    Only x/y used; z from a standard webcam is noisy and hurts accuracy.
    This function is also imported by the training script so both paths
    are guaranteed to use the exact same feature representation.
    """
    pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)  # (21, 2)
    pts -= pts[0]                                                            # wrist → origin
    scale = np.linalg.norm(pts[9])                                          # wrist→middle MCP
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()                                                     # (42,)


class CustomGestureDetector:
    gesture_name = None  # set by subclass — must match a training label

    def __init__(self, model_path: str):
        payload         = joblib.load(model_path)
        self._clf       = payload["model"]
        self._labels    = payload["labels"]   # numpy array of class name strings

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=_HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb      = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect_for_video(mp_image, int(time.time() * 1000))

        for i, handedness in enumerate(result.handedness):
            if handedness and handedness[0].category_name == "Right":
                feat       = normalize_landmarks(result.hand_landmarks[i]).reshape(1, -1)
                proba      = self._clf.predict_proba(feat)[0]
                label_idx  = int(proba.argmax())
                confidence = float(proba[label_idx])
                if self._labels[label_idx] == self.gesture_name and confidence >= CONF_THRESHOLD:
                    self.on_detect(confidence)
                break   # only care about the first right hand

    def on_detect(self, confidence: float):
        raise NotImplementedError

    def release(self):
        self._detector.close()
