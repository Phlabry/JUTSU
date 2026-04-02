import os
import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_MODEL = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task"))


class HandDetector:
    def __init__(self, num_hands=2):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = time.perf_counter_ns() // 1_000_000
        return self._detector.detect_for_video(mp_img, ts_ms)

    def release(self):
        self._detector.close()
