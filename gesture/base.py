import os
import time
import urllib.request
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'gesture_recognizer.task')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading gesture_recognizer.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


class GestureDetector:
    gesture_name = None  # subclasses define this

    def __init__(self):
        _ensure_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def process_frame(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        if result.gestures and result.gestures[0][0].category_name == self.gesture_name:
            self.on_detect()

    def on_detect(self):
        raise NotImplementedError

    def release(self):
        self._recognizer.close()
