# tracking/trackhand.py
import time
import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 255, 255)
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task', num_hands=2):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """Detect on original frame, draw on mirrored frame."""
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return self._draw(cv.flip(frame, 1), result)

    def _draw(self, frame, result):
        annotated = np.copy(frame)
        h, w, _ = annotated.shape

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            # Mirror x so landmarks align with the flipped display frame
            points = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for conn in HAND_CONNECTIONS:
                cv.line(annotated, points[conn.start], points[conn.end],
                        CONNECTION_COLOR, 2)

            for pt in points:
                cv.circle(annotated, pt, 5, LANDMARK_COLOR, -1)

            handedness = result.handedness[idx]
            text_x = min(pt[0] for pt in points)
            text_y = min(pt[1] for pt in points) - MARGIN
            cv.putText(annotated, handedness[0].category_name,
                       (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                       FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

        return annotated

    def release(self):
        self.detector.close()
