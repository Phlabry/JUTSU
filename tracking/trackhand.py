import cv2 as cv
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS      = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
LANDMARK_COLOR        = (0, 255, 0)
CONNECTION_COLOR      = (255, 255, 255)
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MARGIN                = 10


class HandTracker:
    def process_frame(self, frame, result):
        return self._draw(cv.flip(frame, 1), result)

    def _draw(self, frame, result):
        annotated = np.copy(frame)
        h, w, _   = annotated.shape

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            points = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for conn in HAND_CONNECTIONS:
                cv.line(annotated, points[conn.start], points[conn.end], CONNECTION_COLOR, 2)

            for pt in points:
                cv.circle(annotated, pt, 5, LANDMARK_COLOR, -1)

            text_x = min(pt[0] for pt in points)
            text_y = min(pt[1] for pt in points) - MARGIN
            cv.putText(annotated, result.handedness[idx][0].category_name,
                       (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                       1, HANDEDNESS_TEXT_COLOR, 1, cv.LINE_AA)

        return annotated
