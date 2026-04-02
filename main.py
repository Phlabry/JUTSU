import cv2 as cv
from camera.feed import open_camera

from tracking.hand_detector import HandDetector
from tracking.trackhand import HandTracker
from state.hollow_purple import HollowPurpleState
from gesture.hollow_purple.charging import ChargingDetector
from gesture.hollow_purple.releasing import ReleasingDetector

DISPLAY_SIZE = (1280, 720)  # (width, height) — set to None to use raw camera resolution

cap      = open_camera(0)
detector = HandDetector()
tracker  = HandTracker()
state    = HollowPurpleState()

charging_detector  = ChargingDetector(state)
releasing_detector = ReleasingDetector(state)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        result = detector.detect(frame)

        charging_detector.process_frame(frame, result)
        releasing_detector.process_frame(frame, result)

        annotated = tracker.process_frame(frame, result)
        annotated = state.render(annotated)

        if DISPLAY_SIZE:
            annotated = cv.resize(annotated, DISPLAY_SIZE)

        cv.imshow("JUTSU", annotated)

        if cv.waitKey(5) & 0xFF == ord("q"):
            break
finally:
    detector.release()
    cap.release()
    cv.destroyAllWindows()
