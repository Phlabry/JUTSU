import cv2 as cv
from camera.feed import open_camera
from tracking.trackhand import HandTracker
from gesture.palm import OpenPalmDetector
from gesture.pointing_up import PointingUpDetector

cap = open_camera(0)
tracker = HandTracker()

# Test gesture detectors
# palm_detector = OpenPalmDetector() 
# pointing_up_detector = PointingUpDetector()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # palm_detector.process_frame(frame)
    # pointing_up_detector.process_frame(frame)
    annotated = tracker.process_frame(frame)
    cv.imshow("JUTSU", annotated)

    if cv.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

# palm_detector.release()
# pointing_up_detector.release()
tracker.release()
cap.release()
cv.destroyAllWindows()
