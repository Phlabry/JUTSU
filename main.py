import cv2 as cv
from camera.feed import open_camera
from tracking.trackhand import HandTracker
from gesture.hollow_purple_charging import ChargingDetector
from gesture.hollow_purple_releasing import ReleasingDetector

cap                = open_camera(0)
tracker            = HandTracker()
charging_detector  = ChargingDetector()
releasing_detector = ReleasingDetector()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    charging_detector.process_frame(frame)
    releasing_detector.process_frame(frame)
    annotated = tracker.process_frame(frame)
    cv.imshow("JUTSU", annotated)

    if cv.waitKey(5) & 0xFF == ord("q"):
        break

charging_detector.release()
releasing_detector.release()
tracker.release()
cap.release()
cv.destroyAllWindows()
