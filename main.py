import cv2 as cv
from camera.feed import open_camera
from tracking.trackhand import HandTracker

cap = open_camera(0)
tracker = HandTracker()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    annotated = tracker.process_frame(frame)
    cv.imshow("JUTSU", annotated)

    if cv.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

tracker.release()
cap.release()
cv.destroyAllWindows()
