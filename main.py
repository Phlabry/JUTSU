import queue
import threading

import cv2 as cv
from camera.feed import open_camera

from tracking.hand_detector import HandDetector
from tracking.trackhand import HandTracker
from state.hollow_purple import HollowPurpleState
from gesture.hollow_purple.charging import ChargingDetector
from gesture.hollow_purple.releasing import ReleasingDetector

# Use all available CPU cores and SIMD paths for every OpenCV call
cv.setNumThreads(-1)
cv.setUseOptimized(True)

DISPLAY_SIZE = (960, 720)   # 4:3 to match 640×480 camera; set to None for raw

cap = open_camera(0)
detector = HandDetector()
tracker = HandTracker()
state = HollowPurpleState()

charging_detector = ChargingDetector(state)
releasing_detector = ReleasingDetector(state)

# ── Background detection thread ──────────────────────────────────────────────
# Runs MediaPipe in parallel with rendering (MediaPipe releases the GIL during
# C-level inference). Main loop submits frames non-blocking and picks up the
# latest result on each iteration — adds at most 1-frame latency to gestures,
# which is imperceptible at 30fps.
_detect_in  = queue.Queue(maxsize=1)
_detect_out = queue.Queue(maxsize=1)

def _detector_worker() -> None:
    while True:
        frame = _detect_in.get()
        if frame is None:
            break
        result = detector.detect(frame)
        try:
            _detect_out.get_nowait()   # discard stale result
        except queue.Empty:
            pass
        _detect_out.put(result)

threading.Thread(target=_detector_worker, daemon=True).start()

# ── Eager asset pre-load (avoids first-frame stall) ──────────────────────────
_ok, _warmup = cap.read()
if _ok:
    _fh, _fw = _warmup.shape[:2]
    _charge_size = max(_fw, _fh, 50 * 28)

    def _prewarm() -> None:
        from effects.hollow_purple import charging as _c
        from effects.hollow_purple import releasing as _r
        from effects import gpu as _gpu
        _c._load_raw()
        _c._build_ready(_charge_size)
        _r._load_frames()
        _r._build_ready(_fw, _fh)
        if _gpu.HAVE_GPU:
            _gpu.warmup(_fw, _fh)

    threading.Thread(target=_prewarm, daemon=True).start()

_last_result = None

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Submit frame to detector (drop oldest if worker is still busy)
        try:
            _detect_in.get_nowait()
        except queue.Empty:
            pass
        _detect_in.put_nowait(frame.copy())

        # Pick up latest detection result (non-blocking — use stale if not ready)
        try:
            _last_result = _detect_out.get_nowait()
        except queue.Empty:
            pass

        result = _last_result
        if result is not None:
            charging_detector.process_frame(frame, result)
            releasing_detector.process_frame(frame, result)
            annotated = tracker.process_frame(frame, result)
        else:
            annotated = frame.copy()

        annotated = state.render(annotated)

        if DISPLAY_SIZE:
            annotated = cv.resize(annotated, DISPLAY_SIZE)

        cv.imshow("JUTSU", annotated)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    _detect_in.put(None)   # signal worker to exit
    detector.release()
    cap.release()
    cv.destroyAllWindows()
