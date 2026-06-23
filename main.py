import queue
import threading

import cv2 as cv
from camera.feed import open_camera
from tracking.hand_detector import HandDetector
from tracking.trackhand import HandTracker
import jutsu as jutsu_registry
from config import ACTIVE_JUTSU

cv.setNumThreads(-1)
cv.setUseOptimized(True)


def run(headless: bool = False, stop_event: threading.Event | None = None) -> None:
    cap      = open_camera(0)
    detector = HandDetector()
    tracker  = HandTracker()

    _ok, _warmup = cap.read()
    _fh, _fw = (_warmup.shape[:2] if _ok else (480, 640))

    _jutsu_modules = [jutsu_registry.load(name, _fw, _fh) for name in ACTIVE_JUTSU]
    all_detectors  = [det for _, dets, _ in _jutsu_modules for det in dets]
    for name in ACTIVE_JUTSU:
        jutsu_registry.init_audio(name)

    DISPLAY_SIZE = None if _fw >= 1280 else (960, 720)
    _out_w = DISPLAY_SIZE[0] if DISPLAY_SIZE else _fw
    _out_h = DISPLAY_SIZE[1] if DISPLAY_SIZE else _fh

    # MediaPipe releases the GIL during C-level inference, so a background thread
    # removes its ~10ms from the critical path with at most 1-frame latency.
    _detect_in  = queue.Queue(maxsize=1)
    _detect_out = queue.Queue(maxsize=1)

    def _detector_worker() -> None:
        while True:
            frame = _detect_in.get()
            if frame is None:
                break
            result = detector.detect(frame)
            try:
                _detect_out.get_nowait()
            except queue.Empty:
                pass
            _detect_out.put(result)

    threading.Thread(target=_detector_worker, daemon=True).start()
    for _, _, _prewarm_fn in _jutsu_modules:
        threading.Thread(target=_prewarm_fn, daemon=True).start()

    _vcam = None
    try:
        import pyvirtualcam
        _vcam = pyvirtualcam.Camera(
            width=_out_w, height=_out_h, fps=30,
            fmt=pyvirtualcam.PixelFormat.BGR,
            print_fps=False,
        )
        print(f"[VCam] streaming to {_vcam.device}")
    except Exception as e:
        print(f"[VCam] not available ({type(e).__name__}: {e})")

    _last_result = None

    try:
        while cap.isOpened():
            if stop_event and stop_event.is_set():
                break

            success, frame = cap.read()
            if not success:
                continue

            try:
                _detect_in.get_nowait()
            except queue.Empty:
                pass
            _detect_in.put_nowait(frame.copy())

            try:
                _last_result = _detect_out.get_nowait()
            except queue.Empty:
                pass

            result = _last_result
            if result is not None:
                for det in all_detectors:
                    det.process_frame(frame, result)
                annotated = tracker.process_frame(frame, result)
            else:
                annotated = frame.copy()

            for _state, _, _ in _jutsu_modules:
                annotated = _state.render(annotated)

            if DISPLAY_SIZE:
                annotated = cv.resize(annotated, DISPLAY_SIZE)

            if not headless:
                cv.imshow("JUTSU", annotated)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            if _vcam is not None:
                _vcam.send(annotated)

    finally:
        _detect_in.put(None)
        if _vcam is not None:
            _vcam.close()
        detector.release()
        cap.release()
        if not headless:
            cv.destroyAllWindows()


if __name__ == "__main__":
    run(headless=False)
