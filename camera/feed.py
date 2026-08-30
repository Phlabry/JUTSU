"""
Webcam capture.

Backend matters more than anything else here.  Over DirectShow this webcam only
negotiates uncompressed YUY2, which the USB bus caps at 10 fps for 720p (30 fps
at 640x480, 5 fps at 1080p).  Media Foundation negotiates a compressed mode and
holds 30 fps at 720p, so MSMF is tried first.  The trade is start-up: every
MSMF property set renegotiates the stream and costs ~3 s, so only the width is
requested and MSMF picks the matching height itself.

The reported CAP_PROP_FPS is not to be trusted — DirectShow claims 60 while
delivering 10 — so the real rate is timed at open and logged.
"""
import time

import cv2 as cv

_BACKENDS = {
    "msmf": cv.CAP_MSMF,
    "dshow": cv.CAP_DSHOW,
}
_BACKEND_NAMES = {cv.CAP_MSMF: "MSMF", cv.CAP_DSHOW: "DirectShow"}


def _settings() -> tuple[int, str]:
    try:
        from config import CAMERA_WIDTH, CAMERA_BACKEND
        return int(CAMERA_WIDTH), str(CAMERA_BACKEND).lower()
    except Exception:
        return 1280, "msmf"


def _configure(cap: cv.VideoCapture, api: int, width: int) -> None:
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    height = round(width * 9 / 16)

    if api == cv.CAP_MSMF:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        return

    for w, h in ((width, height), (1280, 720), (640, 480)):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        if int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) == w:
            break
    for fps_target in (60, 30):
        cap.set(cv.CAP_PROP_FPS, fps_target)
        if int(cap.get(cv.CAP_PROP_FPS)) >= fps_target:
            break


def _probe(cap: cv.VideoCapture) -> tuple[bool, float]:
    """
    (has_real_content, measured_fps).

    A virtual camera fed np.zeros returns exactly 0 for every pixel; a real
    sensor always has at least one pixel > 2 somewhere, even in a dark room.
    """
    has_content = False
    for _ in range(8):
        ok, frame = cap.read()
        if ok and frame is not None and int(frame.max()) > 2:
            has_content = True
            break
    if not has_content:
        return False, 0.0

    t0 = time.perf_counter()
    read = 0
    for _ in range(12):
        if cap.read()[0]:
            read += 1
    elapsed = time.perf_counter() - t0
    return True, (read / elapsed if elapsed > 0 else 0.0)


def open_camera(index: int = 0) -> cv.VideoCapture:
    """
    Open the physical webcam, skipping any device that returns exactly-zero
    frames (i.e. the UnityCapture virtual camera we feed black frames to).
    """
    width, backend = _settings()
    order = [_BACKENDS.get(backend, cv.CAP_MSMF)]
    for api in (cv.CAP_MSMF, cv.CAP_DSHOW):
        if api not in order:
            order.append(api)

    for api in order:
        name = _BACKEND_NAMES.get(api, str(api))
        for idx in range(index, index + 8):
            cap = cv.VideoCapture(idx, api)
            if not cap.isOpened():
                cap.release()
                continue

            _configure(cap, api, width)
            fw = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            has_content, fps = _probe(cap)
            if has_content:
                print(f"[Camera] {name} index {idx} — {fw}x{fh} @ {fps:.0f}fps measured")
                return cap

            print(f"[Camera] {name} index {idx} skipped — {fw}x{fh} all-black (virtual)")
            cap.release()

        print(f"[Camera] {name}: no physical camera found, trying next backend")

    raise RuntimeError(
        "No physical camera found. "
        "Check that your webcam is connected and not in use by another app."
    )
