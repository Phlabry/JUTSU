import cv2 as cv


def open_camera(index: int = 0) -> cv.VideoCapture:
    """
    Open the physical webcam, skipping any device that returns exactly-zero
    frames (i.e. the UnityCapture virtual camera we feed black frames to).
    A real sensor always has noise: at least one pixel > 2 in any frame.
    """
    for idx in range(index, index + 8):
        cap = cv.VideoCapture(idx, cv.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        for w, h in ((1280, 720), (1920, 1080), (640, 480)):
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
            if int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) == w:
                break
        cap.set(cv.CAP_PROP_FPS, 30)

        fw  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))

        # Read up to 8 frames.  A virtual camera fed with np.zeros returns
        # exactly 0 for every pixel.  A real sensor always has at least one
        # pixel > 2 somewhere in the frame (even in a dark room).
        has_content = False
        max_val = 0
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None:
                max_val = max(max_val, int(frame.max()))
                if max_val > 2:
                    has_content = True
                    break

        if has_content:
            print(f"[Camera] {fw}x{fh} @ {fps}fps — index {idx}")
            return cap

        print(f"[Camera] index {idx} skipped — {fw}x{fh} max pixel={max_val} (virtual/black)")
        cap.release()

    raise RuntimeError(
        "No physical camera found. "
        "Check that your webcam is connected and not in use by another app."
    )
