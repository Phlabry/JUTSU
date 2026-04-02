import cv2 as cv

def open_camera(index=0):
    cap = cv.VideoCapture(index, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Request the camera's maximum resolution.
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  3840)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv.CAP_PROP_FPS, 30)

    return cap
