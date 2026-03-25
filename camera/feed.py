# camera\feed.py
import numpy as np
import cv2 as cv

def open_camera(index=0):
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    return cap