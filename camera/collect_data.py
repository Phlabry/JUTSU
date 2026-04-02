"""
python -m camera.collect_data

Controls:
  1       switch label → charge
  2       switch label → release
  r       toggle recording ON / OFF
  q       quit

Only frames where a Right hand is visible are eligible to save.
"""

import os
import time

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feed import open_camera

_ROOT        = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_BASE = os.path.join(_ROOT, "datasets", "hollow_purple")
MODEL_PATH   = os.path.join(_ROOT, "hand_landmarker.task")

LABELS          = {ord("1"): "charge", ord("2"): "release"}
SAVE_EVERY_N    = 5
RESIZE_TO       = (224, 224)  # set to None to keep original resolution
RIGHT_HAND_ONLY = True

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
LANDMARK_COLOR   = (0, 255, 0)
CONNECTION_COLOR = (255, 255, 255)
TEXT_COLOR       = (88, 205, 54)


def _count_existing(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))


def _draw_hud(frame, label: str, recording: bool, counts: dict, right_hand_visible: bool):
    h = frame.shape[0]

    rec_color  = (0, 0, 220)    if recording else (160, 160, 160)
    hand_color = (0, 220, 0)    if right_hand_visible else (0, 100, 220)
    rec_text   = "● REC"        if recording else "■ PAUSED"
    hand_text  = "Right hand ✓" if right_hand_visible else "No right hand"

    cv.putText(frame, f"Label  : {label}",    (10, 32),  cv.FONT_HERSHEY_DUPLEX, 0.7,  (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(frame, rec_text,                (10, 62),  cv.FONT_HERSHEY_DUPLEX, 0.7,  rec_color,       1, cv.LINE_AA)
    cv.putText(frame, hand_text,               (10, 92),  cv.FONT_HERSHEY_DUPLEX, 0.65, hand_color,      1, cv.LINE_AA)

    saved_str = "  ".join(f"{k}: {v}" for k, v in counts.items())
    cv.putText(frame, f"Saved  — {saved_str}", (10, 122), cv.FONT_HERSHEY_DUPLEX, 0.60, (0, 210, 210),  1, cv.LINE_AA)

    cv.putText(frame, "1=charge  2=release  r=toggle  q=quit",
               (10, h - 14), cv.FONT_HERSHEY_DUPLEX, 0.48, (170, 170, 170), 1, cv.LINE_AA)


def _has_right_hand(result) -> bool:
    for handedness in result.handedness:
        if handedness and handedness[0].category_name == "Right":
            return True
    return False


def _detect_and_draw(detector, frame):
    rgb      = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect_for_video(mp_image, time.perf_counter_ns() // 1_000_000)

    mirrored  = cv.flip(frame, 1)
    annotated = np.copy(mirrored)
    h, w, _   = annotated.shape

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        points = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for conn in HAND_CONNECTIONS:
            cv.line(annotated, points[conn.start], points[conn.end], CONNECTION_COLOR, 2)
        for pt in points:
            cv.circle(annotated, pt, 5, LANDMARK_COLOR, -1)
        label_txt = result.handedness[idx][0].category_name
        tx = min(pt[0] for pt in points)
        ty = min(pt[1] for pt in points) - 10
        cv.putText(annotated, label_txt, (tx, ty),
                   cv.FONT_HERSHEY_DUPLEX, 1, TEXT_COLOR, 1, cv.LINE_AA)

    return annotated, result


def main():
    for label in LABELS.values():
        os.makedirs(os.path.join(DATASET_BASE, label), exist_ok=True)

    cap = open_camera(0)

    options  = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    current_label = "charge"
    recording     = False
    frame_idx     = 0
    saved_counts  = {label: _count_existing(os.path.join(DATASET_BASE, label))
                     for label in LABELS.values()}

    print("Data collector ready.")
    print(f"  Existing images: {saved_counts}")
    print("  Press 1/2 to pick a label, R to start/stop, Q to quit.\n")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame_idx        += 1
            annotated, result = _detect_and_draw(detector, frame)
            right_visible     = _has_right_hand(result)

            if recording and frame_idx % SAVE_EVERY_N == 0:
                if not RIGHT_HAND_ONLY or right_visible:
                    save_frame = cv.flip(frame, 1)
                    if RESIZE_TO:
                        save_frame = cv.resize(save_frame, RESIZE_TO)
                    ts   = int(time.time() * 1000)
                    name = f"{ts}_{saved_counts[current_label]:05d}.jpg"
                    cv.imwrite(os.path.join(DATASET_BASE, current_label, name), save_frame)
                    saved_counts[current_label] += 1

            _draw_hud(annotated, current_label, recording, saved_counts, right_visible)
            cv.imshow("Data Collector — JUTSU", annotated)

            key = cv.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key in LABELS:
                current_label = LABELS[key]
                print(f"  Label → {current_label}")
            elif key == ord("r"):
                recording = not recording
                print(f"  Recording {'ON' if recording else 'OFF'}")
    finally:
        detector.close()
        cap.release()
        cv.destroyAllWindows()

    print("\nCollection complete.")
    for label, count in saved_counts.items():
        print(f"  {label}: {count} images")


if __name__ == "__main__":
    main()
