# JUTSU
**J.U.T.S.U. — Joint Upper-limb Tracking & Spatial User-interfaces**

A real-time computer vision system that recognizes hand gestures and triggers visual effects on a live webcam feed.

---

## Overview

JUTSU uses hand tracking to simulate techniques inspired by anime. It detects hand poses and state transitions, mapping them to visual effects rendered onto the camera feed.

Currently implemented: **Hollow Purple** — hold the charge gesture to summon a pulsing purple orb, release to fire it.

---

## Tech Stack

* Python, OpenCV, MediaPipe, scikit-learn

Planned: v4l2loopback (virtual camera output), audio effects

---

## Architecture

```
assets/          # sprites and animation frames
camera/          # webcam capture, dataset collection
config/          # settings and flags
datasets/        # gesture training images (gitignored)
effects/
  default/       # built-in MediaPipe gesture effects
  hollow_purple/ # hollow purple rendering
gesture/
  default/       # built-in MediaPipe gesture detectors
  hollow_purple/ # hollow purple gesture classification
models/          # trained classifiers (gitignored)
output/          # virtual camera streaming (planned)
state/           # state machines (idle → charging → released)
tracking/        # MediaPipe hand detection, shared per frame
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python mediapipe scikit-learn joblib
```

Train the gesture model (required before first run):
```bash
python models/hollow_purple_training.py
```

Run:
```bash
python main.py
```

Collect new training data:
```bash
python camera/collect_data.py
```

---

## License

TBD

## Author

Aleksandre Khvadagadze
