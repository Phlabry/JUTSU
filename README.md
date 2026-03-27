# JUTSU
**J.U.T.S.U. — Joint Upper-limb Tracking & Spatial User-interfaces**

A real-time computer vision system that recognizes hand gestures and triggers specific visual effects on a live webcam feed.

---

## Overview

JUTSU uses hand tracking to simulate techniques inspired by anime. It detects hand poses and transitions, mapping them to visual effects rendered onto the camera feed and streamed through a virtual camera.

---

## Tech Stack

* Python, OpenCV, MediaPipe, v4l2loopback

Optional future: Unity + Socket/IPC for advanced rendering

---

## Architecture

```
assets/       # images and sprites
camera/       # webcam capture
config/       # settings and flags
dataset/      # gesture training images (gitignored)
effects/      # visual rendering for techniques
gesture/      # gesture recognition logic
models/       # trained gesture classifiers (gitignored)
output/       # virtual camera streaming
state/        # idle → charging → released state machine
tracking/     # MediaPipe hand landmark detection
```

---

## License

TBD

## Author

Aleksandre Khvadagadze