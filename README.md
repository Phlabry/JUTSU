# JUTSU
**J.U.T.S.U. — Joint Upper-limb Tracking & Spatial User-interfaces**

A real-time computer vision system that recognizes hand gestures and triggers specifc visual effects on a live webcam feed.

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
camera/       # webcam capture
tracking/     # MediaPipe hand landmark detection
gesture/      # gesture recognition logic
state/        # idle → charging → released state machine
effects/      # visual rendering for techniques
output/       # virtual camera streaming
config/       # settings and flags
assets/       # images and sprites
```

---

## License

TBD

## Author

Aleksandre Khvadagadze