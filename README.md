# JUTSU

**J.U.T.S.U. — Joint Upper-limb Tracking & Spatial User-interfaces**

A real-time computer vision system that recognizes hand gestures and triggers anime-inspired abilities directly on a live webcam feed.

---

## Overview

JUTSU is a gesture-driven interaction engine that uses real-time hand tracking to simulate “techniques” inspired by anime. By detecting specific hand poses and transitions, the system maps user input to visual effects rendered directly onto a camera feed.

The output is streamed through a virtual camera, allowing the effects to be used in applications like Discord, Zoom, or OBS in real time.

---

## Core Idea

The system follows a simple pipeline:

Camera Input → Hand Tracking → Gesture Recognition → Ability System → Visual Effects → Virtual Camera Output

At its core, JUTSU transforms physical hand gestures into interactive, visual actions.

---

## Features (Version 1.0)

* Real-time hand tracking using computer vision
* Custom gesture recognition (rule-based, no training required)
* State-based ability system (idle → charge → release)
* Hollow Purple technique prototype:
  * Gesture to initiate charge
  * Effect follows hand in real time
  * Release gesture triggers animation
* Live rendering directly onto webcam feed
* Virtual camera output for external applications

---

## Planned Features

* Additional techniques (Red, Blue, Rasengan, etc.)
* Gesture sequence recognition (e.g., Naruto-style hand signs)
* Sound effects and audio triggers
* Visual filters (e.g., Gojo “Six Eyes” effect)
* Configurable settings (sensitivity, toggles, debug mode)
* Multiple ability system with modular architecture
* Improved visual effects (particles, glow, animation polish)
* User segmentation for advanced effects (e.g., shadow clones)

---

## Tech Stack (Planned)

* Python
* OpenCV (camera input and rendering)
* MediaPipe (hand tracking and landmarks)
* v4l2loopback (virtual camera output on Linux)

Optional future stack:

* Unity (for advanced visual effects rendering)
* Socket/IPC communication between CV and rendering engine

---

## System Architecture

```
camera/       # webcam capture (OpenCV feed)
tracking/     # MediaPipe hand landmark detection
gesture/      # gesture recognition logic
state/        # idle → charging → released state machine
effects/      # visual rendering for techniques
output/       # virtual camera streaming (v4l2loopback)
config/       # settings and flags
assets/       # images and sprites for effects
```

---

## How It Works

1. The webcam feed is captured in real time.
2. Hand landmarks are detected using MediaPipe.
3. Gestures are interpreted based on landmark positions and movement.
4. A state machine determines the current ability phase.
5. Visual effects are rendered and attached to tracked hand positions.
6. The final frame is streamed to a virtual camera device.

---

## Motivation

This project combines computer vision, real-time systems, and interactive graphics into a single cohesive system. It is designed to explore gesture-based interfaces in a way that is both technically challenging and creatively engaging.

Rather than being a static demo, JUTSU aims to become a flexible engine for gesture-driven interactions.

---

## Current Status

Early development — building core tracking and gesture recognition systems.

---

## Future Direction

JUTSU is designed to evolve beyond a single technique into a generalized gesture interaction framework, capable of supporting multiple abilities, effects, and input styles.

---

## License

TBD

---

## Author

Aleksandre Khvadagadze
