# JUTSU

Real-time hand gesture recognition that overlays anime-style visual effects on a virtual webcam so Discord (or any video app) sees the effects live.

Currently implemented: **Hollow Purple** — hold the charge pose to summon a pulsing energy orb, release to fire a screen-wide wave blast. Sound effects route through a virtual mic so teammates hear the charge and release.

---

## How it works

```
Physical webcam
   └─ MediaPipe hand tracking (background thread)
      └─ Custom sklearn ensemble (SVM + HGB + MLP)
         └─ HollowPurpleState machine  (IDLE → CHARGING → RELEASED)
            └─ GPU-accelerated effects (CuPy + OpenCV)
               └─ pyvirtualcam → UnityCapture virtual camera
                                   └─ Discord sees it
```

The service idles at ~0% CPU/GPU with the physical camera **off**. The camera turns on only when a consumer (Discord, OBS, etc.) opens the Unity Video Capture device. It shuts off again when the consumer closes it.

---

## Tech stack

| Area | Library |
|------|---------|
| Hand tracking | MediaPipe HandLandmarker |
| Gesture classification | scikit-learn (SVM + HistGradientBoosting + MLP ensemble) |
| Visual effects | OpenCV + NumPy + CuPy (GPU) |
| Virtual camera output | pyvirtualcam + UnityCapture |
| Audio effects | pygame (playback) + VB-Cable (mic routing) |
| System tray | pystray |

---

## Directory structure

```
assets/
  hollow_purple/
    audio/       # Charge.wav, Release.wav
    charging/    # sprite PNG sequence (60 frames)
    release/     # sprite PNG sequence (30 frames)
audio/           # VirtualMicMixer — merges effects + mic into VB-Cable
camera/          # open_camera() + dataset collection script
effects/
  hollow_purple/ # charging orb + starburst; release waves + rings
gesture/
  hollow_purple/ # ChargingDetector, ReleasingDetector
jutsu/           # jutsu registry (load / init_audio)
scripts/
  setup_autostart.py   # register service.py to run at Windows login
  clean_pycache.py     # delete all __pycache__ / .pyc files
state/
  hollow_purple/ # HollowPurpleState machine + audio triggers
tracking/        # MediaPipe HandLandmarker wrapper
models/          # trained .pkl classifiers  [gitignored]
datasets/        # gesture training images   [gitignored]
```

Entry points:

| File | Purpose |
|------|---------|
| `service.py` | Headless production service (auto-start at login) |
| `tray.py` | System tray launcher wrapping `main.py` |
| `main.py` | Debug runner — shows a live window, optional vcam output |
| `config.py` | All tunable settings |

---

## Prerequisites

1. **UnityCapture** — virtual DirectShow camera driver  
   https://github.com/schellingb/UnityCapture → `Install.bat` (run as Admin) → reboot

2. **VB-CABLE** — virtual audio cable for mic routing  
   https://vb-audio.com/Cable/ → install → reboot

3. **Python 3.11+** with a `.venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python mediapipe scikit-learn joblib pyvirtualcam pystray pillow pygame sounddevice soundfile cupy-cuda12x
```

4. **Train the gesture model** (one-time, requires a recorded dataset):

```bash
python models/hollow_purple/training.py
```

---

## Running

### Production (background service)

Register to auto-start at Windows login:

```bash
python scripts/setup_autostart.py
```

Or launch manually without registering:

```bash
python service.py
```

Open Discord → Settings → Voice & Video → select **Unity Video Capture** → turn on camera. The pipeline starts automatically.

**Hotkey:** `Ctrl+Shift+Alt+J` — toggle the pipeline on/off from anywhere.

### Development / testing

```bash
python main.py        # live window, optional vcam output
```

### Data collection and training

```bash
python camera/collect_data.py   # record gesture samples
python datasets/review.py       # review and clean samples
python models/hollow_purple/training.py   # retrain
```

### Utilities

```bash
python scripts/clean_pycache.py              # remove __pycache__ / .pyc
python scripts/setup_autostart.py --remove  # unregister autostart
```

---

## Gesture reference

| Gesture | Description |
|---------|-------------|
| Charge | Open hand toward camera, fingers spread — hold to build up |
| Release | Close fist quickly while charged |

The charge must hold for at least 12 consecutive frames before a release registers.

---

## Author

Phlabry