# JUTSU - Joint Upper-limb Tracking System for UI augmentation

Real-time hand gesture recognition that overlays anime-style visual effects on a virtual webcam so Discord (or any video app) sees the effects live.

Currently implemented:

- **Hollow Purple** — hold the charge pose to summon a pulsing energy orb, release to fire a screen-wide wave blast. Sound effects route through a virtual mic so teammates hear the charge and release.
- **Sukuna's Cleave** — flick the wrist or fingers in any direction to slash a full-frame manga-style ink cut across the screen. Direction is tracked from hand velocity, so diagonal, vertical, and horizontal slashes all work. Multiple cleaves can chain without waiting for the previous one to finish.

---

## How it works

```
Physical webcam
   └─ MediaPipe hand tracking (background thread)
      ├─ Custom sklearn ensemble (SVM + HGB + MLP)  ← Hollow Purple
      │     └─ HollowPurpleState  (IDLE → CHARGING → RELEASED)
      │           └─ GPU-accelerated orb + wave effects
      └─ Velocity-based flick detector (no ML)      ← Cleave
            └─ CleaveState  (chains concurrent slashes)
                  └─ Manga-style B&W slash + shear displacement
      └─ pyvirtualcam → UnityCapture virtual camera
                          └─ Discord sees it
```

The service idles at ~0% CPU/GPU with the physical camera **off**. The camera turns on only when a consumer (Discord, OBS, etc.) opens the Unity Video Capture device. It shuts off again when the consumer closes it.

`watcher.py` wraps `service.py` with auto-reload: any `.py` change triggers a 1-second debounced restart, so edits go live without manually restarting.

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
  cleave/
    audio/       # cleave.wav (slash sound)
    frames/      # Blender-rendered B&W slash PNGs (8 frames, RGBA 1024×1024)
audio/           # VirtualMicMixer — merges effects + mic into VB-Cable
camera/          # open_camera() + dataset collection script
effects/
  hollow_purple/ # charging orb + starburst; release waves + rings
  cleave/        # full-frame slash renderer (Python geometry, frame-edge to frame-edge)
gesture/
  hollow_purple/ # ChargingDetector, ReleasingDetector (ML-based)
  cleave/        # FlickDetector (velocity-based, no ML)
jutsu/           # jutsu registry (load / init_audio)
scripts/
  setup_autostart.py   # register watcher.py to run at Windows login
  clean_pycache.py     # delete all __pycache__ / .pyc files
state/
  hollow_purple/ # HollowPurpleState machine + audio triggers
  cleave/        # CleaveState — list of concurrent active slashes
tracking/        # MediaPipe HandLandmarker wrapper
models/          # trained .pkl classifiers  [gitignored]
datasets/        # gesture training images   [gitignored]
```

Entry points:

| File | Purpose |
|------|---------|
| `watcher.py` | Production entry point — runs service.py with auto-reload on code changes |
| `service.py` | Headless pipeline — idles at ~0% CPU, starts on consumer connect |
| `tray.py` | System tray launcher |
| `main.py` | Debug runner — live window, optional vcam output |
| `config.py` | All tunable settings (`ACTIVE_JUTSU` list, resolution, mic device) |

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

4. **Train the gesture model** (one-time, required for Hollow Purple only):

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
python watcher.py
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

| Jutsu | Trigger | Description |
|-------|---------|-------------|
| Hollow Purple | Charge pose (hold) | Thumb over index and middle finger, ring and pinky out |
| Hollow Purple | Release (while charged) | Upside down open hand |
| Cleave | Wrist flick | Any fast wrist movement — direction from palm velocity |
| Cleave | Finger flick | Index + middle finger snap while wrist stays still |

---

## Author

Phlabry
