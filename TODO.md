# JUTSU — Project TODO

## Hollow Purple Effect

### Visual

- [x] Thick, space-distorting release waves (4 waves, quadratic profile, sigma=90px)
- [x] Charging effect: purple tint + starburst arms + slow waves
- [x] Charging tint leans white/opaque — `(55, 35, 65)` additive overlay
- [x] Release sprite animation (30 frames, loaded from `assets/hollow_purple/release/`)
- [x] Radial energy streaks on release
- [x] Expanding thin ring lines on release (5 speeds, colored)
- [x] Sprite fade mask (edge roll-off + core suppression) for charging animation

### Performance

- [x] Pre-allocated numpy buffers — zero per-frame allocation in all render paths
- [x] Half-resolution starburst rendering (240×320 → single upscale to full res)
- [x] Background MediaPipe thread (Queue maxsize=1, non-blocking) — removes ~10ms from critical path
- [x] `waitKey(1)` — minimal display loop overhead
- [x] GPU acceleration via CuPy (RTX 5070 Ti, CUDA 12) for release waves — 7.6× faster than CPU
- [x] GPU acceleration for charging waves — same CuPy path, half-res
- [x] `os.add_dll_directory()` trick to load NVIDIA runtime DLLs from pip venv packages
- [x] `cupyx.scipy.ndimage.gaussian_filter` on GPU (replaces `cv.GaussianBlur` in GPU path)
- [x] GPU warmup in prewarm thread (pre-JITs all CUDA kernels before first frame)
- [x] Distance map cache — recomputed only when center moves >2–3px
- [x] OpenCV `setNumThreads(-1)` + `setUseOptimized(True)`
- [x] Display window 960×720 (4:3 matching 640×480 camera)

### Gesture Detection

- [x] 83-D feature vector: XY(42) + Z(21) + extension ratios(5) + bend cosines(10) + inter-tip distances(5)
- [x] 10× data augmentation: mirror-x, ±6°/±12° rotations, 2 noise levels, 2 scale variants
- [x] Soft-voting ensemble: SVM (RBF) + HistGradientBoosting (150 trees) + MLP (256→128→64)
- [x] HGB weights 2× in ensemble (best on tabular pose data)
- [x] 5-fold stratified CV: 98.99% accuracy (charge: 99.1%, release: 98.8%)
- [x] Temporal probability smoothing: 7-frame ring buffer in `CustomGestureDetector`
- [x] Confidence threshold 0.70 (temporal smoothing absorbs noise)
- [x] Model: 2.7MB, inference ~2.16ms

### State Machine

- [x] IDLE → CHARGING → RELEASED flow
- [x] `HollowPurpleState.render()` dispatches to correct effect module

## Backlog / Ideas

- [ ] Second hand detection: two-handed "push" release variant
- [ ] Sound effects (charging hum, release boom)
- [ ] Additional gestures / jutsu effects
- [ ] Config file for tunable parameters (wave speed, colors, thresholds)
- [ ] Retrain model with more diverse dataset for better real-world robustness
