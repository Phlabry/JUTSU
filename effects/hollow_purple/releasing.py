import math
import os

import cv2 as cv
import numpy as np

from effects.gpu import HAVE_GPU, cp, gaussian_filter as gpu_gaussian

_ASSET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'hollow_purple', 'release')
)

_frames:       list[np.ndarray] = []
_frames_ready: list[np.ndarray] = []
_ready_shape:  tuple = (0, 0)

# ── Distance map cache ────────────────────────────────────────────────────────
# CPU version  : (cx, cy, fh, fw, np.ndarray)
# GPU version  : (cx, cy, fh, fw, cp.ndarray)  — kept in GPU memory
_rel_dist_cpu: tuple | None = None
_rel_dist_gpu: tuple | None = None

# ── GPU wave buffers (per resolution) ────────────────────────────────────────
_gpu_draw:  dict[tuple, object] = {}   # cp.ndarray float32 (fh,fw,3)

# ── CPU wave buffers (fallback, per resolution) ───────────────────────────────
_wave_draw: dict[tuple, np.ndarray] = {}
_wave_res:  dict[tuple, np.ndarray] = {}
_wave_f32:  dict[tuple, np.ndarray] = {}
_wave_prof: dict[tuple, np.ndarray] = {}
_wave_tmp:  dict[tuple, np.ndarray] = {}


def _ensure_cpu_bufs(fh: int, fw: int) -> None:
    key = (fh, fw)
    if key not in _wave_draw:
        _wave_draw[key] = np.zeros((fh, fw, 3), dtype=np.uint8)
        _wave_res[key]  = np.empty((fh, fw, 3), dtype=np.uint8)
        _wave_f32[key]  = np.empty((fh, fw, 3), dtype=np.float32)
        _wave_prof[key] = np.empty((fh, fw), dtype=np.float32)
        _wave_tmp[key]  = np.empty((fh, fw), dtype=np.float32)


def _ensure_gpu_bufs(fh: int, fw: int) -> None:
    key = (fh, fw)
    if key not in _gpu_draw:
        _gpu_draw[key] = cp.zeros((fh, fw, 3), dtype=cp.float32)


def _load_frames() -> None:
    global _frames
    if _frames:
        return
    for i in range(1, 31):
        path = os.path.join(_ASSET_DIR, f"frame_{i:04d}.png")
        img = cv.imread(path)
        if img is not None:
            _frames.append(img)


def _build_ready(fw: int, fh: int) -> None:
    global _frames_ready, _ready_shape
    if (fw, fh) == _ready_shape and _frames_ready:
        return
    size  = max(fw, fh)
    x_off = (size - fw) // 2
    y_off = (size - fh) // 2
    _frames_ready = []
    for img in _frames:
        resized = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
        cropped = resized[y_off:y_off + fh, x_off:x_off + fw]
        _frames_ready.append(np.clip(cropped.astype(np.float32) * 0.80, 0, 255).astype(np.uint8))
    _ready_shape = (fw, fh)


# ── Wave definitions ──────────────────────────────────────────────────────────
# (phase_fraction, (B, G, R), brightness)
_WAVE_DEFS = [
    (0.00, (255,  60, 210), 0.42),
    (0.25, (255, 180, 255), 0.30),
    (0.50, (220,  30, 170), 0.38),
    (0.75, (255,  90, 230), 0.26),
]
_WAVE_SIGMA = 90.0


def _draw_waves_gpu(frame: np.ndarray, dist_gpu, t: float, tail_fade: float,
                    fh: int, fw: int) -> np.ndarray:
    """GPU path: all wave math on RTX, one download at the end."""
    key  = (fh, fw)
    draw = _gpu_draw[key]
    draw[:] = 0

    diag       = math.hypot(fw, fh)
    wave_speed = diag / 110.0

    for phase_frac, color_bgr, brightness in _WAVE_DEFS:
        r = (t - phase_frac) * wave_speed * 4.0
        if r < -_WAVE_SIGMA:
            continue
        diff  = cp.abs(dist_gpu - r)
        prof  = cp.clip(1.0 - diff / _WAVE_SIGMA, 0.0, 1.0) ** 2
        scale = brightness * tail_fade
        # Vectorised: add all 3 channels at once, no Python channel loop
        draw += prof[:, :, cp.newaxis] * (
            cp.array(color_bgr, dtype=cp.float32) * scale
        )

    cp.clip(draw, 0, 255, out=draw)
    gpu_gaussian(draw, sigma=[2.5, 2.5, 0], output=draw)
    overlay = draw.astype(cp.uint8).get()   # single download ~0.07ms
    return cv.add(frame, overlay)


def _draw_waves_cpu(frame: np.ndarray, dist: np.ndarray, t: float,
                    tail_fade: float, fh: int, fw: int) -> np.ndarray:
    """CPU fallback — zero per-frame allocation."""
    key  = (fh, fw)
    draw = _wave_draw[key];  draw[:] = 0
    res  = _wave_res[key]
    f32  = _wave_f32[key]
    prof = _wave_prof[key]
    tmp  = _wave_tmp[key]

    diag       = math.hypot(fw, fh)
    wave_speed = diag / 110.0

    for phase_frac, color_bgr, brightness in _WAVE_DEFS:
        r = (t - phase_frac) * wave_speed * 4.0
        if r < -_WAVE_SIGMA:
            continue
        np.subtract(dist, r, out=prof)
        np.abs(prof, out=prof)
        np.divide(prof, _WAVE_SIGMA, out=prof)
        np.subtract(1.0, prof, out=prof)
        np.clip(prof, 0.0, 1.0, out=prof)
        np.multiply(prof, prof, out=prof)
        scaled = brightness * tail_fade
        for ch, val in enumerate(color_bgr):
            np.copyto(f32[:, :, ch], draw[:, :, ch], casting='unsafe')
            np.multiply(prof, val * scaled, out=tmp)
            f32[:, :, ch] += tmp
            np.clip(f32[:, :, ch], 0, 255, out=f32[:, :, ch])
            np.copyto(draw[:, :, ch], f32[:, :, ch], casting='unsafe')

    cv.GaussianBlur(draw, (9, 9), 2.5, dst=res)
    return cv.add(frame, res)


def render(frame: np.ndarray, center: tuple, radius: int, alpha: float) -> np.ndarray:
    global _rel_dist_cpu, _rel_dist_gpu

    _load_frames()
    if not _frames:
        return frame

    fh, fw = frame.shape[:2]
    _build_ready(fw, fh)

    cx, cy = int(center[0]), int(center[1])

    # ── Update distance maps when center moves > 3 px ────────────────────────
    cpu_stale = (_rel_dist_cpu is None
                 or _rel_dist_cpu[2] != fh or _rel_dist_cpu[3] != fw
                 or abs(_rel_dist_cpu[0] - cx) > 3 or abs(_rel_dist_cpu[1] - cy) > 3)

    if cpu_stale:
        Y, X  = np.ogrid[:fh, :fw]
        dist_np = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        _rel_dist_cpu = (cx, cy, fh, fw, dist_np)
        if HAVE_GPU:
            _ensure_gpu_bufs(fh, fw)
            _rel_dist_gpu = (cx, cy, fh, fw, cp.asarray(dist_np))

    tail_fade = max(0.0, min(1.0, (alpha + 0.25) / 0.25)) if alpha < 0 else 1.0
    t = 1.0 - max(0.0, alpha)   # 0 at release, grows toward 1

    # ── Sprite overlay ────────────────────────────────────────────────────────
    if alpha > 0 and _frames_ready:
        n = len(_frames_ready)
        sprite_idx = min(int((1.0 - alpha) * n), n - 1)
        frame = cv.add(frame, _frames_ready[sprite_idx])

    diag           = int(math.hypot(fw, fh))
    ring_intensity = max(0.0, alpha) ** 0.6 * tail_fade

    # ── Fast thin expanding rings (CPU lines — negligible cost) ───────────────
    for speed, color, max_thick in (
        (0.55, (240,  70, 255), 3),
        (1.05, (255, 200, 255), 2),
        (1.65, (165,  25, 215), 2),
        (0.85, (255, 255, 255), 2),
        (2.10, (130,  15, 195), 1),
    ):
        r = int(diag * t * speed)
        if r <= 0:
            continue
        brightness = ring_intensity * (speed ** -0.4)
        c = tuple(int(v * min(1.0, brightness)) for v in color)
        thickness = max(1, int(max_thick * ring_intensity * 2))
        cv.circle(frame, (cx, cy), r, c, thickness, cv.LINE_AA)

    # ── Radial energy streaks ─────────────────────────────────────────────────
    streak_intensity = max(0.0, 1.0 - t * 2.5) * tail_fade
    if streak_intensity > 0.05:
        streak_r   = int(diag * t * 0.45)
        streak_len = max(12, int(diag * 0.07))
        for deg in range(0, 360, 20):
            rad      = math.radians(deg)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            x1 = cx + int(streak_r * cos_r)
            y1 = cy - int(streak_r * sin_r)
            x2 = cx + int((streak_r + streak_len) * cos_r)
            y2 = cy - int((streak_r + streak_len) * sin_r)
            b  = int(230 * streak_intensity)
            cv.line(frame, (x1, y1), (x2, y2), (b // 2, b // 6, b), 1, cv.LINE_AA)

    # ── Slow thick waves — GPU if available, CPU fallback ─────────────────────
    if HAVE_GPU and _rel_dist_gpu is not None:
        frame = _draw_waves_gpu(frame, _rel_dist_gpu[4], t, tail_fade, fh, fw)
    else:
        _ensure_cpu_bufs(fh, fw)
        frame = _draw_waves_cpu(frame, _rel_dist_cpu[4], t, tail_fade, fh, fw)

    return frame
