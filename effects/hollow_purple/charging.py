import math
import os

import cv2 as cv
import numpy as np

from effects.gpu import HAVE_GPU, cp, gaussian_filter as gpu_gaussian

_ASSET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'hollow_purple', 'charging')
)

_frames_raw:   list[np.ndarray] = []
_frames_ready: list[np.ndarray] = []
_ready_size:   int = 0

_frame_idx: int = 0
_tint:      np.ndarray | None = None
_star_dist:     tuple | None = None    # (cx2, cy2, fh2, fw2, np.ndarray)
_star_dist_gpu: tuple | None = None   # (cx2, cy2, fh2, fw2, cp.ndarray) or None

# GPU wave draw buffer at half resolution — keyed by (fh2, fw2)
_gpu_wdraw: dict[tuple, object] = {}  # cp.ndarray float32

# Fixed random spark positions (seeded for reproducibility)
_rng = np.random.default_rng(42)
_SPARK_ANGLES = _rng.uniform(0, 2 * math.pi, 40).tolist()
_SPARK_R_FRAC = _rng.uniform(0.20, 0.92, 40).tolist()
_SPARK_PHASES = _rng.uniform(0, 2 * math.pi, 40).tolist()

# ── Per-(fh2,fw2) buffer pools — all computation runs at half resolution ─────
# Starburst layers: 4 × uint8 draw canvas, 4 × uint8 result, 4 × float32 workspace
# Aux 2D: fade_buf and scratch (both float32, (fh2,fw2)) — no per-frame alloc
# Combined: single uint8 half-res overlay before upscale
# Wave: draw/res/f32/prof/tmp for wave computation
# Upscale: full-res overlay buffer (uint8, (fh,fw,3))
_draw_h: dict[tuple, list[np.ndarray]] = {}
_res_h:  dict[tuple, list[np.ndarray]] = {}
_f32_h:  dict[tuple, list[np.ndarray]] = {}
_fade_h: dict[tuple, np.ndarray]       = {}   # (fh2,fw2) float32
_scr_h:  dict[tuple, np.ndarray]       = {}   # (fh2,fw2) float32
_comb_h: dict[tuple, np.ndarray]       = {}   # (fh2,fw2,3) uint8 combined layers
_wdraw:  dict[tuple, np.ndarray]       = {}
_wres:   dict[tuple, np.ndarray]       = {}
_wf32:   dict[tuple, np.ndarray]       = {}
_wprof:  dict[tuple, np.ndarray]       = {}
_wtmp:   dict[tuple, np.ndarray]       = {}
_overlay_full: dict[tuple, np.ndarray] = {}   # keyed by (fh,fw) — full-res uint8


def _ensure_bufs(fh: int, fw: int) -> tuple:
    """Return (fh2, fw2) and ensure all half-res buffers exist."""
    fh2, fw2 = fh // 2, fw // 2
    key2 = (fh2, fw2)
    if key2 not in _draw_h:
        _draw_h[key2] = [np.zeros((fh2, fw2, 3), dtype=np.uint8) for _ in range(4)]
        _res_h[key2]  = [np.empty((fh2, fw2, 3), dtype=np.uint8)  for _ in range(4)]
        _f32_h[key2]  = [np.empty((fh2, fw2, 3), dtype=np.float32) for _ in range(4)]
        _fade_h[key2] = np.empty((fh2, fw2), dtype=np.float32)
        _scr_h[key2]  = np.empty((fh2, fw2), dtype=np.float32)
        _comb_h[key2] = np.empty((fh2, fw2, 3), dtype=np.uint8)
        _wdraw[key2]  = np.zeros((fh2, fw2, 3), dtype=np.uint8)
        _wres[key2]   = np.empty((fh2, fw2, 3), dtype=np.uint8)
        _wf32[key2]   = np.empty((fh2, fw2, 3), dtype=np.float32)
        _wprof[key2]  = np.empty((fh2, fw2), dtype=np.float32)
        _wtmp[key2]   = np.empty((fh2, fw2), dtype=np.float32)
    key_full = (fh, fw)
    if key_full not in _overlay_full:
        _overlay_full[key_full] = np.empty((fh, fw, 3), dtype=np.uint8)
    for b in _draw_h[key2]:
        b[:] = 0
    return fh2, fw2


def _load_raw() -> None:
    global _frames_raw
    if _frames_raw:
        return
    for i in range(1, 61):
        path = os.path.join(_ASSET_DIR, f"frame_{i:04d}.png")
        img = cv.imread(path)
        if img is not None and img.mean() < 50:
            _frames_raw.append(img)


def _build_ready(size: int) -> None:
    global _frames_ready, _ready_size
    if size == _ready_size and _frames_ready:
        return

    cp = size // 2
    Y, X = np.ogrid[:size, :size]
    d = np.sqrt((X - cp) ** 2 + (Y - cp) ** 2).astype(np.float32)

    edge     = np.clip((cp - d) / (cp * 0.05), 0.0, 1.0)[:, :, None]
    core_r   = max(60, size // 8)
    core_sup = np.clip(d / core_r, 0.0, 1.0) ** 3
    mask     = (edge * core_sup[:, :, None]).astype(np.float32)

    _frames_ready = []
    for img in _frames_raw:
        resized = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
        _frames_ready.append(np.clip(resized.astype(np.float32) * mask, 0, 255).astype(np.uint8))
    _ready_size = size


def _composite(frame: np.ndarray, sprite: np.ndarray, center: tuple) -> np.ndarray:
    fh, fw = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    x1, y1 = cx - sw // 2, cy - sh // 2
    x2, y2 = x1 + sw, y1 + sh

    sx1 = max(0, -x1);  sy1 = max(0, -y1)
    sx2 = sw - max(0, x2 - fw);  sy2 = sh - max(0, y2 - fh)
    fx1 = max(0, x1);  fy1 = max(0, y1)
    fx2 = fx1 + (sx2 - sx1);  fy2 = fy1 + (sy2 - sy1)

    if sx2 <= sx1 or sy2 <= sy1:
        return frame

    frame[fy1:fy2, fx1:fx2] = cv.add(frame[fy1:fy2, fx1:fx2], sprite[sy1:sy2, sx1:sx2])
    return frame


def _draw_starburst(frame: np.ndarray, center: tuple, base_radius: int, idx: int) -> np.ndarray:
    global _star_dist, _star_dist_gpu
    fh, fw = frame.shape[:2]
    fh2, fw2 = _ensure_bufs(fh, fw)
    key2 = (fh2, fw2)

    # All coordinates and lengths are halved — everything runs at half resolution
    cx2, cy2 = int(center[0]) // 2, int(center[1]) // 2
    L = max(72, (base_radius * 7) // 2)   # half of max(145, base_radius*7)
    t  = idx

    if (_star_dist is None
            or _star_dist[2] != fh2 or _star_dist[3] != fw2
            or abs(_star_dist[0] - cx2) > 2 or abs(_star_dist[1] - cy2) > 2):
        Y, X = np.ogrid[:fh2, :fw2]
        dist2 = np.sqrt((X - cx2) ** 2 + (Y - cy2) ** 2).astype(np.float32)
        _star_dist = (cx2, cy2, fh2, fw2, dist2)
        if HAVE_GPU:
            if key2 not in _gpu_wdraw:
                _gpu_wdraw[key2] = cp.zeros((fh2, fw2, 3), dtype=cp.float32)
            _star_dist_gpu = (cx2, cy2, fh2, fw2, cp.asarray(dist2))
    dist2 = _star_dist[4]

    draws = _draw_h[key2]
    ress  = _res_h[key2]
    f32s  = _f32_h[key2]
    fade  = _fade_h[key2]

    # _bake: blur → fade → clip — zero allocations
    def _bake(di: int, blur_k: int, max_len: float, flicker: float = 1.0) -> np.ndarray:
        cv.GaussianBlur(draws[di], (blur_k, blur_k), 0, dst=ress[di])
        np.copyto(f32s[di], ress[di], casting='unsafe')
        np.copyto(fade, dist2)
        np.divide(fade, max(max_len, 1.0), out=fade)
        np.subtract(1.0, fade, out=fade)
        np.clip(fade, 0.0, flicker, out=fade)
        np.multiply(f32s[di], fade[:, :, None], out=f32s[di])
        np.clip(f32s[di], 0, 255, out=f32s[di])
        np.copyto(ress[di], f32s[di], casting='unsafe')
        return ress[di]

    # ── Layer A: Main arms ───────────────────────────────────────────────────
    main_Ls = [L * (1.0 + 0.22 * math.sin(t * 0.29 + i * 1.4)) for i in range(4)]
    for i, (base_ang, arm_L) in enumerate(zip([0.0, 90.0, 180.0, 270.0], main_Ls)):
        ang = math.radians(base_ang + 3.0 * math.sin(t * 0.41 + i * 2.3))
        cv.line(draws[0], (cx2, cy2),
                (cx2 + int(arm_L * math.cos(ang)), cy2 - int(arm_L * math.sin(ang))),
                (255, 255, 255), 1, cv.LINE_AA)
    la = _bake(0, 9, max(main_Ls))

    # ── Layer BC: Secondary diagonals + rotating crosses ────────────────────
    sec_Ls = [L * 0.5 * (1.0 + 0.18 * math.sin(t * 0.37 + i * 0.9)) for i in range(4)]
    for ang_deg, arm_L in zip([45.0, 135.0, 225.0, 315.0], sec_Ls):
        ang = math.radians(ang_deg)
        cv.line(draws[1], (cx2, cy2),
                (cx2 + int(arm_L * math.cos(ang)), cy2 - int(arm_L * math.sin(ang))),
                (230, 100, 255), 1, cv.LINE_AA)
    rot_L = L * 0.40
    r1 = (t *  2.2) % 360
    r2 = (360 - t * 1.5) % 360
    for i in range(4):
        a1 = math.radians(r1 + i * 90)
        a2 = math.radians(r2 + i * 90)
        cv.line(draws[1], (cx2, cy2),
                (cx2 + int(rot_L        * math.cos(a1)), cy2 - int(rot_L        * math.sin(a1))),
                (165, 28, 210), 1, cv.LINE_AA)
        cv.line(draws[1], (cx2, cy2),
                (cx2 + int(rot_L * 0.65 * math.cos(a2)), cy2 - int(rot_L * 0.65 * math.sin(a2))),
                (95, 12, 155),  1, cv.LINE_AA)
    lbc = _bake(1, 7, max(max(sec_Ls), rot_L))

    # ── Layer D: Tertiary arms + sparks ──────────────────────────────────────
    tert_L     = L * 0.25
    tert_flick = 0.85 + 0.15 * abs(math.sin(t * 0.53))
    for deg in [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]:
        ang = math.radians(deg)
        cv.line(draws[2], (cx2, cy2),
                (cx2 + int(tert_L * math.cos(ang)), cy2 - int(tert_L * math.sin(ang))),
                (190, 50, 200), 1, cv.LINE_AA)
    for angle, r_frac, phase in zip(_SPARK_ANGLES, _SPARK_R_FRAC, _SPARK_PHASES):
        v = math.sin(t * 0.7 + phase)
        if v > 0.30:
            sx = cx2 + int(L * r_frac * math.cos(angle))
            sy = cy2 - int(L * r_frac * math.sin(angle))
            if 0 <= sx < fw2 and 0 <= sy < fh2:
                b = int((v - 0.30) / 0.70 * 250)
                draws[2][sy, sx] = (min(255, b // 2), min(255, b // 6), min(255, b + 10))
    ld = _bake(2, 5, max(L * max(_SPARK_R_FRAC), tert_L), tert_flick)

    # ── Layer E: Center glow ─────────────────────────────────────────────────
    if 0 <= cy2 < fh2 and 0 <= cx2 < fw2:
        draws[3][cy2, cx2] = (255, 255, 255)
    cv.GaussianBlur(draws[3], (11, 11), 3.5, dst=ress[3])

    # ── Combine all layers at half-res, add slow waves, upscale once ─────────
    comb = _comb_h[key2]
    cv.add(la, lbc, dst=comb)
    cv.add(comb, ld, dst=comb)
    cv.add(comb, ress[3], dst=comb)

    _dist_gpu = _star_dist_gpu[4] if _star_dist_gpu is not None else None
    comb = _draw_charge_waves(comb, dist2, _dist_gpu, idx, fh2, fw2, _scr_h[key2], key2)

    overlay = _overlay_full[(fh, fw)]
    cv.resize(comb, (fw, fh), dst=overlay, interpolation=cv.INTER_LINEAR)
    return cv.add(frame, overlay)


_CHARGE_WAVE_DEFS = [
    (0,  (190,  40, 230), 0.32),
    (30, (230,  90, 255), 0.22),
    (60, (150,  20, 205), 0.28),
]
_CHARGE_WAVE_SIGMA = 35.0


def _draw_charge_waves(canvas: np.ndarray, dist_cpu: np.ndarray, dist_gpu,
                       idx: int, fh2: int, fw2: int,
                       prof_buf: np.ndarray, key2: tuple) -> np.ndarray:
    """3 slow purple waves at half resolution — GPU path if available."""
    diag_h     = math.hypot(fw2, fh2)
    wave_speed = diag_h * 0.011

    if HAVE_GPU and dist_gpu is not None:
        draw = _gpu_wdraw[key2];  draw[:] = 0
        for offset, color, brightness in _CHARGE_WAVE_DEFS:
            r    = ((idx + offset) * wave_speed) % (diag_h * 1.2)
            diff = cp.abs(dist_gpu - r)
            prof = cp.clip(1.0 - diff / _CHARGE_WAVE_SIGMA, 0.0, 1.0) ** 2
            draw += prof[:, :, cp.newaxis] * (cp.array(color, dtype=cp.float32) * brightness)
        cp.clip(draw, 0, 255, out=draw)
        gpu_gaussian(draw, sigma=[1.5, 1.5, 0], output=draw)
        return cv.add(canvas, draw.astype(cp.uint8).get())

    # CPU fallback — zero per-frame allocation
    wdraw = _wdraw[key2];  wdraw[:] = 0
    wres  = _wres[key2]
    wf32  = _wf32[key2]
    tmp   = _wtmp[key2]
    prof  = _wprof[key2]

    for offset, color, brightness in _CHARGE_WAVE_DEFS:
        r = ((idx + offset) * wave_speed) % (diag_h * 1.2)
        np.subtract(dist_cpu, r, out=prof)
        np.abs(prof, out=prof)
        np.divide(prof, _CHARGE_WAVE_SIGMA, out=prof)
        np.subtract(1.0, prof, out=prof)
        np.clip(prof, 0.0, 1.0, out=prof)
        np.multiply(prof, prof, out=prof)
        for ch, val in enumerate(color):
            np.copyto(wf32[:, :, ch], wdraw[:, :, ch], casting='unsafe')
            np.multiply(prof, val * brightness, out=tmp)
            wf32[:, :, ch] += tmp
            np.clip(wf32[:, :, ch], 0, 255, out=wf32[:, :, ch])
            np.copyto(wdraw[:, :, ch], wf32[:, :, ch], casting='unsafe')

    cv.GaussianBlur(wdraw, (5, 5), 1.5, dst=wres)
    return cv.add(canvas, wres)


def _add_tint(frame: np.ndarray) -> np.ndarray:
    global _tint
    if _tint is None or _tint.shape != frame.shape:
        _tint = np.full(frame.shape, (55, 35, 65), dtype=np.uint8)
    return cv.add(frame, _tint)


def render(frame: np.ndarray, center: tuple, base_radius: int) -> np.ndarray:
    global _frame_idx
    _load_raw()
    if not _frames_raw:
        return frame

    fh, fw = frame.shape[:2]
    size = max(fw, fh, base_radius * 28)
    _build_ready(size)

    sprite = _frames_ready[_frame_idx % len(_frames_ready)]
    _frame_idx += 1

    frame = _add_tint(frame)
    frame = _composite(frame, sprite, center)
    frame = _draw_starburst(frame, center, base_radius, _frame_idx)
    return frame
