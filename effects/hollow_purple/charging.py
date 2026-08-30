import math
import os

import cv2 as cv
import numpy as np

from effects.gpu import HAVE_GPU, cp, gaussian_filter as gpu_gaussian

_ASSET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'hollow_purple', 'charging')
)

# Source PNGs are 512x512 and the orb mask is defined in fractions of the sprite
# (edge falloff at 5% of the radius, core suppression at 1/8), so it is identical
# at every on-screen size.  The masked frames are therefore baked once, here, and
# _composite() scales the visible crop per frame — one resize of at most a
# frame's worth of pixels, instead of rebuilding all 60 sprites every time the
# hand moves a pixel closer to the camera.
_MASK_SIZE = 512

_frames_raw:   list[np.ndarray] = []
_frames_ready: list[np.ndarray] = []

_frame_idx: int = 0
_tint_c: dict[tuple, np.ndarray] = {}
_star_dist:     tuple | None = None
_star_dist_gpu: tuple | None = None

_gpu_wdraw: dict[tuple, object] = {}

_rng = np.random.default_rng(42)
_SPARK_ANGLES = _rng.uniform(0, 2 * math.pi, 40).tolist()
_SPARK_R_FRAC = _rng.uniform(0.20, 0.92, 40).tolist()
_SPARK_PHASES = _rng.uniform(0, 2 * math.pi, 40).tolist()

_draw_c: dict[tuple, list[np.ndarray]] = {}
_res_c:  dict[tuple, list[np.ndarray]] = {}
_f32_c:  dict[tuple, list[np.ndarray]] = {}
_fade_c: dict[tuple, np.ndarray]       = {}
_scr_c:  dict[tuple, np.ndarray]       = {}
_comb_c: dict[tuple, np.ndarray]       = {}
_wdraw:  dict[tuple, np.ndarray]       = {}
_wres:   dict[tuple, np.ndarray]       = {}
_wf32:   dict[tuple, np.ndarray]       = {}
_wprof:  dict[tuple, np.ndarray]       = {}
_wtmp:   dict[tuple, np.ndarray]       = {}
_overlay_full: dict[tuple, np.ndarray] = {}

# Starburst canvas the effect was tuned against: half of a 640x360 frame.
_REF_CANVAS_W = 320.0
_ARM_FLOOR_FRAC = 72.0 / _REF_CANVAS_W    # minimum arm length, as a canvas fraction


def _canvas_size(fw: int, fh: int) -> tuple[int, int]:
    """
    Working size for the starburst.  The arms are blurred lines, so drawing them
    small and upscaling costs nothing visually while keeping the per-frame cost
    flat as the output resolution grows.  Never exceeds half the frame.
    """
    try:
        from config import STARBURST_WIDTH
        want = int(STARBURST_WIDTH)
    except Exception:
        want = 320
    cw = max(64, min(fw // 2, want))
    ch = max(36, round(fh * cw / fw))
    return cw, ch


def _ensure_bufs(cw: int, ch: int, fw: int, fh: int) -> None:
    key = (ch, cw)
    if key not in _draw_c:
        _draw_c[key] = [np.zeros((ch, cw, 3), dtype=np.uint8) for _ in range(4)]
        _res_c[key]  = [np.empty((ch, cw, 3), dtype=np.uint8)  for _ in range(4)]
        _f32_c[key]  = [np.empty((ch, cw, 3), dtype=np.float32) for _ in range(4)]
        _fade_c[key] = np.empty((ch, cw), dtype=np.float32)
        _scr_c[key]  = np.empty((ch, cw), dtype=np.float32)
        _comb_c[key] = np.empty((ch, cw, 3), dtype=np.uint8)
        _wdraw[key]  = np.zeros((ch, cw, 3), dtype=np.uint8)
        _wres[key]   = np.empty((ch, cw, 3), dtype=np.uint8)
        _wf32[key]   = np.empty((ch, cw, 3), dtype=np.float32)
        _wprof[key]  = np.empty((ch, cw), dtype=np.float32)
        _wtmp[key]   = np.empty((ch, cw), dtype=np.float32)
    key_full = (fh, fw)
    if key_full not in _overlay_full:
        _overlay_full[key_full] = np.empty((fh, fw, 3), dtype=np.uint8)
    for b in _draw_c[key]:
        b[:] = 0


def _load_raw() -> None:
    global _frames_raw
    if _frames_raw:
        return
    for i in range(1, 61):
        path = os.path.join(_ASSET_DIR, f"frame_{i:04d}.png")
        img = cv.imread(path)
        if img is not None and img.mean() < 50:
            _frames_raw.append(img)


def _build_ready() -> None:
    """Bake the radial mask into every frame once, at _MASK_SIZE."""
    global _frames_ready
    if _frames_ready:
        return
    _load_raw()

    size   = _MASK_SIZE
    centre = size // 2
    Y, X = np.ogrid[:size, :size]
    d = np.sqrt((X - centre) ** 2 + (Y - centre) ** 2).astype(np.float32)

    edge     = np.clip((centre - d) / (centre * 0.05), 0.0, 1.0)[:, :, None]
    core_sup = np.clip(d / (size / 8.0), 0.0, 1.0) ** 3
    mask     = (edge * core_sup[:, :, None]).astype(np.float32)

    ready = []
    for img in _frames_raw:
        if img.shape[0] != size or img.shape[1] != size:
            img = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
        ready.append(np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8))
    _frames_ready = ready


def _composite(frame: np.ndarray, sprite: np.ndarray, center: tuple, size: int) -> np.ndarray:
    """Draw `sprite` centred on `center`, scaled to `size` px, clipped to the frame."""
    fh, fw = frame.shape[:2]
    src = sprite.shape[0]
    cx, cy = int(center[0]), int(center[1])

    x1, y1 = cx - size // 2, cy - size // 2

    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(fw, x1 + size), min(fh, y1 + size)
    if fx2 <= fx1 or fy2 <= fy1:
        return frame

    # Matching rectangle in sprite space, rounded outwards so the visible region
    # is always fully covered.
    s = src / size
    sx1 = max(0, min(src - 1, int((fx1 - x1) * s)))
    sy1 = max(0, min(src - 1, int((fy1 - y1) * s)))
    sx2 = max(sx1 + 1, min(src, math.ceil((fx2 - x1) * s)))
    sy2 = max(sy1 + 1, min(src, math.ceil((fy2 - y1) * s)))

    patch = cv.resize(sprite[sy1:sy2, sx1:sx2], (fx2 - fx1, fy2 - fy1),
                      interpolation=cv.INTER_LINEAR)
    frame[fy1:fy2, fx1:fx2] = cv.add(frame[fy1:fy2, fx1:fx2], patch)
    return frame


def _draw_starburst(frame: np.ndarray, center: tuple, base_radius: int, idx: int) -> np.ndarray:
    global _star_dist, _star_dist_gpu
    fh, fw = frame.shape[:2]
    cw, ch = _canvas_size(fw, fh)
    _ensure_bufs(cw, ch, fw, fh)
    key = (ch, cw)

    # All coordinates run on the small canvas — centre and arm lengths scale with it
    scale = cw / fw
    cx2, cy2 = int(center[0] * scale), int(center[1] * scale)
    L = max(_ARM_FLOOR_FRAC * cw, base_radius * 7 * scale)
    t  = idx

    if (_star_dist is None
            or _star_dist[2] != ch or _star_dist[3] != cw
            or abs(_star_dist[0] - cx2) > 2 or abs(_star_dist[1] - cy2) > 2):
        Y, X = np.ogrid[:ch, :cw]
        dist2 = np.sqrt((X - cx2) ** 2 + (Y - cy2) ** 2).astype(np.float32)
        _star_dist = (cx2, cy2, ch, cw, dist2)
        if HAVE_GPU:
            if key not in _gpu_wdraw:
                _gpu_wdraw[key] = cp.zeros((ch, cw, 3), dtype=cp.float32)
            _star_dist_gpu = (cx2, cy2, ch, cw, cp.asarray(dist2))
    dist2 = _star_dist[4]

    draws = _draw_c[key]
    ress  = _res_c[key]
    f32s  = _f32_c[key]
    fade  = _fade_c[key]

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

    main_Ls = [L * (1.0 + 0.22 * math.sin(t * 0.29 + i * 1.4)) for i in range(4)]
    for i, (base_ang, arm_L) in enumerate(zip([0.0, 90.0, 180.0, 270.0], main_Ls)):
        ang = math.radians(base_ang + 3.0 * math.sin(t * 0.41 + i * 2.3))
        cv.line(draws[0], (cx2, cy2),
                (cx2 + int(arm_L * math.cos(ang)), cy2 - int(arm_L * math.sin(ang))),
                (255, 255, 255), 1, cv.LINE_AA)
    la = _bake(0, 9, max(main_Ls))

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
            if 0 <= sx < cw and 0 <= sy < ch:
                b = int((v - 0.30) / 0.70 * 250)
                draws[2][sy, sx] = (min(255, b // 2), min(255, b // 6), min(255, b + 10))
    ld = _bake(2, 5, max(L * max(_SPARK_R_FRAC), tert_L), tert_flick)

    if 0 <= cy2 < ch and 0 <= cx2 < cw:
        draws[3][cy2, cx2] = (255, 255, 255)
    cv.GaussianBlur(draws[3], (11, 11), 3.5, dst=ress[3])

    comb = _comb_c[key]
    cv.add(la, lbc, dst=comb)
    cv.add(comb, ld, dst=comb)
    cv.add(comb, ress[3], dst=comb)

    _dist_gpu = _star_dist_gpu[4] if _star_dist_gpu is not None else None
    comb = _draw_charge_waves(comb, dist2, _dist_gpu, idx, cw, ch, _scr_c[key], key)

    # The ambient purple tint rides along in the overlay: adding it here, on the
    # small canvas, saves a second full-resolution pass over the frame.  uint8
    # addition saturates either way, so the result is the same.
    comb = cv.add(comb, _tint(key))

    overlay = _overlay_full[(fh, fw)]
    cv.resize(comb, (fw, fh), dst=overlay, interpolation=cv.INTER_LINEAR)
    return cv.add(frame, overlay)


_CHARGE_WAVE_DEFS = [
    (0,  (190,  40, 230), 0.32),
    (30, (230,  90, 255), 0.22),
    (60, (150,  20, 205), 0.28),
]
# Ring thickness as a fraction of the canvas width, so on-screen thickness stays
# the same whatever STARBURST_WIDTH is set to (35 px on the reference canvas).
_CHARGE_WAVE_SIGMA_FRAC = 35.0 / _REF_CANVAS_W


def _draw_charge_waves(canvas: np.ndarray, dist_cpu: np.ndarray, dist_gpu,
                       idx: int, cw: int, ch: int,
                       prof_buf: np.ndarray, key: tuple) -> np.ndarray:
    diag       = math.hypot(cw, ch)
    wave_speed = diag * 0.011
    sigma      = max(1.0, _CHARGE_WAVE_SIGMA_FRAC * cw)

    if HAVE_GPU and dist_gpu is not None:
        draw = _gpu_wdraw[key];  draw[:] = 0
        for offset, color, brightness in _CHARGE_WAVE_DEFS:
            r    = ((idx + offset) * wave_speed) % (diag * 1.2)
            diff = cp.abs(dist_gpu - r)
            prof = cp.clip(1.0 - diff / sigma, 0.0, 1.0) ** 2
            draw += prof[:, :, cp.newaxis] * (cp.array(color, dtype=cp.float32) * brightness)
        cp.clip(draw, 0, 255, out=draw)
        gpu_gaussian(draw, sigma=[1.5, 1.5, 0], output=draw)
        return cv.add(canvas, draw.astype(cp.uint8).get())

    wdraw = _wdraw[key];  wdraw[:] = 0
    wres  = _wres[key]
    wf32  = _wf32[key]
    tmp   = _wtmp[key]
    prof  = _wprof[key]

    for offset, color, brightness in _CHARGE_WAVE_DEFS:
        r = ((idx + offset) * wave_speed) % (diag * 1.2)
        np.subtract(dist_cpu, r, out=prof)
        np.abs(prof, out=prof)
        np.divide(prof, sigma, out=prof)
        np.subtract(1.0, prof, out=prof)
        np.clip(prof, 0.0, 1.0, out=prof)
        np.multiply(prof, prof, out=prof)
        for chan, val in enumerate(color):
            np.copyto(wf32[:, :, chan], wdraw[:, :, chan], casting='unsafe')
            np.multiply(prof, val * brightness, out=tmp)
            wf32[:, :, chan] += tmp
            np.clip(wf32[:, :, chan], 0, 255, out=wf32[:, :, chan])
            np.copyto(wdraw[:, :, chan], wf32[:, :, chan], casting='unsafe')

    cv.GaussianBlur(wdraw, (5, 5), 1.5, dst=wres)
    return cv.add(canvas, wres)


def _tint(key: tuple) -> np.ndarray:
    tint = _tint_c.get(key)
    if tint is None:
        ch, cw = key
        tint = _tint_c[key] = np.full((ch, cw, 3), (55, 35, 65), dtype=np.uint8)
    return tint


def render(frame: np.ndarray, center: tuple, base_radius: int) -> np.ndarray:
    global _frame_idx
    _build_ready()
    if not _frames_ready:
        return frame

    fh, fw = frame.shape[:2]
    size = max(fw, fh, base_radius * 28)

    sprite = _frames_ready[_frame_idx % len(_frames_ready)]
    _frame_idx += 1

    frame = _composite(frame, sprite, center, size)
    frame = _draw_starburst(frame, center, base_radius, _frame_idx)
    return frame
