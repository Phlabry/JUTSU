"""
Sukuna's Cleave — slash renderer.

Blade geometry is drawn in Python (frame-edge to frame-edge, any angle).
The 8 Blender-rendered frames drive the opacity schedule:
  frame 1 (t<0.125): white core only
  frame 2-4 (t 0.125-0.5): both at full opacity
  frames 5-8 (t 0.5-1.0): both fade out

Drawing procedurally guarantees the slash always reaches both frame edges
regardless of angle — the rotation-and-crop approach clips diagonals.
"""
import math
import os
import random

import cv2 as cv
import numpy as np

_NUM_FRAMES = 8
_NUM_SPARKS = 40

# Blade width in pixels (screen space, not world space)
_OUTER_HW = 11   # outer black half-width
_INNER_HW = 3    # inner white half-width


# ── Opacity schedule (mirrors Blender keyframes) ──────────────────────────────

def _white_alpha(t: float) -> float:
    if t <= 0.50:
        return 1.0
    return max(0.0, (0.875 - t) / 0.375)

def _black_alpha(t: float) -> float:
    if t < 0.125:
        return 0.0
    if t <= 0.50:
        return 1.0
    return max(0.0, (0.875 - t) / 0.375)


# ── Frame boundary intersection ───────────────────────────────────────────────

def _frame_endpoints(cos_a: float, sin_a: float, fw: int, fh: int):
    """
    Returns (p1, p2) — the two points where the slash line (through frame
    center, direction (cos_a, sin_a)) intersects the frame boundary.
    """
    cx, cy = fw / 2.0, fh / 2.0
    eps = 1e-9
    ts = []

    if abs(cos_a) > eps:
        for x_edge in (0.0, float(fw)):
            t = (x_edge - cx) / cos_a
            y = cy + t * sin_a
            if -1 <= y <= fh + 1:
                ts.append(t)

    if abs(sin_a) > eps:
        for y_edge in (0.0, float(fh)):
            t = (y_edge - cy) / sin_a
            x = cx + t * cos_a
            if -1 <= x <= fw + 1:
                ts.append(t)

    if len(ts) < 2:
        return None, None

    t1, t2 = min(ts), max(ts)
    p1 = (int(cx + t1 * cos_a), int(cy + t1 * sin_a))
    p2 = (int(cx + t2 * cos_a), int(cy + t2 * sin_a))
    return p1, p2


# ── Blade polygon ─────────────────────────────────────────────────────────────

def _blade_poly(p1, p2, perp_x: float, perp_y: float, hw: int) -> np.ndarray:
    """Axis-aligned rectangle perpendicular to the slash direction."""
    return np.array([
        [int(p1[0] + hw * perp_x), int(p1[1] + hw * perp_y)],
        [int(p1[0] - hw * perp_x), int(p1[1] - hw * perp_y)],
        [int(p2[0] - hw * perp_x), int(p2[1] - hw * perp_y)],
        [int(p2[0] + hw * perp_x), int(p2[1] + hw * perp_y)],
    ], dtype=np.int32)


# ── Spark init ────────────────────────────────────────────────────────────────

def _init_sparks(cos_a: float, sin_a: float, fw: int, fh: int) -> list[dict]:
    cx, cy = fw / 2.0, fh / 2.0
    perp_x, perp_y = -sin_a, cos_a
    diag = math.hypot(fw, fh)
    sparks = []
    for _ in range(_NUM_SPARKS):
        along  = random.uniform(-diag * 0.5, diag * 0.5)
        off    = random.uniform(-6, 6)
        px     = cx + along * cos_a + off * perp_x
        py     = cy + along * sin_a + off * perp_y
        vx     = perp_x * random.uniform(-3, 3) + cos_a * random.uniform(-1, 1)
        vy     = perp_y * random.uniform(-3, 3) + sin_a * random.uniform(-1, 1)
        sparks.append({
            "x": px, "y": py,
            "vx": vx, "vy": vy,
            "life": random.uniform(0.25, 0.85),
            "size": random.randint(1, 2),
        })
    return sparks


# ── Public API ────────────────────────────────────────────────────────────────

def prewarm(fw: int, fh: int) -> None:
    pass   # no assets to preload


def render(frame: np.ndarray, angle: float, t: float, sparks: list) -> np.ndarray:
    """
    angle : atan2(dy, dx) of the normalised flick direction
    t     : 0.0 (triggered) → 1.0 (fully expired)
    sparks: mutable list, populated on first call
    """
    fh, fw = frame.shape[:2]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    perp_x, perp_y = -sin_a, cos_a

    if not sparks:
        sparks.extend(_init_sparks(cos_a, sin_a, fw, fh))

    p1, p2 = _frame_endpoints(cos_a, sin_a, fw, fh)
    if p1 is None:
        return frame

    b_alpha = _black_alpha(t)
    w_alpha = _white_alpha(t)

    # ── Draw outer black blade ────────────────────────────────────────────────
    if b_alpha > 0.01:
        outer = _blade_poly(p1, p2, perp_x, perp_y, _OUTER_HW)
        overlay = frame.copy()
        cv.fillPoly(overlay, [outer], (0, 0, 0))
        frame = cv.addWeighted(overlay, b_alpha, frame, 1.0 - b_alpha, 0)

    # ── Draw inner white core ─────────────────────────────────────────────────
    if w_alpha > 0.01:
        inner = _blade_poly(p1, p2, perp_x, perp_y, _INNER_HW)
        overlay = frame.copy()
        cv.fillPoly(overlay, [inner], (255, 255, 255))
        frame = cv.addWeighted(overlay, w_alpha, frame, 1.0 - w_alpha, 0)

    # ── Perpendicular shear displacement ──────────────────────────────────────
    if t < 0.5:
        offset_px = int(_OUTER_HW * 0.9 * (1.0 - t / 0.5))
        if offset_px > 0:
            frame = _apply_shear(frame, cos_a, sin_a, perp_x, perp_y,
                                 offset_px, fw, fh)

    # ── Sparks ────────────────────────────────────────────────────────────────
    for sp in sparks:
        if sp["life"] <= 0:
            continue
        sp["x"]    += sp["vx"]
        sp["y"]    += sp["vy"]
        sp["life"] -= 1.0 / _NUM_FRAMES

        brightness = max(0.0, sp["life"]) * (1.0 - t)
        if brightness < 0.05:
            continue
        sx, sy = int(sp["x"]), int(sp["y"])
        if 0 <= sx < fw and 0 <= sy < fh:
            b = int(255 * brightness)
            cv.circle(frame, (sx, sy), sp["size"], (b, b, b), -1, cv.LINE_AA)

    return frame


def _apply_shear(frame, cos_a, sin_a, perp_x, perp_y,
                 offset_px, fw, fh) -> np.ndarray:
    """Shift the two halves of the frame apart perpendicular to the slash."""
    cx, cy = fw / 2.0, fh / 2.0
    ys, xs = np.mgrid[0:fh, 0:fw]
    # signed distance from the slash centre line
    side = (xs - cx) * perp_x + (ys - cy) * perp_y

    shift_pos = np.float32([[1, 0,  perp_x * offset_px],
                             [0, 1,  perp_y * offset_px]])
    shift_neg = np.float32([[1, 0, -perp_x * offset_px],
                             [0, 1, -perp_y * offset_px]])

    half_pos = cv.warpAffine(frame, shift_pos, (fw, fh),
                              borderMode=cv.BORDER_REPLICATE)
    half_neg = cv.warpAffine(frame, shift_neg, (fw, fh),
                              borderMode=cv.BORDER_REPLICATE)

    mask = (side >= 0)[..., np.newaxis]
    return np.where(mask, half_pos, half_neg)
