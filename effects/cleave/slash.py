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

# Blade width, tuned against a 640-wide frame and scaled from there so the slash
# covers the same fraction of the picture at any resolution.
_REF_WIDTH = 640.0
_OUTER_HW = 11   # outer black half-width
_INNER_HW = 3    # inner white half-width

_mask_buf: dict[tuple, np.ndarray] = {}


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
    k = fw / _REF_WIDTH   # spread and speed are in pixels, so they scale with the frame
    sparks = []
    for _ in range(_NUM_SPARKS):
        along  = random.uniform(-diag * 0.5, diag * 0.5)
        off    = random.uniform(-6, 6) * k
        px     = cx + along * cos_a + off * perp_x
        py     = cy + along * sin_a + off * perp_y
        vx     = (perp_x * random.uniform(-3, 3) + cos_a * random.uniform(-1, 1)) * k
        vy     = (perp_y * random.uniform(-3, 3) + sin_a * random.uniform(-1, 1)) * k
        sparks.append({
            "x": px, "y": py,
            "vx": vx, "vy": vy,
            "life": random.uniform(0.25, 0.85),
            "size": max(1, round(random.randint(1, 2) * k)),
        })
    return sparks


# ── Alpha-blended polygon, restricted to its bounding box ─────────────────────

def _blend_poly(frame: np.ndarray, poly: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    """
    cv.addWeighted over the whole frame costs three full-resolution passes for
    what is usually a thin band, so blend inside the polygon's bounding box only.
    """
    fh, fw = frame.shape[:2]
    bx, by, bw, bh = cv.boundingRect(poly)
    x1, y1 = max(0, bx), max(0, by)
    x2, y2 = min(fw, bx + bw), min(fh, by + bh)
    if x2 <= x1 or y2 <= y1:
        return frame

    roi     = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv.fillPoly(overlay, [poly - np.array([[x1, y1]], dtype=np.int32)], color)
    frame[y1:y2, x1:x2] = cv.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)
    return frame


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

    k        = fw / _REF_WIDTH
    outer_hw = max(2, round(_OUTER_HW * k))
    inner_hw = max(1, round(_INNER_HW * k))

    # ── Draw outer black blade ────────────────────────────────────────────────
    if b_alpha > 0.01:
        outer = _blade_poly(p1, p2, perp_x, perp_y, outer_hw)
        frame = _blend_poly(frame, outer, (0, 0, 0), b_alpha)

    # ── Draw inner white core ─────────────────────────────────────────────────
    if w_alpha > 0.01:
        inner = _blade_poly(p1, p2, perp_x, perp_y, inner_hw)
        frame = _blend_poly(frame, inner, (255, 255, 255), w_alpha)

    # ── Perpendicular shear displacement ──────────────────────────────────────
    if t < 0.5:
        offset_px = int(outer_hw * 0.9 * (1.0 - t / 0.5))
        if offset_px > 0:
            frame = _apply_shear(frame, p1, p2, perp_x, perp_y,
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


def _apply_shear(frame, p1, p2, perp_x, perp_y,
                 offset_px, fw, fh) -> np.ndarray:
    """
    Shift the two halves of the frame apart perpendicular to the slash.

    The dividing line already runs edge to edge through p1/p2, so the positive
    half-plane is just the quad that extends from it past the far corner —
    fillPoly draws that in a fraction of a millisecond, where evaluating the
    signed distance over a coordinate grid cost 20+ ms at 720p.
    """
    shift_pos = np.float32([[1, 0,  perp_x * offset_px],
                             [0, 1,  perp_y * offset_px]])
    shift_neg = np.float32([[1, 0, -perp_x * offset_px],
                             [0, 1, -perp_y * offset_px]])

    half_pos = cv.warpAffine(frame, shift_pos, (fw, fh),
                              borderMode=cv.BORDER_REPLICATE)
    out      = cv.warpAffine(frame, shift_neg, (fw, fh),
                              borderMode=cv.BORDER_REPLICATE)

    key = (fh, fw)
    mask = _mask_buf.get(key)
    if mask is None:
        mask = _mask_buf[key] = np.empty((fh, fw), dtype=np.uint8)
    mask[:] = 0

    far  = fw + fh
    poly = np.array([
        [p1[0], p1[1]],
        [p2[0], p2[1]],
        [int(p2[0] + perp_x * far), int(p2[1] + perp_y * far)],
        [int(p1[0] + perp_x * far), int(p1[1] + perp_y * far)],
    ], dtype=np.int32)
    cv.fillPoly(mask, [poly], 255)

    cv.copyTo(half_pos, mask, out)
    return out
