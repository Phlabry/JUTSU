import collections
import math
import time

_PALM_IDX = [0, 1, 5, 9, 13, 17]  # wrist, thumb-CMC, 4× MCP

# Speed threshold as a fraction of frame width per second.
# Resolution-independent: 0.45 = "palm moves 45% of frame width / sec".
# At 640px → 288 px/s.  At 1280px → 576 px/s.
_SPEED_THRESH = 0.41  # wrist-body flick (lower is safe because EMA kills noise)
_FINGER_THRESH = 0.28  # fingertip-relative flick (wrist locked)
_STILL_THRESH = 0.12  # max wrist speed for "wrist locked" mode

_SPEED_WIN = 0.080  # 80ms look-back for speed check
_DIR_WIN = 0.040  # 40ms look-back for direction (most recent motion)
_MIN_WIN = 0.030  # don't compute if buffer spans less than this

_COOLDOWN = 0.055  # 55ms minimum between triggers

# Max centroid jump in a single frame as a fraction of frame width.
# Legitimate fast flicks at 60fps top out around 5-8% per frame.
# Tracking artifacts routinely jump 30-100%. Reject anything above this.
_MAX_JUMP_NORM = 0.18

# EMA smoothing on the palm centroid before it enters the velocity buffer.
# Damps single-frame jitter spikes to 40% of their magnitude while only
# reducing multi-frame real flick peaks by ~20%.
_EMA_ALPHA = 0.40

# Direction consistency: speed-window vector vs dir-window vector must agree
# within this many degrees. Jitter mid-window makes them disagree wildly.
_MAX_DIR_DIVERGE_DEG = 90


class FlickDetector:
    def __init__(self, state):
        self._state = state
        self._buf = collections.deque(maxlen=32)  # (x_px, y_px, t)
        self._tip_buf = collections.deque(maxlen=32)  # (x_px, y_px, t)
        self._fw = 640  # updated each frame
        self._last_fire = 0.0

    # ── Frame callback ────────────────────────────────────────────────────────

    def process_frame(self, frame, result):
        h, w = frame.shape[:2]
        self._fw = w

        right_found = False
        for i, handedness in enumerate(result.handedness):
            if handedness and handedness[0].category_name == "Right":
                lms = result.hand_landmarks[i]
                now = time.monotonic()

                # Palm centroid in pixels, mirror-flipped to match display
                cx = (w - 1) - sum(lms[j].x * w for j in _PALM_IDX) / len(_PALM_IDX)
                cy = sum(lms[j].y * h for j in _PALM_IDX) / len(_PALM_IDX)

                # EMA smoothing: blend 40% new / 60% previous.
                # Damps single-frame jitter spikes before they enter the
                # velocity buffer, without killing multi-frame real flicks.
                if self._buf:
                    px, py, _ = self._buf[-1]
                    cx = _EMA_ALPHA * cx + (1.0 - _EMA_ALPHA) * px
                    cy = _EMA_ALPHA * cy + (1.0 - _EMA_ALPHA) * py

                # Drop tracking artifacts: if the centroid teleports more than
                # _MAX_JUMP_NORM * frame_width in a single frame it's a bad
                # landmark reading, not real motion. Skip the sample entirely.
                if self._buf:
                    last_x, last_y, _ = self._buf[-1]
                    if math.hypot(cx - last_x, cy - last_y) > w * _MAX_JUMP_NORM:
                        right_found = True
                        break  # discard this frame, keep existing buffer

                self._buf.append((cx, cy, now))

                # Fingertip centroid (index + middle), same mirror flip
                tx = (w - 1) - (lms[8].x + lms[12].x) * w / 2
                ty = (lms[8].y + lms[12].y) * h / 2
                self._tip_buf.append((tx, ty, now))

                if now - self._last_fire >= _COOLDOWN:
                    self._check_trigger()

                right_found = True
                break

        if not right_found:
            self._buf.clear()
            self._tip_buf.clear()

    # ── Detection ─────────────────────────────────────────────────────────────

    def _check_trigger(self):
        fw = self._fw

        wrist_spd, w_dx, w_dy = self._vec(self._buf, _SPEED_WIN)
        wrist_norm = wrist_spd / fw

        finger_spd, f_dx, f_dy = self._finger_relative_vec()
        finger_norm = finger_spd / fw

        if wrist_norm >= _SPEED_THRESH:
            # Direction consistency: speed window and dir window must agree.
            # Jitter mid-window makes them point in opposite directions.
            _, d_dx, d_dy = self._vec(self._buf, _DIR_WIN)
            if self._dirs_agree(w_dx, w_dy, d_dx, d_dy):
                self._fire(d_dx, d_dy)
        elif wrist_norm < _STILL_THRESH and finger_norm >= _FINGER_THRESH:
            self._fire(f_dx, f_dy)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _vec(buf, window) -> tuple[float, float, float]:
        """(speed_px_s, dx_px_s, dy_px_s) over `window` seconds."""
        if len(buf) < 2:
            return 0.0, 0.0, 0.0
        x1, y1, t1 = buf[-1]
        for i in range(len(buf) - 2, -1, -1):
            dt = t1 - buf[i][2]
            if dt >= window:
                dx = (x1 - buf[i][0]) / dt
                dy = (y1 - buf[i][1]) / dt
                return math.hypot(dx, dy), dx, dy
        dt = t1 - buf[0][2]
        if dt < _MIN_WIN:
            return 0.0, 0.0, 0.0
        dx = (x1 - buf[0][0]) / dt
        dy = (y1 - buf[0][1]) / dt
        return math.hypot(dx, dy), dx, dy

    def _finger_relative_vec(self) -> tuple[float, float, float]:
        """Fingertip velocity relative to wrist (isolates finger-only motion)."""
        n = min(len(self._buf), len(self._tip_buf))
        if n < 2:
            return 0.0, 0.0, 0.0
        t1 = self._tip_buf[-1][2]
        rx1 = self._tip_buf[-1][0] - self._buf[-1][0]
        ry1 = self._tip_buf[-1][1] - self._buf[-1][1]
        for i in range(n - 2, -1, -1):
            dt = t1 - self._tip_buf[i][2]
            if dt >= _SPEED_WIN:
                rx0 = self._tip_buf[i][0] - self._buf[i][0]
                ry0 = self._tip_buf[i][1] - self._buf[i][1]
                dx = (rx1 - rx0) / dt
                dy = (ry1 - ry0) / dt
                return math.hypot(dx, dy), dx, dy
        dt = t1 - self._tip_buf[0][2]
        if dt < _MIN_WIN:
            return 0.0, 0.0, 0.0
        rx0 = self._tip_buf[0][0] - self._buf[0][0]
        ry0 = self._tip_buf[0][1] - self._buf[0][1]
        dx = (rx1 - rx0) / dt
        dy = (ry1 - ry0) / dt
        return math.hypot(dx, dy), dx, dy

    @staticmethod
    def _dirs_agree(ax, ay, bx, by) -> bool:
        """True if two velocity vectors point within _MAX_DIR_DIVERGE_DEG of each other."""
        ma = math.hypot(ax, ay)
        mb = math.hypot(bx, by)
        if ma < 1e-6 or mb < 1e-6:
            return False
        cos_a = (ax * bx + ay * by) / (ma * mb)
        return cos_a >= math.cos(math.radians(_MAX_DIR_DIVERGE_DEG))

    def _fire(self, dx: float, dy: float):
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return
        self._state.on_flick(dx / mag, dy / mag)
        self._last_fire = time.monotonic()
        self._buf.clear()
        self._tip_buf.clear()
