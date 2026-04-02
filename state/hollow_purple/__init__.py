from enum import Enum, auto

import numpy as np

from effects.hollow_purple import charging as charging_fx
from effects.hollow_purple import releasing as releasing_fx

_CHARGE_TIMEOUT = 10   # frames without right hand in frame before dropping to IDLE
_CHARGE_MIN     = 12   # accumulated charge hits required before release is armed
_EXPAND_PX      = 150  # px the release circle grows per frame
_MAX_RADIUS     = 2000


class _State(Enum):
    IDLE     = auto()
    CHARGING = auto()
    RELEASED = auto()


class HollowPurpleState:
    #   IDLE ──(charge detected)──► CHARGING
    #   CHARGING ──(release detected, _charge_hits >= _CHARGE_MIN)──► RELEASED
    #   CHARGING ──(no hand for _CHARGE_TIMEOUT frames)──► IDLE
    #   RELEASED ──(animation done)──► IDLE

    def __init__(self):
        self._state          = _State.IDLE
        self._charge_misses  = 0
        self._charge_hits    = 0   # total confirmed-charge frames in this session
        self._last_top_px    = None
        self._last_base_r    = 40
        self._release_radius = 0

    def update_position(self, top_px: tuple, base_radius: int):
        if self._state is _State.CHARGING:
            self._last_top_px = top_px
            self._last_base_r = base_radius

    def on_charge(self, top_px: tuple, base_radius: int):
        if self._state in (_State.IDLE, _State.CHARGING):
            self._state         = _State.CHARGING
            self._charge_misses = 0
            self._charge_hits  += 1
            self._last_top_px   = top_px
            self._last_base_r   = base_radius

    def on_charge_lost(self):
        if self._state is _State.CHARGING:
            self._charge_misses += 1
            if self._charge_misses >= _CHARGE_TIMEOUT:
                self._state       = _State.IDLE
                self._charge_hits = 0

    def on_release(self):
        if self._state is _State.CHARGING and self._charge_hits >= _CHARGE_MIN:
            self._state          = _State.RELEASED
            self._charge_hits    = 0
            self._release_radius = self._last_base_r

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self._state is _State.CHARGING and self._last_top_px:
            frame = charging_fx.render(frame, self._last_top_px, self._last_base_r)

        elif self._state is _State.RELEASED and self._last_top_px:
            self._release_radius += _EXPAND_PX
            alpha = 1.0 - (self._release_radius / _MAX_RADIUS)
            if alpha <= 0:
                self._state = _State.IDLE
            else:
                frame = releasing_fx.render(frame, self._last_top_px, self._release_radius, alpha)

        return frame
