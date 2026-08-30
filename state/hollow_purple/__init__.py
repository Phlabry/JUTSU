import os
from enum import Enum, auto

import numpy as np

from effects.hollow_purple import charging as charging_fx
from effects.hollow_purple import releasing as releasing_fx

_CHARGE_TIMEOUT = 10
_CHARGE_MIN = 12
_EXPAND_PX = 50
_MAX_RADIUS = 2000

_AUDIO_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "hollow_purple", "audio"
    )
)

_audio_ready = False
_charge_ch = None
_release_ch = None
_charge_snd = None
_release_snd = None

_vmm = None
_vmm_charge_id: int | None = None


def _init_audio() -> bool:
    global _audio_ready, _charge_ch, _release_ch, _charge_snd, _release_snd
    if _audio_ready:
        return True
    try:
        import pygame

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        _charge_snd = pygame.mixer.Sound(os.path.join(_AUDIO_DIR, "Charge.wav"))
        _release_snd = pygame.mixer.Sound(os.path.join(_AUDIO_DIR, "Release.wav"))
        _charge_snd.set_volume(0.025)
        _release_snd.set_volume(0.0875)
        _charge_ch = pygame.mixer.Channel(0)
        _release_ch = pygame.mixer.Channel(1)
        _audio_ready = True
        return True
    except Exception as e:
        print(f"[Audio] init failed: {e}")
        return False


def _ensure_vmm():
    global _vmm
    if _vmm is not None:
        return _vmm
    try:
        from audio.virtual_mic import VirtualMicMixer
        from config import MIC_DEVICE

        _vmm = VirtualMicMixer(_AUDIO_DIR, mic_device=MIC_DEVICE)
        _vmm.start()
    except Exception as e:
        print(f"[VirtualMic] init failed: {e}")
    return _vmm


def _master() -> float:
    try:
        from config import AUDIO_VOLUME
        return float(AUDIO_VOLUME)
    except Exception:
        return 1.0


def _start_charge() -> None:
    global _vmm_charge_id
    if not _init_audio():
        return
    mv = _master()
    if not _charge_ch.get_busy():
        _charge_ch.set_volume(mv)
        _charge_ch.play(_charge_snd, loops=-1)
    vmm = _ensure_vmm()
    if vmm and _vmm_charge_id is None:
        _vmm_charge_id = vmm.play("charge", volume=0.25 * mv, loop=True)


def _start_release() -> None:
    if not _audio_ready:
        return
    mv = _master()
    _release_ch.play(_release_snd)
    if _vmm:
        _vmm.play("release", volume=0.79 * mv)


def _set_charge_volume(ratio: float) -> None:
    mv = _master()
    if _audio_ready:
        _charge_ch.set_volume(ratio * mv)
    if _vmm and _vmm_charge_id is not None:
        _vmm.set_volume(_vmm_charge_id, ratio * 0.25 * mv)


def _stop_all() -> None:
    global _vmm_charge_id
    if _audio_ready:
        _charge_ch.set_volume(1.0)
        _charge_ch.stop()
        _release_ch.stop()
    if _vmm:
        _vmm.stop_all()
    _vmm_charge_id = None


def shutdown() -> None:
    """
    Hand back the audio devices before a hot reload drops this module — the
    replacement opens its own virtual-mic stream and the two can't share one.
    """
    global _vmm, _vmm_charge_id, _audio_ready
    try:
        _stop_all()
    except Exception:
        pass
    if _vmm is not None:
        try:
            _vmm.close()
        except Exception:
            pass
    _vmm = None
    _vmm_charge_id = None
    _audio_ready = False


class _State(Enum):
    IDLE = auto()
    CHARGING = auto()
    RELEASED = auto()


class HollowPurpleState:
    #   IDLE ──(charge detected)──► CHARGING
    #   CHARGING ──(release detected, _charge_hits >= _CHARGE_MIN)──► RELEASED
    #   CHARGING ──(no hand for _CHARGE_TIMEOUT frames)──► IDLE
    #   RELEASED ──(animation done)──► IDLE

    def __init__(self):
        self._state = _State.IDLE
        self._charge_misses = 0
        self._charge_hits = 0
        self._last_top_px = None
        self._last_base_r = 40
        self._release_radius = 0

    def update_position(self, top_px: tuple, base_radius: int):
        if self._state is _State.CHARGING:
            self._last_top_px = top_px
            self._last_base_r = base_radius

    def on_charge(self, top_px: tuple, base_radius: int):
        if self._state in (_State.IDLE, _State.CHARGING):
            was_idle = self._state is _State.IDLE
            self._state = _State.CHARGING
            self._charge_misses = 0
            self._charge_hits += 1
            self._last_top_px = top_px
            self._last_base_r = base_radius
            if was_idle:
                _start_charge()

    def on_charge_lost(self):
        if self._state is _State.CHARGING:
            self._charge_misses += 1
            if self._charge_misses >= _CHARGE_TIMEOUT:
                self._state = _State.IDLE
                self._charge_hits = 0
                _stop_all()

    def on_release(self):
        if self._state is _State.CHARGING and self._charge_hits >= _CHARGE_MIN:
            self._state = _State.RELEASED
            self._charge_hits = 0
            self._release_radius = self._last_base_r
            _start_release()

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self._state is _State.CHARGING and self._last_top_px:
            frame = charging_fx.render(frame, self._last_top_px, self._last_base_r)

        elif self._state is _State.RELEASED and self._last_top_px:
            self._release_radius += _EXPAND_PX
            alpha = 1.0 - (self._release_radius / _MAX_RADIUS)
            if alpha <= -0.50:
                self._state = _State.IDLE
                _stop_all()
            else:
                if alpha < 0:
                    _set_charge_volume(max(0.0, (alpha + 0.50) / 0.50))
                frame = releasing_fx.render(
                    frame, self._last_top_px, self._release_radius, alpha
                )

        return frame
