import math
import os
import time
from dataclasses import dataclass, field

from effects.cleave import slash as slash_fx

_DURATION = 0.270
_SND_VOL  = 0.85   # jutsu-specific base volume; multiplied by config.AUDIO_VOLUME

_AUDIO_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "cleave", "audio")
)

_audio_ready = False
_snd = None

_vmm = None
_VMM_SLASH = "cleave/slash"


def _init_audio() -> bool:
    global _audio_ready, _snd, _vmm
    if _audio_ready:
        return True
    try:
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        if not os.path.isdir(_AUDIO_DIR):
            return False

        exts = (".wav", ".mp3", ".ogg", ".flac")
        files = [f for f in os.listdir(_AUDIO_DIR) if f.lower().endswith(exts)]
        if not files:
            return False

        path = os.path.join(_AUDIO_DIR, files[0])
        _snd = pygame.mixer.Sound(path)
        _audio_ready = True
    except Exception as e:
        print(f"[Cleave audio] init failed: {e}")
        return False

    # Also route the slash into VB-CABLE so OBS/Discord pick it up.
    try:
        from audio.virtual_mic import get_mixer
        vmm = get_mixer()
        if vmm is not None and vmm.load(_VMM_SLASH, path):
            _vmm = vmm
    except Exception as e:
        print(f"[Cleave audio] virtual mic unavailable: {e}")
    return True


def _play() -> None:
    if not (_audio_ready and _snd is not None):
        return
    try:
        from config import AUDIO_VOLUME
        vol = _SND_VOL * AUDIO_VOLUME
    except Exception:
        vol = _SND_VOL
    _snd.set_volume(vol)
    _snd.play()
    if _vmm is not None:
        _vmm.play(_VMM_SLASH, volume=vol)


def shutdown() -> None:
    """Drop the loaded sound before a hot reload replaces this module."""
    global _snd, _audio_ready, _vmm
    try:
        if _snd is not None:
            _snd.stop()
    except Exception:
        pass
    _snd = None
    _vmm = None
    _audio_ready = False


@dataclass
class _Slash:
    angle:      float
    start_time: float = field(default_factory=time.monotonic)
    sparks:     list  = field(default_factory=list)

    @property
    def t(self) -> float:
        return min(1.0, (time.monotonic() - self.start_time) / _DURATION)

    @property
    def expired(self) -> bool:
        return self.t >= 1.0


class CleaveState:
    def __init__(self):
        self._active: list[_Slash] = []

    def on_flick(self, dx: float, dy: float) -> None:
        self._active.append(_Slash(angle=math.atan2(dy, dx)))
        _play()

    def render(self, frame):
        self._active = [s for s in self._active if not s.expired]
        for slash in self._active:
            frame = slash_fx.render(frame, slash.angle, slash.t, slash.sparks)
        return frame
