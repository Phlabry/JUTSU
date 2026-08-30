def create(fw: int, fh: int):
    from state.hollow_purple import HollowPurpleState
    from gesture.hollow_purple.charging import ChargingDetector
    from gesture.hollow_purple.releasing import ReleasingDetector

    state     = HollowPurpleState()
    detectors = [ChargingDetector(state), ReleasingDetector(state)]

    def prewarm():
        from effects.hollow_purple import charging as _c
        from effects.hollow_purple import releasing as _r
        from effects import gpu as _gpu
        _c._build_ready()
        _r._load_frames()
        _r._build_ready(fw, fh)
        if _gpu.HAVE_GPU:
            _gpu.warmup(fw, fh)

    return state, detectors, prewarm


def init_audio() -> None:
    from state.hollow_purple import _ensure_vmm
    _ensure_vmm()
