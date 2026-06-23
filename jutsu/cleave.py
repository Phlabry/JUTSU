def create(fw: int, fh: int):
    from state.cleave import CleaveState
    from gesture.cleave.flick import FlickDetector
    from effects.cleave import slash as slash_fx

    state     = CleaveState()
    detectors = [FlickDetector(state)]

    def prewarm():
        slash_fx.prewarm(fw, fh)

    return state, detectors, prewarm


def init_audio() -> None:
    from state.cleave import _init_audio
    _init_audio()
