from gesture.custom_base import CustomGestureDetector
from gesture.hollow_purple import _MODEL


class ReleasingDetector(CustomGestureDetector):
    gesture_name = "release"

    def __init__(self, state):
        super().__init__(_MODEL)
        self._state = state

    def on_detect(self, confidence: float, top_px: tuple, base_radius: int):
        self._state.on_release()
