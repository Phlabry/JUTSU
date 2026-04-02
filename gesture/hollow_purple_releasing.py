import os
from gesture.custom_base import CustomGestureDetector
from effects.hollow_purple_releasing import ReleasingEffect

_MODEL = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "hollow_purple.pkl"))


class ReleasingDetector(CustomGestureDetector):
    gesture_name = "release"

    def __init__(self):
        super().__init__(_MODEL)
        self._effect = ReleasingEffect()

    def on_detect(self, confidence: float):
        self._effect.trigger()
