import os
from gesture.custom_base import CustomGestureDetector
from effects.hollow_purple_charging import ChargingEffect

_MODEL = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "hollow_purple.pkl"))


class ChargingDetector(CustomGestureDetector):
    gesture_name = "charge"

    def __init__(self):
        super().__init__(_MODEL)
        self._effect = ChargingEffect()

    def on_detect(self, confidence: float):
        self._effect.trigger()
