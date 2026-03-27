from gesture.base import GestureDetector
from effects.pointing_up import PointingUpEffect

class PointingUpDetector(GestureDetector):
    gesture_name = 'Pointing_Up'

    def __init__(self):
        super().__init__()
        self._effect = PointingUpEffect()

    def on_detect(self):
        self._effect.trigger()
