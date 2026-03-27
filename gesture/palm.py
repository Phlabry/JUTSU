from gesture.base import GestureDetector
from effects.palm import OpenPalmEffect

class OpenPalmDetector(GestureDetector):
    gesture_name = 'Open_Palm'

    def __init__(self):
        super().__init__()
        self._effect = OpenPalmEffect()

    def on_detect(self):
        self._effect.trigger()
