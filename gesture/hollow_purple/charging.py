from gesture.custom_base import CustomGestureDetector
from gesture.hollow_purple import _MODEL


class ChargingDetector(CustomGestureDetector):
    gesture_name = "charge"

    def __init__(self, state):
        super().__init__(_MODEL)
        self._state = state

    def on_hand_visible(self, top_px: tuple, base_radius: int):
        self._state.update_position(top_px, base_radius)

    def on_detect(self, confidence: float, top_px: tuple, base_radius: int):
        self._state.on_charge(top_px, base_radius)

    def on_no_detect(self):
        self._state.on_charge_lost()
