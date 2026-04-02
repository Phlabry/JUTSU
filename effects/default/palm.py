from effects.base import Effect


class OpenPalmEffect(Effect):
    def trigger(self):
        print("Open hand detected")
