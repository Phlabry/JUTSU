class Effect:
    def trigger(self):
        raise NotImplementedError


class FrameEffect:
    def render(self, frame, **kwargs):
        raise NotImplementedError
