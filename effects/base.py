class Effect:
    def trigger(self):
        raise NotImplementedError


class FrameEffect:
    """Per-frame visual renderer attached to a state machine.
    Subclasses define their own render() signature."""
    def render(self, frame, **kwargs):
        raise NotImplementedError
