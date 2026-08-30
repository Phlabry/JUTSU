# Every jutsu the app knows about, and whether it is switched on.
# Flip a value and save — the running service picks it up without a restart.
ACTIVE_JUTSU: dict[str, bool] = {
    "hollow_purple": False,
    "cleave": True,
}


def enabled_jutsu() -> list[str]:
    """Names of the jutsu currently switched on, in declaration order."""
    return [name for name, on in ACTIVE_JUTSU.items() if on]


# Master volume multiplier applied on top of each jutsu's own levels.
# 1.0 = full, 0.5 = half, 0.0 = silent.
AUDIO_VOLUME: float = 0.05

# None = Windows default mic. Set to a name substring to pick a specific device,
# e.g. "SteelSeries Sonar - Chat". Run service.py once to see the full device list.
MIC_DEVICE: str | None = None

# Capture width requested from the webcam; the height follows the sensor's 16:9
# mode. Over DirectShow this camera only offers uncompressed YUY2, which caps
# 720p at 10fps — Media Foundation gets 30fps out of it, at the cost of ~7s to
# open the device instead of ~5s. Set CAMERA_BACKEND to "dshow" to trade the
# frame rate back for the faster start.
CAMERA_WIDTH: int = 1280
CAMERA_BACKEND: str = "msmf"  # "msmf" or "dshow"

VIRTUAL_CAM_WIDTH: int = 1280
VIRTUAL_CAM_HEIGHT: int = 720

# Effects are composited at this size. Keeping it equal to the virtual cam size
# means the frame is never upscaled, so the output stays as sharp as the webcam.
PROC_WIDTH: int = 1280
PROC_HEIGHT: int = 720

# MediaPipe only needs enough pixels to find a hand, and it returns normalised
# landmarks, so detection runs on its own smaller frame. Raising this costs
# inference time without making the picture any sharper.
DETECT_WIDTH: int = 640
DETECT_HEIGHT: int = 360

# Width of the off-screen canvas the charge starburst is drawn on before it is
# upscaled onto the frame. Cost scales with its area, and 320 reproduces what
# the old 640x360 pipeline showed. Raise it for crisper arms if you have
# headroom; the [Pipeline] heartbeat in service.log reports the frame budget.
STARBURST_WIDTH: int = 320
