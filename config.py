ACTIVE_JUTSU: list[str] = ["cleave"]

# None = Windows default mic. Set to a name substring to pick a specific device,
# e.g. "SteelSeries Sonar - Chat". Run service.py once to see the full device list.
MIC_DEVICE: str | None = None

VIRTUAL_CAM_WIDTH: int = 1280
VIRTUAL_CAM_HEIGHT: int = 720

# MediaPipe and effects run at this size; output is upscaled to virtual cam size.
PROC_WIDTH: int = 640
PROC_HEIGHT: int = 360
