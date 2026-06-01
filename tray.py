import threading

import pystray
from PIL import Image, ImageDraw

_stop_event = threading.Event()
_pipeline: threading.Thread | None = None
_icon_ref: pystray.Icon | None = None


def _make_icon(active: bool = False) -> Image.Image:
    img  = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill    = (160, 0, 220) if active else (70, 0, 100)
    outline = (220, 100, 255) if active else (140, 60, 180)
    draw.ellipse([4, 4, 60, 60], fill=fill, outline=outline, width=3)
    return img


def _run_pipeline() -> None:
    from main import run
    run(headless=True, stop_event=_stop_event)
    if _icon_ref:
        _icon_ref.icon  = _make_icon(active=False)
        _icon_ref.title = "JUTSU"


def _activate(icon, _item) -> None:
    global _pipeline
    if _pipeline and _pipeline.is_alive():
        return
    _stop_event.clear()
    _pipeline = threading.Thread(target=_run_pipeline, daemon=True)
    _pipeline.start()
    icon.icon  = _make_icon(active=True)
    icon.title = "JUTSU — Active"


def _deactivate(icon, _item) -> None:
    _stop_event.set()
    icon.icon  = _make_icon(active=False)
    icon.title = "JUTSU"


def _quit(icon, _item) -> None:
    _stop_event.set()
    icon.stop()


if __name__ == "__main__":
    icon = pystray.Icon(
        name="JUTSU",
        icon=_make_icon(active=False),
        title="JUTSU",
        menu=pystray.Menu(
            pystray.MenuItem("Activate", _activate, default=True),
            pystray.MenuItem("Deactivate", _deactivate),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", _quit),
        ),
    )
    _icon_ref = icon
    icon.run()
