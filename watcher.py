"""
Wraps service.py with auto-reload. Register this as the autostart entry instead
of service.py (setup_autostart.py already does this).

The service hot-reloads its own jutsu stack in place — effects, gestures, state
and config all go live without dropping the virtual camera (see hotreload.py).
So this only restarts the process for the handful of files that restructure the
process itself and therefore can't be swapped into it.
"""
import os
import subprocess
import sys
import threading
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.join(ROOT, "service.py")
_log_file = open(os.path.join(ROOT, "watcher.log"), "a", buffering=1, encoding="utf-8")

from hotreload import COLD_FILES

# Editing watcher.py itself can't be handled from in here — restart it by hand.
_RESTART_FILES = COLD_FILES - {"watcher.py"}
_DEBOUNCE = 1.0  # seconds to wait after last change before restarting

_proc: subprocess.Popen | None = None
_timer: threading.Timer | None = None
_lock = threading.Lock()


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line)
    _log_file.write(line + "\n")


def _start() -> None:
    global _proc
    _log("[Watcher] starting service.py")
    _proc = subprocess.Popen([sys.executable, SERVICE])


def _do_restart() -> None:
    global _proc, _timer
    with _lock:
        _timer = None
        if _proc and _proc.poll() is None:
            _log("[Watcher] stopping service.py")
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _start()


def _schedule_restart(path: str) -> None:
    global _timer
    with _lock:
        if _timer:
            _timer.cancel()
        _log(f"[Watcher] changed: {os.path.relpath(path, ROOT)} — restarting in {_DEBOUNCE}s")
        _timer = threading.Timer(_DEBOUNCE, _do_restart)
        _timer.daemon = True
        _timer.start()


class _Handler(FileSystemEventHandler):
    def _relevant(self, path: str) -> bool:
        if not path.endswith(".py"):
            return False
        try:
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        except ValueError:
            return False
        return rel in _RESTART_FILES

    def on_modified(self, event):
        if not event.is_directory and self._relevant(event.src_path):
            _schedule_restart(event.src_path)

    def on_created(self, event):
        self.on_modified(event)


if __name__ == "__main__":
    _start()

    observer = Observer()
    observer.schedule(_Handler(), ROOT, recursive=True)
    observer.start()
    _log(f"[Watcher] watching {ROOT} for changes to {sorted(_RESTART_FILES)}")

    try:
        while True:
            time.sleep(2)
            if _proc and _proc.poll() is not None:
                _log("[Watcher] service.py exited unexpectedly — restarting in 5s")
                time.sleep(5)
                _start()
    except KeyboardInterrupt:
        _log("[Watcher] shutting down")
        observer.stop()
        if _proc and _proc.poll() is None:
            _proc.terminate()

    observer.join()
