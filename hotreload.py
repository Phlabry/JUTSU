"""
In-process hot reload for the running service.

watcher.py restarts the whole process when a .py file changes, which drops the
virtual camera for a couple of seconds and re-loads every asset.  Almost no edit
needs that: effects, gestures, state machines and config can all be swapped into
a live process.  This module watches the project and reports when something
hot-reloadable has changed and settled; service.py then rebuilds the jutsu stack
on a background thread, keeps streaming from the old one meanwhile, and swaps
the two over between frames.

A module that defines a module-level shutdown() gets it called just before it is
dropped, so it can hand back anything the replacement will want to open again
(audio streams, in practice).
"""
import importlib
import os
import sys
import threading
import time

_DEBOUNCE = 0.6  # quiet period after the last write before a reload is offered

# Editing these restructures the process itself, so they can't be swapped in
# place.  watcher.py restarts the process for them instead.
COLD_FILES = frozenset({"service.py", "watcher.py", "hotreload.py", "tray.py"})

_IGNORE_DIRS = {".venv", "__pycache__", "datasets", ".git", ".vscode", "scripts"}

_lock = threading.Lock()
_pending_since: float | None = None
_last_rel: str = ""
_root: str = ""
_started = False


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


def _relevant(path: str) -> str | None:
    """Project-relative path if this file can be hot-reloaded, else None."""
    if not path.endswith(".py"):
        return None
    try:
        rel = os.path.relpath(path, _root)
    except ValueError:
        return None
    parts = rel.split(os.sep)
    if parts[0] == ".." or any(p in _IGNORE_DIRS for p in parts):
        return None
    if rel.replace(os.sep, "/") in COLD_FILES:
        return None
    return rel


def _note(path: str) -> None:
    global _pending_since, _last_rel
    rel = _relevant(path)
    if rel is None:
        return
    with _lock:
        _pending_since = time.monotonic()
        _last_rel = rel


def _poll_loop() -> None:
    """Fallback when watchdog isn't installed: compare mtimes once a second."""
    seen: dict[str, float] = {}
    first = True
    while True:
        for dirpath, dirnames, filenames in os.walk(_root):
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if seen.get(path) != mtime:
                    seen[path] = mtime
                    if not first:
                        _note(path)
        first = False
        time.sleep(1.0)


def start(root: str) -> None:
    """Begin watching `root` for .py changes.  Safe to call more than once."""
    global _root, _started
    if _started:
        return
    _root = os.path.abspath(root)
    _started = True

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory:
                    _note(event.src_path)

            on_created = on_modified

        observer = Observer()
        observer.daemon = True
        observer.schedule(_Handler(), _root, recursive=True)
        observer.start()
    except Exception:
        threading.Thread(target=_poll_loop, daemon=True).start()


def consume() -> str | None:
    """
    The most recently edited hot-reloadable file, once writing has settled —
    then nothing again until the next change.  Editors save in bursts, hence the
    debounce.
    """
    global _pending_since
    with _lock:
        if _pending_since is None or time.monotonic() - _pending_since < _DEBOUNCE:
            return None
        _pending_since = None
        return _last_rel


# ---------------------------------------------------------------------------
# Module swap
# ---------------------------------------------------------------------------


def _top_level_names() -> set[str]:
    """
    Import names that belong to this project.  Filtering on these first matters:
    reading __file__ off an arbitrary sys.modules entry can execute a lazily
    loaded third-party module (cupy.testing does exactly that) and blow up.
    """
    names = set()
    for entry in os.listdir(_root):
        if entry in _IGNORE_DIRS or entry.startswith("."):
            continue
        full = os.path.join(_root, entry)
        if entry.endswith(".py"):
            names.add(entry[:-3])
        elif os.path.isdir(full):
            try:
                if any(f.endswith(".py") for f in os.listdir(full)):
                    names.add(entry)
            except OSError:
                pass
    return names


def drop_project_modules() -> int:
    """
    Forget every project module so the next import reads the edited source.

    Objects built from the old modules keep working — a function holds its own
    module globals — which is what lets the pipeline carry on rendering with the
    outgoing jutsu stack while the replacement is being built.
    """
    top = _top_level_names()
    dropped = 0

    for name, mod in list(sys.modules.items()):
        if name in ("__main__", __name__) or mod is None:
            continue
        if name.split(".")[0] not in top:
            continue

        try:
            path = getattr(mod, "__file__", None)
        except Exception:
            continue

        if path:
            try:
                rel = os.path.relpath(os.path.abspath(path), _root)
            except ValueError:
                continue
            parts = rel.split(os.sep)
            if parts[0] == ".." or any(p in _IGNORE_DIRS for p in parts):
                continue
            if rel.replace(os.sep, "/") in COLD_FILES:
                continue

            release = getattr(mod, "shutdown", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
        # path is None for the namespace packages (effects/, gesture/) — dropping
        # them costs nothing and keeps the tree consistent.

        del sys.modules[name]
        dropped += 1

    importlib.invalidate_caches()
    return dropped
