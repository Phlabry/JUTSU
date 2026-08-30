"""
Works around a broken MediaPipe wheel on Windows.

mediapipe 0.10.30's ctypes loader ends every load with

    _shared_lib.free.argtypes = [ctypes.c_void_p]

which assumes `free` is reachable through the task library.  On Linux and macOS
it is — the dynamic loader resolves it out of libc via libmediapipe.so/.dylib.
Windows has no such global symbol namespace and libmediapipe.dll exports no
`free` of its own, so the lookup raises

    AttributeError: function 'free' not found

before any task object can be built.  Nothing in the project can catch it: it
fires inside create_from_options().

The fix is to load the DLL ourselves, hang msvcrt's `free` off it, and hand the
result to MediaPipe as its already-loaded library.  MediaPipe then skips its own
load, and the `.argtypes` assignment lands harmlessly on the msvcrt pointer.

MediaPipe allocates the buffers this frees (error strings, result structs) with
its own CRT rather than msvcrt's, which is a heap mismatch on paper.  In
practice both route to the process default heap on Windows 10+, and this has
been running against 0.10.30 without incident.  Revisit if crashes ever show up
around task teardown or error paths.

No-op off Windows, on a MediaPipe old enough to predate the ctypes bindings, and
on any future wheel that exports `free` properly — so it can stay in place after
the upstream fix lands.
"""

import ctypes
import os
from importlib import resources

_applied = False


def apply() -> bool:
    """Pre-load the task DLL with a working `free`.  True if it was needed."""
    global _applied
    if _applied or os.name != "nt":
        return False
    _applied = True  # one attempt per process, success or not

    try:
        from mediapipe.tasks.python.core import mediapipe_c_bindings as bindings
    except ImportError:
        return False  # older MediaPipe, no ctypes bindings to fix

    if bindings._shared_lib is not None:
        return False  # already loaded by someone who got there first

    try:
        lib_path = str(resources.files("mediapipe.tasks.c") / "libmediapipe.dll")
        lib = ctypes.CDLL(lib_path)
    except (ImportError, ModuleNotFoundError, OSError):
        return False  # let MediaPipe fail on its own terms

    try:
        lib.free
        return False  # wheel is fine, stay out of the way
    except AttributeError:
        pass

    free = ctypes.CDLL("msvcrt").free  # keeps the msvcrt CDLL alive by reference
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    lib.free = free

    bindings._shared_lib = lib
    return True
