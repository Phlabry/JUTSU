"""
GPU acceleration layer for effects.

Handles NVIDIA DLL discovery on Windows, initialises CuPy, and exposes
a minimal interface used by the effect modules. Falls back to CPU silently
if CuPy is not installed or the GPU is unavailable.
"""
import glob
import os

HAVE_GPU: bool = False
cp       = None        # cupy module or None
gaussian_filter = None  # cupyx.scipy.ndimage.gaussian_filter or None


def _init() -> None:
    global HAVE_GPU, cp, gaussian_filter

    # ── Windows: register every NVIDIA package DLL directory ─────────────────
    # CuPy ships runtime libs as separate pip packages (nvidia-cuda-*-cu12).
    # os.add_dll_directory() makes them visible to LoadLibrary without PATH edits.
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            nvidia_root = list(spec.submodule_search_locations)[0]
            for dll in glob.glob(os.path.join(nvidia_root, "**", "*.dll"), recursive=True):
                try:
                    os.add_dll_directory(os.path.dirname(dll))
                except (AttributeError, OSError):
                    pass
    except Exception:
        pass

    try:
        import cupy as _cp
        from cupyx.scipy.ndimage import gaussian_filter as _gf

        # Verify the device is reachable and prime the JIT for ops we use
        _t = _cp.zeros((8, 8), dtype=_cp.float32)
        _cp.abs(_t - 1.0, out=_t)
        _cp.clip(_t, 0.0, 1.0, out=_t)
        _cp.cuda.Stream.null.synchronize()

        cp              = _cp
        gaussian_filter = _gf
        HAVE_GPU        = True

        dev = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"[GPU] {dev} — wave effects on GPU")

    except Exception as e:
        print(f"[GPU] CPU fallback ({type(e).__name__}: {e})")


def warmup(fw: int, fh: int) -> None:
    """
    Trigger JIT compilation for all kernels used at runtime.
    Call this once from the prewarm thread so the first rendered frame
    does not stall on kernel compilation.
    """
    if not HAVE_GPU:
        return
    dist = cp.zeros((fh, fw), dtype=cp.float32)
    draw = cp.zeros((fh, fw, 3), dtype=cp.float32)
    for r in (50.0, 200.0, 400.0, 600.0):
        diff  = cp.abs(dist - r)
        prof  = cp.clip(1.0 - diff / 90.0, 0.0, 1.0) ** 2
        draw += prof[:, :, cp.newaxis] * cp.array([100.0, 50.0, 200.0])
    cp.clip(draw, 0, 255, out=draw)
    gaussian_filter(draw, sigma=[2.5, 2.5, 0], output=draw)
    draw.astype(cp.uint8).get()
    cp.cuda.Stream.null.synchronize()


_init()
