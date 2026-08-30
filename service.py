"""
Runs at login (registered via setup_autostart.py).

Idles at ~0% CPU/GPU with the camera OFF.  When any consumer (Discord etc.)
opens the UnityCapture virtual camera the physical camera starts, effects run,
and frames stream.  When the consumer closes it everything shuts down and the
camera light goes off.

Connect detection (idle loop): times cam.send(black).  No consumer →
  OpenFileMappingA fails inside C ext → < 0.5 ms.  Consumer open → 2.7 MB
  memcpy fires → > 1 ms.  send_ms > 1.5 (debounced ×8) → start pipeline.

Disconnect detection (pipeline): two independent mechanisms — either fires:
  1. UnityCapture_Want0 kernel event: created by the consumer's DLL, destroyed
     when the consumer closes the camera.  Checked every 30 frames.
  2. Consecutive fast sends: if 120 sends in a row take < 0.8 ms the consumer
     is no longer pulling frames (memcpy cost disappears without a reader).

Editing any .py outside hotreload.COLD_FILES rebuilds the jutsu stack in place
(see hotreload.py) — the stream never stops and the camera light never blinks.

Logs to service.log (no console when running as pythonw.exe).
"""

import ctypes
from ctypes import wintypes as _wintypes
import importlib
import os
import queue
import threading
import time

import cv2 as cv
import numpy as np
import pyvirtualcam

# Held as module handles rather than from-imports: a hot reload replaces the
# module objects, and _rebind() re-points these at the new ones.
import camera.feed as camera_feed
import config
import hotreload
import jutsu as jutsu_registry
import tracking.hand_detector as hand_detector

_POLL_HZ = 10  # polls per second while pipeline is active
_ROOT = os.path.dirname(os.path.abspath(__file__))
_log_file = open(os.path.join(_ROOT, "service.log"), "a", buffering=1, encoding="utf-8")

_k32 = ctypes.windll.kernel32
_user32 = ctypes.windll.user32


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line)
    _log_file.write(line + "\n")
    _log_file.flush()


# ---------------------------------------------------------------------------
# Consumer detection
# ---------------------------------------------------------------------------

_DEBOUNCE_POLLS = 8  # consecutive same-state reads required before acting


def _want_event_exists() -> bool:
    """
    True while a UnityCapture consumer (Discord etc.) has the virtual camera
    open.  The consumer's DLL creates 'UnityCapture_Want0' when it starts
    capturing and destroys it (all handles closed) when it stops.  This gives
    us a reliable, instantaneous disconnect signal that doesn't depend on frame
    timing.  Returns False both when the event doesn't exist AND when it can't
    be opened for any other reason — safe to treat as 'no consumer'.
    """
    h = _k32.OpenEventA(0x00100000, False, b"UnityCapture_Want0")  # SYNCHRONIZE
    if h:
        _k32.CloseHandle(h)
        return True
    return False


# ---------------------------------------------------------------------------
# Global hotkey  (Ctrl+Shift+Alt+J — toggle pipeline on / off)
# ---------------------------------------------------------------------------

_WM_HOTKEY = 0x0312
_PM_REMOVE = 0x0001
_MOD_CTRL = 0x0002
_MOD_SHIFT = 0x0004
_MOD_ALT = 0x0001
_HOTKEY_ID = 1
_HOTKEY_MODS = _MOD_CTRL | _MOD_SHIFT | _MOD_ALT
_HOTKEY_VK = 0x4A  # 'J'
_HOTKEY_DESC = "Ctrl+Shift+Alt+J"

_hotkey_toggle = threading.Event()  # set by hotkey thread; consumed by main loop
_manual_off = threading.Event()  # set after manual stop; blocks auto-restart until
# consumer goes away (mapping absent → cleared)


def _hotkey_loop() -> None:
    """
    Daemon thread: registers a global hotkey with Windows and fires
    _hotkey_toggle whenever the user presses the combination.
    Uses RegisterHotKey (no extra libraries required).
    """
    if not _user32.RegisterHotKey(None, _HOTKEY_ID, _HOTKEY_MODS, _HOTKEY_VK):
        _log(
            f"[Hotkey] RegisterHotKey failed — {_HOTKEY_DESC} not available "
            "(another app may have claimed it)"
        )
        return
    _log(f"[Hotkey] {_HOTKEY_DESC} → toggle camera pipeline")
    msg = _wintypes.MSG()
    try:
        while True:
            while _user32.PeekMessageA(
                ctypes.byref(msg), None, _WM_HOTKEY, _WM_HOTKEY, _PM_REMOVE
            ):
                if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                    _hotkey_toggle.set()
                    _log(f"[Hotkey] {_HOTKEY_DESC} pressed")
            time.sleep(0.05)
    finally:
        _user32.UnregisterHotKey(None, _HOTKEY_ID)


# ---------------------------------------------------------------------------
# Jutsu stack
# ---------------------------------------------------------------------------


def _render_size() -> tuple[int, int]:
    return config.PROC_WIDTH, config.PROC_HEIGHT


def _detect_size() -> tuple[int, int]:
    return config.DETECT_WIDTH, config.DETECT_HEIGHT


def _cam_size() -> tuple[int, int]:
    return config.VIRTUAL_CAM_WIDTH, config.VIRTUAL_CAM_HEIGHT


def _build_stack(render_size: tuple[int, int], prewarm_sync: bool):
    """(names, modules, detectors) for every jutsu switched on in config."""
    fw, fh = render_size
    names = config.enabled_jutsu()
    modules = [jutsu_registry.load(name, fw, fh) for name in names]

    for name in names:
        jutsu_registry.init_audio(name)

    for _, _, prewarm_fn in modules:
        if prewarm_sync:
            prewarm_fn()
        else:
            threading.Thread(target=prewarm_fn, daemon=True).start()

    detectors = [det for _, dets, _ in modules for det in dets]
    return names, modules, detectors


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------

_swap_slot: list = []  # mailbox: a rebuilt stack waiting for the pipeline to take it
_reload_busy = threading.Event()
_cam_dirty = threading.Event()  # virtual cam geometry changed — main loop reopens it


def _rebind() -> None:
    """Re-point the module handles at the freshly imported project modules."""
    global camera_feed, config, jutsu_registry, hand_detector
    config = importlib.import_module("config")
    jutsu_registry = importlib.import_module("jutsu")
    camera_feed = importlib.import_module("camera.feed")
    hand_detector = importlib.import_module("tracking.hand_detector")


def _reload_idle() -> None:
    """Nothing is streaming, so just re-import: the next pipeline start uses it."""
    cam_before = _cam_size()
    hotreload.drop_project_modules()
    _rebind()
    for name in config.enabled_jutsu():
        jutsu_registry.init_audio(name)
    threading.Thread(target=_prewarm_background, daemon=True).start()
    if _cam_size() != cam_before:
        _cam_dirty.set()
    _log(f"[Reload] active: {', '.join(config.enabled_jutsu()) or '(none)'}")


def _reload_live() -> None:
    """
    Streaming: rebuild off the critical path and post the result for the pipeline
    to swap in.  Objects from the outgoing modules keep rendering until then, so
    a broken edit costs nothing but a log line.
    """
    try:
        cam_before = _cam_size()
        hotreload.drop_project_modules()
        _rebind()

        render_size = _render_size()
        names, modules, detectors = _build_stack(render_size, prewarm_sync=True)
        _swap_slot.append((names, modules, detectors, render_size))

        if _cam_size() != cam_before:
            _cam_dirty.set()
    except Exception as e:
        import traceback

        _log(
            f"[Reload] failed — still running the old code: "
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
    finally:
        _reload_busy.clear()


def _check_reload(streaming: bool) -> None:
    changed = hotreload.consume()
    if not changed:
        return
    if streaming:
        if _reload_busy.is_set():
            return
        _reload_busy.set()
        _log(f"[Reload] {changed} changed — rebuilding while the stream keeps running")
        threading.Thread(target=_reload_live, daemon=True).start()
    else:
        _log(f"[Reload] {changed} changed — reloading")
        try:
            _reload_idle()
        except Exception as e:
            import traceback

            # A half-saved file must not take the virtual camera down with it —
            # the next save gets another go.
            _log(
                f"[Reload] failed — still running the old code: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )


# ---------------------------------------------------------------------------
# Background pre-warm  (runs during idle → first activation is near-instant)
# ---------------------------------------------------------------------------


def _prewarm_background() -> None:
    try:
        fw, fh = _render_size()
        for name in config.enabled_jutsu():
            _, _, prewarm_fn = jutsu_registry.load(name, fw, fh)
            prewarm_fn()

        # Pull the MediaPipe .task file into the OS page cache so that the
        # first real HandDetector() call is fast.
        det = hand_detector.HandDetector()
        det.release()

        _log("[Service] pre-warm done — first activation will be fast")
    except Exception as e:
        _log(f"[Service] pre-warm failed: {e}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(
    stop_event: threading.Event, cam: pyvirtualcam.Camera, forced: bool = False
) -> None:
    """
    forced=True: started manually via hotkey.  Disconnect detection is
    skipped; the pipeline runs until stop_event is set (hotkey pressed again)
    or the safety timeout fires.
    forced=False (default): auto-started by consumer detection; disconnect
    detection runs normally.
    """
    cap = camera_feed.open_camera()
    detector = hand_detector.HandDetector()

    render_size = _render_size()
    detect_size = _detect_size()
    out_size = _cam_size()

    # Anything a reload left behind after the last pipeline stopped is stale now.
    _swap_slot.clear()
    names, jutsu_modules, all_detectors = _build_stack(render_size, prewarm_sync=False)

    detect_in = queue.Queue(maxsize=1)
    detect_out = queue.Queue(maxsize=1)

    def _worker():
        while True:
            frame = detect_in.get()
            if frame is None:
                break
            result = detector.detect(frame)
            try:
                detect_out.get_nowait()
            except queue.Empty:
                pass
            detect_out.put(result)

    threading.Thread(target=_worker, daemon=True).start()

    # cap.read() blocks until the sensor hands over the next frame, so reading on
    # the render thread adds the frame's own work to every camera period and the
    # output settles well below the camera's rate.  A reader thread fetches
    # frame N+1 while frame N is being composited; the depth-1 queue drops
    # anything the renderer couldn't keep up with, so latency stays at one frame.
    frame_q: queue.Queue = queue.Queue(maxsize=1)
    capture_stop = threading.Event()

    def _capture():
        while not capture_stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if not cap.isOpened():
                    break
                continue
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                pass

    capture_thread = threading.Thread(target=_capture, daemon=True)
    capture_thread.start()

    mode = "MANUAL" if forced else "AUTO"
    _log(
        f"[Pipeline] started — camera on  [{mode} mode]  "
        f"render {render_size[0]}x{render_size[1]}, detect "
        f"{detect_size[0]}x{detect_size[1]}, out {out_size[0]}x{out_size[1]}  "
        f"[{', '.join(names) or 'no jutsu'}]"
    )
    last_result = None
    _frame_n = 0
    _want_misses = 0
    _consec_fast = 0
    _pipeline_start = time.time()
    _busy_ms = 0.0

    if not forced:
        _want_at_start = _want_event_exists()
        _log(f"[Pipeline] UnityCapture_Want0 present at start: {_want_at_start}")
    else:
        _want_at_start = False  # forced mode ignores Want0

    try:
        while cap.isOpened() and not stop_event.is_set():
            try:
                frame = frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # A rebuilt stack is waiting: take it between frames.  The old one
            # rendered right up to here, so nothing on screen skips.
            if _swap_slot:
                names, jutsu_modules, all_detectors, render_size = _swap_slot.pop()
                detect_size = _detect_size()
                _log(
                    f"[Reload] live — {', '.join(names) or 'no jutsu'} @ "
                    f"{render_size[0]}x{render_size[1]}"
                )

            _frame_n += 1
            t_frame = time.perf_counter()

            # MediaPipe returns normalised landmarks, so it can work from a much
            # smaller frame than the one the effects are composited onto.
            small = cv.resize(frame, detect_size)
            render = (
                frame
                if (frame.shape[1], frame.shape[0]) == render_size
                else cv.resize(frame, render_size)
            )

            try:
                detect_in.get_nowait()
            except queue.Empty:
                pass
            detect_in.put_nowait(small)

            try:
                last_result = detect_out.get_nowait()
            except queue.Empty:
                pass

            if last_result is not None:
                # Detectors only read the frame's dimensions, so passing the
                # render frame puts the landmark pixels in render space.
                for det in all_detectors:
                    det.process_frame(render, last_result)

            annotated = cv.flip(render, 1)
            for _state, _, _ in jutsu_modules:
                annotated = _state.render(annotated)

            if render_size != out_size:
                annotated = cv.resize(annotated, out_size)

            _busy_ms = (time.perf_counter() - t_frame) * 1000

            t0 = time.perf_counter()
            cam.send(annotated)
            send_ms = (time.perf_counter() - t0) * 1000

            # ── Consumer disconnect detection (AUTO mode only) ────────────────
            if not forced:
                if _frame_n % 30 == 0 and _want_at_start:
                    if not _want_event_exists():
                        _want_misses += 1
                        _log(
                            f"[Pipeline] Want0 gone (miss {_want_misses}/3, "
                            f"send={send_ms:.2f}ms)"
                        )
                        if _want_misses >= 3:
                            _log(
                                "[Pipeline] consumer disconnected (Want0 absent) — stopping"
                            )
                            break
                    else:
                        _want_misses = 0

                if send_ms < 0.8:
                    _consec_fast += 1
                    if _consec_fast >= 120:
                        _log(
                            f"[Pipeline] consumer disconnected "
                            f"({_consec_fast} fast sends, last={send_ms:.2f}ms) — stopping"
                        )
                        break
                else:
                    _consec_fast = 0

            if _frame_n % 150 == 0:
                elapsed = int(time.time() - _pipeline_start)
                _log(
                    f"[Pipeline] heartbeat  frame={_busy_ms:.1f}ms/33.3  "
                    f"send={send_ms:.2f}ms  fast_streak={_consec_fast}  "
                    f"want={'present' if _want_event_exists() else 'ABSENT'}  "
                    f"up={elapsed}s"
                )

    except Exception as e:
        import traceback

        _log(f"[Pipeline] crash: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        detect_in.put(None)
        # Join the reader before releasing: cap.release() while the thread is
        # inside cap.read() takes the driver down with it.
        capture_stop.set()
        capture_thread.join(timeout=2)
        detector.release()
        cap.release()
        _log("[Pipeline] stopped — camera off")


# ---------------------------------------------------------------------------
# Virtual camera
# ---------------------------------------------------------------------------


def _open_virtualcam() -> pyvirtualcam.Camera | None:
    width, height = _cam_size()
    try:
        cam = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=30,
            fmt=pyvirtualcam.PixelFormat.BGR,
            print_fps=False,
            backend="unitycapture",
        )
        _log(f"[Service] virtual camera ready: {cam.device} @ {width}x{height}")
        return cam
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    _log("[Service] starting")
    hotreload.start(_ROOT)
    _log(f"[Service] active jutsu: {', '.join(config.enabled_jutsu()) or '(none)'}")
    for name in config.enabled_jutsu():
        jutsu_registry.init_audio(name)
    threading.Thread(target=_prewarm_background, daemon=True).start()
    threading.Thread(target=_hotkey_loop, daemon=True).start()

    while True:
        cam = _open_virtualcam()

        if cam is None:
            _log(
                "[Service] UnityCapture not found — "
                "github.com/schellingb/UnityCapture → Install.bat (Admin) → reboot"
            )
            time.sleep(30)
            continue

        cam_w, cam_h = _cam_size()
        black = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        stop_event = threading.Event()
        pipeline: threading.Thread | None = None
        was_active = False
        _ticks_active = 0  # consecutive True readings
        _ticks_inactive = 0  # consecutive False readings

        try:
            with cam:
                _log("[Service] idle — waiting for consumer")

                while True:
                    _check_reload(streaming=was_active)

                    # Only the virtual camera's own geometry needs a reopen;
                    # everything else the reload already swapped in place.
                    if _cam_dirty.is_set():
                        _cam_dirty.clear()
                        _log("[Reload] virtual camera size changed — reopening")
                        stop_event.set()
                        if pipeline:
                            pipeline.join(timeout=5)
                        _swap_slot.clear()
                        break

                    # ── Manual toggle via hotkey ──────────────────────────────
                    if _hotkey_toggle.is_set():
                        _hotkey_toggle.clear()
                        if was_active:
                            # Pipeline is running → stop it and block auto-restart.
                            # _manual_off stays set until the consumer goes away,
                            # preventing the idle loop from immediately re-triggering.
                            _manual_off.set()
                            _log("[Hotkey] manual stop")
                            stop_event.set()
                            if pipeline:
                                pipeline.join(timeout=2)
                            pipeline = None
                            was_active = False
                            _ticks_active = 0
                            _ticks_inactive = 0
                            _log(
                                "[Service] idle — waiting for consumer "
                                f"(press {_HOTKEY_DESC} to force-start)"
                            )
                        else:
                            _manual_off.clear()
                            _log(
                                f"[Hotkey] manual start — forced mode, "
                                f"press {_HOTKEY_DESC} again to stop"
                            )
                            stop_event.clear()
                            pipeline = threading.Thread(
                                target=_run_pipeline,
                                args=(stop_event, cam, True),
                                daemon=True,
                            )
                            pipeline.start()
                            was_active = True
                            _ticks_active = _DEBOUNCE_POLLS
                            _ticks_inactive = 0
                        continue

                    # Detection: when idle, time cam.send(black).
                    # No consumer → OpenFileMappingA fails inside C ext → < 0.5 ms.
                    # Consumer open → full 2.7 MB memcpy → > 1 ms.
                    if not was_active:
                        t0 = time.perf_counter()
                        cam.send(black)
                        send_ms = (time.perf_counter() - t0) * 1000

                        # When the consumer goes away, lift the manual-off block so
                        # the next consumer open triggers auto-start normally.
                        if send_ms < 0.5:
                            _manual_off.clear()

                        # Don't auto-start while _manual_off is set (user pressed the
                        # hotkey to stop; they'll use the hotkey to restart).
                        raw = send_ms > 1.5 and not _manual_off.is_set()
                    else:
                        send_ms = 0.0
                        raw = (
                            True  # pipeline detects its own disconnect via send timing
                        )

                    # Debounce: require _DEBOUNCE_POLLS consecutive same-state
                    # reads before we act, to survive brief blips.
                    if raw:
                        _ticks_active += 1
                        _ticks_inactive = 0
                    else:
                        _ticks_inactive += 1
                        _ticks_active = 0

                    if not was_active:
                        is_active = _ticks_active >= _DEBOUNCE_POLLS
                    else:
                        is_active = _ticks_inactive < _DEBOUNCE_POLLS

                    # Detect unexpected pipeline crash
                    if was_active and pipeline and not pipeline.is_alive():
                        stop_event.set()
                        pipeline = None
                        was_active = False
                        _ticks_active = 0
                        _ticks_inactive = 0
                        _log("[Pipeline] ended — returning to idle")
                        continue

                    if is_active and not was_active:
                        _log(
                            f"[Service] consumer detected ({send_ms:.1f} ms send) — starting pipeline"
                        )
                        stop_event.clear()
                        pipeline = threading.Thread(
                            target=_run_pipeline,
                            args=(stop_event, cam),
                            daemon=True,
                        )
                        pipeline.start()

                    elif not is_active and was_active:
                        stop_event.set()
                        if pipeline:
                            pipeline.join(timeout=5)
                        pipeline = None
                        _log("[Service] idle — waiting for consumer")

                    was_active = is_active

                    if not is_active:
                        cam.sleep_until_next_frame()  # ~33 ms → ~30 polls/s
                    else:
                        time.sleep(1 / _POLL_HZ)

        except Exception as e:
            _log(
                f"[Service] virtual camera lost ({type(e).__name__}: {e}), retrying in 5s..."
            )
            stop_event.set()
            if pipeline:
                pipeline.join(timeout=3)
            time.sleep(5)


if __name__ == "__main__":
    main()
