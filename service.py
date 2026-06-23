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

Logs to service.log (no console when running as pythonw.exe).
"""
import ctypes
from ctypes import wintypes as _wintypes
import os
import queue
import threading
import time

import cv2 as cv
import numpy as np
import pyvirtualcam

import jutsu as jutsu_registry
from camera.feed import open_camera
from config import (ACTIVE_JUTSU, VIRTUAL_CAM_WIDTH, VIRTUAL_CAM_HEIGHT,
                    PROC_WIDTH, PROC_HEIGHT)
from tracking.hand_detector import HandDetector

_POLL_HZ  = 10          # polls per second while pipeline is active
_ROOT     = os.path.dirname(os.path.abspath(__file__))
_log_file = open(os.path.join(_ROOT, 'service.log'), 'a', buffering=1, encoding='utf-8')

_k32    = ctypes.windll.kernel32
_user32 = ctypes.windll.user32


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line)
    _log_file.write(line + '\n')
    _log_file.flush()


# ---------------------------------------------------------------------------
# Consumer detection
# ---------------------------------------------------------------------------

_DEBOUNCE_POLLS = 8   # consecutive same-state reads required before acting


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

_WM_HOTKEY   = 0x0312
_PM_REMOVE   = 0x0001
_MOD_CTRL    = 0x0002
_MOD_SHIFT   = 0x0004
_MOD_ALT     = 0x0001
_HOTKEY_ID   = 1
_HOTKEY_MODS = _MOD_CTRL | _MOD_SHIFT | _MOD_ALT
_HOTKEY_VK   = 0x4A   # 'J'
_HOTKEY_DESC = "Ctrl+Shift+Alt+J"

_hotkey_toggle = threading.Event()   # set by hotkey thread; consumed by main loop
_manual_off    = threading.Event()   # set after manual stop; blocks auto-restart until
                                     # consumer goes away (mapping absent → cleared)


def _hotkey_loop() -> None:
    """
    Daemon thread: registers a global hotkey with Windows and fires
    _hotkey_toggle whenever the user presses the combination.
    Uses RegisterHotKey (no extra libraries required).
    """
    if not _user32.RegisterHotKey(None, _HOTKEY_ID, _HOTKEY_MODS, _HOTKEY_VK):
        _log(f"[Hotkey] RegisterHotKey failed — {_HOTKEY_DESC} not available "
             "(another app may have claimed it)")
        return
    _log(f"[Hotkey] {_HOTKEY_DESC} → toggle camera pipeline")
    msg = _wintypes.MSG()
    try:
        while True:
            while _user32.PeekMessageA(
                    ctypes.byref(msg), None,
                    _WM_HOTKEY, _WM_HOTKEY, _PM_REMOVE):
                if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                    _hotkey_toggle.set()
                    _log(f"[Hotkey] {_HOTKEY_DESC} pressed")
            time.sleep(0.05)
    finally:
        _user32.UnregisterHotKey(None, _HOTKEY_ID)


# ---------------------------------------------------------------------------
# Background pre-warm  (runs during idle → first activation is near-instant)
# ---------------------------------------------------------------------------

def _prewarm_background() -> None:
    try:
        for _name in ACTIVE_JUTSU:
            _, _, prewarm_fn = jutsu_registry.load(_name, PROC_WIDTH, PROC_HEIGHT)
            prewarm_fn()

        # Pull the MediaPipe .task file into the OS page cache so that the
        # first real HandDetector() call is fast.
        det = HandDetector()
        det.release()

        _log("[Service] pre-warm done — first activation will be fast")
    except Exception as e:
        _log(f"[Service] pre-warm failed: {e}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(stop_event: threading.Event, cam: pyvirtualcam.Camera,
                  forced: bool = False) -> None:
    """
    forced=True: started manually via hotkey.  Disconnect detection is
    skipped; the pipeline runs until stop_event is set (hotkey pressed again)
    or the safety timeout fires.
    forced=False (default): auto-started by consumer detection; disconnect
    detection runs normally.
    """
    cap      = open_camera()
    detector = HandDetector()

    _jutsu_modules = [jutsu_registry.load(n, PROC_WIDTH, PROC_HEIGHT) for n in ACTIVE_JUTSU]
    all_detectors  = [det for _, dets, _ in _jutsu_modules for det in dets]
    for _, _, prewarm_fn in _jutsu_modules:
        threading.Thread(target=prewarm_fn, daemon=True).start()

    out_size  = (VIRTUAL_CAM_WIDTH, VIRTUAL_CAM_HEIGHT)
    proc_size = (PROC_WIDTH, PROC_HEIGHT)

    detect_in  = queue.Queue(maxsize=1)
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

    mode = "MANUAL" if forced else "AUTO"
    _log(f"[Pipeline] started — camera on  [{mode} mode]")
    last_result    = None
    _frame_n      = 0
    _want_misses  = 0    # consecutive polls where UnityCapture_Want0 was absent
    _consec_fast  = 0    # consecutive sends < 0.8 ms (no-reader signature)
    _pipeline_start = time.time()  # for heartbeat uptime display only

    if not forced:
        _want_at_start = _want_event_exists()
        _log(f"[Pipeline] UnityCapture_Want0 present at start: {_want_at_start}")
    else:
        _want_at_start = False  # forced mode ignores Want0

    try:
        while cap.isOpened() and not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                continue

            _frame_n += 1
            proc = cv.resize(frame, proc_size)

            try:
                detect_in.get_nowait()
            except queue.Empty:
                pass
            detect_in.put_nowait(proc.copy())

            try:
                last_result = detect_out.get_nowait()
            except queue.Empty:
                pass

            if last_result is not None:
                for det in all_detectors:
                    det.process_frame(proc, last_result)

            annotated = cv.flip(proc, 1)
            for _state, _, _ in _jutsu_modules:
                annotated = _state.render(annotated)

            if proc_size != out_size:
                annotated = cv.resize(annotated, out_size)

            t0 = time.perf_counter()
            cam.send(annotated)
            send_ms = (time.perf_counter() - t0) * 1000

            # ── Consumer disconnect detection (AUTO mode only) ────────────────
            if not forced:
                # Mechanism 1: UnityCapture_Want0 kernel event.
                # The consumer's DLL creates this event; destroying it is an
                # immediate, reliable disconnect signal.  Checked every ~1 s.
                if _frame_n % 30 == 0 and _want_at_start:
                    if not _want_event_exists():
                        _want_misses += 1
                        _log(f"[Pipeline] Want0 gone (miss {_want_misses}/3, "
                             f"send={send_ms:.2f}ms)")
                        if _want_misses >= 3:
                            _log("[Pipeline] consumer disconnected (Want0 absent) — stopping")
                            break
                    else:
                        _want_misses = 0

                # Mechanism 2: 120 consecutive fast sends (< 0.8 ms).
                if send_ms < 0.8:
                    _consec_fast += 1
                    if _consec_fast >= 120:
                        _log(f"[Pipeline] consumer disconnected "
                             f"({_consec_fast} fast sends, last={send_ms:.2f}ms) — stopping")
                        break
                else:
                    _consec_fast = 0

            if _frame_n % 150 == 0:
                elapsed = int(time.time() - _pipeline_start)
                _log(f"[Pipeline] heartbeat  send={send_ms:.2f}ms  "
                     f"fast_streak={_consec_fast}  "
                     f"want={'present' if _want_event_exists() else 'ABSENT'}  "
                     f"up={elapsed}s")

            # Do NOT call cam.sleep_until_next_frame().
            # cap.read() already blocks ~33 ms for the next physical-camera frame;
            # adding a second sleep double-waits and drops output to ~15 fps.

    except Exception as e:
        import traceback
        _log(f"[Pipeline] crash: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        detect_in.put(None)
        detector.release()
        cap.release()
        _log("[Pipeline] stopped — camera off")


# ---------------------------------------------------------------------------
# Virtual camera
# ---------------------------------------------------------------------------

def _open_virtualcam() -> pyvirtualcam.Camera | None:
    try:
        cam = pyvirtualcam.Camera(
            width=VIRTUAL_CAM_WIDTH, height=VIRTUAL_CAM_HEIGHT, fps=30,
            fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False,
            backend='unitycapture',
        )
        _log(f"[Service] virtual camera ready: {cam.device}")
        return cam
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    _log("[Service] starting")
    for _name in ACTIVE_JUTSU:
        jutsu_registry.init_audio(_name)
    threading.Thread(target=_prewarm_background, daemon=True).start()
    threading.Thread(target=_hotkey_loop, daemon=True).start()

    black = np.zeros((VIRTUAL_CAM_HEIGHT, VIRTUAL_CAM_WIDTH, 3), dtype=np.uint8)

    while True:
        cam = _open_virtualcam()

        if cam is None:
            _log("[Service] UnityCapture not found — "
                 "github.com/schellingb/UnityCapture → Install.bat (Admin) → reboot")
            time.sleep(30)
            continue

        stop_event      = threading.Event()
        pipeline:        threading.Thread | None = None
        was_active       = False
        _ticks_active    = 0   # consecutive True readings
        _ticks_inactive  = 0   # consecutive False readings

        try:
            with cam:
                _log("[Service] idle — waiting for consumer")

                while True:
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
                            pipeline        = None
                            was_active      = False
                            _ticks_active   = 0
                            _ticks_inactive = 0
                            _log("[Service] idle — waiting for consumer "
                                 f"(press {_HOTKEY_DESC} to force-start)")
                        else:
                            _manual_off.clear()
                            _log(f"[Hotkey] manual start — forced mode, "
                                 f"press {_HOTKEY_DESC} again to stop")
                            stop_event.clear()
                            pipeline = threading.Thread(
                                target=_run_pipeline,
                                args=(stop_event, cam, True),
                                daemon=True,
                            )
                            pipeline.start()
                            was_active    = True
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
                        raw = True  # pipeline detects its own disconnect via send timing

                    # Debounce: require _DEBOUNCE_POLLS consecutive same-state
                    # reads before we act, to survive brief blips.
                    if raw:
                        _ticks_active   += 1
                        _ticks_inactive  = 0
                    else:
                        _ticks_inactive += 1
                        _ticks_active    = 0

                    if not was_active:
                        is_active = _ticks_active   >= _DEBOUNCE_POLLS
                    else:
                        is_active = _ticks_inactive <  _DEBOUNCE_POLLS

                    # Detect unexpected pipeline crash
                    if was_active and pipeline and not pipeline.is_alive():
                        stop_event.set()
                        pipeline        = None
                        was_active      = False
                        _ticks_active   = 0
                        _ticks_inactive = 0
                        _log("[Pipeline] ended — returning to idle")
                        continue

                    if is_active and not was_active:
                        _log(f"[Service] consumer detected ({send_ms:.1f} ms send) — starting pipeline")
                        stop_event.clear()
                        pipeline = threading.Thread(
                            target=_run_pipeline, args=(stop_event, cam), daemon=True,
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
            _log(f"[Service] virtual camera lost ({type(e).__name__}: {e}), retrying in 5s...")
            stop_event.set()
            if pipeline:
                pipeline.join(timeout=3)
            time.sleep(5)


if __name__ == "__main__":
    main()
