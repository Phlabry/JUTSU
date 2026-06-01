"""
Routes real microphone input + sound effects into VB-CABLE so apps like
Discord receive both the user's voice and the sound effects on one input.

Setup: download VB-CABLE from vb-audio.com/Cable (free), install, reboot.
In Discord Audio Settings → Input Device → select "CABLE Output (VB-Audio Virtual Cable)".
"""
import itertools
import os
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf


def _to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1 or data.shape[1] == 1:
        return np.repeat(data.reshape(-1, 1), 2, axis=1)
    return data[:, :2]


def _resample(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return data
    n_out = int(round(len(data) * to_sr / from_sr))
    x_old = np.linspace(0.0, 1.0, len(data))
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.column_stack([
        np.interp(x_new, x_old, data[:, ch]).astype(np.float32)
        for ch in range(data.shape[1])
    ])


class VirtualMicMixer:
    def __init__(self, audio_dir: str, mic_device: str | None = None):
        self._lock   = threading.Lock()
        self._pending: dict = {}
        self._ids    = itertools.count()
        self._stream = None
        self._mic_in = self._find_device(mic_device.lower(), output=False) if mic_device else None

        self._vb_out = self._find_device('cable input', output=True)

        # Use VB-CABLE's preferred rate so no OS resampling is needed
        if self._vb_out is not None:
            self._sr = int(sd.query_devices(self._vb_out, 'output')['default_samplerate'])
        else:
            self._sr = 44100

        def _load(name: str) -> np.ndarray:
            data, file_sr = sf.read(os.path.join(audio_dir, name), dtype='float32', always_2d=True)
            data = _to_stereo(data)
            return _resample(data, file_sr, self._sr)

        try:
            self._sounds = {
                'charge':  _load('Charge.wav'),
                'release': _load('Release.wav'),
            }
        except Exception as e:
            print(f"[VirtualMic] sound load failed: {e}")
            self._sounds = {k: np.zeros((1, 2), dtype=np.float32) for k in ('charge', 'release')}

        if mic_device and self._mic_in is None:
            print(f"[VirtualMic] mic '{mic_device}' not found, using system default")

    @staticmethod
    def _find_device(keyword: str, output: bool) -> int | None:
        ch_key = 'max_output_channels' if output else 'max_input_channels'
        for i, d in enumerate(sd.query_devices()):
            if keyword in d['name'].lower() and d[ch_key] > 0:
                return i
        return None

    @staticmethod
    def print_input_devices() -> None:
        print("[VirtualMic] Available microphone devices:")
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f"  [{i:2d}] {d['name']}")

    def start(self) -> None:
        self.print_input_devices()
        if self._vb_out is None:
            print("[VirtualMic] VB-CABLE not found — download from vb-audio.com/Cable")
            return

        try:
            in_ch = min(int(sd.query_devices(self._mic_in, 'input')['max_input_channels']), 2)
        except Exception:
            in_ch = 1

        def _cb(indata, outdata, frames, time, status):
            mixed = _to_stereo(indata).copy()
            with self._lock:
                done = []
                for sid, s in self._pending.items():
                    end  = min(s['pos'] + frames, len(s['data']))
                    n    = end - s['pos']
                    mixed[:n] += s['data'][s['pos']:end] * s['vol']
                    s['pos']   = end
                    if s['pos'] >= len(s['data']):
                        if s['loop']:
                            s['pos'] = 0
                        else:
                            done.append(sid)
                for sid in done:
                    del self._pending[sid]
            np.clip(mixed, -1.0, 1.0, out=mixed)
            outdata[:] = mixed

        try:
            self._stream = sd.Stream(
                device=(self._mic_in, self._vb_out),
                samplerate=self._sr,
                channels=(in_ch, 2),
                dtype='float32',
                blocksize=1024,
                latency='high',
                callback=_cb,
            )
            self._stream.start()
            name = sd.query_devices(self._vb_out)['name']
            print(f"[VirtualMic] {self._sr}Hz  mic + effects -> {name}")
        except Exception as e:
            print(f"[VirtualMic] stream failed: {e}")
            self._stream = None

    def play(self, sound_key: str, volume: float, loop: bool = False) -> int:
        sid = next(self._ids)
        with self._lock:
            self._pending[sid] = {
                'data': self._sounds[sound_key],
                'pos':  0,
                'vol':  volume,
                'loop': loop,
            }
        return sid

    def set_volume(self, sid: int, volume: float) -> None:
        with self._lock:
            if sid in self._pending:
                self._pending[sid]['vol'] = volume

    def stop(self, sid: int) -> None:
        with self._lock:
            self._pending.pop(sid, None)

    def stop_all(self) -> None:
        with self._lock:
            self._pending.clear()

    def close(self) -> None:
        self.stop_all()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
