"""
python -m scripts.fetch_models

Fetches the stock MediaPipe models the project needs but does not track.

These are third-party binaries with stable URLs, so the repo carries their
hashes instead of the files themselves.  The trained gesture classifiers under
models/ are a different matter — those are irreplaceable without the raw
recordings, so they are committed.

Safe to re-run: a file already present with the right hash is left alone, and
one that fails its hash is re-fetched.
"""

import hashlib
import os
import sys
import tempfile
import urllib.request

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

_BASE = "https://storage.googleapis.com/mediapipe-models"

# filename -> (url, sha256)
MODELS = {
    "hand_landmarker.task": (
        f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, want: str, dest_dir: str = _ROOT) -> bool:
    """Ensure dest_dir/name exists with the expected hash.  True if downloaded."""
    dest = os.path.join(dest_dir, name)

    if os.path.exists(dest):
        if _sha256(dest) == want:
            print(f"{name}: present")
            return False
        print(f"{name}: hash mismatch, re-fetching")

    print(f"{name}: downloading...")
    # Download beside the destination so the rename below can't cross volumes.
    fd, tmp = tempfile.mkstemp(dir=dest_dir, prefix=f".{name}.", suffix=".part")
    os.close(fd)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)

        got = _sha256(tmp)
        if got != want:
            raise RuntimeError(f"{name}: expected sha256 {want}, got {got}")

        os.replace(tmp, dest)  # only publish a file that already verified
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"{name}: ok ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return True


def main() -> int:
    for name, (url, want) in MODELS.items():
        try:
            fetch(name, url, want)
        except Exception as e:
            print(f"{name}: FAILED — {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
