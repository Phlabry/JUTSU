"""
Usage:
    python datasets/review.py <folder>
    python datasets/review.py datasets/hollow_purple/charge

Controls:
    a  keep image
    d  move image to <folder>/bad/
    s  skip
    q  quit
"""

import os
import shutil
import sys

import cv2 as cv

MAX_W = 960
MAX_H = 720
EXTS  = {".jpg", ".jpeg", ".png"}


def _fit(img):
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        img = cv.resize(img, (int(w * scale), int(h * scale)))
    return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python datasets/review.py <folder>")
        sys.exit(1)

    folder  = sys.argv[1]
    bad_dir = os.path.join(folder, "bad")

    images = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in EXTS)
    if not images:
        print(f"No images found in {folder}")
        return

    kept = moved = skipped = 0
    idx  = 0

    while idx < len(images):
        name = images[idx]
        path = os.path.join(folder, name)

        img = cv.imread(path)
        if img is None:
            idx += 1
            continue

        display = _fit(img)
        cv.putText(display, f"[{idx + 1}/{len(images)}]  {name}",
                   (10, 24), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(display, "a=keep  d=bad  s=skip  q=quit",
                   (10, display.shape[0] - 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (170, 170, 170), 1, cv.LINE_AA)
        cv.imshow("Review", display)

        print(f"[{idx + 1}/{len(images)}] {name}")

        key = cv.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            kept += 1
            idx  += 1
        elif key == ord("d"):
            os.makedirs(bad_dir, exist_ok=True)
            shutil.move(path, os.path.join(bad_dir, name))
            images.pop(idx)
            moved += 1
        elif key == ord("s"):
            skipped += 1
            idx     += 1

    cv.destroyAllWindows()
    print(f"\nDone — kept: {kept}  moved to bad/: {moved}  skipped: {skipped}")


if __name__ == "__main__":
    main()
