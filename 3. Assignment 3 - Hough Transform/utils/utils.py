# ⚠️⚠️⚠️ Only used cv2.imread(), cv2.cvtColor() and cv2.imwrite()
import time
import numpy as np
import cv2


class Timer:
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"  [{self.label}] elapsed: {self.elapsed:.4f} s")


def load_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64) / 255.0


def load_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path, img):
    if img.dtype == np.float64 or img.dtype == np.float32:
        out = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        out = img.copy()
    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out)
    print(f"  Saved: {path}")
