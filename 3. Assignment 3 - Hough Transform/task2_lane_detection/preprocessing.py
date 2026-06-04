
import numpy as np
from utils.filters import gaussian_blur


def region_of_interest(image: np.ndarray, vertices=None) -> np.ndarray:
    """
    Mask out everything outside a trapezoidal ROI.
    Default ROI is tuned for the dashcam highway images:
      - Excludes the top 65% (sky / background)
      - Extends to the very bottom of the image
      - Narrows toward the vanishing point at top
    """
    h, w = image.shape[:2]
    if vertices is None:
        vertices = np.array([
            [int(w * 0.05), h - 1],              # bottom-left
            [int(w * 0.40), int(h * 0.65)],       # top-left
            [int(w * 0.60), int(h * 0.65)],       # top-right
            [int(w * 0.95), h - 1],               # bottom-right
        ])

    mask = np.zeros((h, w), dtype=np.float64)
    _fill_convex(mask, vertices)
    if image.ndim == 2:
        return image * mask
    else:
        return image * mask[:, :, None]


def _fill_convex(mask, pts):
    """Scanline fill for a convex polygon."""
    h, w = mask.shape
    ys = pts[:, 1]
    y_min, y_max = max(int(ys.min()), 0), min(int(ys.max()), h - 1)
    n = len(pts)
    for y in range(y_min, y_max + 1):
        x_ints = []
        for i in range(n):
            j = (i + 1) % n
            y1, y2 = pts[i, 1], pts[j, 1]
            if y1 == y2:
                continue
            if (y1 <= y < y2) or (y2 <= y < y1):
                x_int = pts[i, 0] + (y - y1) * \
                    (pts[j, 0] - pts[i, 0]) / (y2 - y1)
                x_ints.append(x_int)
        x_ints.sort()
        for k in range(0, len(x_ints) - 1, 2):
            xa = max(int(np.ceil(x_ints[k])), 0)
            xb = min(int(np.floor(x_ints[k + 1])), w - 1)
            mask[y, xa:xb + 1] = 1.0


def colour_threshold(rgb: np.ndarray) -> np.ndarray:
    """Binary mask highlighting white/yellow lane markings."""
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    white = (r > 180) & (g > 180) & (b > 160) & (np.abs(r - g) < 35)
    yellow = (r > 170) & (g > 150) & (b < 120) & ((r - b) > 60)

    mask = (white | yellow).astype(np.float64)
    return mask


def preprocess_lane_image(rgb: np.ndarray,
                          blur_size: int = 5,
                          blur_sigma: float = 1.4) -> np.ndarray:
    """
    Full preprocessing pipeline combining both approaches:
      1. Grayscale + ROI (captures road edges)
      2. Colour threshold + ROI (captures lane markings)
      3. Merge both for maximum feature coverage
    """
    # Grayscale path
    gray = (0.299 * rgb[:, :, 0].astype(np.float64)
            + 0.587 * rgb[:, :, 1].astype(np.float64)
            + 0.114 * rgb[:, :, 2].astype(np.float64)) / 255.0
    gray_blurred = gaussian_blur(gray, size=blur_size, sigma=blur_sigma)
    gray_roi = region_of_interest(gray_blurred)

    # Colour threshold path (isolates white/yellow markings)
    ct = colour_threshold(rgb)
    ct_blurred = gaussian_blur(ct, size=blur_size, sigma=blur_sigma)
    ct_roi = region_of_interest(ct_blurred)

    # Merge: use colour threshold where available, grayscale elsewhere
    merged = np.maximum(gray_roi, ct_roi * 0.8)
    return merged
