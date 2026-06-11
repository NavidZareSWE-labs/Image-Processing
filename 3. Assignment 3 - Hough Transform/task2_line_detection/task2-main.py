import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import cv2  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


matplotlib.use('Agg')


# ====================== Timer ======================
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


# ====================== Utils ======================
def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path, img):
    if img.dtype in (np.float64, np.float32):
        out = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        out = img.copy()
    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out)
    print(f"  Saved: {path}")


# ====================== Filters ======================
def gaussian_kernel(size, sigma):
    if size % 2 == 0:
        size += 1
    ax = np.arange(size) - size // 2
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / kernel.sum()


def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='reflect')
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i +
                                         image.shape[0], j:j + image.shape[1]]
    return out


def gaussian_blur(image, size=5, sigma=1.4):
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)


# ====================== Preprocessing ======================
def preprocess_lane_image(rgb):
    h, w = rgb.shape[:2]

    gray = (0.299 * rgb[:, :, 0] + 0.587 *
            rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    gray = gaussian_blur(gray, size=5, sigma=1.4)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    white = (r > 180) & (g > 180) & (b > 160)
    yellow = (r > 170) & (g > 150) & (b < 120)
    color_mask = (white | yellow).astype(np.float64)
    color_mask = gaussian_blur(color_mask, size=5, sigma=1.4)

    vertices = np.array([
        [int(w * 0.05), h - 1],
        [int(w * 0.40), int(h * 0.65)],
        [int(w * 0.60), int(h * 0.65)],
        [int(w * 0.95), h - 1]
    ])
    mask = np.zeros((h, w), dtype=np.float64)
    _fill_roi(mask, vertices)

    merged = np.maximum(gray * mask, color_mask * mask * 0.8)
    return merged


def _fill_roi(mask, pts):
    h, w = mask.shape
    y_min = max(int(pts[:, 1].min()), 0)
    y_max = min(int(pts[:, 1].max()), h - 1)
    for y in range(y_min, y_max + 1):
        x_ints = []
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            y1, y2 = pts[i, 1], pts[j, 1]
            if y1 == y2:
                continue
            if (y1 <= y < y2) or (y2 <= y < y1):
                x = pts[i, 0] + (y - y1) * (pts[j, 0] - pts[i, 0]) / (y2 - y1)
                x_ints.append(x)
        if len(x_ints) >= 2:
            x_ints.sort()
            xa = max(int(np.ceil(x_ints[0])), 0)
            xb = min(int(np.floor(x_ints[1])), w - 1)
            mask[y, xa:xb + 1] = 1.0


# ====================== Edge Detection ======================
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def canny_edge_detection(image, blur_size=5, blur_sigma=1.0,
                         low_thresh=0.05, high_thresh=0.15):
    smoothed = gaussian_blur(image, blur_size, blur_sigma)
    gx = convolve2d(smoothed, SOBEL_X)
    gy = convolve2d(smoothed, SOBEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)

    nms = _non_max_suppression(magnitude, direction)
    edges = _hysteresis(nms, low_thresh, high_thresh)
    return edges, magnitude, direction


def _non_max_suppression(magnitude, direction):
    angle_deg = np.rad2deg(direction) % 180.0
    out = np.zeros_like(magnitude)
    h, w = magnitude.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            ang = angle_deg[i, j]
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= ang < 67.5:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            elif 67.5 <= ang < 112.5:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]

            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                out[i, j] = magnitude[i, j]
    return out


def _hysteresis(nms, low_ratio=0.05, high_ratio=0.15):
    high = nms.max() * high_ratio if nms.max() > 0 else 1.0
    low = nms.max() * low_ratio if nms.max() > 0 else 0.1
    strong = nms >= high
    weak = (nms >= low) & (~strong)
    return strong.astype(bool)


# ====================== Hough Lines ======================
def hough_line_transform(edge_map, theta_res=1.0, rho_res=1.0):
    h, w = edge_map.shape
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rhos = np.arange(-diag, diag + 1, rho_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    ey, ex = np.nonzero(edge_map > 0.5 if edge_map.dtype != bool else edge_map)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for ti in range(len(thetas)):
        rho_vals = ex * cos_t[ti] + ey * sin_t[ti]
        rho_idx = np.round((rho_vals - rhos[0]) / rho_res).astype(np.int32)
        valid = (rho_idx >= 0) & (rho_idx < len(rhos))
        np.add.at(accumulator[:, ti], rho_idx[valid], 1)

    return accumulator, thetas, rhos


def extract_line_peaks(accumulator, thetas, rhos, threshold_ratio=0.30, nhood_size=15):
    thresh = accumulator.max() * threshold_ratio
    acc = accumulator.copy().astype(np.float64)
    peaks = []
    for _ in range(30):
        idx = np.unravel_index(acc.argmax(), acc.shape)
        val = acc[idx]
        if val < thresh:
            break
        ri, ti = idx
        peaks.append((rhos[ri], thetas[ti], val))
        r_lo = max(ri - nhood_size, 0)
        r_hi = min(ri + nhood_size + 1, acc.shape[0])
        t_lo = max(ti - nhood_size, 0)
        t_hi = min(ti + nhood_size + 1, acc.shape[1])
        acc[r_lo:r_hi, t_lo:t_hi] = 0
    return peaks


def separate_left_right_lanes(peaks, img_w, img_h, accumulator=None, thetas=None, rhos=None):
    y_bottom = img_h - 1
    y_top = int(img_h * 0.675)

    if accumulator is not None and thetas is not None and rhos is not None:
        left_peaks = _extract_angle_range(accumulator, thetas, rhos, 20, 65)
        right_peaks = _extract_angle_range(accumulator, thetas, rhos, -65, -20)
        all_peaks = left_peaks + right_peaks
    else:
        all_peaks = peaks

    left_candidates = []
    right_candidates = []

    for rho, theta, votes in all_peaks:
        x_bottom = _line_x_at_y(rho, theta, y_bottom)
        x_top = _line_x_at_y(rho, theta, y_top)
        if x_bottom is None or x_top is None:
            continue
        if abs(x_bottom - x_top) < 20:
            continue
        if x_top < img_w * 0.10 or x_top > img_w * 0.90:
            continue

        if (x_bottom < x_top - 10 and img_w * 0.20 < x_bottom < img_w * 0.50):
            left_candidates.append((rho, theta, votes, x_bottom, x_top))
        elif (x_bottom > x_top + 10 and img_w * 0.55 < x_bottom < img_w * 0.85):
            right_candidates.append((rho, theta, votes, x_bottom, x_top))

    left_line = _pick_best_lane(left_candidates, img_w * 0.35, y_top, y_bottom)
    right_line = _pick_best_lane(
        right_candidates, img_w * 0.68, y_top, y_bottom)
    return left_line, right_line


def _line_x_at_y(rho, theta, y):
    cos_t = np.cos(theta)
    if abs(cos_t) < 1e-9:
        return None
    return (rho - y * np.sin(theta)) / cos_t


def _extract_angle_range(accumulator, thetas, rhos, theta_lo, theta_hi):
    theta_degs = np.degrees(thetas)
    cols = np.where((theta_degs >= theta_lo) & (theta_degs <= theta_hi))[0]
    if len(cols) == 0:
        return []
    sub_acc = accumulator[:, cols].copy().astype(np.float64)
    peaks = []
    for _ in range(8):
        idx = np.unravel_index(sub_acc.argmax(), sub_acc.shape)
        ri, ci_local = idx
        val = sub_acc[ri, ci_local]
        if val < 8:
            break
        ci = cols[ci_local]
        peaks.append((rhos[ri], thetas[ci], float(val)))
        r_lo = max(ri - 12, 0)
        r_hi = min(ri + 13, sub_acc.shape[0])
        c_lo = max(ci_local - 12, 0)
        c_hi = min(ci_local + 13, sub_acc.shape[1])
        sub_acc[r_lo:r_hi, c_lo:c_hi] = 0
    return peaks


def _pick_best_lane(candidates, expected_x, y_top, y_bottom):
    if not candidates:
        return None
    scored = []
    for c in candidates:
        dist = abs(c[3] - expected_x)
        if dist > 250:
            continue
        score = -dist + c[2] * 0.1
        scored.append((score, c))
    if not scored:
        return None
    best = max(scored, key=lambda x: x[0])[1]
    return ((int(best[4]), y_top), (int(best[3]), y_bottom))


# ====================== Visualize ======================
def draw_lane_overlay(rgb_img, left_line, right_line, color=(0, 200, 0), alpha=0.3):
    overlay = rgb_img.copy().astype(np.float64)
    h, w = rgb_img.shape[:2]

    if left_line is not None and right_line is not None:
        (lx1, ly1), (lx2, ly2) = left_line
        (rx1, ry1), (rx2, ry2) = right_line
        pts = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2],
                       [rx1, ry1]], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        _fill_polygon(mask, pts, 1)
        region = mask.astype(bool)
        poly_color = np.array(color, dtype=np.float64)
        overlay[region] = overlay[region] * (1 - alpha) + poly_color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _fill_polygon(mask, pts, value):
    h, w = mask.shape
    y_min = max(int(pts[:, 1].min()), 0)
    y_max = min(int(pts[:, 1].max()), h - 1)
    for y in range(y_min, y_max + 1):
        x_ints = []
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            y1, y2 = pts[i, 1], pts[j, 1]
            if y1 == y2:
                continue
            if (y1 <= y < y2) or (y2 <= y < y1):
                x = pts[i, 0] + (y - y1) * (pts[j, 0] - pts[i, 0]) / (y2 - y1)
                x_ints.append(x)
        if len(x_ints) >= 2:
            x_ints.sort()
            xa = max(int(np.ceil(x_ints[0])), 0)
            xb = min(int(np.floor(x_ints[1])), w - 1)
            mask[y, xa:xb + 1] = value


def save_lane_pipeline(original, edges, accumulator, overlay, path, img_name=""):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ['Original', 'Edge Map', 'Hough Accumulator', 'Lane Overlay']
    imgs = [original, edges, accumulator, overlay]
    cmaps = [None, 'gray', 'hot', None]

    for ax, img, t, cm in zip(axes, imgs, titles, cmaps):
        if cm:
            ax.imshow(img, cmap=cm, aspect='auto')
        else:
            ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    if img_name:
        fig.suptitle(f"Lane Detection — {img_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ====================== Main Function ======================
def run():
    IMAGE_DIR = str(BASE_DIR / "images" / "task2_line_detection")
    OUTPUT_DIR = str(BASE_DIR / "output" / "task2")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith('.png')],
                   key=lambda x: int(os.path.splitext(x)[0]))

    print("\n" + "=" * 70)
    print("Autonomous Lane Detection (Hough Lines)")
    print("=" * 70)

    frames = []
    prev_left = None
    prev_right = None

    for i, fname in enumerate(files):
        path = os.path.join(IMAGE_DIR, fname)
        stem = os.path.splitext(fname)[0]

        print(f"\n--- Frame {i + 1}/{len(files)}: {fname} ---")

        rgb = load_image_rgb(path)
        h, w = rgb.shape[:2]

        with Timer("Preprocessing") as t_pre:
            preprocessed = preprocess_lane_image(rgb)

        with Timer("Edge Detection") as t_edge:
            edges, _, _ = canny_edge_detection(preprocessed)

        with Timer("Hough Voting") as t_hough:
            accumulator, thetas, rhos = hough_line_transform(edges)

        with Timer("Post-processing") as t_post:
            peaks = extract_line_peaks(accumulator, thetas, rhos)
            left_line, right_line = separate_left_right_lanes(
                peaks, w, h, accumulator, thetas, rhos
            )

            if left_line and abs(left_line[1][0] - w * 0.35) > 100:
                left_line = None
            if right_line and abs(right_line[1][0] - w * 0.68) > 80:
                right_line = None

            if left_line is None and prev_left is not None:
                left_line = prev_left
            if right_line is None and prev_right is not None:
                right_line = prev_right

            prev_left = left_line
            prev_right = right_line

            overlay = draw_lane_overlay(rgb, left_line, right_line)

        print(f"  Left:  {left_line}")
        print(f"  Right: {right_line}")

        save_image(os.path.join(OUTPUT_DIR, f"{stem}_overlay.png"), overlay)
        frames.append(overlay)

        if i in (0, len(files) // 2, len(files) - 1):
            save_lane_pipeline(rgb, edges.astype(float), accumulator, overlay,
                               os.path.join(OUTPUT_DIR, f"{stem}_pipeline.png"), fname)

    gif_path = os.path.join(OUTPUT_DIR, "lane_detection.gif")
    print(f"\nBuilding GIF with {len(frames)} frames...")
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"GIF saved: {gif_path}")

    print("\n" + "=" * 70)
    print("All Done!")
    print("=" * 70)


if __name__ == "__main__":
    run()
