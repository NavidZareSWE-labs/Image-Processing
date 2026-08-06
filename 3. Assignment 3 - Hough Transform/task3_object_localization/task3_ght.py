import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve1d


def gaussian_blur(image, kernel_size=5, sigma=1.4):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss1d /= gauss1d.sum()
    blurred = convolve1d(image.astype(np.float32), gauss1d, axis=0)
    blurred = convolve1d(blurred, gauss1d, axis=1)
    return blurred


def sobel_filters(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    Gx = cv2.filter2D(image, -1, sobel_x)
    Gy = cv2.filter2D(image, -1, sobel_y)
    magnitude = np.hypot(Gx, Gy)
    angle = np.arctan2(Gy, Gx) * 180.0 / np.pi
    angle[angle < 0] += 180
    return magnitude, angle


def non_max_suppression(mag, ang):
    h, w = mag.shape
    suppressed = np.zeros((h, w), dtype=np.float32)
    ang = ang % 180

    mask_horizontal = ((ang >= 0) & (ang < 22.5)) | ((ang >= 157.5) & (ang <= 180))
    mask_diag1 = (ang >= 22.5) & (ang < 67.5)
    mask_vertical = (ang >= 67.5) & (ang < 112.5)
    mask_diag2 = (ang >= 112.5) & (ang < 157.5)

    mag_shift_left = np.roll(mag, 1, axis=1)
    mag_shift_right = np.roll(mag, -1, axis=1)
    mag_shift_up = np.roll(mag, -1, axis=0)
    mag_shift_down = np.roll(mag, 1, axis=0)
    mag_shift_upright = np.roll(np.roll(mag, -1, axis=0), 1, axis=1)
    mag_shift_downleft = np.roll(np.roll(mag, 1, axis=0), -1, axis=1)
    mag_shift_upleft = np.roll(np.roll(mag, -1, axis=0), -1, axis=1)
    mag_shift_downright = np.roll(np.roll(mag, 1, axis=0), 1, axis=1)

    cond_h = (mag >= mag_shift_left) & (mag >= mag_shift_right)
    cond_d1 = (mag >= mag_shift_upright) & (mag >= mag_shift_downleft)
    cond_v = (mag >= mag_shift_up) & (mag >= mag_shift_down)
    cond_d2 = (mag >= mag_shift_upleft) & (mag >= mag_shift_downright)

    suppressed[mask_horizontal & cond_h] = mag[mask_horizontal & cond_h]
    suppressed[mask_diag1 & cond_d1] = mag[mask_diag1 & cond_d1]
    suppressed[mask_vertical & cond_v] = mag[mask_vertical & cond_v]
    suppressed[mask_diag2 & cond_d2] = mag[mask_diag2 & cond_d2]

    return suppressed


def hysteresis_threshold(suppressed, low_ratio=0.1, high_ratio=0.3):
    high_thresh = suppressed.max() * high_ratio
    low_thresh = high_thresh * low_ratio

    strong = (suppressed >= high_thresh).astype(np.uint8)
    weak = ((suppressed >= low_thresh) & (suppressed < high_thresh)).astype(np.uint8)

    edge_map = strong.copy()
    h, w = suppressed.shape
    for _ in range(5):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j] == 1 and np.any(
                    edge_map[i - 1 : i + 2, j - 1 : j + 2] == 1
                ):
                    edge_map[i, j] = 1
    return edge_map


def compute_edge_map(image, blur=True):
    if blur:
        image = gaussian_blur(image, kernel_size=5, sigma=1.0)
    mag, ang = sobel_filters(image)
    suppressed = non_max_suppression(mag, ang)
    edge_binary = hysteresis_threshold(suppressed)
    edge_angle = ang * edge_binary
    return edge_binary.astype(np.uint8), edge_angle


def build_rtable(edge_binary, edge_angle, reference_point, num_angle_bins=36):
    bin_size = 180.0 / num_angle_bins
    rtable = {i: [] for i in range(num_angle_bins)}

    edge_points = np.argwhere(edge_binary > 0)
    for y, x in edge_points:
        phi = edge_angle[y, x]
        bin_idx = int(phi // bin_size)
        bin_idx = min(bin_idx, num_angle_bins - 1)

        dx = reference_point[0] - x
        dy = reference_point[1] - y
        r = np.hypot(dx, dy)
        alpha = np.arctan2(dy, dx)
        rtable[bin_idx].append((r, alpha))

    return rtable


def generalized_hough_vote(
    edge_binary, edge_angle, rtable, num_angle_bins, image_shape
):
    accumulator = np.zeros(image_shape, dtype=np.int32)
    bin_size = 180.0 / num_angle_bins

    edge_points = np.argwhere(edge_binary > 0)
    for y, x in edge_points:
        phi = edge_angle[y, x]
        bin_idx = int(phi // bin_size)
        bin_idx = min(bin_idx, num_angle_bins - 1)

        vectors = rtable.get(bin_idx, [])
        for r, alpha in vectors:
            xc = int(round(x + r * np.cos(alpha)))
            yc = int(round(y + r * np.sin(alpha)))
            if 0 <= xc < image_shape[1] and 0 <= yc < image_shape[0]:
                accumulator[yc, xc] += 1
    return accumulator


def find_strongest_peak(accumulator):
    max_val = np.max(accumulator)
    if max_val == 0:
        return None, 0
    points = np.argwhere(accumulator == max_val)
    center_y, center_x = np.mean(points, axis=0).astype(int)
    return (center_x, center_y), max_val


def save_visualization(
    original, gray, edge_map, accumulator, detected_center, output_prefix
):
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Test Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Test Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(edge_map, cmap="gray")
    plt.title("Edge Map")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(accumulator, cmap="hot")
    plt.colorbar()
    plt.title("Hough Accumulator Heatmap")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    result = original.copy()
    if detected_center:
        cv2.circle(result, detected_center, 8, (0, 0, 255), -1)
        cv2.circle(result, detected_center, 35, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Final Detection (Center + Bounding Circle)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_visuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(accumulator, cmap="hot")
    plt.colorbar()
    plt.title("Accumulator Heatmap")
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    template_path = "object_localization/template_flower_gray.png"
    test_gray_path = "object_localization/test_flower_gray.png"
    test_original_path = "object_localization/test_flower_original.jpg"

    print("=== Generalized Hough Transform - Task 3 ===")
    print("Loading images...")
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_gray = cv2.imread(test_gray_path, cv2.IMREAD_GRAYSCALE)
    test_original = cv2.imread(test_original_path)

    if template is None or test_gray is None or test_original is None:
        print("ERROR: Images not found. Check paths.")
        return

    print(f"Template size: {template.shape}")
    print(f"Test image size: {test_gray.shape}\n")

    start = time.time()
    edge_template, angle_template = compute_edge_map(template, blur=True)
    elapsed = time.time() - start
    print(f"[1] Edge detection on template: {elapsed:.4f} sec")

    ref_x = template.shape[1] // 2
    ref_y = template.shape[0] // 2
    reference_point = (ref_x, ref_y)
    print(f"    Reference point (center of template): {reference_point}")

    start = time.time()
    num_bins = 36
    rtable = build_rtable(edge_template, angle_template, reference_point, num_bins)
    elapsed = time.time() - start
    print(f"[2] R-Table construction: {elapsed:.4f} sec")

    start = time.time()
    edge_test, angle_test = compute_edge_map(test_gray, blur=True)
    elapsed = time.time() - start
    print(f"[3] Edge detection on test image: {elapsed:.4f} sec")

    start = time.time()
    accumulator = generalized_hough_vote(
        edge_test, angle_test, rtable, num_bins, test_gray.shape
    )
    elapsed = time.time() - start
    print(f"[4] Hough voting: {elapsed:.4f} sec")

    start = time.time()
    detected_center, max_votes = find_strongest_peak(accumulator)
    elapsed = time.time() - start
    print(f"[5] Peak detection: {elapsed:.4f} sec")

    if detected_center:
        print(f"\n=> Detected object center: {detected_center} with {max_votes} votes")
    else:
        print("\n=> No peak found.")

    save_visualization(
        test_original, test_gray, edge_test, accumulator, detected_center, "GHT_result"
    )
    print("\nVisual outputs saved: GHT_result_visuals.png , GHT_result_heatmap.png")

    plt.show()


if __name__ == "__main__":
    main()
