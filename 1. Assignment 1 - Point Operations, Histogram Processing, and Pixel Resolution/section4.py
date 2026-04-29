from utils import compute_norm_cdf, compute_histogram, to_gray, get_r_range
from visualize import plot_rgb_histogram_matching_results_s4, plot_gray_histogram_matching_results_s4, plot_rgb_vs_hsv_equalization_s4, plot_per_channel_histograms_s4, plot_hsv_decomposition_s4
from section3 import calc_ghe
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')

os.makedirs('output/section4', exist_ok=True)


def match_histogram(src, ref):
    cdf_src = compute_norm_cdf(compute_histogram(src))
    cdf_ref = compute_norm_cdf(compute_histogram(ref))

    cdf_src_vec = cdf_src[:, np.newaxis]
    cdf_ref_vec = cdf_ref[np.newaxis, :]
    diff = np.abs(cdf_ref_vec - cdf_src_vec)
    pixel_map = np.argmin(diff, axis=1).astype(np.uint8)

    return pixel_map[src]


def run_histogram_matching():
    print('\n[4.1] Histogram Matching (Specification)')

    # ─── Grayscale pairs ────────────────────────────────────────────
    desired_gray_path = 'Images/Section 4/grayscale/desired.png'
    ref_raw = cv2.imread(desired_gray_path, cv2.IMREAD_UNCHANGED)
    if ref_raw is None:
        print('  [WARNING] Cannot read grayscale desired image')
        return
    ref_gray = to_gray(ref_raw)

    for src_name, src_path in [
        ('initial_1', 'Images/Section 4/grayscale/initial_1.tif'),
        ('initial_2', 'Images/Section 4/grayscale/initial_2.tif'),
    ]:
        img_src = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img_src is None:
            print(f'  [WARNING] Cannot read {src_path}')
            continue
        src = to_gray(img_src)
        matched = match_histogram(src, ref_gray)

        # Transformation curve  z = G^{-1}(T(r))
        cdf_src = compute_norm_cdf(compute_histogram(src))
        cdf_ref = compute_norm_cdf(compute_histogram(ref_gray))
        diff_m = np.abs(cdf_ref[np.newaxis, :] - cdf_src[:, np.newaxis])
        lut_vals = np.argmin(diff_m, axis=1).astype(np.float64)   # (256,)

        hist_src = compute_histogram(src)
        hist_ref = compute_histogram(ref_gray)
        hist_matched = compute_histogram(matched)
        r_range = get_r_range()

        plot_gray_histogram_matching_results_s4(
            ref_gray, src_name, src, matched, cdf_src, cdf_ref, lut_vals, hist_src, hist_ref, hist_matched, r_range)

    # ─── Color image pairs (channel-by-channel matching) ────────────
    desired_rgb_path = 'Images/Section 4/rgb/desired.jpg'
    ref_rgb_raw = cv2.imread(desired_rgb_path)
    if ref_rgb_raw is None:
        print('  [WARNING] Cannot read RGB desired image')
        return

    for src_name, src_path in [
        ('initial_2', 'Images/Section 4/rgb/initial_2.jpg'),
        ('initial_1', 'Images/Section 4/rgb/initial_1.webp'),
    ]:
        img_src = cv2.imread(src_path)
        if img_src is None:
            print(f'  [WARNING] Cannot read {src_path}')
            continue

        # Channel-by-channel histogram matching (B, G, R)
        matched_channels = []
        for ch in range(3):
            matched_ch = match_histogram(img_src[:, :, ch],
                                         ref_rgb_raw[:, :, ch])
            matched_channels.append(matched_ch)
        matched_rgb = np.stack(matched_channels, axis=2)

        channel_names = ['Blue', 'Green', 'Red']
        channel_cols = ['#3498db', '#2ecc71', '#e74c3c']

        plot_rgb_histogram_matching_results_s4(
            src_name, img_src, ref_rgb_raw, matched_rgb, channel_names, channel_cols)


# ============================================================
# 4.2  RGB <-> HSV conversion
# ============================================================


def rgb_to_hsv(rgb):
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    # --- Value ---
    V = Cmax.copy()

    # --- Saturation ---
    safe_cmax = np.where(Cmax > 1e-9, Cmax, 1.0)   # avoid division by zero
    S = np.where(Cmax > 1e-9, delta / safe_cmax, 0.0)

    # --- Hue ---
    H = np.zeros_like(R)
    eps = 1e-9  # avoid division by zero

    mask_r = (Cmax == R) & (delta > eps)
    mask_g = (Cmax == G) & (delta > eps)
    mask_b = (Cmax == B) & (delta > eps)

    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2.0)
    H[mask_b] = 60.0 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4.0)
    H = H % 360.0

    return np.stack([H, S, V], axis=2)


def hsv_to_rgb(hsv):
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    chroma = V * S
    secondary = chroma * (1.0 - np.abs((H / 60.0) % 2.0 - 1.0))  # intermediate
    lightness_match = V - chroma

    R1 = np.zeros_like(H)
    G1 = np.zeros_like(H)
    B1 = np.zeros_like(H)

    # Boolean sector masks
    s = [
        (H >= 0) & (H < 60),
        (H >= 60) & (H < 120),
        (H >= 120) & (H < 180),
        (H >= 180) & (H < 240),
        (H >= 240) & (H < 300),
        (H >= 300) & (H < 360),
    ]
    # (R', G', B') per sector
    rgb_prime = [
        (chroma, secondary, 0),
        (secondary, chroma, 0),
        (0, chroma, secondary),
        (0, secondary, chroma),
        (secondary, 0, chroma),
        (chroma, 0, secondary),
    ]
    for mask, (r_val, g_val, b_val) in zip(s, rgb_prime):
        if isinstance(r_val, np.ndarray):
            R1[mask] = r_val[mask]
        else:
            R1[mask] = r_val
        if isinstance(g_val, np.ndarray):
            G1[mask] = g_val[mask]
        else:
            G1[mask] = g_val
        if isinstance(b_val, np.ndarray):
            B1[mask] = b_val[mask]
        else:
            B1[mask] = b_val

    R = np.clip(R1 + lightness_match, 0.0, 1.0)
    G = np.clip(G1 + lightness_match, 0.0, 1.0)
    B = np.clip(B1 + lightness_match, 0.0, 1.0)

    return np.stack([R, G, B], axis=2)


# ─── High-level colour HE helpers ───────────────────────────

def apply_ghe_per_channel(bgr):
    channels_eq = [calc_ghe(bgr[:, :, ch]) for ch in range(3)]
    return np.stack(channels_eq, axis=2)


def eq_v_channel(bgr):
    # BGR -> RGB float [0, 1]
    rgb = bgr[:, :, ::-1]
    rgb_float = rgb.astype(np.float64) / 255.0

    hsv = rgb_to_hsv(rgb_float)

    v_channel_float = hsv[:, :, 2]
    v_channel_scaled = v_channel_float * 255.0
    V_uint8 = np.clip(v_channel_scaled, 0, 255).astype(np.uint8)
    V_eq = calc_ghe(V_uint8)

    hsv_eq = hsv.copy()
    v_eq_normalized = V_eq.astype(np.float64) / 255.0
    hsv_eq[:, :, 2] = v_eq_normalized

    rgb_eq = hsv_to_rgb(hsv_eq)

    # RGB float -> BGR uint8
    bgr_eq_float = rgb_eq[:, :, ::-1]
    bgr_eq_scaled = bgr_eq_float * 255.0
    bgr_eq = np.clip(bgr_eq_scaled, 0, 255).astype(np.uint8)
    return bgr_eq


def run_color_operations():
    print('\n[4.2] Point Operations on Color Images (RGB <-> HSV)')

    # Use both color source images
    for src_name, src_path in [
        ('initial_2', 'Images/Section 4/rgb/initial_2.jpg'),
        ('initial_1', 'Images/Section 4/rgb/initial_1.webp'),
    ]:
        img = cv2.imread(src_path)
        if img is None:
            print(f'  [WARNING] Cannot read {src_path}')
            continue
        print(f'  Processing {src_name}  {img.shape[:2]} ...')

        eq_rgb = apply_ghe_per_channel(img)
        eq_v = eq_v_channel(img)

        # ── Main comparison figure ──────────────────────────────────
        plot_rgb_vs_hsv_equalization_s4(src_name, img, eq_rgb, eq_v)

        # ── Per-channel histogram comparison ────────────────────────
        ch_names = ['Blue', 'Green', 'Red']
        ch_cols = ['#3498db', '#2ecc71', '#e74c3c']
        r_range = get_r_range()

        plot_per_channel_histograms_s4(
            src_name, img, eq_rgb, eq_v, ch_names, ch_cols, r_range)

        # ── HSV decomposition figure ─────────────────────────────────
        rgb_float = img[:, :, ::-1].astype(np.float64) / 255.0
        hsv = rgb_to_hsv(rgb_float)
        H_chan = hsv[:, :, 0] / 360.0         # normalise for display
        S_chan = hsv[:, :, 1]
        V_chan = hsv[:, :, 2]
        V_eq_norm = calc_ghe(
            np.clip(V_chan * 255, 0, 255).astype(np.uint8)
        ).astype(np.float64) / 255.0

        plot_hsv_decomposition_s4(
            src_name, img, H_chan, S_chan, V_chan, V_eq_norm)


def run_section4():
    print('\n' + '=' * 60)
    print('SECTION 4: Histogram Matching & Color Spaces')
    print('=' * 60)

    run_histogram_matching()
    run_color_operations()

    print('\n[Section 4] All outputs saved to output/section4/')


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_section4()
