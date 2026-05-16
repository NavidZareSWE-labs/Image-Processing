from utils import get_r_range, compute_mse, to_gray
from visualize import plot_transform_s2, plot_gamma_curves_family_s2, plot_piecewise_linear_stretching_s2, plot_all_bitplanes_s2, plot_bitplane_reconstruction_comparison_s2
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')

os.makedirs('output/section2', exist_ok=True)


def apply_negative(img):
    # s = L - 1 - r
    # L - 1 = (2 ^ 8 - 1) = 256 - 1 = 255
    return (255 - img.astype(np.int32)).clip(0, 255).astype(np.uint8)


def run_negative():
    print("\n[2.1a] Image Negative")

    # --- Grayscale ---
    moon = cv2.imread('Images/Section 2/moon.tif', cv2.IMREAD_UNCHANGED)
    neg_moon = apply_negative(moon)
    r_range = get_r_range()
    plot_transform_s2(moon, neg_moon,
                      r_range, 255 - r_range,
                      "Image Negative  s = L - 1 - r  (Grayscale: moon.tif)",
                      subtitles=("Original (moon)", "Negative"),
                      save_path="output/section2/negative_moon.png")

    # --- Color ---
    peppers = cv2.imread('Images/Section 2/peppers.png')
    neg_peppers = apply_negative(peppers)
    plot_transform_s2(peppers, neg_peppers,
                      r_range, 255 - r_range,
                      "Image Negative  s = L - 1 - r  (Color: peppers.png)",
                      subtitles=("Original (peppers)", "Negative"),
                      save_path="output/section2/negative_peppers.png")


def apply_log(img_gray, c=None):
    # s = c * log(1 + r)
    # c normalised so that r=255 maps to s=255:
    # c = 255 / log(256)
    r = img_gray.astype(np.float64)
    # log1p(r) = log(1+r), numerically stable / avoid precision loss
    s = c * np.log1p(r)
    return np.clip(s, 0, 255).astype(np.uint8)


def run_log_transform():
    print("\n[2.1b] Logarithmic Transformation")

    #  high dynamic range image (Fourier spectrum image)
    img = cv2.imread('Images/Section 2/2.png')
    img_gray = to_gray(img)

    c = 255.0 / np.log(256.0)
    transformed = apply_log(img_gray, c)

    r_range = get_r_range()
    s = np.clip(c * np.log1p(r_range), 0, 255)

    plot_transform_s2(img_gray, transformed,
                      r_range, s,
                      f"Logarithmic Transform  s = c·log(1+r),  c = {c:.2f}  (2.png)",
                      subtitles=("Original", "Log-transformed"),
                      save_path="output/section2/log_transform.png")


def apply_gamma(img_gray, gamma):
    # s = 255 * (r / 255) ^ gamma
    # gamma < 1 -> brightens dark(under-exposed) images
    # gamma > 1 -> darkens bright(over-exposed) images
    # cause r is normalized c is not used.

    r_norm = img_gray.astype(np.float64) / 255.0
    s_norm = np.power(r_norm, gamma)
    s_norm = np.clip(s_norm * 255.0, 0, 255).astype(np.uint8)
    return s_norm


def run_gamma_correction():
    print("\n[2.1c] Power-Law (Gamma) Correction")

    # Under-exposed (dark) image -> gamma < 1 brightens it
    under_exposed_image = cv2.imread(
        'Images/Section 2/trees.tif', cv2.IMREAD_UNCHANGED)
    under_exposed_gray = to_gray(under_exposed_image)

    # Over-exposed (washed-out) image -> gamma > 1 darkens it
    over_exposed_image = cv2.imread('Images/Section 2/low contrast.jpg')
    if over_exposed_image.ndim == 3:
        over_exposed_gray = to_gray(over_exposed_image)
    else:
        over_exposed_gray = over_exposed_image

    r_range = get_r_range()

    test_cases = [
        (under_exposed_gray,  0.40, "gamma=0.40",  "gamma_under_exposed",
         "Under-exposed (trees.tif)  gamma=0.40 < 1 -> brightens"),
        (under_exposed_gray,  0.25, "gamma=0.25",  "gamma_under_exposed_025",
         "Under-exposed (trees.tif)  gamma=0.25 < 1 -> brightens strongly"),
        (over_exposed_gray, 2.50, "gamma=2.50",  "gamma_over_exposed",
         "Over-exposed (low contrast.jpg)  gamma=2.50 > 1 -> darkens"),
        (over_exposed_gray, 1.80, "gamma=1.80",  "gamma_over_exposed_180",
         "Over-exposed (low contrast.jpg)  gamma=1.80 > 1 -> darkens moderately"),
    ]

    for img_gray, gamma_val, plt_lable, fname, title in test_cases:
        gamma_corrected_image = apply_gamma(img_gray, gamma_val)
        s_curve = np.clip(255.0 * np.power(r_range / 255.0, gamma_val), 0, 255)
        plot_transform_s2(img_gray, gamma_corrected_image,
                          r_range, s_curve,
                          f"Gamma Correction  s = 255·(r/255)^γ  - {title}",
                          subtitles=(
                              "Original", f"Gamma-corrected ({plt_lable})"),
                          save_path=f"output/section2/{fname}.png")

    plot_gamma_curves_family_s2(r_range)


def apply_piecewise_linear(img_gray):
    # Piecewise-Linear Contrast Stretching with 3 segments:
    # Segment 1  [0,   85) -> maps to [0,  30)   (compress deep shadows)
    # Segment 2  [85, 170) -> maps to [30, 220)  (stretch mid-tones)
    # Segment 3  [170, 255] -> maps to [220, 255] (clip highlights slightly)

    # Define break-points: (r_in, s_out) pairs including (0,0) and (255,255)
    r_breakpoints = [0, 85, 170, 255]
    s_breakpoints = [0, 30, 220, 255]

    t_arr = np.zeros(256, dtype=np.float64)
    for idx in range(len(r_breakpoints) - 1):
        r0, r1 = r_breakpoints[idx], r_breakpoints[idx + 1]
        s0, s1 = s_breakpoints[idx], s_breakpoints[idx + 1]
        slope = (s1 - s0) / (r1 - r0)
        for r in range(r0, r1 + 1):
            t_arr[r] = s0 + slope * (r - r0)

    t_arr = np.clip(t_arr, 0, 255).astype(np.uint8)
    return t_arr[img_gray]


def run_piecewise_linear():
    print("\n[2.1d] Piecewise-Linear Contrast Stretching")

    low_con = cv2.imread('Images/Section 2/low contrast.jpg')
    img_gray = to_gray(low_con)
    enhanced = apply_piecewise_linear(img_gray)

    r_range = get_r_range()
    plot_piecewise_linear_stretching_s2(img_gray, enhanced, r_range)


def extract_bitplanes(img_gray):
    # >> Right‑shift
    bitplanes = [(img_gray >> k) & 1 for k in range(8)]
    return bitplanes


def run_bitplane_slicing():
    print("\n[2.2] Bit-Plane Slicing")

    path = 'Images/Section 2/einstein.jpg'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_gray = to_gray(img)

    bitplanes = extract_bitplanes(img_gray)

    plot_all_bitplanes_s2(bitplanes)

    # ----- Reconstruct from top 4 MSBs (planes 7,6,5,4) -----
    recon_msb = np.zeros_like(img_gray, dtype=np.uint8)
    for k in range(4, 8):
        recon_msb = recon_msb + (bitplanes[k] * (2 ** k)).astype(np.uint8)

    # ----- Reconstruct from bottom 4 LSBs (planes 3,2,1,0) -----
    recon_lsb = np.zeros_like(img_gray, dtype=np.uint8)
    for k in range(0, 4):
        recon_lsb = recon_lsb + (bitplanes[k] * (2 ** k)).astype(np.uint8)

    mse_msb = compute_mse(img_gray, recon_msb)
    mse_lsb = compute_mse(img_gray, recon_lsb)

    # Comparison figure
    plot_bitplane_reconstruction_comparison_s2(
        img_gray, bitplanes, recon_msb, recon_lsb, mse_msb, mse_lsb)
    print(f"\n  MSE (original vs. top-4-MSB reconstruction): {mse_msb:.2f}")
    print(f"  MSE (original vs. bot-4-LSB reconstruction): {mse_lsb:.2f}")


def run_section2():
    print("\n" + "=" * 60)
    print("SECTION 2: Point Operations & Bit-Plane Slicing")
    print("=" * 60)

    run_negative()
    run_log_transform()
    run_gamma_correction()
    run_piecewise_linear()
    run_bitplane_slicing()

    print("\n[Section 2] All outputs saved to output/section2/")


if __name__ == "__main__":
    run_section2()
