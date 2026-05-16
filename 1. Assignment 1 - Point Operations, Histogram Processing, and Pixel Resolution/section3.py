from utils import to_gray, compute_norm_cdf, compute_histogram, get_r_range
from visualize import plot_histogram_and_cdf_s3, plot_histogram_comparison_s3, plot_histogram_equalization_s3, plot_zoomed_patch_comparison_s3
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')

os.makedirs('output/section3', exist_ok=True)


def calc_ghe(gray):
    hist = compute_histogram(gray)
    cdf = compute_norm_cdf(hist)
    eq_map = np.floor(255.0 * cdf).astype(np.uint8)
    return eq_map[gray]


def run_ghe():
    print('\n[3.2] Global Histogram Equalization')

    images = {
        'low_contrast':  'Images/Section 3/low contrast.jpg',
        'moon':          'Images/Section 3/moon.tif',
        'trees':         'Images/Section 3/trees.tif',
        'einstein':      'Images/Section 3/einstein.jpg',
    }

    for name, path in images.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'  [WARNING] Cannot read {path}')
            continue
        img_gray = to_gray(img)
        img_eq = calc_ghe(img_gray)

        hist_orig = compute_histogram(img_gray)
        hist_eq = compute_histogram(img_eq)

        # --- Compute CDF overlay for transformation curve ---
        cdf_orig = compute_norm_cdf(hist_orig)
        r_range = get_r_range()
        lut_curve = np.floor(255.0 * cdf_orig)

        plot_histogram_comparison_s3(name, img_gray, img_eq, hist_orig,
                                     hist_eq, cdf_orig, r_range, lut_curve)


def calc_lhe(img_gray, window=15):
    height, width = img_gray.shape
    pad = window // 2
    padded = np.pad(img_gray, pad, mode='edge')
    result = np.empty_like(img_gray)

    total_pixels_in_window = float(window * window)

    for i in range(height):
        # (window height, img_w + 2·pad)
        strip = padded[i: i + window, :]

        h = compute_histogram(strip[:, :window]).astype(np.float64)

        for j in range(width):
            if j > 0:
                # Sliding window optimization: Avoid recalculating the full window.
                # Shift right by subtracting the exiting left column and adding the entering right column.

                h -= compute_histogram(strip[:, j - 1])
                h += compute_histogram(strip[:, j + window - 1])

            cdf = np.cumsum(h) / total_pixels_in_window
            local_mapping_table = np.floor(255.0 * cdf).astype(np.uint8)
            result[i, j] = local_mapping_table[img_gray[i, j]]

    return result


def run_lhe():
    print('\n[3.3] Local Histogram Equalization (LHE, win=15)')

    # Select images that benefit most from local enhancement
    images = {
        'moon':         'Images/Section 3/moon.tif',
        'trees':        'Images/Section 3/trees.tif',
        'med1':         'Images/Section 3/med1.jpg',
        'low_contrast': 'Images/Section 3/low contrast.jpg',
    }

    for name, path in images.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'  [WARNING] Cannot read {path}')
            continue
        img_gray = to_gray(img)

        img_gray_height, img_gray_width = img_gray.shape
        print(f'  [{name}]  {img_gray_height}x{img_gray_width}  - running LHE ...')

        gray_ghe = calc_ghe(img_gray)
        gray_lhe = calc_lhe(img_gray, window=15)

        # --- Side-by-side comparison ---
        plot_histogram_equalization_s3(name, img_gray, gray_ghe, gray_lhe)

    # --- Zoomed-in patch comparison for moon (reveals hidden details) ---
    img = cv2.imread('Images/Section 3/moon.tif', cv2.IMREAD_UNCHANGED)
    img_gray = to_gray(img)
    gray_ghe = calc_ghe(img_gray)
    gray_lhe = calc_lhe(img_gray, window=15)

    plot_zoomed_patch_comparison_s3(img_gray, gray_ghe, gray_lhe)


def run_section3():
    print('\n' + '=' * 60)
    print('SECTION 3: Step-by-Step Histogram Equalization')
    print('=' * 60)
    print('\n[3.1] Histogram Calculation & Plotting')

    images = [
        ('low_contrast', 'Images/Section 3/low contrast.jpg'),
        ('moon',         'Images/Section 3/moon.tif'),
        ('einstein',     'Images/Section 3/einstein.jpg'),
        ('mandrill',     'Images/Section 3/mandrill.png'),
    ]
    for name, path in images:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'  [WARNING] Cannot read {path}')
            continue
        img_gray = to_gray(img)
        plot_histogram_and_cdf_s3(
            img_gray,
            title=f'Histogram & Normalised CDF - {name}',
            save_path=f'output/section3/hist_{name}.png',
        )

    run_ghe()
    run_lhe()

    print('\n[Section 3] All outputs saved to output/section3/')


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_section3()
