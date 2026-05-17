import os
import sys
import time
import numpy as np
import scipy.io
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from visualize import (
    save_image_row, save_robustness_plot
)
from utils import (
    convolve2d, load_grayscale, normalise_to_uint8, clip_uint8,
    gaussian_kernel, gaussian_kernel_size,
    IMG_T1, OUT_S1, ensure_dirs
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


SOBEL_KERNEL_X = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

SOBEL_KERNEL_Y = np.array([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]], dtype=np.float64)

PREWITT_KERNEL_X = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float64)

PREWITT_KERNEL_Y = np.array([[-1, -1, -1],
                             [0,  0,  0],
                             [1,  1,  1]], dtype=np.float64)

ROBERTS_KERNEL_X = np.array([[1,  0],
                             [0, -1]], dtype=np.float64)

ROBERTS_KERNEL_Y = np.array([[0, 1],
                             [-1, 0]], dtype=np.float64)

LAPLACIAN_KERNEL = np.array([[0,  1,  0],
                             [1, -4,  1],
                             [0,  1,  0]], dtype=np.float64)


def sobel(gray):
    Gx = convolve2d(gray.astype(np.float64), SOBEL_KERNEL_X)
    Gy = convolve2d(gray.astype(np.float64), SOBEL_KERNEL_Y)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, mag


def prewitt(gray):
    Gx = convolve2d(gray.astype(np.float64), PREWITT_KERNEL_X)
    Gy = convolve2d(gray.astype(np.float64), PREWITT_KERNEL_Y)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, mag


def roberts(gray):
    Gx = convolve2d(gray.astype(np.float64), ROBERTS_KERNEL_X)
    Gy = convolve2d(gray.astype(np.float64), ROBERTS_KERNEL_Y)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, mag


def laplacian(gray):
    lap_raw = convolve2d(gray.astype(np.float64), LAPLACIAN_KERNEL)
    edge_map = normalise_to_uint8(np.abs(lap_raw))
    sharpened = clip_uint8(gray.astype(np.float64) - 1.0 * lap_raw)
    return lap_raw, edge_map, sharpened


def _non_maximum_suppression(magnitude, direction):

    # Gradient angle in degrees, wrapped to [0, 180).
    angle_deg = np.rad2deg(direction) % 180.0

    # Quantise into 4 bins. The default bin (0) covers [0, 22.5) and
    # [157.5, 180), which both correspond to a near-horizontal gradient.
    direction_bin = np.zeros_like(angle_deg, dtype=np.uint8)
    direction_bin[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 1   # ~45 deg
    direction_bin[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 2  # vertical
    direction_bin[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 3  # ~135 deg

    padded = np.pad(magnitude, 1, mode='constant', constant_values=0)

    # Eight one-pixel-shifted views of `magnitude`. Read each slice as
    # "the pixel <direction> of (i,j)" - e.g. `left[i,j]` is magnitude[i,j-1].
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    up_left = padded[:-2, :-2]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    down_right = padded[2:, 2:]

    # For each pixel, pick its two neighbours along the gradient direction:
    #   bin 0 (horizontal gradient): left      / right
    #   bin 1 (45 deg gradient):     up-left   / down-right
    #   bin 2 (vertical gradient):   up        / down
    #   bin 3 (135 deg gradient):    up-right  / down-left
    n1 = np.where(direction_bin == 0, left,
         np.where(direction_bin == 1, up_left,
         np.where(direction_bin == 2, up, up_right)))
    n2 = np.where(direction_bin == 0, right,
         np.where(direction_bin == 1, down_right,
         np.where(direction_bin == 2, down, down_left)))

    keep = (magnitude >= n1) & (magnitude >= n2)
    output = np.where(keep, magnitude, 0.0)

    # Remove border pixels
    output[0, :] = 0
    output[-1, :] = 0
    output[:,  0] = 0
    output[:, -1] = 0

    return output


def _hysteresis_threshold(nms, low_thresh, high_thresh):

    STRONG = np.uint8(255)
    WEAK = np.uint8(128)

    output = np.zeros(nms.shape, dtype=np.uint8)
    output[nms >= high_thresh] = STRONG
    output[(nms >= low_thresh) & (nms < high_thresh)] = WEAK

    while True:
        strong_mask = (output == STRONG)
        padded_mask = np.pad(strong_mask, 1, mode='constant',
                             constant_values=False)

        # Each slice below is `strong_mask` shifted by one pixel in one of the
        # eight directions. ORing all eight gives, for every pixel, whether
        # any of its 3x3 neighbours is currently strong.
        has_strong_neighbour = (
            padded_mask[:-2, :-2] | padded_mask[:-2, 1:-1] | padded_mask[:-2, 2:] |
            padded_mask[1:-1, :-2] | padded_mask[1:-1, 2:] |
            padded_mask[2:, :-2] | padded_mask[2:, 1:-1] | padded_mask[2:, 2:]
        )

        promote = (output == WEAK) & has_strong_neighbour
        if not promote.any():
            break
        output[promote] = STRONG

    # Any WEAK pixel that was never promoted becomes a non-edge.
    output[output == WEAK] = 0
    return output


def canny(gray, sigma=1.0, low_ratio=0.05, high_ratio=0.15):
    #  Gaussian smoothing
    ksize = gaussian_kernel_size(sigma)
    kernel = gaussian_kernel(ksize, sigma)
    blurred = convolve2d(gray.astype(np.float64), kernel)
    blurred_u8 = clip_uint8(blurred)

    #  Gradient magnitude & direction (Sobel)
    Gx, Gy, mag = sobel(blurred_u8)
    direction = np.arctan2(Gy, Gx)       # radians in [-pi, pi]

    #  Non-maximum suppression
    nms = _non_maximum_suppression(mag, direction)

    #  Hysteresis thresholding
    max_nms = nms.max() if nms.max() > 0 else 1.0
    low_thresh = low_ratio * max_nms
    high_thresh = high_ratio * max_nms
    edges = _hysteresis_threshold(nms, low_thresh, high_thresh)

    return {
        'blurred': blurred_u8,
        'magnitude': mag,
        'direction': direction,
        'nms': nms,
        'edges': edges,
        'sigma': sigma,
        'low_thresh': low_thresh,
        'high_thresh': high_thresh,
    }


def _load_gt_consensus(mat_path):
    mat = scipy.io.loadmat(mat_path, squeeze_me=False)
    gt = mat['groundTruth']
    num_annotators = gt.shape[1]
    first_boundary = gt[0, 0]['Boundaries'][0, 0]
    H, W = first_boundary.shape
    consensus = np.zeros((H, W), dtype=np.uint8)
    for i in range(num_annotators):
        boundaries = gt[0, i]['Boundaries'][0, 0]
        consensus = np.logical_or(consensus, boundaries > 0).astype(np.uint8)
    return consensus


def threshold_edge_map(magnitude, percentile=90.0):
    nonzero = magnitude[magnitude > 0]
    if len(nonzero) == 0:
        return np.zeros_like(magnitude, dtype=np.uint8)
    thresh = np.percentile(nonzero, percentile)
    return (magnitude >= thresh).astype(np.uint8)


def _dilate_binary(binary, radius):
    out = binary.astype(bool)
    H, W = out.shape
    rad = int(radius)

    # Expand edges horizontally.
    # First add empty columns on the left and right so shifting stays in bounds.
    # Then slide across neighboring columns and combine them with OR,
    # so nearby edge pixels spread sideways.
    padded_h = np.pad(out, ((0, 0), (rad, rad)),
                      mode='constant', constant_values=False)
    horizontal = np.zeros_like(out)
    for offset in range(2 * rad + 1):
        horizontal = horizontal | padded_h[:, offset:offset + W]

    # Vertical sweep on the horizontally-dilated result.
    padded_v = np.pad(horizontal, ((rad, rad), (0, 0)),
                      mode='constant', constant_values=False)
    vertical = np.zeros_like(out)
    for offset in range(2 * rad + 1):
        vertical = vertical | padded_v[offset:offset + H, :]

    return vertical.astype(np.uint8)


def evaluate_edge_detector(pred_binary, gt_binary, tolerance_px: int = 2):
    pred = (pred_binary > 0).astype(np.uint8)
    gt = (gt_binary > 0).astype(np.uint8)

    n_pred = int(pred.sum())
    n_gt = int(gt.sum())

    if n_pred == 0 or n_gt == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'loc_error': float('inf')}

    # Precision / Recall with dilation-based matching.
    gt_dilated = _dilate_binary(gt,   tolerance_px)
    pred_dilated = _dilate_binary(pred, tolerance_px)

    tp_precision = int((pred * gt_dilated).sum())
    tp_recall = int((gt * pred_dilated).sum())
    precision = tp_precision / n_pred
    recall = tp_recall / n_gt
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Localisation error.
    # `distance_transform_edt(gt == 0)` returns, for every pixel, the Euclidean
    # distance to the nearest pixel where gt == 1 (i.e. the nearest GT edge).
    # We then average that distance over the predicted edge pixels.
    # Average distance = localization error. -> Smaller = better.
    dist_to_nearest_gt = distance_transform_edt(gt == 0)
    mean_loc = float(dist_to_nearest_gt[pred == 1].mean())

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loc_error': mean_loc,
    }


def add_gaussian_noise(gray, sigma):
    noise = np.random.normal(0, sigma, gray.shape)
    noisy = gray.astype(np.float64) + noise
    return clip_uint8(noisy)


def run_section1():
    ensure_dirs()
    print("\n" + "=" * 70)
    print("  TASK 1 - Edge Detection & Multi-Scale Analysis")
    print("=" * 70)
    image_names = ['100007', '69007']

    for img_name in image_names:
        jpg_path = os.path.join(IMG_T1, f'{img_name}.jpg')
        mat_path = os.path.join(IMG_T1, f'{img_name}.mat')

        print(f"\n{'-'*60}")
        print(f"  Image: {img_name}.jpg")
        print(f"{'-'*60}")

        gray = load_grayscale(jpg_path)
        print(f"  Loaded grayscale  shape={gray.shape}  dtype={gray.dtype}")
        print(f"  Pixel range: [{gray.min()}, {gray.max()}]")

        # -- 1.1a  Sobel ------------------------------------------------------
        print("\n  [1.1] Classical Operators")
        t0 = time.time()
        Gx_sob, Gy_sob, mag_sob = sobel(gray)
        t_sobel = time.time() - t0
        mag_sob_u8 = normalise_to_uint8(mag_sob)
        print(f"    Sobel     -> magnitude range [{mag_sob.min():.2f}, {mag_sob.max():.2f}]  "
              f"time={t_sobel*1000:.1f} ms")

        # -- 1.1b  Prewitt ----------------------------------------------------
        t0 = time.time()
        Gx_pre, Gy_pre, mag_pre = prewitt(gray)
        t_prewitt = time.time() - t0
        mag_pre_u8 = normalise_to_uint8(mag_pre)
        print(f"    Prewitt   -> magnitude range [{mag_pre.min():.2f}, {mag_pre.max():.2f}]  "
              f"time={t_prewitt*1000:.1f} ms")

        # -- 1.1c  Roberts ----------------------------------------------------
        t0 = time.time()
        Gx_rob, Gy_rob, mag_rob = roberts(gray)
        t_roberts = time.time() - t0
        mag_rob_u8 = normalise_to_uint8(mag_rob)
        print(f"    Roberts   -> magnitude range [{mag_rob.min():.2f}, {mag_rob.max():.2f}]  "
              f"time={t_roberts*1000:.1f} ms")

        # -- 1.1d  Laplacian --------------------------------------------------
        t0 = time.time()
        lap_raw, lap_edge, lap_sharp = laplacian(gray)
        t_lap = time.time() - t0
        print(f"    Laplacian -> signed range [{lap_raw.min():.2f}, {lap_raw.max():.2f}]  "
              f"time={t_lap*1000:.1f} ms")
        print(
            f"              Sharpened pixel range: [{lap_sharp.min()}, {lap_sharp.max()}]")

        # Save classical operators figure
        save_image_row(
            images=[gray, mag_sob_u8, mag_pre_u8,
                    mag_rob_u8, lap_edge, lap_sharp],
            titles=['Original', 'Sobel', 'Prewitt', 'Roberts',
                    'Laplacian Edge', 'Laplacian Sharp'],
            output_path=os.path.join(
                OUT_S1, f'{img_name}_classical_operators.png'),
            suptitle=f'Task 1.1 - Classical Edge Detectors ({img_name})'
        )

        # Sobel Gx / Gy figure
        save_image_row(
            images=[normalise_to_uint8(Gx_sob),
                    normalise_to_uint8(Gy_sob),
                    mag_sob_u8],
            titles=['Sobel Gx (normalised)',
                    'Sobel Gy (normalised)', 'Sobel Magnitude'],
            output_path=os.path.join(
                OUT_S1, f'{img_name}_sobel_components.png'),
            suptitle=f'Task 1.1 - Sobel Components ({img_name})'
        )

        # -- 1.2  Canny Multi-Scale -------------------------------------------
        print(
            "\n  [1.2] Canny Edge Detection Pipeline - Multi-Scale (sigma = 0.5, 1.5, 3.0)")
        sigmas = [0.5, 1.5, 3.0]
        canny_results = []
        for sigma in sigmas:
            t0 = time.time()
            res = canny(gray, sigma=sigma, low_ratio=0.05, high_ratio=0.15)
            t_c = time.time() - t0
            canny_results.append(res)
            n_edges = int((res['edges'] > 0).sum())
            print(f"    sigma={sigma:.1f}  kernel={gaussian_kernel_size(sigma)}x{gaussian_kernel_size(sigma)}  "
                  f"low={res['low_thresh']:.2f}  high={res['high_thresh']:.2f}  "
                  f"edges={n_edges}  time={t_c*1000:.1f} ms")

        canny_imgs = [gray] + [r['edges'] for r in canny_results]
        canny_titles = ['Original'] + [f'Canny sigma={s}' for s in sigmas]
        save_image_row(
            images=canny_imgs, titles=canny_titles,
            output_path=os.path.join(
                OUT_S1, f'{img_name}_canny_multiscale.png'),
            suptitle=f'Task 1.2 - Canny Multi-Scale ({img_name})'
        )

        # Intermediate Canny pipeline stages for sigma=1.5
        res_mid = canny_results[1]
        save_image_row(
            images=[gray,
                    res_mid['blurred'],
                    normalise_to_uint8(res_mid['magnitude']),
                    normalise_to_uint8(res_mid['nms']),
                    res_mid['edges']],
            titles=['Input', 'Gaussian (sigma=1.5)', 'Gradient Magnitude',
                    'After NMS', 'Final Edges'],
            output_path=os.path.join(
                OUT_S1, f'{img_name}_canny_pipeline_stages.png'),
            suptitle=f'Task 1.2 - Canny Pipeline Stages sigma=1.5 ({img_name})'
        )

        # -- 1.3  Quantitative Evaluation -------------------------------------
        print("\n  [1.3] Quantitative Evaluation vs. Ground Truth")
        gt_binary = _load_gt_consensus(mat_path)
        print(f"    Ground truth shape: {gt_binary.shape}  "
              f"edge pixels: {int(gt_binary.sum())}")

        sobel_binary = threshold_edge_map(mag_sob, percentile=90.0)
        canny_binary = (canny_results[1]['edges'] > 0).astype(np.uint8)

        for method_name, pred_bin in [('Sobel (90th pct)', sobel_binary),
                                      ('Canny (sigma=1.5)', canny_binary)]:
            metrics = evaluate_edge_detector(
                pred_bin, gt_binary, tolerance_px=2)
            print(f"\n    -- {method_name} --")
            print(f"       Precision      : {metrics['precision']:.4f}")
            print(f"       Recall         : {metrics['recall']:.4f}")
            print(f"       F1 Score       : {metrics['f1']:.4f}")
            print(f"       Loc. Error (px): {metrics['loc_error']:.3f}")

        # -- 1.3d  Robustness to Noise ----------------------------------------
        print(
            "\n  [1.3] Robustness to Noise (adding Gaussian noise sigma: [0, 30])")
        np.random.seed(42)
        noise_sigmas = [0, 10, 20, 30]
        sob_prec_list, sob_rec_list = [], []
        can_prec_list, can_rec_list = [], []

        for ns in noise_sigmas:
            noisy_img = add_gaussian_noise(gray, sigma=ns) if ns > 0 else gray
            # Sobel on noisy image
            _, _, mag_n = sobel(noisy_img)
            sob_bin = threshold_edge_map(mag_n, percentile=90.0)
            m_sob = evaluate_edge_detector(sob_bin, gt_binary)
            sob_prec_list.append(m_sob['precision'])
            sob_rec_list.append(m_sob['recall'])
            # Canny on noisy image (sigma=1.5)
            can_res = canny(noisy_img, sigma=1.5)
            can_bin = (can_res['edges'] > 0).astype(np.uint8)
            m_can = evaluate_edge_detector(can_bin, gt_binary)
            can_prec_list.append(m_can['precision'])
            can_rec_list.append(m_can['recall'])
            print(f"    Noise sigma={ns:2d}  "
                  f"Sobel P={m_sob['precision']:.3f} R={m_sob['recall']:.3f}  "
                  f"Canny P={m_can['precision']:.3f} R={m_can['recall']:.3f}")

        save_robustness_plot(
            noise_levels=noise_sigmas, precision_list=sob_prec_list,
            recall_list=sob_rec_list,  method_name=f'Sobel ({img_name})',
            output_path=os.path.join(
                OUT_S1, f'{img_name}_robustness_sobel.png')
        )
        save_robustness_plot(
            noise_levels=noise_sigmas, precision_list=can_prec_list,
            recall_list=can_rec_list,  method_name=f'Canny sigma=1.5 ({img_name})',
            output_path=os.path.join(
                OUT_S1, f'{img_name}_robustness_canny.png')
        )

        sob_m = evaluate_edge_detector(sobel_binary, gt_binary)
        canny_m = evaluate_edge_detector(canny_binary, gt_binary)
        cell_data = [
            [f"{sob_m['precision']:.4f}", f"{sob_m['recall']:.4f}",
             f"{sob_m['f1']:.4f}",        f"{sob_m['loc_error']:.3f} px"],
            [f"{canny_m['precision']:.4f}", f"{canny_m['recall']:.4f}",
             f"{canny_m['f1']:.4f}",        f"{canny_m['loc_error']:.3f} px"],
        ]
        print(cell_data)

    print("\n  [INFO] Task 1 complete. All outputs saved to:", OUT_S1)


if __name__ == '__main__':
    run_section1()