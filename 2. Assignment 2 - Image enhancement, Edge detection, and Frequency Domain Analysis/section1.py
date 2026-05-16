import os
import sys
import time
import numpy as np
import scipy.io
from pathlib import Path
from visualize import (
    save_image_row, save_robustness_plot, save_metrics_table
)
from utils import (
    convolve2d, load_grayscale, normalise_to_uint8, clip_uint8,
    gaussian_kernel, gaussian_kernel_size,
    IMG_T1, OUT_S1, ensure_dirs
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


SOBEL_KX = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

SOBEL_KY = np.array([[-1, -2, -1],
                     [0,  0,  0],
                     [1,  2,  1]], dtype=np.float64)


PREWITT_KX = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float64)

PREWITT_KY = np.array([[-1, -1, -1],
                       [0,  0,  0],
                       [1,  1,  1]], dtype=np.float64)


ROBERTS_KX = np.array([[1,  0],
                       [0, -1]], dtype=np.float64)

ROBERTS_KY = np.array([[0, 1],
                       [-1, 0]], dtype=np.float64)


LAPLACIAN_K = np.array([[0,  1,  0],
                        [1, -4,  1],
                        [0,  1,  0]], dtype=np.float64)


def sobel(gray):
    Gx = convolve2d(gray.astype(np.float64), SOBEL_KX)
    Gy = convolve2d(gray.astype(np.float64), SOBEL_KY)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, mag


def prewitt(gray):
    Gx = convolve2d(gray.astype(np.float64), PREWITT_KX)
    Gy = convolve2d(gray.astype(np.float64), PREWITT_KY)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, mag


def roberts(gray):
    Gx = convolve2d(gray.astype(np.float64), ROBERTS_KX)
    Gy = convolve2d(gray.astype(np.float64), ROBERTS_KY)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, mag


def laplacian(gray):
    lap_raw = convolve2d(gray.astype(np.float64), LAPLACIAN_K)
    edge_map = normalise_to_uint8(np.abs(lap_raw))
    sharpened = clip_uint8(gray.astype(np.float64) - 1.0 * lap_raw)
    return lap_raw, edge_map, sharpened


def _non_maximum_suppression(magnitude,
                             direction):
    H, W = magnitude.shape
    output = np.zeros((H, W), dtype=np.float64)

    # Convert angle to degrees and wrap to [0°, 180°)
    angle_deg = np.rad2deg(direction) % 180.0

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            ang = angle_deg[i, j]

            if (0.0 <= ang < 22.5) or (157.5 <= ang < 180.0):
                # 0° - compare left/right
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= ang < 67.5:
                # 45° - compare top-left / bottom-right
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            elif 67.5 <= ang < 112.5:
                # 90° - compare top/bottom
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                # 135° - compare top-right / bottom-left
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]

            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                output[i, j] = magnitude[i, j]

    return output


def _hysteresis_threshold(nms,
                          low_thresh,
                          high_thresh):
    from numpy.lib.stride_tricks import as_strided
    STRONG = np.uint8(255)
    WEAK = np.uint8(128)
    output = np.zeros(nms.shape, dtype=np.uint8)
    output[nms >= high_thresh] = STRONG
    output[(nms >= low_thresh) & (nms < high_thresh)] = WEAK

    H, W = output.shape
    changed = True
    while changed:
        strong_mask = (output == STRONG).astype(np.uint8)
        # Build 3x3 sliding-window view of the strong mask
        padded = np.pad(strong_mask, 1, mode='constant', constant_values=0)
        padded = np.ascontiguousarray(padded)
        win_shape = (H, W, 3, 3)
        win_strides = (padded.strides[0], padded.strides[1],
                       padded.strides[0], padded.strides[1])
        wins = as_strided(padded, shape=win_shape, strides=win_strides)
        # Any pixel whose 3x3 neighbourhood contains a strong pixel
        has_strong_neighbour = wins.max(axis=(2, 3)).astype(bool)
        # Promote weak pixels that have a strong neighbour
        promote = (output == WEAK) & has_strong_neighbour
        if promote.any():
            output[promote] = STRONG
        else:
            changed = False

    output[output == WEAK] = 0
    return output


def canny(gray,
          sigma=1.0,
          low_ratio=0.05,
          high_ratio=0.15):
    # Step 1 - Gaussian smoothing
    ksize = gaussian_kernel_size(sigma)
    kernel = gaussian_kernel(ksize, sigma)
    blurred = convolve2d(gray.astype(np.float64), kernel)
    blurred_u8 = clip_uint8(blurred)

    # Step 2 - Gradient magnitude & direction (Sobel)
    Gx, Gy, mag = sobel(blurred_u8)
    direction = np.arctan2(Gy, Gx)       # radians in [-π, π]

    # Step 3 - Non-maximum suppression
    nms = _non_maximum_suppression(mag, direction)

    # Step 4 - Hysteresis thresholding
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


def _load_gt_consensus(mat_path: str):

    mat = scipy.io.loadmat(mat_path, squeeze_me=False)
    gt = mat['groundTruth']           # shape (1, n_annotators)
    n = gt.shape[1]
    first_b = gt[0, 0]['Boundaries'][0, 0]
    H, W = first_b.shape
    consensus = np.zeros((H, W), dtype=np.uint8)
    for i in range(n):
        b = gt[0, i]['Boundaries'][0, 0]
        consensus = np.logical_or(consensus, b > 0).astype(np.uint8)
    return consensus


def threshold_edge_map(magnitude,
                       percentile=90.0):

    nonzero = magnitude[magnitude > 0]
    if len(nonzero) == 0:
        return np.zeros_like(magnitude, dtype=np.uint8)
    thresh = np.percentile(nonzero, percentile)
    return (magnitude >= thresh).astype(np.uint8)


def _dilate_binary(binary, radius: int):

    out = binary.astype(np.float64)
    H, W = binary.shape
    r = int(radius)

    # Horizontal max-sweep via stride tricks
    padded = np.pad(out, ((0, 0), (r, r)), mode='constant', constant_values=0)
    from numpy.lib.stride_tricks import as_strided
    win_shape = (H, W, 2 * r + 1)
    win_strides = (padded.strides[0], padded.strides[1], padded.strides[1])
    wins = as_strided(padded, shape=win_shape, strides=win_strides)
    out = wins.max(axis=2)

    # Vertical max-sweep
    padded = np.pad(out, ((r, r), (0, 0)), mode='constant', constant_values=0)
    win_shape = (H, W, 2 * r + 1)
    win_strides = (padded.strides[0], padded.strides[1], padded.strides[0])
    wins = as_strided(np.ascontiguousarray(padded),
                      shape=win_shape, strides=win_strides)
    out = wins.max(axis=2)

    return (out > 0).astype(np.uint8)


def evaluate_edge_detector(pred_binary,
                           gt_binary,
                           tolerance_px: int = 2):

    pred = (pred_binary > 0).astype(np.uint8)
    gt = (gt_binary > 0).astype(np.uint8)

    n_pred = int(pred.sum())
    n_gt = int(gt.sum())

    if n_pred == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loc_error'('inf')}
    if n_gt == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loc_error'('inf')}

    # Dilation-based Precision & Recall (vectorised, no Python pixel loops)
    gt_dilated = _dilate_binary(gt,   tolerance_px)
    pred_dilated = _dilate_binary(pred, tolerance_px)

    tp_precision = int((pred * gt_dilated).sum())
    tp_recall = int((gt * pred_dilated).sum())
    precision = tp_precision / n_pred
    recall = tp_recall / n_gt
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Localisation Error - sampled pairwise distances (pred -> nearest GT)
    pred_pts = np.column_stack(np.where(pred == 1))   # (N_pred, 2)
    gt_pts = np.column_stack(np.where(gt == 1))   # (N_gt,   2)

    MAX_SAMPLE = 3000
    if len(pred_pts) > MAX_SAMPLE:
        idx = np.random.choice(len(pred_pts), MAX_SAMPLE, replace=False)
        pred_pts_s = pred_pts[idx]
    else:
        pred_pts_s = pred_pts

    BLOCK = 1000
    loc_errors = []
    for start in range(0, len(pred_pts_s), BLOCK):
        chunk = pred_pts_s[start:start + BLOCK].astype(np.float64)
        gt_f = gt_pts.astype(np.float64)
        diff = chunk[:, np.newaxis, :] - gt_f[np.newaxis, :, :]
        dist2 = (diff ** 2).sum(axis=2)
        loc_errors.append(np.sqrt(dist2.min(axis=1)))

    mean_loc = float(np.concatenate(loc_errors).mean())

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

    # We process both images; use 100007 as primary for evaluation.
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

        # -- 1.1b  Prewitt -----------------------------------------------------
        t0 = time.time()
        Gx_pre, Gy_pre, mag_pre = prewitt(gray)
        t_prewitt = time.time() - t0
        mag_pre_u8 = normalise_to_uint8(mag_pre)
        print(f"    Prewitt   -> magnitude range [{mag_pre.min():.2f}, {mag_pre.max():.2f}]  "
              f"time={t_prewitt*1000:.1f} ms")

        # -- 1.1c  Roberts -----------------------------------------------------
        t0 = time.time()
        Gx_rob, Gy_rob, mag_rob = roberts(gray)
        t_roberts = time.time() - t0
        mag_rob_u8 = normalise_to_uint8(mag_rob)
        print(f"    Roberts   -> magnitude range [{mag_rob.min():.2f}, {mag_rob.max():.2f}]  "
              f"time={t_roberts*1000:.1f} ms")

        # -- 1.1d  Laplacian ---------------------------------------------------
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
            images=[normalise_to_uint8(
                Gx_sob), normalise_to_uint8(Gy_sob), mag_sob_u8],
            titles=['Sobel Gx (normalised)',
                    'Sobel Gy (normalised)', 'Sobel Magnitude'],
            output_path=os.path.join(
                OUT_S1, f'{img_name}_sobel_components.png'),
            suptitle=f'Task 1.1 - Sobel Components ({img_name})'
        )

        # -- 1.2  Canny Multi-Scale --------------------------------------------
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

        # Also save intermediate Canny pipeline stages for sigma=1.5
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

        # Threshold Sobel magnitude at 90th percentile
        sobel_binary = threshold_edge_map(mag_sob, percentile=90.0)
        canny_binary = (canny_results[1]['edges'] > 0).astype(np.uint8)

        for method_name, pred_bin in [('Sobel (90th pct)', sobel_binary),
                                      ('Canny (sigma=1.5)',    canny_binary)]:
            metrics = evaluate_edge_detector(
                pred_bin, gt_binary, tolerance_px=2)
            print(f"\n    -- {method_name} --")
            print(f"       Precision      : {metrics['precision']:.4f}")
            print(f"       Recall         : {metrics['recall']:.4f}")
            print(f"       F1 Score       : {metrics['f1']:.4f}")
            print(f"       Loc. Error (px): {metrics['loc_error']:.3f}")

        # -- 1.3d  Robustness to Noise -----------------------------------------
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

        # Metrics table for both methods (clean image)
        sob_m = evaluate_edge_detector(sobel_binary, gt_binary)
        canny_m = evaluate_edge_detector(canny_binary, gt_binary)
        cell_data = [
            [f"{sob_m['precision']:.4f}", f"{sob_m['recall']:.4f}",
             f"{sob_m['f1']:.4f}",        f"{sob_m['loc_error']:.3f} px"],
            [f"{canny_m['precision']:.4f}", f"{canny_m['recall']:.4f}",
             f"{canny_m['f1']:.4f}",        f"{canny_m['loc_error']:.3f} px"],
        ]

    print("\n  [INFO] Task 1 complete. All outputs saved to:", OUT_S1)


if __name__ == '__main__':
    run_section1()
