# ⚠️⚠️⚠️ Only used cv2.cvtColor for imshow()

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")

try:
    from visualize import print_metrics_table_s1, plot_metrics_bar_s1
    from utils import compute_mse, compute_psnr
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from visualize import print_metrics_table_s1, plot_metrics_bar_s1
    from utils import compute_mse, compute_psnr

os.makedirs("output/section1", exist_ok=True)


def downsample(img, N):
    return img[::N, ::N].copy()


def upsample_nn(img_down, target_H, target_W, N):
    src_height, src_width = img_down.shape[:2]

    tgt_row_idx = np.arange(target_H)
    tgt_col_idx = np.arange(target_W)

    i_map = np.clip((tgt_row_idx / N).astype(int), 0, src_height - 1)
    j_map = np.clip((tgt_col_idx / N).astype(int), 0, src_width - 1)

    result = img_down[np.ix_(i_map, j_map)]

    return result.astype(np.uint8)


def upsample_bilinear(img_down, target_H, target_W, N):

    src_height, src_width = img_down.shape[:2]
    is_color = img_down.ndim == 3

    fx = np.arange(target_H) / N
    fy = np.arange(target_W) / N

    i0 = np.clip(np.floor(fx).astype(int), 0, src_height - 1)
    i1 = np.clip(i0 + 1, 0, src_height - 1)
    j0 = np.clip(np.floor(fy).astype(int), 0, src_width - 1)
    j1 = np.clip(j0 + 1, 0, src_width - 1)

    dx = (fx - np.floor(fx))[:, np.newaxis]
    dy = (fy - np.floor(fy))[np.newaxis, :]

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    def _interp_channel(ch):
        ch = ch.astype(np.float64)
        I00 = ch[np.ix_(i0, j0)]
        I10 = ch[np.ix_(i1, j0)]
        I01 = ch[np.ix_(i0, j1)]
        I11 = ch[np.ix_(i1, j1)]
        return np.clip(w00 * I00 + w10 * I10 + w01 * I01 + w11 * I11, 0, 255)

    if is_color:
        channels = [
            _interp_channel(img_down[:, :, c]) for c in range(img_down.shape[2])
        ]
        result = np.stack(channels, axis=2)
    else:
        result = _interp_channel(img_down)

    return result.astype(np.uint8)


def upsample_bicubic(img_down, target_H, target_W, N):
    def _k_cubic(t):
        t = np.abs(t)
        return np.where(
            t <= 1.0,
            1.5 * t**3 - 2.5 * t**2 + 1.0,
            np.where(t < 2.0, -0.5 * t**3 + 2.5 * t**2 - 4.0 * t + 2.0, 0.0),
        )

    src_height, src_width = img_down.shape[:2]
    is_color = img_down.ndim == 3

    fx = np.arange(target_H) / N
    fy = np.arange(target_W) / N

    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)

    dx = fx - i0
    dy = fy - j0

    offsets = [-1, 0, 1, 2]
    wx = [_k_cubic(dx - m) for m in offsets]
    wy = [_k_cubic(dy - n) for n in offsets]

    def _interp_channel(ch):
        ch = ch.astype(np.float64)
        acc = np.zeros((target_H, target_W), dtype=np.float64)
        for mi, m in enumerate(offsets):
            i_idx = np.clip(i0 + m, 0, src_height - 1)
            for ni, n in enumerate(offsets):
                j_idx = np.clip(j0 + n, 0, src_width - 1)

                w = wx[mi][:, np.newaxis] * wy[ni][np.newaxis, :]

                pixel_val = ch[np.ix_(i_idx, j_idx)]
                acc += w * pixel_val
        return np.clip(acc, 0.0, 255.0)

    if is_color:
        channels = [
            _interp_channel(img_down[:, :, c]) for c in range(img_down.shape[2])
        ]
        result = np.stack(channels, axis=2)
    else:
        result = _interp_channel(img_down)

    return result.astype(np.uint8)


def crop_patch(img, row_start, col_start, patch_h=60, patch_w=60):
    """Extract a patch; clamp to image bounds."""
    r1 = min(row_start, img.shape[0] - patch_h)
    c1 = min(col_start, img.shape[1] - patch_w)
    return img[r1 : r1 + patch_h, c1 : c1 + patch_w]


def process_image(img_path, img_name, factors=(2, 4, 8)):
    original = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if original is None:
        print(f"  [WARNING] Could not read {img_path}")
        return {}

    is_color = original.ndim == 3
    tgt_height, tgt_width = original.shape[:2]

    methods = {
        "Nearest\nNeighbour": upsample_nn,
        "Bilinear": upsample_bilinear,
        "Bicubic": upsample_bicubic,
    }

    metrics = {}

    for scale_fact in factors:
        downsampled = downsample(original, scale_fact)
        downsampled_height, downsampled_width = downsampled.shape[:2]
        print(
            f"  N={scale_fact}: original {tgt_height}x{tgt_width} -> downsampled {downsampled_height}x{downsampled_width}"
        )

        plt_num_cols = 1 + len(methods)
        plt_fig, plt_axes = plt.subplots(1, plt_num_cols, figsize=(5 * plt_num_cols, 5))
        plt_fig.suptitle(
            f"{img_name} | Downsampling factor N={scale_fact}", fontsize=12
        )

        def _show(ax, im, title, is_col):
            if is_col:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        _show(plt_axes[0], original, "Original\n(reference)", is_color)

        for col_idx, (method_name, upsample_fn) in enumerate(methods.items(), start=1):
            upsampled = upsample_fn(downsampled, tgt_height, tgt_width, scale_fact)
            mse = compute_mse(original, upsampled)
            psnr = compute_psnr(original, upsampled)
            clean_method_name = method_name.replace("\n", " ")
            metrics[(scale_fact, clean_method_name)] = (mse, psnr)
            plt_label = f"{method_name}\nMSE={mse:.2f} | PSNR={psnr:.2f} dB"
            _show(plt_axes[col_idx], upsampled, plt_label, is_color)

        plt.tight_layout()
        out_path = f"output/section1/{img_name}_N{scale_fact}_comparison.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()

        # ---- Zoomed-in patches figure ----
        patch_start_y, patch_start_x = tgt_height // 4, tgt_width // 4
        plt_fig2, plt_axes2 = plt.subplots(
            1, plt_num_cols, figsize=(4 * plt_num_cols, 4)
        )
        plt_fig2.suptitle(f"{img_name} | N={scale_fact} - Zoomed Patch", fontsize=11)

        _show(
            plt_axes2[0],
            crop_patch(original, patch_start_y, patch_start_x),
            "Original\npatch",
            is_color,
        )
        for col_idx, (method_name, upsample_fn) in enumerate(methods.items(), start=1):
            upsampled = upsample_fn(downsampled, tgt_height, tgt_width, scale_fact)
            _show(
                plt_axes2[col_idx],
                crop_patch(upsampled, patch_start_y, patch_start_x),
                method_name + "\npatch",
                is_color,
            )

        plt.tight_layout()
        patch_path = f"output/section1/{img_name}_N{scale_fact}_patch.png"
        plt.savefig(patch_path, dpi=150, bbox_inches="tight")
        plt.close()

    return metrics


def run_section1():
    print("\n" + "=" * 60)
    print("SECTION 1: Pixel Resolution & Interpolation")
    print("=" * 60)

    images = {
        "cameraman": "Images/Section 1/cameraman.tif",
        "mandrill": "Images/Section 1/mandrill.bmp",
        "peppers": "Images/Section 1/peppers.png",
        "sparseresidential": "Images/Section 1/sparseresidential_6.jpg",
    }

    for name, path in images.items():
        print(f"\n--- Processing: {name} ---")
        metrics = process_image(path, name)
        print_metrics_table_s1(metrics, name)
        plot_metrics_bar_s1(metrics, name)

    print("\n[Section 1] All outputs saved to output/section1/")


if __name__ == "__main__":
    run_section1()
