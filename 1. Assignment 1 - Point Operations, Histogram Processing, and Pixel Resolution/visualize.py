# ⚠️⚠️⚠️ Only used cv2.cvtColor for imshow()
from utils import compute_norm_cdf, compute_histogram, get_r_range
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")


def compute_mse(img_a, img_b):
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


# ---------------------------------------------------
# Secttion 0
# ---------------------------------------------------


def plot_result_s0(
    SM,
    LG_zeros,
    LG_inserted,
    title="Diamond Embedding",
    filename="diamond_embedding.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    def show_matrix(ax, mat, t):
        ax.imshow(mat != 0, cmap="Blues", aspect="equal", vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if v == 0 else "white",
                )
        ax.set_title(t, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    show_matrix(axes[0], SM, f"Smaller Matrix SM\n({SM.shape[0]}x{SM.shape[1]})")
    show_matrix(
        axes[1],
        LG_zeros,
        f"Larger Matrix LG (zeros)\n({LG_zeros.shape[0]}x{LG_zeros.shape[1]})",
    )
    show_matrix(axes[2], LG_inserted, f"Result: SM embedded in LG")

    plt.tight_layout()

    # Prevent overwriting of previous images
    base, ext = os.path.splitext(filename)
    save_path = f"output/utils/{filename}"
    counter = 1
    while os.path.exists(save_path):
        save_path = f"output/utils/{base}_{counter}{ext}"
        counter += 1

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Section 0] Saved {save_path}")


# ---------------------------------------------------
# Secttion 1
# ---------------------------------------------------


def print_metrics_table_s1(all_metrics, img_name):
    print(f"\n  {'Method':<22} {'N=2':>18} {'N=4':>18} {'N=8':>18}")
    print("  " + "-" * 78)
    for method in ["Nearest Neighbour", "Bilinear", "Bicubic"]:
        row = f"  {method:<22}"
        for N in [2, 4, 8]:
            key = (N, method)
            if key in all_metrics:
                mse, psnr = all_metrics[key]
                row += f"  MSE={mse:7.2f}/{psnr:.1f}dB"
            else:
                row += f"  {'N/A':>18}"
        print(row)


def plot_metrics_bar_s1(all_metrics, img_name):
    factors = [2, 4, 8]
    method_names = ["Nearest Neighbour", "Bilinear", "Bicubic"]
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    x = np.arange(len(factors))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (mname, col) in enumerate(zip(method_names, colors)):
        psnr_vals = [all_metrics.get((N, mname), (0, 0))[1] for N in factors]
        ax.bar(
            x + i * width,
            psnr_vals,
            width,
            label=mname,
            color=col,
            edgecolor="black",
            linewidth=0.5,
        )
        for j, v in enumerate(psnr_vals):
            ax.text(x[j] + i * width, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)

    ax.set_xlabel("Downsampling Factor N", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title(f"{img_name} - PSNR Comparison", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"N={n}" for n in factors])
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/section1/{img_name}_psnr_chart.png", dpi=130)
    plt.close()


# ---------------------------------------------------
# Secttion 2
# ---------------------------------------------------


def plot_transform_s2(
    original,
    transformed,
    curve_r,
    curve_s,
    title,
    subtitles=("Original", "Enhanced"),
    save_path=None,
    cmap="gray",
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def _show(ax, im, t):
        if im.ndim == 3:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(t, fontsize=10)
        ax.axis("off")

    _show(axes[0], original, subtitles[0])
    _show(axes[1], transformed, subtitles[1])

    axes[2].plot(curve_r, curve_s, "b-", linewidth=2)
    axes[2].plot([0, 255], [0, 255], "k--", linewidth=1, alpha=0.4)
    axes[2].set_xlabel("Input intensity r", fontsize=10)
    axes[2].set_ylabel("Output intensity s", fontsize=10)
    axes[2].set_title("Transformation Curve  s = T(r)", fontsize=10)
    axes[2].set_xlim(0, 255)
    axes[2].set_ylim(0, 255)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.close()


def plot_gamma_curves_family_s2(r_range):
    # Explanation figure: family of gamma curves
    fig, ax = plt.subplots(figsize=(7, 5))
    gammas = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    cmap_lines = plt.cm.coolwarm(np.linspace(0, 1, len(gammas)))
    for g, col in zip(gammas, cmap_lines):
        s = 255.0 * np.power(r_range / 255.0, g)
        ax.plot(r_range, s, color=col, linewidth=1.8, label=f"γ={g}")
    ax.plot([0, 255], [0, 255], "k--", linewidth=1.2, alpha=0.5, label="γ=1 (identity)")
    ax.fill_between(
        r_range,
        r_range,
        255 * np.power(r_range / 255.0, 0.5),
        alpha=0.08,
        color="blue",
        label="Brightening region (γ<1)",
    )
    ax.fill_between(
        r_range,
        r_range,
        255 * np.power(r_range / 255.0, 2.0),
        alpha=0.08,
        color="red",
        label="Darkening region (γ>1)",
    )
    ax.set_xlabel("Input r", fontsize=11)
    ax.set_ylabel("Output s", fontsize=11)
    ax.set_title(
        "Power-Law (Gamma) Family of Curves\n"
        "γ<1 expands dark regions; γ>1 compresses them",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("output/section2/gamma_curves_family.png", dpi=130)
    plt.close()
    print("  Saved output/section2/gamma_curves_family.png")


def plot_piecewise_linear_stretching_s2(gray, enhanced, r_range):
    r_breakpoints = [0, 85, 170, 255]
    s_breakpoints = [0, 30, 220, 255]
    s = np.zeros(256)
    for seg in range(len(r_breakpoints) - 1):
        r0, r1 = r_breakpoints[seg], r_breakpoints[seg + 1]
        s0, s1 = s_breakpoints[seg], s_breakpoints[seg + 1]
        slope = (s1 - s0) / (r1 - r0)
        mask = (r_range >= r0) & (r_range <= r1)
        s[mask] = s0 + slope * (r_range[mask] - r0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Piecewise-Linear Contrast Stretching (low contrast.jpg)", fontsize=12)
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Enhanced")
    axes[1].axis("off")

    for i in range(len(r_breakpoints) - 1):
        axes[2].plot(
            r_breakpoints[i : i + 2], s_breakpoints[i : i + 2], "b-o", linewidth=2
        )
    axes[2].plot([0, 255], [0, 255], "k--", alpha=0.4)
    colors_seg = ["#e74c3c", "#3498db", "#2ecc71"]
    seg_labels = [
        "Seg 1: compress shadows\n[0,85]->[0,30]",
        "Seg 2: stretch mid-tones\n[85,170]->[30,220]",
        "Seg 3: clip highlights\n[170,255]->[220,255]",
    ]
    for i, (r0, r1, s0, s1) in enumerate(
        zip(
            r_breakpoints[:-1], r_breakpoints[1:], s_breakpoints[:-1], s_breakpoints[1:]
        )
    ):
        axes[2].fill_betweenx(
            [s0, s1], [r0, r1], alpha=0.15, color=colors_seg[i], label=seg_labels[i]
        )
    axes[2].set_xlabel("Input r")
    axes[2].set_ylabel("Output s")
    axes[2].set_title("Transformation s = T(r)")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].set_xlim(0, 255)
    axes[2].set_ylim(0, 255)
    plt.tight_layout()
    plt.savefig("output/section2/piecewise_linear.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved output/section2/piecewise_linear.png")


def plot_all_bitplanes_s2(bitplanes):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Bit-Plane Slicing - einstein.jpg", fontsize=13)
    for k in range(8):
        ax = axes[k // 4][k % 4]
        ax.imshow(bitplanes[k] * 255, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"Bit-plane {k}  ({'MSB' if k == 7 else 'LSB' if k == 0 else ''})",
            fontsize=9,
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("output/section2/bitplanes_all.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved output/section2/bitplanes_all.png")


def plot_bitplane_reconstruction_comparison_s2(
    img_gray, bitplanes, recon_msb, recon_lsb, mse_msb, mse_lsb
):
    # Changed from (1, 4) to (1, 3) and adjusted figsize width from 16 to 12
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
    fig2.suptitle("Bit-Plane Reconstruction - einstein.jpg", fontsize=13)

    for ax, im, title in zip(
        axes2,
        [img_gray, recon_msb, recon_lsb],  # Removed recon_lsb_vis
        [
            "Original",
            f"Top 4 MSBs (planes 7-4)\nMSE={mse_msb:.2f}",
            # Removed the 4th title
            f"Bottom 4 LSBs (planes 3-0)\nMSE={mse_lsb:.2f}",
        ],
    ):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "output/section2/bitplane_reconstruction.png", dpi=130, bbox_inches="tight"
    )
    plt.close()
    print("  Saved output/section2/bitplane_reconstruction.png")

    # ------ Visualise significance of each plane ------
    mse_vals = []
    psnr_vals = []
    labels = []
    for top_k in range(1, 9):  # use top top_k planes (from MSB downward)
        recon = np.zeros_like(img_gray, dtype=np.uint8)
        for k in range(8 - top_k, 8):
            recon = recon + (bitplanes[k] * (2**k)).astype(np.uint8)
        mse_v = compute_mse(img_gray, recon)

        if mse_v > 0:
            psnr_v = 10 * np.log10(255**2 / mse_v)
        else:
            psnr_v = 100.0  # Capped value for drawing the bar

        mse_vals.append(mse_v)
        psnr_vals.append(psnr_v)
        labels.append(f"Top {top_k}")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(labels, psnr_vals, color="steelblue", edgecolor="black", linewidth=0.5)
    ax3.set_xlabel("Number of top MSBs used")
    ax3.set_ylabel("PSNR (dB)")
    ax3.set_title("Reconstruction Quality vs. Number of MSBs (einstein.jpg)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(psnr_vals):
        if np.isfinite(v):
            ax3.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("output/section2/bitplane_psnr.png", dpi=130)
    plt.close()
    print("  Saved output/section2/bitplane_psnr.png")


# ---------------------------------------------------
# Secttion 3
# ---------------------------------------------------


def plot_histogram_and_cdf_s3(gray, title, save_path):
    """Plot the image alongside its histogram bar chart and normalised CDF."""
    h = compute_histogram(gray)
    cdf = compute_norm_cdf(h)
    r = np.arange(256)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].bar(r, h, width=1, color="steelblue", edgecolor="none")
    axes[1].set_xlabel("Intensity  r")
    axes[1].set_ylabel("Count  h(r)")
    axes[1].set_title("Histogram  h(r)")
    axes[1].set_xlim(0, 255)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    axes[2].plot(r, cdf, color="tomato", linewidth=2)
    axes[2].set_xlabel("Intensity  r")
    axes[2].set_ylabel("CDF  p(r)")
    axes[2].set_title("Normalised CDF  p(r)")
    axes[2].set_xlim(0, 255)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_histogram_comparison_s3(
    name, img_gray, img_eq, hist_orig, hist_eq, cdf_orig, r_range, lut_curve
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Global Histogram Equalization - {name}", fontsize=13, fontweight="bold"
    )

    # Row 0: images + transformation curve
    axes[0, 0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_eq, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Equalized Image")
    axes[0, 1].axis("off")

    axes[0, 2].plot(r_range, lut_curve, "b-", linewidth=2)
    axes[0, 2].plot([0, 255], [0, 255], "k--", linewidth=1, alpha=0.4)
    axes[0, 2].set_xlabel("Input intensity  r")
    axes[0, 2].set_ylabel("Output intensity  T(r)")
    axes[0, 2].set_title("Equalization Mapping  T(r) = floor(255·CDF(r))")
    axes[0, 2].set_xlim(0, 255)
    axes[0, 2].set_ylim(0, 255)
    axes[0, 2].grid(True, linestyle="--", alpha=0.4)

    # Row 1: histograms
    axes[1, 0].bar(r_range, hist_orig, width=1, color="steelblue", edgecolor="none")
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_xlim(0, 255)

    axes[1, 1].bar(r_range, hist_eq, width=1, color="tomato", edgecolor="none")
    axes[1, 1].set_title("Equalized Histogram")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_xlim(0, 255)

    cdf_eq = compute_norm_cdf(hist_eq)
    axes[1, 2].plot(r_range, cdf_orig, label="Original CDF", linewidth=2)
    axes[1, 2].plot(r_range, cdf_eq, label="Equalized CDF", linewidth=2, linestyle="--")
    axes[1, 2].set_xlabel("Intensity")
    axes[1, 2].set_ylabel("CDF")
    axes[1, 2].set_title("CDF Before vs. After")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].set_xlim(0, 255)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = f"output/section3/ghe_{name}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_histogram_equalization_s3(name, img_gray, gray_ghe, gray_lhe):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"GHE vs. LHE (15x15) - {name}", fontsize=13, fontweight="bold")

    for ax, im, title in zip(
        axes[0],
        [img_gray, gray_ghe, gray_lhe],
        ["Original", "Global HE", "Local HE  (15x15 window)"],
    ):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

        # Histograms
    r_range = get_r_range()
    for ax, im, title, col in zip(
        axes[1],
        [img_gray, gray_ghe, gray_lhe],
        ["Original Histogram", "GHE Histogram", "LHE Histogram"],
        ["steelblue", "tomato", "forestgreen"],
    ):
        ax.bar(r_range, compute_histogram(im), width=1, color=col, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 255)

    plt.tight_layout()
    out_path = f"output/section3/lhe_{name}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_zoomed_patch_comparison_s3(img_gray, gray_ghe, gray_lhe):
    img_gray_height, img_gray_width = img_gray.shape
    pr, pc = img_gray_height // 4, img_gray_width // 4
    ph, pw = img_gray_height // 3, img_gray_width // 3

    def crop(im):
        return im[pr : pr + ph, pc : pc + pw]

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("Zoomed Patch Comparison - moon.tif", fontsize=12)
    for ax, im, t in zip(
        axes2,
        [crop(img_gray), crop(gray_ghe), crop(gray_lhe)],
        ["Original (patch)", "GHE (patch)", "LHE 15x15 (patch)"],
    ):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("output/section3/lhe_moon_patch.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved output/section3/lhe_moon_patch.png")


# ---------------------------------------------------
# Secttion 4
# ---------------------------------------------------


def plot_rgb_histogram_matching_results_s4(
    src_name, img_src, ref_rgb_raw, matched_rgb, channel_names, channel_cols
):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        f"Color Histogram Matching - source: {src_name}  ->  desired.jpg",
        fontsize=13,
        fontweight="bold",
    )

    for ax, im, title in zip(
        axes[0, :3],
        [img_src, ref_rgb_raw, matched_rgb],
        ["Source  A", "Reference  B", "Matched Output"],
    ):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        # Transformation curves per channel
    for ch, (cname, col) in enumerate(zip(channel_names, channel_cols)):
        cdf_s = compute_norm_cdf(compute_histogram(img_src[:, :, ch]))
        cdf_r = compute_norm_cdf(compute_histogram(ref_rgb_raw[:, :, ch]))
        diff = np.abs(cdf_r[np.newaxis, :] - cdf_s[:, np.newaxis])
        lv = np.argmin(diff, axis=1).astype(np.float64)
        axes[0, 3].plot(np.arange(256), lv, color=col, linewidth=1.8, label=cname)
    axes[0, 3].plot([0, 255], [0, 255], "k--", alpha=0.3)
    axes[0, 3].set_title("Transformation Curves per Channel")
    axes[0, 3].set_xlabel("Input  r")
    axes[0, 3].set_ylabel("Output  z")
    axes[0, 3].legend(fontsize=8)
    axes[0, 3].set_xlim(0, 255)
    axes[0, 3].set_ylim(0, 255)
    axes[0, 3].grid(True, linestyle="--", alpha=0.4)

    # Histograms: one column per channel (source, ref, matched)
    for ch, (cname, col) in enumerate(zip(channel_names, channel_cols)):
        ax = axes[1, ch]
        r = np.arange(256)
        ax.bar(
            r,
            compute_histogram(img_src[:, :, ch]),
            width=1,
            color=col,
            edgecolor="none",
            alpha=0.5,
            label="Source",
        )
        ax.bar(
            r,
            compute_histogram(ref_rgb_raw[:, :, ch]),
            width=1,
            color="gray",
            edgecolor="none",
            alpha=0.4,
            label="Reference",
        )
        ax.bar(
            r,
            compute_histogram(matched_rgb[:, :, ch]),
            width=1,
            color=col,
            edgecolor="none",
            alpha=0.9,
            label="Matched",
        )
        ax.set_title(f"{cname} channel")
        ax.set_xlabel("Intensity")
        ax.set_xlim(0, 255)
        ax.legend(fontsize=7)

    axes[1, 3].axis("off")

    plt.tight_layout()
    out = f"output/section4/hist_match_color_{src_name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_gray_histogram_matching_results_s4(
    ref_gray,
    src_name,
    src,
    matched,
    cdf_src,
    cdf_ref,
    lut_vals,
    h_src,
    h_ref,
    h_matched,
    r,
):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        f"Histogram Matching - source: {src_name}  ->  reference: desired (grayscale)",
        fontsize=13,
        fontweight="bold",
    )

    # Row 0: images + transformation curve
    for ax, im, title in zip(
        axes[0, :3],
        [src, ref_gray, matched],
        ["Source  A", "Reference  B", "Matched Output  z = G⁻¹(T(r))"],
    ):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    axes[0, 3].plot(r, lut_vals, "b-", linewidth=2)
    axes[0, 3].plot([0, 255], [0, 255], "k--", linewidth=1, alpha=0.4, label="Identity")
    axes[0, 3].set_xlabel("Input  r")
    axes[0, 3].set_ylabel("Output  z")
    axes[0, 3].set_title("Transformation Curve  z = G⁻¹(T(r))")
    axes[0, 3].set_xlim(0, 255)
    axes[0, 3].set_ylim(0, 255)
    axes[0, 3].grid(True, linestyle="--", alpha=0.4)

    # Row 1: histograms + CDFs
    for ax, h, title, col in zip(
        axes[1, :3],
        [h_src, h_ref, h_matched],
        ["Source Histogram", "Reference Histogram", "Matched Histogram"],
        ["steelblue", "tomato", "forestgreen"],
    ):
        ax.bar(r, h, width=1, color=col, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 255)

        # Overlay CDFs for comparison
    axes[1, 3].plot(r, cdf_src, label="CDF source", linewidth=2, color="steelblue")
    axes[1, 3].plot(r, cdf_ref, label="CDF reference", linewidth=2, color="tomato")
    axes[1, 3].plot(
        r,
        compute_norm_cdf(h_matched),
        label="CDF matched",
        linewidth=2,
        color="forestgreen",
        linestyle="--",
    )
    axes[1, 3].set_xlabel("Intensity")
    axes[1, 3].set_ylabel("CDF")
    axes[1, 3].set_title("CDF Comparison")
    axes[1, 3].legend(fontsize=8)
    axes[1, 3].set_xlim(0, 255)
    axes[1, 3].set_ylim(0, 1)
    axes[1, 3].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = f"output/section4/hist_match_gray_{src_name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_rgb_vs_hsv_equalization_s4(src_name, img, eq_rgb, eq_v):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"HE on RGB Channels vs. HE on V Channel - {src_name}",
        fontsize=13,
        fontweight="bold",
    )

    captions = [
        "Original",
        "HE on R, G, B independently\n(colour balance shifted)",
        "HE on V channel (HSV)\n(hue & saturation preserved)",
    ]
    for ax, im, cap in zip(axes, [img, eq_rgb, eq_v], captions):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(cap, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out = f"output/section4/color_he_{src_name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_per_channel_histograms_s4(
    src_name, img, eq_rgb, eq_v, ch_names, ch_cols, r_range
):
    fig2, axes2 = plt.subplots(3, 3, figsize=(16, 11))
    fig2.suptitle(
        f"Per-Channel Histograms - {src_name}\n"
        "Columns: Original | HE on RGB | HE on V (HSV)",
        fontsize=12,
        fontweight="bold",
    )

    for ch in range(3):
        for col_idx, (image, col_title) in enumerate(
            [
                (img, "Original"),
                (eq_rgb, "HE on RGB"),
                (eq_v, "HE on V (HSV)"),
            ]
        ):
            ax = axes2[ch, col_idx]
            ax.bar(
                r_range,
                compute_histogram(image[:, :, ch]),
                width=1,
                color=ch_cols[ch],
                edgecolor="none",
                alpha=0.85,
            )
            ax.set_title(f"{ch_names[ch]} - {col_title}", fontsize=9)
            ax.set_xlabel("Intensity")
            ax.set_xlim(0, 255)

    plt.tight_layout()
    out2 = f"output/section4/color_channel_hist_{src_name}.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out2}")


def plot_hsv_decomposition_s4(src_name, img, H_chan, S_chan, V_chan, V_eq_norm):
    fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))
    fig3.suptitle(f"HSV Decomposition - {src_name}", fontsize=12)
    for ax, im, cmap, title in zip(
        axes3,
        [img, H_chan, S_chan, V_chan, V_eq_norm],
        [None, "hsv", "gray", "gray", "gray"],
        [
            "Original",
            "Hue  H / 360",
            "Saturation  S",
            "Value  V  (original)",
            "Value  V  (equalized)",
        ],
    ):
        if cmap is None:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out3 = f"output/section4/hsv_decomposition_{src_name}.png"
    plt.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out3}")
