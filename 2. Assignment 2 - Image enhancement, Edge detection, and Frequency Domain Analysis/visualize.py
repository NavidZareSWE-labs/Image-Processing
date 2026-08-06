from utils import _prepare_display
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")


def save_image_row(
    images: list,
    titles: list,
    output_path: str,
    suptitle: str = "",
    cmap: str = "gray",
    col_width: float = 3.8,
    row_height: float = 3.8,
) -> None:

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(col_width * n, row_height))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        disp = _prepare_display(np.array(img))
        ax.imshow(disp, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Saved -> {output_path}")


def save_image_grid(
    images: list,
    titles: list,
    output_path: str,
    ncols: int = 4,
    suptitle: str = "",
    cmap: str = "gray",
    col_width: float = 3.5,
    row_height: float = 3.5,
) -> None:

    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(col_width * ncols, row_height * nrows)
    )
    axes = np.array(axes).flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        disp = _prepare_display(np.array(img))
        axes[i].imshow(disp, cmap=cmap, vmin=0, vmax=255)
        axes[i].set_title(title, fontsize=8, pad=4)
        axes[i].axis("off")
    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Saved -> {output_path}")


def save_spectrum_figure(
    original: np.ndarray,
    magnitude_spectrum: np.ndarray,
    phase_spectrum: np.ndarray,
    output_path: str,
    img_name: str = "",
) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(_prepare_display(original), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Grayscale", fontsize=9)
    axes[0].axis("off")

    log_mag = np.log1p(magnitude_spectrum)
    im1 = axes[1].imshow(log_mag, cmap="viridis")
    axes[1].set_title("Log-Magnitude Spectrum (centred)", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(phase_spectrum, cmap="hsv")
    axes[2].set_title("Phase Spectrum (centred)", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    title = (
        f"Frequency Domain Analysis - {img_name}"
        if img_name
        else "Frequency Domain Analysis"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Saved -> {output_path}")


def save_benchmark_plot(
    sizes: list, naive_times: list, fft_times: list, output_path: str
):

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        sizes,
        naive_times,
        "o-",
        color="crimson",
        lw=2,
        ms=8,
        label="Naïve 2-D DFT  O(N⁴)",
    )
    ax.plot(
        sizes,
        fft_times,
        "s-",
        color="steelblue",
        lw=2,
        ms=8,
        label="2-D FFT         O(N² log N)",
    )
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Image size N (square NxN)", fontsize=11)
    ax.set_ylabel("Execution time (seconds, log scale)", fontsize=11)
    ax.set_title(
        "Naïve DFT vs FFT - Execution Time Benchmark", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Annotate speedup factors
    for N, t_n, t_f in zip(sizes, naive_times, fft_times):
        speedup = t_n / max(t_f, 1e-9)
        ax.annotate(
            f"{speedup:.0f}x",
            xy=(N, t_f),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="steelblue",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Saved -> {output_path}")


def save_noise_restoration_grid(
    clean: np.ndarray,
    noisy_images: list,
    restored_images: list,
    noisy_labels: list,
    restored_labels: list,
    output_path: str,
    suptitle: str = "",
) -> None:

    n_noisy = len(noisy_images)
    n_rest = len(restored_images)
    ncols = max(n_noisy, n_rest)
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols + 1, figsize=(3.5 * (ncols + 1), 3.5 * nrows))
    axes = np.atleast_2d(axes)

    # Row 0: clean + noisy
    axes[0, 0].imshow(_prepare_display(clean), cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Clean Original", fontsize=8)
    axes[0, 0].axis("off")
    for i, (ni, lbl) in enumerate(zip(noisy_images, noisy_labels)):
        axes[0, i + 1].imshow(_prepare_display(ni), cmap="gray", vmin=0, vmax=255)
        axes[0, i + 1].set_title(lbl, fontsize=8)
        axes[0, i + 1].axis("off")

    # Row 1: restored
    axes[1, 0].axis("off")
    for i, (ri, lbl) in enumerate(zip(restored_images, restored_labels)):
        axes[1, i + 1].imshow(_prepare_display(ri), cmap="gray", vmin=0, vmax=255)
        axes[1, i + 1].set_title(lbl, fontsize=8)
        axes[1, i + 1].axis("off")

    for j in range(n_rest + 1, ncols + 1):
        axes[1, j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Saved -> {output_path}")


def save_robustness_plot(
    noise_levels: list,
    precision_list: list,
    recall_list: list,
    method_name: str,
    output_path: str,
):

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        noise_levels,
        precision_list,
        "o-",
        color="royalblue",
        lw=2,
        ms=7,
        label="Precision",
    )
    ax.plot(
        noise_levels, recall_list, "s--", color="tomato", lw=2, ms=7, label="Recall"
    )
    ax.set_xlabel("Gaussian Noise sigma (added to original)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Robustness to Noise - {method_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Saved -> {output_path}")
