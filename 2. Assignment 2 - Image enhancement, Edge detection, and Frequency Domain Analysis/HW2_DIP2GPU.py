import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from math import cos, sin, pi, exp, sqrt


# GPU کانولوشن
def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    s = 2 * sigma * sigma
    total = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = exp(-(x * x + y * y) / s) / (pi * s)
            total += kernel[i, j]
    return (kernel / total).astype(np.float32)


def conv2d_gpu(image_gpu, kernel_gpu):
    kh, kw = kernel_gpu.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = image_gpu.shape
    padded = cp.pad(image_gpu, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    from cupy.lib.stride_tricks import as_strided

    shape = (h, w, kh, kw)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    windows = as_strided(padded, shape=shape, strides=strides)
    return cp.tensordot(windows, kernel_gpu, axes=((2, 3), (0, 1)))


# عملیات اصلی GPU
def task3_gpu(image_path, k_unsharp=0.8, gauss_sigma=1.2, lowpass_cutoff=40):

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing: {base_name}")

    img = plt.imread(image_path)
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255
    original = img.copy()
    h, w = original.shape
    print(f"Image size: {h} x {w}")

    img_gpu = cp.asarray(original)

    # sharpening
    print("Spatial sharpening on GPU...")
    kern_gauss = gaussian_kernel(5, gauss_sigma)
    kern_gauss_gpu = cp.asarray(kern_gauss)
    blurred_gpu = conv2d_gpu(img_gpu, kern_gauss_gpu)
    unsharp_gpu = img_gpu + k_unsharp * (img_gpu - blurred_gpu)
    unsharp_gpu = cp.clip(unsharp_gpu, 0, 255)
    unsharp = cp.asnumpy(unsharp_gpu).astype(np.uint8)

    kern_hp = cp.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=cp.float32)
    highpass_gpu = conv2d_gpu(img_gpu, kern_hp)
    highpass_gpu = cp.clip(highpass_gpu, 0, 255)
    highpass_spatial = cp.asnumpy(highpass_gpu).astype(np.uint8)

    # Frequency domain
    print("Frequency domain on GPU (cuFFT)...")
    new_h = 1 << (h - 1).bit_length()
    new_w = 1 << (w - 1).bit_length()
    img_pad = cp.zeros((new_h, new_w), dtype=cp.float32)
    img_pad[:h, :w] = img_gpu
    F = cp.fft.fft2(img_pad)
    F_shifted = cp.fft.fftshift(F)
    magnitude_log = cp.asnumpy(cp.log1p(cp.abs(F_shifted)))

    # gaussian_mask
    rows, cols = new_h, new_w
    crow, ccol = rows // 2, cols // 2
    ideal = cp.zeros((rows, cols), dtype=cp.float32)
    gaussian_mask = cp.zeros((rows, cols), dtype=cp.float32)
    for i in range(rows):
        for j in range(cols):
            d = sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if d <= lowpass_cutoff:
                ideal[i, j] = 1.0
            gaussian_mask[i, j] = exp(-(d**2) / (2 * lowpass_cutoff**2))
    ideal_hp = 1 - ideal
    gaussian_hp_mask = 1 - gaussian_mask

    def ifft_crop(F_filt_shifted):
        F_filt = cp.fft.ifftshift(F_filt_shifted)
        img_filt = cp.real(cp.fft.ifft2(F_filt))
        img_filt = cp.clip(img_filt, 0, 255)
        return cp.asnumpy(img_filt[:h, :w]).astype(np.uint8)

    img_ideal_lp = ifft_crop(F_shifted * ideal)
    img_gaussian_lp = ifft_crop(F_shifted * gaussian_mask)
    img_ideal_hp = ifft_crop(F_shifted * ideal_hp)
    img_gaussian_hp = ifft_crop(F_shifted * gaussian_hp_mask)

    # phase
    mag = cp.abs(F_shifted)
    phase = cp.angle(F_shifted)
    img_mag = ifft_crop(mag * cp.exp(1j * 0))
    img_phase = ifft_crop(1.0 * cp.exp(1j * phase))

    # High-freq Energy
    F_cpu = cp.asnumpy(F_shifted)
    total_energy = np.sum(np.abs(F_cpu) ** 2)
    high_energy = 0
    max_radius = sqrt(crow**2 + ccol**2)
    threshold = 0.5 * max_radius
    for i in range(rows):
        for j in range(cols):
            d = sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if d > threshold:
                high_energy += np.abs(F_cpu[i, j]) ** 2
    energy_ratio = high_energy / total_energy if total_energy > 0 else 0
    print(f"High-freq Energy: {high_energy:.2e}, Ratio: {energy_ratio:.6f}")

    # CPU
    print("Running benchmark (Naive DFT vs FFT) on CPU...")
    sizes = [32, 64, 128]
    times_naive = []
    times_fft_gpu = []  # FFT روی GPU
    for n in sizes:
        if n <= min(h, w):
            test_img = original[:n, :n].copy()
        else:
            test_img = np.zeros((n, n))
            test_img[:h, :w] = original

        # Naive DFT on CPU
        start = time.perf_counter()
        dft_naive = np.zeros((n, n), dtype=complex)
        for u in range(n):
            for v in range(n):
                s = 0 + 0j
                for x in range(n):
                    for y in range(n):
                        angle = -2 * pi * (u * x / n + v * y / n)
                        s += test_img[x, y] * complex(cos(angle), sin(angle))
                dft_naive[u, v] = s
        times_naive.append(time.perf_counter() - start)

        # FFT on GPU
        start = time.perf_counter()
        _ = cp.fft.fft2(cp.asarray(test_img))
        times_fft_gpu.append(time.perf_counter() - start)

    # رسم نمودار
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times_naive, "o-", label="Naive DFT O(N^4) (CPU)")
    plt.plot(sizes, times_fft_gpu, "s-", label="FFT O(N^2 log N) (GPU)")
    plt.xlabel("Image Size N (N×N)")
    plt.ylabel("Time (seconds)")
    plt.title(f"Benchmark: Naive DFT vs FFT - {base_name}")
    plt.legend()
    plt.grid(True)
    bench_filename = os.path.join(output_dir, f"{base_name}_fft_benchmark.png")
    plt.savefig(bench_filename, dpi=200)
    plt.show()
    print(f"Benchmark plot saved: {bench_filename}")

    # نمایش ۱۲ خروجی
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.15, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap="gray")
    ax1.set_title("1) Original")
    ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(unsharp, cmap="gray")
    ax2.set_title("2) Spatial Sharpening")
    ax2.axis("off")
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(highpass_spatial, cmap="gray")
    ax3.set_title("3) Spatial High-pass")
    ax3.axis("off")
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(magnitude_log, cmap="gray")
    ax4.set_title("4) Log-Magnitude Spectrum")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(img_ideal_lp, cmap="gray")
    ax5.set_title("5) Ideal Low-pass")
    ax5.axis("off")
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(img_gaussian_lp, cmap="gray")
    ax6.set_title("6) Gaussian Low-pass")
    ax6.axis("off")
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(img_ideal_hp, cmap="gray")
    ax7.set_title("7) Ideal High-pass")
    ax7.axis("off")
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(img_gaussian_hp, cmap="gray")
    ax8.set_title("8) Gaussian High-pass")
    ax8.axis("off")

    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(img_mag, cmap="gray")
    ax9.set_title("9) Magnitude only")
    ax9.axis("off")
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(img_phase, cmap="gray")
    ax10.set_title("10) Phase only")
    ax10.axis("off")
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.text(
        0.1,
        0.5,
        f"High-freq Energy: {high_energy:.2e}\nEnergy Ratio: {energy_ratio:.4f}",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax11.axis("off")
    ax11.set_title("11) Frequency Metrics")
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.text(
        0.1,
        0.5,
        f"GPU Accelerated (CuPy)\nImage: {base_name}",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax12.axis("off")
    ax12.set_title("12) Info")

    results_filename = os.path.join(output_dir, f"{base_name}_task3_results.png")
    plt.savefig(results_filename, dpi=250, bbox_inches="tight")
    plt.show()
    print(f"Results image saved: {results_filename}")

    print("\n===== Frequency Evaluation Results =====")
    print(f"High-Frequency Energy: {high_energy:.2e}")
    print(f"Spectral Energy Ratio (High/Total): {energy_ratio:.6f}")
    print("\n===== Transformation Times =====")
    for i, n in enumerate(sizes):
        print(
            f"N={n:3d} : Naive DFT = {times_naive[i]:.6f} sec, FFT (GPU) = {times_fft_gpu[i]:.6f} sec"
        )
    print(f"\nAll outputs saved in folder: {output_dir}")


if __name__ == "__main__":
    image_list = ["noise.jpg", "noise2.jpg", "pnois1.jpg"]
    for img_path in image_list:
        if os.path.exists(img_path):
            task3_gpu(img_path, k_unsharp=0.8, gauss_sigma=1.2, lowpass_cutoff=40)
        else:
            print(f"File not found: {img_path}")
