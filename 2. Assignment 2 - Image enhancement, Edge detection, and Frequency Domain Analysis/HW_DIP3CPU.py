import numpy as np
import matplotlib.pyplot as plt
import time
import os
from math import cos, sin, pi, exp, sqrt

def zero_pad(image, pad_h, pad_w):
    h, w = image.shape
    padded = np.zeros((h + 2*pad_h, w + 2*pad_w), dtype=image.dtype)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    return padded

def mirror_pad(image, pad_h, pad_w):
    h, w = image.shape
    padded = np.zeros((h + 2*pad_h, w + 2*pad_w), dtype=image.dtype)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    for i in range(pad_h):
        padded[pad_h-1-i, pad_w:pad_w+w] = image[i, :]
        padded[pad_h+h+i, pad_w:pad_w+w] = image[h-1-i, :]
    for j in range(pad_w):
        padded[:, pad_w-1-j] = padded[:, pad_w+j]
        padded[:, pad_w+w+j] = padded[:, pad_w+w-1-j]
    return padded

def my_conv2d(image, kernel, padding='zero'):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    if padding == 'zero':
        padded = zero_pad(image, pad_h, pad_w)
    elif padding == 'mirror':
        padded = mirror_pad(image, pad_h, pad_w)
    else:
        raise ValueError("padding must be 'zero' or 'mirror'")
    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    s = 2 * sigma * sigma
    total = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = exp(-(x*x + y*y) / s) / (pi * s)
            total += kernel[i, j]
    return kernel / total

def fft1d(x):
    N = len(x)
    if N == 1:
        return x
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    length = 2
    while length <= N:
        ang = 2 * pi / length
        wlen = complex(cos(ang), sin(ang))
        for i in range(0, N, length):
            w = 1+0j
            half = length // 2
            for j in range(i, i+half):
                u = x[j]
                v = x[j + half] * w
                x[j] = u + v
                x[j + half] = u - v
                w *= wlen
        length <<= 1
    return x

def ifft1d(X):
    N = len(X)
    conjugated = np.conjugate(X)
    y = fft1d(conjugated)
    y = np.conjugate(y) / N
    return y

def fft2d(image):
    """FFT دو بعدی با استفاده از تفکیک‌پذیری"""
    rows, cols = image.shape
    f_row = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        f_row[i, :] = fft1d(image[i, :].astype(complex))
    f_col = np.zeros((rows, cols), dtype=complex)
    for j in range(cols):
        f_col[:, j] = fft1d(f_row[:, j])
    return f_col

def ifft2d(spectrum):
    rows, cols = spectrum.shape
    if_row = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        if_row[i, :] = ifft1d(spectrum[i, :])
    if_col = np.zeros((rows, cols), dtype=complex)
    for j in range(cols):
        if_col[:, j] = ifft1d(if_row[:, j])
    return if_col

def fftshift2d(spectrum):
    rows, cols = spectrum.shape
    r_mid, c_mid = rows//2, cols//2
    shifted = np.zeros_like(spectrum)
    shifted[:r_mid, :c_mid] = spectrum[r_mid:, c_mid:]
    shifted[:r_mid, c_mid:] = spectrum[r_mid:, :c_mid]
    shifted[r_mid:, :c_mid] = spectrum[:r_mid, c_mid:]
    shifted[r_mid:, c_mid:] = spectrum[:r_mid, :c_mid]
    return shifted

def pad_to_power2(image):
    h, w = image.shape
    new_h = 1 << (h-1).bit_length()
    new_w = 1 << (w-1).bit_length()
    padded = np.zeros((new_h, new_w), dtype=image.dtype)
    padded[:h, :w] = image
    return padded, (h, w)

def crop_center(img, target_shape):
    h, w = img.shape
    th, tw = target_shape
    if h < th or w < tw:
        raise ValueError(f"Crop error: image {img.shape} smaller than target {target_shape}")
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return img[start_h:start_h+th, start_w:start_w+tw]

def ideal_lowpass_mask(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.float64)
    center = (rows//2, cols//2)
    for i in range(rows):
        for j in range(cols):
            d = sqrt((i - center[0])**2 + (j - center[1])**2)
            if d <= cutoff:
                mask[i, j] = 1.0
    return mask

def gaussian_lowpass_mask(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.float64)
    center = (rows//2, cols//2)
    for i in range(rows):
        for j in range(cols):
            d2 = (i - center[0])**2 + (j - center[1])**2
            mask[i, j] = exp(-d2 / (2 * cutoff**2))
    return mask

def highpass_from_lowpass(lowpass_mask):
    return 1 - lowpass_mask

def high_frequency_energy_ratio(spectrum, cutoff_ratio=0.5):
    rows, cols = spectrum.shape
    center = (rows//2, cols//2)
    max_radius = sqrt(center[0]**2 + center[1]**2)
    threshold = cutoff_ratio * max_radius
    total_energy = 0.0
    high_energy = 0.0
    for i in range(rows):
        for j in range(cols):
            d = sqrt((i - center[0])**2 + (j - center[1])**2)
            val = abs(spectrum[i, j])**2
            total_energy += val
            if d > threshold:
                high_energy += val
    ratio = high_energy / total_energy if total_energy > 0 else 0
    return high_energy, ratio

def task3_manual(image_path, k_unsharp=0.8, gauss_sigma=1.2, lowpass_cutoff=40):
    output_dir = "outputs_manual"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing: {base_name}")

    img = plt.imread(image_path)
    if img.ndim == 3:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    img = img.astype(np.float64)
    if img.max() <= 1.0:
        img = img * 255
    original = img.copy()
    h, w = original.shape
    print(f"Image size: {h} x {w}")

    print("Spatial sharpening (using zero padding)...")
    gauss_kern = gaussian_kernel(5, gauss_sigma)
    blurred = my_conv2d(original, gauss_kern, padding='zero')
    unsharp = original + k_unsharp * (original - blurred)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

    highpass_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=np.float64)
    highpass_spatial = my_conv2d(original, highpass_kernel, padding='zero')
    highpass_spatial = np.clip(highpass_spatial, 0, 255).astype(np.uint8)

    print("Frequency domain using manual FFT (this may take a while)...")
    img_padded, orig_size = pad_to_power2(original)
    padded_h, padded_w = img_padded.shape
    print(f"Padded size: {padded_h} x {padded_w}")

    start_fft = time.perf_counter()
    F = fft2d(img_padded)
    F_shifted = fftshift2d(F)
    magnitude_log = np.log1p(np.abs(F_shifted))
    print(f"Manual FFT time: {time.perf_counter() - start_fft:.2f} sec")

    ideal_lp = ideal_lowpass_mask((padded_h, padded_w), lowpass_cutoff)
    gaussian_lp = gaussian_lowpass_mask((padded_h, padded_w), lowpass_cutoff)
    ideal_hp = highpass_from_lowpass(ideal_lp)
    gaussian_hp = highpass_from_lowpass(gaussian_lp)

    def reconstruct_crop(filtered_shifted):
        unshifted = fftshift2d(filtered_shifted)
        img_filtered = np.real(ifft2d(unshifted))
        img_filtered = np.clip(img_filtered, 0, 255)
        return crop_center(img_filtered, orig_size).astype(np.uint8)

    print("Reconstructing filtered images...")
    img_ideal_lp = reconstruct_crop(F_shifted * ideal_lp)
    img_gaussian_lp = reconstruct_crop(F_shifted * gaussian_lp)
    img_ideal_hp = reconstruct_crop(F_shifted * ideal_hp)
    img_gaussian_hp = reconstruct_crop(F_shifted * gaussian_hp)

    magnitude_only = np.abs(F_shifted)
    phase_only = np.angle(F_shifted)

    img_mag_only = reconstruct_crop(magnitude_only * np.exp(1j * 0))
    img_phase_only = reconstruct_crop(1.0 * np.exp(1j * phase_only))


    high_energy, energy_ratio = high_frequency_energy_ratio(F_shifted, cutoff_ratio=0.5)
    print(f"High-freq Energy: {high_energy:.2e}, Ratio: {energy_ratio:.6f}")

    print("Running benchmark (Naive DFT vs Manual FFT) on small sizes...")
    sizes = [32, 64, 128]
    times_naive = []
    times_manual_fft = []
    for n in sizes:
        if n <= min(h, w):
            test_img = original[:n, :n].copy()
        else:
            test_img = np.zeros((n, n))
            test_img[:h, :w] = original

        # Naive DFT
        start = time.perf_counter()
        dft_naive = np.zeros((n, n), dtype=complex)
        for u in range(n):
            for v in range(n):
                s = 0+0j
                for x in range(n):
                    for y in range(n):
                        angle = -2 * pi * (u*x/n + v*y/n)
                        s += test_img[x, y] * complex(cos(angle), sin(angle))
                dft_naive[u, v] = s
        times_naive.append(time.perf_counter() - start)

        # FFT
        start = time.perf_counter()
        _ = fft2d(test_img.astype(np.float64))
        times_manual_fft.append(time.perf_counter() - start)

    # نمودار
    plt.figure(figsize=(8,5))
    plt.plot(sizes, times_naive, 'o-', label='Naive DFT O(N^4)')
    plt.plot(sizes, times_manual_fft, 's-', label='Manual FFT O(N^2 log N)')
    plt.xlabel('Image Size N (N×N)')
    plt.ylabel('Time (seconds)')
    plt.title(f'Benchmark: Naive DFT vs Manual FFT - {base_name}')
    plt.legend()
    plt.grid(True)
    bench_filename = os.path.join(output_dir, f"{base_name}_fft_benchmark.png")
    plt.savefig(bench_filename, dpi=200)
    plt.show()
    print(f"Benchmark plot saved: {bench_filename}")

    # خروجی 
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.15, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(original, cmap='gray'); ax1.set_title('1) Original'); ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(unsharp, cmap='gray'); ax2.set_title('2) Spatial Sharpening'); ax2.axis('off')
    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(highpass_spatial, cmap='gray'); ax3.set_title('3) Spatial High-pass'); ax3.axis('off')
    ax4 = fig.add_subplot(gs[0, 3]); ax4.imshow(magnitude_log, cmap='gray'); ax4.set_title('4) Log-Magnitude Spectrum'); ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 0]); ax5.imshow(img_ideal_lp, cmap='gray'); ax5.set_title('5) Ideal Low-pass'); ax5.axis('off')
    ax6 = fig.add_subplot(gs[1, 1]); ax6.imshow(img_gaussian_lp, cmap='gray'); ax6.set_title('6) Gaussian Low-pass'); ax6.axis('off')
    ax7 = fig.add_subplot(gs[1, 2]); ax7.imshow(img_ideal_hp, cmap='gray'); ax7.set_title('7) Ideal High-pass'); ax7.axis('off')
    ax8 = fig.add_subplot(gs[1, 3]); ax8.imshow(img_gaussian_hp, cmap='gray'); ax8.set_title('8) Gaussian High-pass'); ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, 0]); ax9.imshow(img_mag_only, cmap='gray'); ax9.set_title('9) Magnitude only'); ax9.axis('off')
    ax10 = fig.add_subplot(gs[2, 1]); ax10.imshow(img_phase_only, cmap='gray'); ax10.set_title('10) Phase only'); ax10.axis('off')

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.text(0.1, 0.5, f'High-freq Energy: {high_energy:.2e}\nEnergy Ratio: {energy_ratio:.4f}',
              fontsize=10, bbox=dict(facecolor='white', edgecolor='black', alpha=1))
    ax11.axis('off')
    ax11.set_title('11) Frequency Metrics')

    ax12 = fig.add_subplot(gs[2, 3])
    bench_text = "Benchmark (sec)\n" + "\n".join([f"N={sizes[i]}: Naive={times_naive[i]:.2f}, Manual FFT={times_manual_fft[i]:.4f}" for i in range(len(sizes))])
    ax12.text(0.1, 0.5, bench_text, fontsize=9, bbox=dict(facecolor='white', edgecolor='black', alpha=1))
    ax12.axis('off')
    ax12.set_title('12) Time Comparison')

    plt.tight_layout(pad=1.5)
    results_filename = os.path.join(output_dir, f"{base_name}_task3_results.png")
    plt.savefig(results_filename, dpi=250, bbox_inches='tight')
    plt.show()
    print(f"Results image saved: {results_filename}")

    print("\n===== Frequency Evaluation Results =====")
    print(f"High-Frequency Energy: {high_energy:.2e}")
    print(f"Spectral Energy Ratio (High/Total): {energy_ratio:.6f}")
    print("\n===== Transformation Times =====")
    for i, n in enumerate(sizes):
        print(f"N={n:3d} : Naive DFT = {times_naive[i]:.6f} sec, Manual FFT = {times_manual_fft[i]:.6f} sec")
    print(f"\nAll outputs saved in folder: {output_dir}")


if __name__ == "__main__":
    image_list = ["noise.jpg", "noise2.jpg", "pnois1.jpg"]
    for img_path in image_list:
        if os.path.exists(img_path):
            task3_manual(img_path, k_unsharp=0.8, gauss_sigma=1.2, lowpass_cutoff=40)
        else:
            print(f"File not found: {img_path}")