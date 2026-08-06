import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
#  Utility: Mirror padding and manual convolution
# ============================================================
def mirror_pad(img, pad):
    return np.pad(img, pad, mode="reflect")


def convolve2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    img_pad = mirror_pad(img, (pad_h, pad_w))
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(img_pad[i : i + kh, j : j + kw] * kernel)
    return out


# ============================================================
#  Noise generation
# ============================================================
def add_gaussian_noise(img, var):
    sigma = np.sqrt(var)
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)


def add_salt_pepper_noise(img, density):
    noisy = img.copy()
    total_pixels = img.size
    num_noisy = int(density * total_pixels)
    # pepper (0)
    coords = [np.random.randint(0, i, num_noisy // 2) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    # salt (1)
    coords = [np.random.randint(0, i, num_noisy - num_noisy // 2) for i in img.shape]
    noisy[coords[0], coords[1]] = 1
    return noisy


# ============================================================
#  Restoration filters (3x3 kernels)
# ============================================================
def mean_filter(img, kernel_size=3):
    k = kernel_size
    pad = k // 2
    img_pad = mirror_pad(img, pad)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.mean(img_pad[i : i + k, j : j + k])
    return out


def median_filter(img, kernel_size=3):
    k = kernel_size
    pad = k // 2
    img_pad = mirror_pad(img, pad)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.median(img_pad[i : i + k, j : j + k])
    return out


def gaussian_kernel(size, sigma=1.0):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_filter(img, kernel_size=3, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(img, kernel)


# ============================================================
#  Metrics
# ============================================================
def mse(original, restored):
    return np.mean((original - restored) ** 2)


def ssim(img1, img2):
    x = img1 * 255.0
    y = img2 * 255.0
    K1, K2 = 0.01, 0.03
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    window = np.ones((11, 11)) / 121.0
    mu_x = convolve2d(x, window)
    mu_y = convolve2d(y, window)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = convolve2d(x * x, window) - mu_x_sq
    sigma_y_sq = convolve2d(y * y, window) - mu_y_sq
    sigma_xy = convolve2d(x * y, window) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = num / den
    return np.mean(ssim_map)


# ============================================================
#  Main experiment with visual outputs
# ============================================================
if __name__ == "__main__":
    # List your three images (adjust filenames as needed)
    image_files = ["5226523_orig.jpg", "grayscale.jpg", "kP0u2.png"]
    # Choose the first one for the visual comparison figures
    vis_image_index = 0  # change if you want a different one

    gaussian_levels = [0.01, 0.05, 0.1]
    sp_densities = [0.05, 0.10, 0.20]

    filters = {
        "Mean": mean_filter,
        "Median": median_filter,
        "Gaussian": gaussian_filter,
    }

    # Store all results if you want to compute averages later
    all_results = []

    for idx, img_path in enumerate(image_files):
        try:
            img_pil = Image.open(img_path).convert("L")
        except FileNotFoundError:
            print(f"Image {img_path} not found. Skipping.")
            continue

        img = np.array(img_pil, dtype=np.float64) / 255.0

        print(f"\n===== Results for {img_path} =====")
        print(f"{'Noise Type':<15} {'Level':<8} {'Filter':<10} {'MSE':<12} {'SSIM':<8}")
        print("-" * 55)

        # Gaussian noise loop
        for var in gaussian_levels:
            noisy = add_gaussian_noise(img, var)
            for name, filt_func in filters.items():
                restored = filt_func(noisy)
                m = mse(img, restored)
                s = ssim(img, restored)
                all_results.append((img_path, "Gaussian", var, name, m, s))
                print(
                    f"{'Gaussian':<15} {f'{var:.2f}':<8} {name:<10} {m:<12.6f} {s:<8.4f}"
                )

        # Salt & Pepper noise loop
        for density in sp_densities:
            noisy = add_salt_pepper_noise(img, density)
            for name, filt_func in filters.items():
                restored = filt_func(noisy)
                m = mse(img, restored)
                s = ssim(img, restored)
                all_results.append((img_path, "Salt & Pepper", density, name, m, s))
                print(
                    f"{'Salt & Pepper':<15} {f'{density:.2f}':<8} {name:<10} {m:<12.6f} {s:<8.4f}"
                )

    # ================== Generate visual comparison figures ==================
    # Use the first image (index 0) for visualisation
    vis_path = image_files[vis_image_index]
    img_vis = np.array(Image.open(vis_path).convert("L"), dtype=np.float64) / 255.0

    # --- Gaussian noise, variance = 0.05 ---
    noisy_gauss = add_gaussian_noise(img_vis, 0.05)
    mean_gauss = mean_filter(noisy_gauss)
    median_gauss = median_filter(noisy_gauss)
    gauss_gauss = gaussian_filter(noisy_gauss)

    # --- Salt & Pepper noise, density = 0.10 ---
    noisy_sp = add_salt_pepper_noise(img_vis, 0.10)
    mean_sp = mean_filter(noisy_sp)
    median_sp = median_filter(noisy_sp)
    gauss_sp = gaussian_filter(noisy_sp)

    # Create figure 1: Gaussian noise restoration
    fig1, axes1 = plt.subplots(1, 5, figsize=(20, 5))
    titles = [
        "Original",
        "Noisy (Gauss σ²=0.05)",
        "Mean filter",
        "Median filter",
        "Gaussian filter",
    ]
    images = [img_vis, noisy_gauss, mean_gauss, median_gauss, gauss_gauss]
    for ax, title, im in zip(axes1, titles, images):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("restoration_gaussian.png", dpi=150)
    plt.close()

    # Create figure 2: Salt & Pepper noise restoration
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 5))
    titles2 = [
        "Original",
        "Noisy (S&P d=0.10)",
        "Mean filter",
        "Median filter",
        "Gaussian filter",
    ]
    images2 = [img_vis, noisy_sp, mean_sp, median_sp, gauss_sp]
    for ax, title, im in zip(axes2, titles2, images2):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("restoration_saltpepper.png", dpi=150)
    plt.close()

    print(
        "\nVisual comparisons saved as 'restoration_gaussian.png' and 'restoration_saltpepper.png'."
    )
    print("All tasks completed.")
