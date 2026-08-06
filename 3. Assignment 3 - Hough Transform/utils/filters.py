import numpy as np


def gaussian_kernel(kernel_size, sigma):
    if kernel_size % 2 == 0:
        kernel_size += 1

    offsets = np.arange(kernel_size, dtype=np.float64) - kernel_size // 2
    gaussian_1d = np.exp(-(offsets**2) / (2.0 * sigma**2))
    # Create 2D kernel via outer product of 1D Gaussian (exploiting separability)
    kernel_2d = np.outer(gaussian_1d, gaussian_1d)

    return kernel_2d / kernel_2d.sum()


def convolve2d(image, kernel):
    kernal_h, kernal_w = kernel.shape
    pad_h, pad_w = kernal_h // 2, kernal_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    H, W = image.shape[:2]
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(kernal_h):
        for j in range(kernal_w):
            out += kernel[i, j] * padded[i : i + H, j : j + W]
    return out


def gaussian_blur(image, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)
