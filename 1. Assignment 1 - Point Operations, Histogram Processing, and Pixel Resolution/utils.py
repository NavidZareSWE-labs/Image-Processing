import numpy as np


def get_r_range():
    return np.arange(256, dtype=np.float64)


def compute_mse(img_a, img_b):
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def compute_psnr(img_a, img_b, max_val=255.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR tells you how closely a degraded/processed image matches the original
    image.

    Interpretation:
        * Higher PSNR = Better Quality: A higher number means there is very
        little noise compared to the signal. The images look very similar.
        * Lower PSNR = Worse Quality: A lower number means the error/noise
        is large. The image is likely blurry, pixelated, or distorted.
    """
    mse = compute_mse(img_a, img_b)
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(max_val**2 / mse)


def to_gray(img):
    if img.ndim == 2:
        return img.astype(np.uint8)
    r = img[:, :, 2].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    b = img[:, :, 0].astype(np.float64)

    #    Apply the standard Luminance Formula to calculate grayscale.
    #    Why aren't the weights equal (e.g., 33% each)?
    #    Because human eyes do not perceive all colors with the same brightness.
    #    Our eyes are highly sensitive to Green (so it gets the biggest weight, ~59%),
    #    moderately sensitive to Red (~30%), and poorly sensitive to Blue (~11%).
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def compute_histogram(gray):
    hist = np.zeros(256, dtype=np.int64)

    # .ravel() returns a flattened (1‑D) view of the array.
    flat = gray.ravel()
    for val in flat:
        hist[val] += 1

    return hist


def compute_norm_cdf(h):
    cdf = np.zeros(len(h), dtype=np.float64)

    running_sum = 0
    for i in range(len(h)):
        running_sum += h[i]
        cdf[i] = running_sum

    if cdf[-1] > 0:
        total = cdf[-1]
        for i in range(len(cdf)):
            cdf[i] /= total
    norm_cdf = cdf
    return norm_cdf
