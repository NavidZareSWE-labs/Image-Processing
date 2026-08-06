from pathlib import Path
import numpy as np
import cv2
import math

# ===================== Directory Layout =====================


BASE_DIR = Path(__file__).resolve().parent

# Image Input Paths
IMG_T1 = BASE_DIR / "Images" / "Task_1_Edge_Detection"
IMG_T2 = BASE_DIR / "Images" / "Task_2_Image_Restoration"
IMG_T3 = BASE_DIR / "Images" / "Task_3_Image_Enhancement"

# Output Paths
OUT_S1 = BASE_DIR / "output" / "section1"
OUT_S2 = BASE_DIR / "output" / "section2"
OUT_S3 = BASE_DIR / "output" / "section3"


def ensure_dirs():
    # Assuming OUT_S1, OUT_S2, and OUT_S3 are already Path objects
    for d in [OUT_S1, OUT_S2, OUT_S3]:
        d.mkdir(parents=True, exist_ok=True)


def zero_pad(image, pad_h, pad_w):
    return np.pad(
        image.astype(np.float64),
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )


def mirror_pad(image, pad_h, pad_w):
    return np.pad(
        image.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="edge"
    )


def convolve2d(image, kernel, padding="zero"):
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Image and kernel must be 2D arrays.")

    H, W = image.shape
    kernal_h, kernal_w = kernel.shape

    pad_h = kernal_h // 2
    pad_w = kernal_w // 2

    if padding == "zero":
        padded = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
        )
    elif padding == "mirror":
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    else:
        raise ValueError("Padding mode must be 'zero' or 'mirror'.")

    output = np.zeros((H, W), dtype=np.float64)

    flipped_kernel = kernel[::-1, ::-1].astype(np.float64)

    # Slide the kernel across the image
    for y in range(H):
        for x in range(W):
            window = padded[y : y + kernal_h, x : x + kernal_w]
            output[y, x] = np.sum(window * flipped_kernel)
    return output


def load_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def load_grayscale(path):
    img = load_bgr(path)
    if img.ndim == 2:
        return img
    b = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    r = img[:, :, 2].astype(np.float64)

    #    Apply the standard Luminance Formula to calculate grayscale.
    #    Why aren't the weights equal (e.g., 33% each)?
    #    Because human eyes do not perceive all colors with the same brightness.
    #    Our eyes are highly sensitive to Green (so it gets the biggest weight, ~59%),
    #    moderately sensitive to Red (~30%), and poorly sensitive to Blue (~11%).
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def clip_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def normalise_to_uint8(arr):
    a_min, a_max = arr.min(), arr.max()
    if a_max == a_min:
        return np.zeros_like(arr, dtype=np.uint8)
    stretched = (arr - a_min) / (a_max - a_min) * 255.0
    return stretched.astype(np.uint8)


def gaussian_kernel(kernel_size, sigma):
    if kernel_size % 2 == 0:
        kernel_size += 1

    offsets = np.arange(kernel_size, dtype=np.float64) - kernel_size // 2
    gaussian_1d = np.exp(-(offsets**2) / (2.0 * sigma**2))
    # Create 2D kernel via outer product of 1D Gaussian (exploiting separability)
    kernel_2d = np.outer(gaussian_1d, gaussian_1d)

    return kernel_2d / kernel_2d.sum()


def gaussian_kernel_size(sigma):
    # Calculates kernel size to cover 3 standard deviations (99.7% of the Gaussian curve).
    # Ensures the resulting size is an odd integer (to have a true center pixel)
    # and enforces a minimum size of 3x3.
    size = int(2 * np.ceil(3 * sigma) + 1)
    if size % 2 == 0:
        size += 1
    return max(size, 3)


def crop_center_square(image, N):
    H, W = image.shape
    # pad if necessary
    pad_h = max(0, N - H)
    pad_w = max(0, N - W)
    if pad_h > 0 or pad_w > 0:
        image = zero_pad(image, pad_h // 2 + pad_h % 2, pad_w // 2 + pad_w % 2)
        H, W = image.shape

    crop_h = (H - N) // 2
    crop_w = (W - N) // 2
    return image[crop_h : crop_h + N, crop_w : crop_w + N].astype(np.float64)


def next_power_of_2(n):
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


def pad_to_power_of_2(image):
    H, W = image.shape
    H2 = next_power_of_2(H)
    W2 = next_power_of_2(W)
    padded = np.zeros((H2, W2), dtype=np.float64)
    padded[:H, :W] = image
    return padded


def compute_mse(img_a, img_b):
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def _box_filter_fast(image, radius):
    height, width = image.shape
    window_size = 2 * radius + 1
    window_area = window_size * window_size

    padded_image = np.pad(image, radius, mode="edge").astype(np.float64)

    # 2-D cumulative sum (creates an Integral Image)
    integral_image = padded_image.cumsum(axis=0).cumsum(axis=1)

    # Integral image extraction via four corners
    # The valid region for calculating the sum starts at index `radius`
    bottom_right = integral_image[
        2 * radius : 2 * radius + height, 2 * radius : 2 * radius + width
    ]
    top_left = integral_image[0:height, 0:width]
    top_right = integral_image[0:height, 2 * radius : 2 * radius + width]
    bottom_left = integral_image[2 * radius : 2 * radius + height, 0:width]

    window_sum = bottom_right - top_right - bottom_left + top_left

    return window_sum / window_area


def compute_ssim(ref_image, test_image, window_size=11, dynamic_range=255):

    # Standard SSIM constants
    k1, k2 = 0.01, 0.03
    c1 = (k1 * dynamic_range) ** 2  # prevent division by zero
    c2 = (k2 * dynamic_range) ** 2  # prevent division by zero

    radius = window_size // 2

    # Cast images to float for precision
    ref_float = ref_image.astype(np.float64)
    test_float = test_image.astype(np.float64)

    # Local means (mu)
    mean_ref = _box_filter_fast(ref_float, radius)
    mean_test = _box_filter_fast(test_float, radius)

    mean_ref_sq = mean_ref * mean_ref
    mean_test_sq = mean_test * mean_test
    mean_ref_test = mean_ref * mean_test

    # Local variances (sigma squared) and covariance
    # Cov(X,Y)=E[XY]−E[X]E[Y]
    var_ref = _box_filter_fast(ref_float * ref_float, radius) - mean_ref_sq
    var_test = _box_filter_fast(test_float * test_float, radius) - mean_test_sq
    cov_ref_test = _box_filter_fast(ref_float * test_float, radius) - mean_ref_test

    # SSIM formula components
    numerator = (2.0 * mean_ref_test + c1) * (2.0 * cov_ref_test + c2)
    denominator = (mean_ref_sq + mean_test_sq + c1) * (var_ref + var_test + c2)

    # Calculate SSIM map and return mean
    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


def _prepare_display(img):
    if img.dtype != np.uint8:
        min, max = img.min(), img.max()
        if max > min:
            img = ((img - min) / (max - min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    return img
