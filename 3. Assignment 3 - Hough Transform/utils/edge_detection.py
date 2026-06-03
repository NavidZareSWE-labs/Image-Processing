import numpy as np
from .filters import convolve2d, gaussian_blur


SOBEL_KERNEL_X = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

SOBEL_KERNEL_Y = np.array([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]], dtype=np.float64)


def sobel_edge_detection(image):
    gx = convolve2d(image.astype(np.float64), SOBEL_KERNEL_X)
    gy = convolve2d(image.astype(np.float64), SOBEL_KERNEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction, gx, gy


def _non_maximum_suppression(magnitude, direction):
    # Gradient angle in degrees, wrapped to [0, 180).
    angle_deg = np.rad2deg(direction) % 180.0

    # Quantise into 4 bins.
    direction_bin = np.zeros_like(angle_deg, dtype=np.uint8)
    direction_bin[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 1   # ~45 deg
    direction_bin[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 2   # vertical
    direction_bin[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 3   # ~135 deg

    padded = np.pad(magnitude, 1, mode='constant', constant_values=0)

    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    up_left = padded[:-2, :-2]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    down_right = padded[2:, 2:]

    n1 = np.where(direction_bin == 0, left,
                  np.where(direction_bin == 1, up_left,
                           np.where(direction_bin == 2, up, up_right)))
    n2 = np.where(direction_bin == 0, right,
                  np.where(direction_bin == 1, down_right,
                           np.where(direction_bin == 2, down, down_left)))

    output = np.where((magnitude >= n1) & (magnitude >= n2), magnitude, 0.0)

    # Remove border pixels
    output[0, :] = 0
    output[-1, :] = 0
    output[:, 0] = 0
    output[:, -1] = 0

    return output


def _double_threshold(nms, low_ratio=0.05, high_ratio=0.15):
    max_val = nms.max() if nms.max() > 0 else 1.0
    high_thresh = max_val * high_ratio
    low_thresh = max_val * low_ratio

    strong = nms >= high_thresh
    weak = (nms >= low_thresh) & (~strong)
    return strong, weak


def _hysteresis(strong, weak):
    STRONG = np.uint8(255)
    WEAK = np.uint8(128)

    output = np.zeros(strong.shape, dtype=np.uint8)
    output[strong] = STRONG
    output[weak] = WEAK

    while True:
        strong_mask = (output == STRONG)
        padded_mask = np.pad(strong_mask, 1, mode='constant',
                             constant_values=False)

        has_strong_neighbour = (
            padded_mask[:-2, :-2] | padded_mask[:-2, 1:-1] | padded_mask[:-2, 2:] |
            padded_mask[1:-1, :-2] | padded_mask[1:-1, 2:] |
            padded_mask[2:, :-2] | padded_mask[2:,
                                               1:-1] | padded_mask[2:,  2:]
        )

        promote = (output == WEAK) & has_strong_neighbour
        if not promote.any():
            break
        output[promote] = STRONG

    output[output == WEAK] = 0
    return output.astype(bool)


def canny_edge_detection(image, blur_size=5, blur_sigma=1.4,
                         low_thresh=0.05, high_thresh=0.15):
    # 1. Gaussian smooth
    smoothed = gaussian_blur(image, blur_size, blur_sigma)

    # 2. Sobel gradients
    magnitude, direction, _, _ = sobel_edge_detection(smoothed)

    # 3. Non-maximum suppression
    nms = _non_maximum_suppression(magnitude, direction)

    # 4. Double threshold
    strong, weak = _double_threshold(nms, low_thresh, high_thresh)

    # 5. Hysteresis
    edges = _hysteresis(strong, weak)

    return edges, magnitude, direction
