import numpy as np


def compute_histogram(gray):
    gray = np.asarray(gray)
    flat = gray.ravel()

    if flat.size == 0:
        hist = np.zeros(256, dtype=np.float64)
        return hist

    hist_size = max(256, int(flat.max()) + 1)
    hist = np.zeros(hist_size, dtype=np.float64)

    for val in flat:
        hist[val] += 1.0

    return hist


def auto_threshold(gray, return_curve=False):
    histogram = compute_histogram(gray)
    prob = histogram / histogram.sum()
    intensities = np.arange(256)

    weight_bg = np.cumsum(prob)
    weight_fg = 1.0 - weight_bg
    cumulative_mean = np.cumsum(prob * intensities)
    total_mean = cumulative_mean[-1]

    eps = 1e-12
    mean_bg = cumulative_mean / np.maximum(weight_bg, eps)
    mean_fg = (total_mean - cumulative_mean) / np.maximum(weight_fg, eps)
    between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_class_variance[(weight_bg < eps) | (weight_fg < eps)] = 0.0

    best_threshold = int(np.argmax(between_class_variance))
    if return_curve:
        return best_threshold, between_class_variance
    return best_threshold


def binarize_auto_threshold(gray, foreground="bright"):
    threshold = auto_threshold(gray)
    gray = np.asarray(gray)
    if foreground == "bright":
        mask = gray > threshold
    elif foreground == "dark":
        mask = gray <= threshold
    else:
        raise ValueError("foreground must be 'bright' or 'dark'")
    return mask, threshold
