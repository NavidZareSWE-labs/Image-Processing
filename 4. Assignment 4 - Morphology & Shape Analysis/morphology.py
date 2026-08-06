import numpy as np

_PAIR_TEMPLATES = {
    "horizontal": (
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
        (1, 1),
        (0, 1),
    ),
    "vertical": (
        np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
        (1, 1),
        (1, 0),
    ),
    "diag_main": (
        np.array([[0, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0]]),
        (1, 1),
        (1, 1),
    ),
    "diag_anti": (
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]]),
        (1, 2),
        (1, -1),
    ),
}

# ------------------- Structuring Elements -------------------


def make_structuringElements(shape="square", size=3):
    if shape == "rect":
        rows, cols = size
        return np.ones((rows, cols), dtype=bool)

    if size % 2 == 0:
        raise ValueError(
            "size must be odd so the Structuring Elements has a unique centre"
        )

    if shape == "square":
        return np.ones((size, size), dtype=bool)

    if shape == "cross":
        se = np.zeros((size, size), dtype=bool)
        center = size // 2
        se[center, :] = True
        se[:, center] = True
        return se

    if shape == "disk":
        radius = size // 2
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        return (xx * xx + yy * yy) <= (radius * radius)

    raise ValueError(f"undefined Structuring Elements shape '{shape}'")


# ------------------- Erosion & Dilation -------------------
def erode(image, se, origin=None, border_value=False):
    image = np.asarray(image, dtype=bool)
    se = np.asarray(se, dtype=bool)
    se_h, se_w = se.shape
    origin_y, origin_x = (se_h // 2, se_w // 2) if origin is None else origin

    padded = np.pad(
        image,
        ((origin_y, se_h - 1 - origin_y), (origin_x, se_w - 1 - origin_x)),
        mode="constant",
        constant_values=border_value,
    )
    img_h, img_w = image.shape
    result = np.ones((img_h, img_w), dtype=bool)
    for dy in range(se_h):
        for dx in range(se_w):
            if se[dy, dx]:
                result &= padded[dy : dy + img_h, dx : dx + img_w]
    return result


def dilate(image, se, origin=None):
    image = np.asarray(image, dtype=bool)
    se = np.asarray(se, dtype=bool)
    se_h, se_w = se.shape
    origin_y, origin_x = (se_h // 2, se_w // 2) if origin is None else origin

    padded = np.pad(
        image,
        ((se_h - 1 - origin_y, origin_y), (se_w - 1 - origin_x, origin_x)),
        mode="constant",
        constant_values=False,
    )
    img_h, img_w = image.shape
    result = np.zeros((img_h, img_w), dtype=bool)
    for dy in range(se_h):
        for dx in range(se_w):
            if se[dy, dx]:
                result |= padded[
                    se_h - 1 - dy : se_h - 1 - dy + img_h,
                    se_w - 1 - dx : se_w - 1 - dx + img_w,
                ]
    return result


# ------------------- Opening -------------------
def opening(image, se):
    return dilate(erode(image, se), se)


# ------------------- Hit-or-Miss Transform -------------------
def hit_or_miss(image, template, origin=None):
    image = np.asarray(image, dtype=bool)
    template = np.asarray(template, dtype=int)
    se_h, se_w = template.shape
    if origin is None:
        origin = (se_h // 2, se_w // 2)
    foreground = template == 1
    background = template == 0
    foreground_fit = erode(image, foreground, origin)
    background_fit = erode(~image, background, origin, border_value=True)
    return foreground_fit & background_fit


# ------------------- Isolated Zero-Pairs (3.1) -------------------


def find_isolated_zero_pairs(matrix):
    values = np.asarray(matrix, dtype=int)
    foreground = values == 0
    pairs = []
    for template, origin, partner_offset in _PAIR_TEMPLATES.values():
        hits = hit_or_miss(foreground, template, origin)
        rows, cols = np.where(hits)
        offset_row, offset_col = partner_offset
        for row, col in zip(rows, cols):
            first = (int(row), int(col))
            second = (int(row + offset_row), int(col + offset_col))
            pairs.append(sorted([list(first), list(second)]))
    pairs.sort()
    return pairs


# ------------------- Thinning / Skeletonisation (3.2) -------------------
def golay_thinning_templates():
    return [
        np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]]),
        np.array([[-1, 0, 0], [1, 1, 0], [-1, 1, -1]]),
        np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]]),
        np.array([[-1, 1, -1], [1, 1, 0], [-1, 0, 0]]),
        np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]]),
        np.array([[-1, 1, -1], [0, 1, 1], [0, 0, -1]]),
        np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]]),
        np.array([[0, 0, -1], [0, 1, 1], [-1, 1, -1]]),
    ]


def thin_once(image, templates):
    result = np.asarray(image, dtype=bool)
    for template in templates:
        result = result & ~hit_or_miss(result, template)
    return result


def thinning(image, templates=None, max_iter=200):
    if templates is None:
        templates = golay_thinning_templates()
    current = np.asarray(image, dtype=bool)
    for sweep in range(1, max_iter + 1):
        swept = thin_once(current, templates)
        if np.array_equal(swept, current):
            return current, sweep
        current = swept
    return current, max_iter


# ------------------- Pruning (3.3) -------------------
def endpoint_templates():
    return [
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ]


def find_endpoints(skeleton, templates=None):
    if templates is None:
        templates = endpoint_templates()
    endpoints = np.zeros_like(skeleton, dtype=bool)
    for template in templates:
        endpoints |= hit_or_miss(skeleton, template)
    return endpoints


def prune(skeleton, n=8):
    endpoint_masks = endpoint_templates()
    skeleton = np.asarray(skeleton, dtype=bool)

    trimmed = skeleton.copy()
    for _ in range(n):
        trimmed = trimmed & ~find_endpoints(trimmed, endpoint_masks)

    surviving_endpoints = find_endpoints(trimmed, endpoint_masks)

    se = make_structuringElements("square", 3)
    regrown = surviving_endpoints.copy()
    for _ in range(n):
        regrown = dilate(regrown, se) & skeleton

    return trimmed | regrown
