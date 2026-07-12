import numpy as np


def make_binary_matrix(shape=None, like=None, p=0.7, dtype=np.uint8,
                       min_size=2, max_size=10, seed=None):
    rng = np.random.default_rng(seed)

    if shape is not None:
        out_shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
    elif like is not None:
        out_shape = like.shape
    else:
        out_shape = tuple(rng.integers(min_size, max_size + 1, size=2))

    return (rng.random(out_shape) < p).astype(dtype)
