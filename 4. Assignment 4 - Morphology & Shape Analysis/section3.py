import visualize as vis
from binarization import binarize_auto_threshold
import morphology as mp
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from utils import *

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


TOY_MATRIX = np.array(
    [
        [1, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
    ]
)

DECIMATION = 3
PRUNE_ITERS = 8


def maybe_replace(matrix, prob=0.3, seed=None):
    rng = np.random.default_rng(seed)
    if rng.random() < prob:
        return make_binary_matrix(like=matrix, seed=seed)
    return matrix


def run(input_dir=BASE_DIR / "input_images", out_dir=BASE_DIR / "output" / "section3"):
    print("\n" + "=" * 78)
    print("PART 3 -- SKELETONISATION & PRUNING")
    print("=" * 78)

    # ---- 3.1  Isolated Zero-Pairs (Hit-or-Miss) ----
    print("\n" + "-" * 78)
    print("3.1  Isolated pairs of connected zeros (Hit-or-Miss warm-up)")
    print("-" * 78)
    print("\nInput toy matrix (0 = the value we search pairs of):")
    TEST_MATRIX = maybe_replace(TOY_MATRIX)
    for row in TEST_MATRIX:
        print("     " + " ".join(str(int(v)) for v in row))

    t0 = time.perf_counter()
    pairs = mp.find_isolated_zero_pairs(TEST_MATRIX)
    dt = time.perf_counter() - t0
    print(f"\nDetected {len(pairs)} isolated zero-pair(s) " f"in {dt * 1e3:.3f} ms:")
    for index, pair in enumerate(pairs, start=1):
        (r1, c1), (r2, c2) = pair[0], pair[1]
        print(f"   pair {index}: ({r1}, {c1}) -- ({r2}, {c2})")

    out31 = f"{out_dir}/3_1_isolated_pairs.png"
    vis.plot_matrix_pairs(TEST_MATRIX, pairs, out31)
    print(f"[save] {out31}")

    # ---- 3.2  Skeletonisation (thinning) ----
    print("\n" + "-" * 78)
    print("3.2  Skeletonisation of the PCB copper tracks (iterative thinning)")
    print("-" * 78)

    reference_gray = cv2.cvtColor(
        cv2.imread(f"{input_dir}/PCB/reference.jpg"), cv2.COLOR_BGR2GRAY
    )
    mask_full, threshold = binarize_auto_threshold(reference_gray, "bright")
    print(
        f"\n[input] reference {reference_gray.shape[1]}x{reference_gray.shape[0]}, "
        f"Otsu t* = {int(threshold)}, copper px = {int(mask_full.sum())}"
    )

    mask = mask_full[::DECIMATION, ::DECIMATION]
    print(
        f"[decimate] factor {DECIMATION} (NumPy slicing) -> "
        f"{mask.shape[1]}x{mask.shape[0]}, copper px = {int(mask.sum())}"
    )

    tracks = mp.opening(mask, mp.make_structuringElements("square", 3))

    t0 = time.perf_counter()
    skeleton, sweeps = mp.thinning(tracks)
    thinning_time = time.perf_counter() - t0
    print(f"[thinning] converged in {sweeps} sweeps, {thinning_time:.3f} s")
    print(
        f"           cleaned-track px = {int(tracks.sum())}, "
        f"skeleton px = {int(skeleton.sum())}"
    )

    # ---- 3.3  Pruning ----
    print("\n" + "-" * 78)
    print(f"3.3  Pruning the skeleton ({PRUNE_ITERS} endpoint-thinning iters)")
    print("-" * 78)

    endpoints_before = int(mp.find_endpoints(skeleton).sum())
    t0 = time.perf_counter()
    pruned = mp.prune(skeleton, n=PRUNE_ITERS)
    prune_time = time.perf_counter() - t0
    endpoints_after = int(mp.find_endpoints(pruned).sum())
    print(f"\n[prune] {prune_time:.3f} s")
    print(f"        skeleton px {int(skeleton.sum())} -> pruned px {int(pruned.sum())}")
    print(
        f"        free endpoints {endpoints_before} -> {endpoints_after} "
        f"(parasitic spurs removed)"
    )
    print(
        f"        pruned skeleton is a subset of the raw skeleton: "
        f"{bool((pruned & ~skeleton).sum() == 0)}"
    )

    out33 = f"{out_dir}/3_3_skeleton_pruning.png"
    vis.plot_skeleton_comparison(tracks, skeleton, pruned, out33)
    print(f"[save] {out33}")

    return {
        "isolated_pairs": pairs,
        "decimation": DECIMATION,
        "thinning_iters": int(sweeps),
        "thinning_time": float(thinning_time),
        "prune_time": float(prune_time),
        "skeleton_px": int(skeleton.sum()),
        "pruned_px": int(pruned.sum()),
        "endpoints_before": endpoints_before,
        "endpoints_after": endpoints_after,
    }


if __name__ == "__main__":
    run()
