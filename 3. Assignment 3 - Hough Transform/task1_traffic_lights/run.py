import os
import numpy as np
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from visualize import (draw_circles_on_image, save_traffic_pipeline,  # noqa: E402
                       save_grayscale, save_heatmap)
from task1_traffic_lights.classifier import classify_circles  # noqa: E402
from task1_traffic_lights.hough_circles import hough_circle_transform  # noqa: E402
from utils.filters import gaussian_blur  # noqa: E402
from utils.edge_detection import canny_edge_detection  # noqa: E402
from utils.utils import Timer, load_image_gray, load_image_rgb, save_image  # noqa: E402


def _select_traffic_light_triplet(circles):
    if len(circles) <= 3:
        return circles

    from itertools import combinations

    cand_circles = list(circles)
    best_three_circle = None
    best_score = -float('inf')

    for combo in combinations(cand_circles, 3):
        three_cir = sorted(combo, key=lambda c: c[1])

        # --- Vertical alignment ---
        x_coords = [c[0] for c in three_cir]
        x_spread = max(x_coords) - min(x_coords)

        top_to_middle_dist = three_cir[1][1] - three_cir[0][1]
        middle_to_bottom_dist = three_cir[2][1] - \
            three_cir[1][1]
        if top_to_middle_dist < 15 or middle_to_bottom_dist < 15:
            continue

        spacing_ratio = min(top_to_middle_dist, middle_to_bottom_dist) / \
            max(top_to_middle_dist, middle_to_bottom_dist)

        #   Must be around Similar sizes
        radiuses = [c[2] for c in three_cir]
        r_mean = np.mean(radiuses)
        r_range = max(radiuses) - min(radiuses)

        #  Span: total y range should be reasonable
        y_span = three_cir[2][1] - three_cir[0][1]
        # too compact
        if y_span < 50:
            continue

        # --- Score ---
        # Prefer: tight x-alignment, even spacing, large radius,
        #         low radius variation
        score = (
            -x_spread * 2.0            # penalise x-misalignment heavily
            + spacing_ratio * 100       # reward even spacing
            + r_mean * 1.5              # reward larger circles
            - r_range * 3.0             # penalise radius inconsistency
            # - three_cir[0][1] * 0.5   # bias: prior assumption is that traffic light images are most fo the times at the top of the image, based on the problem(image) you can use this criteria or don't, for now I commented out this line as the algorithm works without it.
        )

        if score > best_score:
            best_score = score
            best_three_circle = list(three_cir)

    # Post-processing (optional):
    # The algorithm currently works without it but if there was a wery weird image that needs this algorithm, you can use it. What we are doing is basically perfectly vertically aligning the x_coords and radiuses of the circles

    # if best_three_circle is not None:
    #     radiuses = [c[2] for c in best_three_circle]
    #     median_r = int(np.median(radiuses))
    #     # Store (cx, cy, display_r, classify_r)
    #     best_three_circle = [(c[0], c[1], median_r, c[2])
    #                          for c in best_three_circle]

    #     # Align x-coordinates: traffic light bulbs are physically
    #     # stacked vertically (in most regular images), so all three share roughly the same x.
    #     # Use the median x (robust to outlier).
    #     median_x = int(np.median([c[0] for c in best_three_circle]))
    #     best_three_circle = [(median_x, c[1], c[2], c[3])
    #                          for c in best_three_circle]

        return best_three_circle

    return circles


def run(image_dir=None, output_dir=None):
    if image_dir is None:
        image_dir = str(BASE_DIR / "images" / "task1_traffic_lights")
    if output_dir is None:
        output_dir = str(BASE_DIR / "output" / "task1")
    os.makedirs(output_dir, exist_ok=True)

    # (blur_sigma, canny_low, canny_high, r_min, r_max, vote_ratio)
    configs = {
        'red_light_1':   (1.4, 0.04, 0.12, 25, 60, 0.50),
        'red_light_2':   (1.6, 0.04, 0.12, 30, 90, 0.40),
        'green_light_1': (1.2, 0.03, 0.10, 30, 80, 0.35),
    }
    default_config = (1.4, 0.05, 0.15, 15, 80, 0.45)

    files = sorted([f for f in os.listdir(image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print("=" * 70)
    print("PART 1: Traffic Light Detection & State Classification")
    print("=" * 70)

    timing_table = []

    for fname in files:
        path = os.path.join(image_dir, fname)
        stem = os.path.splitext(fname)[0]
        cfg = configs.get(stem, default_config)
        blur_sigma, canny_lo, canny_hi, r_min, r_max, vote_ratio = cfg

        print(f"\n--- Processing: {fname} ---")
        print(f"  Config: blur_sigma={blur_sigma}, canny=({canny_lo},{canny_hi}), "
              f"r=[{r_min},{r_max}], vote_ratio={vote_ratio}")

        rgb = load_image_rgb(path)
        gray = load_image_gray(path)
        height, width = gray.shape
        print(f"  Image size: {width}x{height}")

        # ----  Edge Detection ----
        with Timer("Edge Detection") as t_edge:
            edges, magnitude, direction = canny_edge_detection(
                gray, blur_size=5, blur_sigma=blur_sigma,
                low_thresh=canny_lo, high_thresh=canny_hi
            )
        edge_count = edges.sum()
        print(f"  Edge pixels: {edge_count} "
              f"({100 * edge_count / (height * width):.2f}% of image)")

        # ----  Hough Circle Voting ----
        with Timer("Hough Circle Voting") as t_hough:
            circles, acc_best = hough_circle_transform(
                edges, direction,
                r_min=r_min, r_max=r_max, r_step=1,
                vote_threshold_ratio=vote_ratio
            )

        # ---- Post-process: select exactly 3 circles ----
        raw_count = len(circles)
        # Filter out circles too close to image borders
        circles = [(cx, cy, rad) for (cx, cy, rad) in circles
                   if cy > rad * 0.6 and cy < height - rad * 0.6
                   and cx > rad * 0.6 and cx < width - rad * 0.6]
        circles = _select_traffic_light_triplet(circles)
        print(f"  Raw circles: {raw_count} -> triplet: {len(circles)}")

        if circles and len(circles[0]) == 4:
            display_circles = [(cx, cy, dr) for (cx, cy, dr, cr) in circles]
            classify_circles_list = [(cx, cy, cr)
                                     for (cx, cy, dr, cr) in circles]
        else:
            display_circles = circles
            classify_circles_list = circles

        # ----  Classification  ----
        with Timer("State Classification") as t_class:
            labels = classify_circles(rgb, classify_circles_list)

        print(f"  Results:")
        for (cx, cy, r), lbl in zip(display_circles, labels):
            print(f"    Circle center=({cx},{cy}), r={r} => {lbl}")

        timing_table.append({
            'image': fname,
            'edge_det': t_edge.elapsed,
            'hough': t_hough.elapsed,
            'classify': t_class.elapsed,
            'total': t_edge.elapsed + t_hough.elapsed + t_class.elapsed,
        })

        save_grayscale(edges.astype(float), os.path.join(output_dir, f"{stem}_edges.png"),
                       title=f"Edge Map - {fname}")
        save_heatmap(acc_best, os.path.join(output_dir, f"{stem}_accumulator.png"),
                     title=f"Hough Accumulator - {fname}")

        overlay = draw_circles_on_image(rgb, display_circles, labels)
        save_image(os.path.join(output_dir, f"{stem}_overlay.png"), overlay)

        save_traffic_pipeline(
            rgb, edges.astype(float), acc_best, overlay,
            os.path.join(output_dir, f"{stem}_pipeline.png"),
            img_name=fname
        )

    print("\n" + "=" * 70)
    print("PART 1 - Timing Summary")
    print(f"{'Image':<22} {'Edge(s)':>8} {'Hough(s)':>9} {'Class(s)':>9} {'Total(s)':>9}")
    print("-" * 60)
    for t in timing_table:
        print(f"{t['image']:<22} {t['edge_det']:>8.3f} {t['hough']:>9.3f} "
              f"{t['classify']:>9.3f} {t['total']:>9.3f}")
    print("=" * 70)


if __name__ == "__main__":
    run()
