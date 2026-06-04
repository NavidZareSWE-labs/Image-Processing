
import os
import numpy as np
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import imageio.v2 as imageio  # noqa: E402

from utils.utils import Timer, load_image_gray, load_image_rgb, save_image  # noqa: E402
from utils.edge_detection import canny_edge_detection  # noqa: E402
from task2_lane_detection.preprocessing import preprocess_lane_image  # noqa: E402
from task2_lane_detection.hough_lines import (
    hough_line_transform, extract_line_peaks,
    separate_left_right_lanes
)  # noqa: E402
from visualize import draw_lane_overlay, save_lane_pipeline, save_heatmap  # noqa: E402


def run(image_dir=None, output_dir=None):
    if image_dir is None:
        image_dir = str(BASE_DIR / "images" / "task2_line_detection")
    if output_dir is None:
        output_dir = str(BASE_DIR / "output" / "task2")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                   key=lambda x: int(os.path.splitext(x)[0]))

    print("\n" + "=" * 70)
    print("PART 2: Autonomous Lane Detection (Hough Lines)")
    print("=" * 70)

    frames = []
    timing_table = []
    prev_left = None
    prev_right = None

    for i, fname in enumerate(files):
        path = os.path.join(image_dir, fname)
        stem = os.path.splitext(fname)[0]

        print(f"\n--- Frame {i + 1}/{len(files)}: {fname} ---")

        rgb = load_image_rgb(path)
        h, w = rgb.shape[:2]

        # ---- 1. Preprocessing ----
        with Timer("Preprocessing") as t_pre:
            preprocessed = preprocess_lane_image(rgb)

        # ---- 2. Edge Detection ----
        with Timer("Edge Detection") as t_edge:
            edges, mag, dirn = canny_edge_detection(
                preprocessed, blur_size=5, blur_sigma=1.0,
                low_thresh=0.05, high_thresh=0.15
            )

        # ---- 3. Hough Line Transform ----
        with Timer("Hough Line Voting") as t_hough:
            accumulator, thetas, rhos = hough_line_transform(
                edges, theta_res=1.0, rho_res=1.0
            )

        # ---- 4. Peak extraction + lane overlay ----
        with Timer("Post-processing") as t_post:
            peaks = extract_line_peaks(
                accumulator, thetas, rhos,
                threshold_ratio=0.30, nhood_size=15
            )
            left_line, right_line = separate_left_right_lanes(
                peaks, w, h,
                accumulator=accumulator, thetas=thetas, rhos=rhos
            )

            # Position validation: reject detections far from expected
            expected_left = w * 0.35
            expected_right = w * 0.68
            if left_line is not None:
                lx = left_line[1][0]
                if abs(lx - expected_left) > 100:
                    left_line = None
            if right_line is not None:
                rx = right_line[1][0]
                if abs(rx - expected_right) > 80:
                    right_line = None

            # Temporal smoothing
            if left_line is None and prev_left is not None:
                left_line = prev_left
            if right_line is None and prev_right is not None:
                right_line = prev_right

            if left_line is not None:
                prev_left = left_line
            if right_line is not None:
                prev_right = right_line

            overlay = draw_lane_overlay(rgb, left_line, right_line,
                                        color=(0, 200, 0), alpha=0.3)

        print(f"  Left:  {left_line}")
        print(f"  Right: {right_line}")

        timing_table.append({
            'image': fname, 'preproc': t_pre.elapsed,
            'edge_det': t_edge.elapsed, 'hough': t_hough.elapsed,
            'postproc': t_post.elapsed,
            'total': t_pre.elapsed + t_edge.elapsed + t_hough.elapsed + t_post.elapsed,
        })

        save_image(os.path.join(output_dir, f"{stem}_overlay.png"), overlay)
        frames.append(overlay)

        if i in (0, len(files) // 2, len(files) - 1):
            save_lane_pipeline(
                rgb, edges.astype(float), accumulator, overlay,
                os.path.join(output_dir, f"{stem}_pipeline.png"),
                img_name=fname
            )

    # Build GIF
    gif_path = os.path.join(output_dir, "lane_detection.gif")
    print(f"\n  Building GIF ({len(frames)} frames @ 2 fps) ...")
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"  Saved: {gif_path}")

    # Timing summary
    print("\n" + "=" * 70)
    avg = np.mean([t['total'] for t in timing_table])
    print(f"  Average per-frame: {avg:.3f} s")
    print("=" * 70)


if __name__ == "__main__":
    run()
