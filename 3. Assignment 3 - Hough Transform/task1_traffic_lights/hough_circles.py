import numpy as np


def hough_circle_transform(edge_map,
                           gradient_dir,
                           r_min=10,
                           r_max=80,
                           r_step=1,
                           vote_threshold_ratio=0.45):
    H, W = edge_map.shape
    radiuses = np.arange(r_min, r_max + 1, r_step)
    n_rad = len(radiuses)

    accumulator = np.zeros((H, W, n_rad), dtype=np.int32)

    # Get edge pixel coordinates
    edge_y, edge_x = np.nonzero(edge_map)
    edge_theta = gradient_dir[edge_y, edge_x]

    print(f"    Edge pixels used for voting: {len(edge_y)}")
    print(f"    Radius range: [{r_min}, {r_max}], step={r_step}, "
          f"bins={n_rad}")

    # Voting: for each radius, compute cand centres (Like Slide 63)
    # along +/- gradient direction
    for rad_idx, rad in enumerate(radiuses):
        # Two cand centres per edge pixel (both sides of gradient)
        for dir_sign in (+1, -1):
            cand_center_x = (edge_x + dir_sign * rad *
                             np.cos(edge_theta)).astype(np.int32)
            cand_center_y = (edge_y + dir_sign * rad *
                             np.sin(edge_theta)).astype(np.int32)

            # Mask valid coordinates - remove out of bounds
            valid_centers = (cand_center_x >= 0) & (
                cand_center_x < W) & (cand_center_y >= 0) & (cand_center_y < H)
            valid_center_x, valid_center_y = cand_center_x[
                valid_centers], cand_center_y[valid_centers]

            # Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’. For addition ufunc, this method is equivalent to a[indices] += b, except that results are accumulated for elements that are indexed more than once. For example, a[[0,0]] += 1 will only increment the first element once because of buffering, whereas add.at(a, [0,0], 1) will increment the first element twice.
            # Read More : https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
            np.add.at(accumulator[:, :, rad_idx],
                      (valid_center_y, valid_center_x), 1)

    # ---- Peak detection - Local-max check ----
    max_val = accumulator.max()
    threshold = max_val * vote_threshold_ratio
    print(f"    Accumulator max votes: {max_val}, "
          f"threshold: {threshold:.1f}")

    circles = []
    flat_idx = accumulator.argmax()
    best_coords = np.unravel_index(
        flat_idx,
        accumulator.shape
    )  # tuple of (x,y, rad_idx)
    best_rad_idx = best_coords[2]
    acc_best = accumulator[:, :, best_rad_idx].astype(np.float64)

    for rad_idx, rad in enumerate(radiuses):
        acc_slice = accumulator[:, :, rad_idx]
        if acc_slice.max() < threshold:
            continue

        #  Local maximum -  3x3 neighborhood
        padded = np.pad(acc_slice, 1, mode='constant', constant_values=0)
        is_peak = np.ones_like(acc_slice, dtype=bool)
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                row_start = 1 + row_offset
                row_end = H + 1 + row_offset
                col_start = 1 + col_offset
                col_end = W + 1 + col_offset
                neighbor = padded[row_start:row_end,
                                  col_start:col_end]
                # &= A pixel stays True only if it passed every single neighbour check.
                is_peak &= acc_slice >= neighbor

        is_peak &= acc_slice >= threshold
        peak_y, peak_x = np.nonzero(is_peak)
        for y, x in zip(peak_y, peak_x):
            votes = int(acc_slice[y, x])
            circles.append((x, y, rad, votes))

    # Non-maximum suppression: remove overlapping circles
    circles = remove_dup_circles(circles, min_dist=r_min)
    print(f"    Circles detected (after NMS): {len(circles)}")

    circles_out = [(c[0], c[1], c[2]) for c in circles]
    return circles_out, acc_best


def remove_dup_circles(circles, min_dist=15):
    if not circles:
        return []

    circles = sorted(circles, key=lambda c: c[3], reverse=True)
    keep = []
    for cand_circle in circles:
        cand_x, cand_y, cand_rad, cand_votes = cand_circle
        is_rejected = False

        for k in keep:
            accepted_x, accepted_y, accepted_rad, accepted_votes = k
            dist = np.sqrt((cand_x - accepted_x) ** 2 +
                           (cand_y - accepted_y) ** 2)
            min_dist_threshold = min_dist + 0.5 * (cand_rad + accepted_rad)
            if dist < min_dist_threshold:
                is_rejected = True
                break
        if not is_rejected:
            keep.append(cand_circle)
    return keep
