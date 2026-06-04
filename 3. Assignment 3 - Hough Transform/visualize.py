from matplotlib.patches import Circle as MplCircle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


def save_grayscale(img, path, title="", cmap='gray'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_heatmap(accumulator, path, title="Hough Accumulator",
                 xlabel="", ylabel=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(accumulator, cmap='hot', aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_side_by_side(images, titles, path, figsize=None):
    n = len(images)
    if figsize is None:
        figsize = (5 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, t in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_heatmap(accumulator, path, title="Hough Accumulator",
                 xlabel="", ylabel=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(accumulator, cmap='hot', aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)

# ------------------------------------------------------------------
# Task 1 — Traffic lights
# ------------------------------------------------------------------


def draw_circles_on_image(rgb_img, circles, labels):

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(rgb_img)

    label_colour = {
        'RED':    'red',
        'GREEN':  'lime',
        'YELLOW': 'yellow',
        'UNLIT':  'white',
    }

    for (cx, cy, r), lbl in zip(circles, labels):
        lc = label_colour.get(lbl, 'white')

        # Thick magenta circle outline
        circ = MplCircle((cx, cy), r, fill=False, edgecolor='magenta',
                         linewidth=3)
        ax.add_patch(circ)

        # Small yellow center dot
        ax.plot(cx, cy, 'o', color='yellow', markersize=3, zorder=5)

        # Colour-coded text label above the circle
        ax.text(cx, cy - r - 8, lbl, color=lc, fontsize=14,
                fontweight='bold', ha='center', va='bottom')

    ax.set_title("Detected Traffic Lights", fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    # render to array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def save_traffic_pipeline(original, edges, accumulator_slice, overlay,
                          path, img_name=""):
    """Save 4-step pipeline figure for Part 1."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ['Original', 'Edge Map', 'Hough Accumulator (best r)',
              'Detected Circles']
    imgs = [original, edges, accumulator_slice, overlay]
    cmaps = [None, 'gray', 'hot', None]
    for ax, img, t, cm in zip(axes, imgs, titles, cmaps):
        if cm:
            ax.imshow(img, cmap=cm, aspect='auto')
        else:
            ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    if img_name:
        fig.suptitle(f"Traffic Light Detection — {img_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)

# ------------------------------------------------------------------
# Part 2 — Lane detection
# ------------------------------------------------------------------


def draw_lane_overlay(rgb_img, left_line, right_line, color=(0, 255, 0),
                      alpha=0.3):
    """
    Draw the drivable-area polygon between left & right lane lines.
    Only the green polygon is drawn (no separate coloured lane lines)
    to match the TA reference output.
    left_line, right_line : ((x1,y1),(x2,y2))
    Returns annotated RGB copy (uint8).
    """
    overlay = rgb_img.copy().astype(np.float64)
    h, w = rgb_img.shape[:2]

    if left_line is not None and right_line is not None:
        (lx1, ly1), (lx2, ly2) = left_line
        (rx1, ry1), (rx2, ry2) = right_line

        # polygon corners: top-left, bottom-left, bottom-right, top-right
        pts = np.array([
            [lx1, ly1], [lx2, ly2],
            [rx2, ry2], [rx1, ry1]
        ], dtype=np.int32)

        # fill polygon onto a mask
        mask = np.zeros((h, w), dtype=np.uint8)
        _fill_polygon(mask, pts, 1)
        region = mask.astype(bool)
        poly_color = np.array(color, dtype=np.float64)
        overlay[region] = overlay[region] * (1 - alpha) + poly_color * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def _fill_polygon(mask, pts, value):
    """Scanline polygon fill (simple, convex polygon)."""
    h, w = mask.shape
    ys = pts[:, 1]
    y_min, y_max = max(int(ys.min()), 0), min(int(ys.max()), h - 1)
    n = len(pts)
    for y in range(y_min, y_max + 1):
        x_ints = []
        for i in range(n):
            j = (i + 1) % n
            y1, y2 = pts[i, 1], pts[j, 1]
            if y1 == y2:
                continue
            if (y1 <= y < y2) or (y2 <= y < y1):
                x_int = pts[i, 0] + (y - y1) * \
                    (pts[j, 0] - pts[i, 0]) / (y2 - y1)
                x_ints.append(x_int)
        x_ints.sort()
        for k in range(0, len(x_ints) - 1, 2):
            xa = max(int(np.ceil(x_ints[k])), 0)
            xb = min(int(np.floor(x_ints[k + 1])), w - 1)
            mask[y, xa:xb + 1] = value


def _draw_line_on_img(img, line, color, thickness):
    """Bresenham-ish thick line drawing."""
    (x1, y1), (x2, y2) = line
    out = img.copy()
    h, w = out.shape[:2]
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return out
    xs = np.linspace(x1, x2, length).astype(int)
    ys = np.linspace(y1, y2, length).astype(int)
    for dx in range(-thickness // 2, thickness // 2 + 1):
        for dy in range(-thickness // 2, thickness // 2 + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            xx = np.clip(xs + dx, 0, w - 1)
            out[yy, xx] = color
    return out


def save_lane_pipeline(original, edges, accumulator, overlay, path,
                       img_name=""):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ['Original', 'Edge Map', 'Hough Accumulator', 'Lane Overlay']
    imgs = [original, edges, accumulator, overlay]
    cmaps = [None, 'gray', 'hot', None]
    for ax, img, t, cm in zip(axes, imgs, titles, cmaps):
        if cm:
            ax.imshow(img, cmap=cm, aspect='auto')
        else:
            ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    if img_name:
        fig.suptitle(f"Lane Detection — {img_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
