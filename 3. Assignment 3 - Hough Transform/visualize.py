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
