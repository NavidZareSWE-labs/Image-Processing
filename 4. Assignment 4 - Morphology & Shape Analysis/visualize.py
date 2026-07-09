

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")


# ---- Helpers ----
def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---- Isolated-Pairs Plot (3.1) ----
def plot_matrix_pairs(matrix, pairs, path):
    _ensure_dir(path)
    M = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(M, cmap="gray_r", vmin=0, vmax=1)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, str(M[r, c]), ha="center", va="center",
                    color="red" if M[r, c] == 0 else "black", fontsize=14)
    colors = ["tab:green", "tab:blue", "tab:orange", "tab:purple",
              "tab:cyan", "tab:olive"]
    for k, pair in enumerate(pairs):
        col = colors[k % len(colors)]
        rs = [p[0] for p in pair]
        cs = [p[1] for p in pair]
        ax.add_patch(Rectangle((min(cs) - 0.45, min(rs) - 0.45),
                               (max(cs) - min(cs)) + 0.9,
                               (max(rs) - min(rs)) + 0.9,
                               fill=False, edgecolor=col, linewidth=2.5))
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_title(f"Part 3.1 -- isolated zero-pairs found: {len(pairs)}")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---- Skeleton Comparison Plot (3.3) ----
def plot_skeleton_comparison(tracks, raw_skel, pruned_skel, path):
    _ensure_dir(path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(tracks, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Cleaned copper tracks (input)")
    axes[1].imshow(raw_skel, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Raw skeleton (thinning)")
    axes[2].imshow(pruned_skel, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Pruned skeleton")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Part 3 -- skeletonisation and pruning", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
