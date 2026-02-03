"""Plotting utilities for migrama."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def plot_pattern_bboxes(
    image: np.ndarray,
    records: list,
    fov_idx: int,
    output_path: Path | str,
    dpi: int = 150,
) -> None:
    """Plot pattern image with bounding box overlays.

    Parameters
    ----------
    image : np.ndarray
        Pattern image (2D grayscale)
    records : list
        List of PatternRecord objects with x, y, w, h, cell attributes
    fov_idx : int
        FOV index for title
    output_path : Path | str
        Output PNG file path
    dpi : int
        Output resolution (default 150)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image, cmap='gray')
    ax.set_title(f"FOV {fov_idx}: {len(records)} patterns detected")

    for r in records:
        rect = mpatches.Rectangle(
            (r.x, r.y), r.w, r.h,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            r.x + r.w / 2, r.y - 5, str(r.cell),
            ha='center', va='bottom', color='lime', fontsize=8, fontweight='bold'
        )

    ax.set_axis_off()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='black')
    plt.close(fig)
