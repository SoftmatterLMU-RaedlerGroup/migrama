"""Visual tests for analyze functionality with watershed counting.

These tests run pattern detection and cell counting on real data,
producing overlay images for manual inspection.
"""

import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytest

from migrama.core import CellposeCounter
from migrama.core.cell_source import Nd2CellFovSource
from migrama.core.pattern import PatternDetector
from migrama.core.pattern.source import Nd2PatternFovSource
from tests.data import FOURCELL_20250812


# Output directory for visual verification
PLOTS_DIR = Path(__file__).parent / "_plots"


def plot_frame_with_counts(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    counts: list[int],
    fov: int,
    frame: int,
    output_path: Path,
) -> None:
    """Plot a single frame with bboxes and cell counts overlaid.

    Parameters
    ----------
    image : np.ndarray
        Nuclear channel image (Y, X)
    bboxes : list[tuple[int, int, int, int]]
        List of (x, y, w, h) bounding boxes
    counts : list[int]
        Cell count for each bbox
    fov : int
        FOV index for title
    frame : int
        Frame index for title
    output_path : Path
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display image with percentile normalization
    vmin, vmax = np.percentile(image, [1, 99])
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)

    # Draw bounding boxes with counts
    for i, ((x, y, w, h), count) in enumerate(zip(bboxes, counts, strict=True)):
        # Color based on count (green=4, yellow=close, red=far, cyan=empty)
        if count == 0:
            color = "cyan"  # Empty pattern - no cells detected
        elif count == 4:
            color = "lime"
        elif count in (3, 5):
            color = "yellow"
        else:
            color = "red"

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add count label at center of bbox
        ax.text(
            x + w / 2, y + h / 2,
            str(count),
            color=color,
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.7},
        )

        # Add cell index label above bbox
        ax.text(
            x + w / 2, y - 5,
            f"#{i}",
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title(f"FOV {fov}, Frame {frame} - Merged channels with cell counts", color="white")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


@pytest.mark.skipif(
    not FOURCELL_20250812.exists(),
    reason="Test data not available",
)
class TestAnalyzeFunctional:
    """Functional tests for analyze with visual output."""

    def test_cellpose_counting_random_fov(self, tmp_path: Path):
        """Test Cellpose counting on a random FOV sampling every 20 frames.

        Randomly selects a FOV, detects patterns, then counts cells
        at every 20th frame. Produces overlay images with merged channels
        (normalized and summed) showing bboxes and cell counts.
        """
        # Output directory
        output_dir = PLOTS_DIR / "analyze" / FOURCELL_20250812.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load pattern source and detect patterns
        pattern_source = Nd2PatternFovSource(FOURCELL_20250812.patterns_nd2)
        detector = PatternDetector(source=pattern_source)

        # Load cell source
        cell_source = Nd2CellFovSource(FOURCELL_20250812.cells_nd2)

        # Sample a small set of FOVs to check (avoid iterating through all)
        random.seed(42)  # Reproducible randomness
        sample_fovs = random.sample(range(pattern_source.n_fovs), min(10, pattern_source.n_fovs))

        # Find FOVs that have patterns
        fovs_with_patterns = []
        for fov_idx in sample_fovs:
            records = detector.detect_fov(fov_idx)
            if len(records) >= 3:  # At least 3 patterns
                fovs_with_patterns.append((fov_idx, records))
                break  # Take the first good one to save time

        assert len(fovs_with_patterns) > 0, "No FOVs with patterns found in sample"

        # Use the first FOV found with patterns
        fov_idx, records = fovs_with_patterns[0]
        bboxes = [(r.x, r.y, r.w, r.h) for r in records]

        print(f"\nSelected FOV {fov_idx} with {len(bboxes)} patterns")

        # Sample every 20th frame
        n_frames = cell_source.n_frames
        frame_indices = list(range(0, n_frames, 20))

        print(f"Sampling {len(frame_indices)} frames (every 20th)")

        # Initialize counter
        counter = CellposeCounter()

        # Get FOV data
        fov_data = cell_source.get_fov(fov_idx)  # Shape: (T, C, H, W)
        nuclei_channel = 1  # Channel 1 is nuclei (matches Analyzer default)

        # Process each sampled frame
        for frame_idx in frame_indices:
            # Get all channels and merge (normalize then sum)
            all_channels = fov_data[frame_idx]  # Shape: (C, H, W)
            # Normalize each channel to 0-1 range then sum
            normalized = []
            for c in range(all_channels.shape[0]):
                ch = all_channels[c].astype(np.float32)
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    ch = (ch - ch_min) / (ch_max - ch_min)
                else:
                    ch = np.zeros_like(ch)
                normalized.append(ch)
            merged_image = np.sum(normalized, axis=0)

            # Extract crops for each bbox (use nuclei channel for counting)
            nuclei_image = fov_data[frame_idx, nuclei_channel]
            crops = []
            for x, y, w, h in bboxes:
                crop = nuclei_image[y:y + h, x:x + w]
                crops.append(crop)

            result = counter.count_nuclei(crops)
            counts = result.counts

            # Plot merged image with counts overlay
            output_path = output_dir / f"fov{fov_idx:03d}_frame{frame_idx:03d}.png"
            plot_frame_with_counts(
                merged_image,
                bboxes,
                counts,
                fov_idx,
                frame_idx,
                output_path,
            )
            print(f"Frame {frame_idx}: counts = {counts} -> {output_path.name}")

        # Summary
        print(f"\n{'=' * 50}")
        print(f"Output directory: {output_dir}")
        print(f"Generated {len(frame_indices)} overlay images")
