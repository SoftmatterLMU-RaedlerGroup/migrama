"""Functional tests for pattern detection with visual verification.

These tests run pattern detection on real data and produce overlay images
for manual inspection of bounding box accuracy.
"""

import csv
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytest

from migrama.core.pattern import PatternDetector
from migrama.core.pattern.source import Nd2PatternFovSource
from tests.data import FOURCELL_20250812


# Output directory for visual verification
# Structure: tests/_plots/{command_name}/{dataset_name}/
PLOTS_DIR = Path(__file__).parent / "_plots"


def load_csv_bboxes(csv_path: Path) -> dict[int, list[tuple[int, int, int, int]]]:
    """Load bounding boxes from CSV, grouped by FOV.

    Returns:
        Dict mapping fov -> list of (x, y, w, h) tuples
    """
    bboxes_by_fov: dict[int, list[tuple[int, int, int, int]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fov = int(row["fov"])
            bbox = (int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
            if fov not in bboxes_by_fov:
                bboxes_by_fov[fov] = []
            bboxes_by_fov[fov].append(bbox)
    return bboxes_by_fov


def plot_fov_with_bboxes(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    fov: int,
    output_path: Path,
) -> None:
    """Plot pattern image with bounding box overlays.

    Args:
        image: Pattern image array (Y, X)
        bboxes: List of (x, y, w, h) bounding boxes
        fov: FOV index for title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display image with percentile normalization
    vmin, vmax = np.percentile(image, [1, 99])
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)

    # Draw bounding boxes
    for i, (x, y, w, h) in enumerate(bboxes):
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        # Add cell index label
        ax.text(
            x + w / 2, y - 5,
            str(i),
            color="lime",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title(f"FOV {fov}: {len(bboxes)} patterns detected")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


@pytest.mark.skipif(
    not FOURCELL_20250812.exists(),
    reason="Test data not available",
)
class TestPatternFunctional:
    """Functional tests that produce visual output for verification."""

    def test_pattern_detection_sampled_fovs(self, tmp_path: Path):
        """Detect patterns on sampled FOVs and save overlay images.

        Tests FOVs 0, 10, 20, ... and produces overlay images showing
        detected bounding boxes on the pattern images.
        """
        # Output path: tests/_plots/pattern/{dataset_name}/
        output_dir = PLOTS_DIR / "pattern" / FOURCELL_20250812.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup
        csv_path = output_dir / "patterns.csv"
        source = Nd2PatternFovSource(FOURCELL_20250812.patterns_nd2)
        detector = PatternDetector(source=source)

        # Sample FOVs: 0, 10, 20, ...
        sampled_fovs = list(range(0, source.n_fovs, 10))

        # Detect patterns for sampled FOVs
        all_records = []
        for fov in sampled_fovs:
            records = detector.detect_fov(fov)
            all_records.extend(records)
            print(f"FOV {fov}: {len(records)} patterns")

        # Save CSV
        detector.save_csv(all_records, csv_path)
        assert csv_path.exists()

        # Load bboxes from CSV
        bboxes_by_fov = load_csv_bboxes(csv_path)

        # Generate overlay images
        for fov in sampled_fovs:
            # Get pattern image for this FOV
            for fov_id, image in source.iter_fovs():
                if fov_id == fov:
                    break

            bboxes = bboxes_by_fov.get(fov, [])
            output_path = output_dir / f"fov_{fov:03d}.png"
            plot_fov_with_bboxes(image, bboxes, fov, output_path)
            print(f"Saved: {output_path}")

        # Summary
        print(f"\n{'='*50}")
        print(f"Detected {len(all_records)} patterns across {len(sampled_fovs)} FOVs")
        print(f"Output directory: {output_dir}")
        print(f"CSV file: {csv_path}")
