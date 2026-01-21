"""End-to-end tests for extract functionality.

These tests run the full workflow: pattern detection -> analyze -> extract,
producing OME-Zarr files and visualization plots for manual inspection.
"""

import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zarr

from migrama.analyze import Analyzer
from migrama.core.cell_source import Nd2CellFovSource
from migrama.core.pattern import PatternDetector
from migrama.core.pattern.source import Nd2PatternFovSource
from migrama.extract import Extractor
from tests.data import FOURCELL_20250812


# Output directory for test results (gitignored)
RESULTS_DIR = Path(__file__).parent / "_results"


def log(msg: str) -> None:
    """Print and flush immediately for real-time output."""
    print(msg, flush=True)


def plot_extract_frame(
    raw_data: np.ndarray,
    cell_masks: np.ndarray,
    nuclei_masks: np.ndarray,
    frame_idx: int,
    output_path: Path,
) -> None:
    """Plot a single frame with all raw channels and segmentation overlays.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw image data with shape (C, H, W)
    cell_masks : np.ndarray
        Cell segmentation mask with shape (H, W)
    nuclei_masks : np.ndarray
        Nuclei segmentation mask with shape (H, W)
    frame_idx : int
        Frame index for title
    output_path : Path
        Path to save the figure
    """
    n_channels = raw_data.shape[0]

    # Layout: raw channels + cell_mask + nuclei_mask + overlay
    n_cols = n_channels + 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Plot raw channels
    for c in range(n_channels):
        ax = axes[c]
        channel_data = raw_data[c]
        vmin, vmax = np.percentile(channel_data, [1, 99])
        ax.imshow(channel_data, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"Channel {c}")
        ax.axis("off")

    # Plot cell mask
    ax_cell = axes[n_channels]
    ax_cell.imshow(cell_masks, cmap="nipy_spectral", interpolation="nearest")
    ax_cell.set_title("Cell Mask")
    ax_cell.axis("off")

    # Plot nuclei mask
    ax_nuclei = axes[n_channels + 1]
    ax_nuclei.imshow(nuclei_masks, cmap="nipy_spectral", interpolation="nearest")
    ax_nuclei.set_title("Nuclei Mask")
    ax_nuclei.axis("off")

    # Plot RGB overlay with masks
    ax_overlay = axes[n_channels + 2]
    h, w = raw_data.shape[1], raw_data.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Normalize each channel for RGB display
    for c in range(min(n_channels, 3)):
        ch = raw_data[c].astype(np.float32)
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch = (ch - ch_min) / (ch_max - ch_min)
        rgb[:, :, c] = ch

    ax_overlay.imshow(np.clip(rgb, 0, 1))

    # Overlay cell boundaries
    from scipy import ndimage
    cell_edges = ndimage.sobel(cell_masks.astype(float)) != 0
    nuclei_edges = ndimage.sobel(nuclei_masks.astype(float)) != 0

    # Create colored overlay for edges
    overlay_rgb = np.clip(rgb.copy(), 0, 1)
    overlay_rgb[cell_edges] = [0, 1, 0]  # Green for cell boundaries
    overlay_rgb[nuclei_edges] = [1, 0, 0]  # Red for nuclei boundaries

    ax_overlay.imshow(overlay_rgb)
    ax_overlay.set_title("Overlay")
    ax_overlay.axis("off")

    fig.suptitle(f"Frame {frame_idx}", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@pytest.mark.skipif(
    not FOURCELL_20250812.exists(),
    reason="Test data not available",
)
class TestExtractFunctional:
    """Functional tests for the full extract workflow with visual output."""

    def test_full_workflow_random_fov(self):
        """Test full workflow: detect -> analyze -> extract on random FOV.

        Randomly selects a FOV with patterns, runs the complete pipeline,
        and generates visualization plots every 20 frames showing all
        raw channels and segmentation masks.

        Run with: pytest tests/test_extract.py -v -s
        """
        # Configure logging to show progress from migrama modules
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(name)s - %(message)s",
            stream=sys.stdout,
            force=True,
        )

        # Output directories
        output_dir = RESULTS_DIR / "extract" / FOURCELL_20250812.name
        output_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Output file paths
        patterns_csv = output_dir / "patterns.csv"
        analysis_csv = output_dir / "analysis.csv"
        cache_zarr = output_dir / "cache.ome.zarr"
        extracted_zarr = output_dir / "extracted.ome.zarr"

        # === Step 1: Pattern Detection ===
        log("\n=== Step 1: Pattern Detection ===")
        pattern_source = Nd2PatternFovSource(FOURCELL_20250812.patterns_nd2)
        detector = PatternDetector(source=pattern_source)

        # Sample FOVs to find one with patterns
        random.seed(42)  # Reproducible randomness
        sample_fovs = random.sample(
            range(pattern_source.n_fovs),
            min(10, pattern_source.n_fovs)
        )

        # Find a FOV with patterns
        selected_fov = None
        detected_records = []
        for fov_idx in sample_fovs:
            log(f"  Checking FOV {fov_idx}...")
            records = detector.detect_fov(fov_idx)
            if len(records) >= 1:
                selected_fov = fov_idx
                detected_records = records  # Use all patterns
                break

        assert selected_fov is not None, "No FOVs with patterns found in sample"
        log(f"Selected FOV {selected_fov} with {len(detected_records)} patterns")

        # Save patterns CSV (filtered to selected FOV and limited patterns)
        detector.save_csv(detected_records, str(patterns_csv))
        log(f"Saved patterns to: {patterns_csv}")

        # === Step 2: Analyze ===
        log("\n=== Step 2: Analyze (this may take a while - segmenting all frames) ===")
        cell_source = Nd2CellFovSource(FOURCELL_20250812.cells_nd2)
        log(f"Total frames in timelapse: {cell_source.n_frames}")

        analyzer = Analyzer(
            source=cell_source,
            csv_path=str(patterns_csv),
            cache_path=str(cache_zarr),
            nuclei_channel=1,
            cell_channels=[0],  # Phase contrast channel
            merge_method="none",
            n_cells=4,
        )

        analysis_records = analyzer.analyze(str(analysis_csv))
        log(f"Analyzed {len(analysis_records)} patterns")
        log(f"Saved analysis CSV to: {analysis_csv}")
        log(f"Saved mask cache to: {cache_zarr}")

        # Check which records have valid t0/t1 ranges
        valid_records = [r for r in analysis_records if r.t0 >= 0 and r.t1 >= r.t0]
        log(f"Records with valid time ranges: {len(valid_records)}")
        for r in analysis_records:
            log(f"  Pattern cell={r.cell} fov={r.fov}: t0={r.t0}, t1={r.t1}")

        # === Step 3: Extract ===
        log("\n=== Step 3: Extract ===")
        extractor = Extractor(
            source=cell_source,
            analysis_csv=str(analysis_csv),
            output_path=str(extracted_zarr),
            nuclei_channel=1,
            cell_channels=[0],
            merge_method="none",
            cache_path=str(cache_zarr),
        )

        # Extract with minimum 10 frames to ensure we have enough for plots
        n_sequences = extractor.extract(min_frames=10)
        log(f"Extracted {n_sequences} sequences to: {extracted_zarr}")

        # === Step 4: Generate Visualization Plots ===
        log("\n=== Step 4: Generate Visualization Plots ===")

        if n_sequences == 0:
            log("No sequences extracted, skipping plots")
            return

        # Open the extracted zarr store
        root = zarr.open(str(extracted_zarr), mode="r")

        # Find the first sequence to visualize
        sequence_key = None
        for key in root.keys():
            if key.startswith("fov"):
                sequence_key = key
                break

        assert sequence_key is not None, "No sequences found in extracted zarr"

        seq_group = root[sequence_key]
        raw_data = np.asarray(seq_group["data"])  # Shape: (T, C, H, W)
        cell_masks = np.asarray(seq_group["cell_masks"])  # Shape: (T, H, W)
        nuclei_masks = np.asarray(seq_group["nuclei_masks"])  # Shape: (T, H, W)

        n_frames = raw_data.shape[0]
        log(f"Sequence {sequence_key}: {n_frames} frames, shape {raw_data.shape}")

        # Sample every 20th frame
        frame_indices = list(range(0, n_frames, 20))
        log(f"Generating plots for {len(frame_indices)} frames (every 20th)")

        for frame_idx in frame_indices:
            output_path = plots_dir / f"{sequence_key}_frame{frame_idx:03d}.png"
            plot_extract_frame(
                raw_data=raw_data[frame_idx],
                cell_masks=cell_masks[frame_idx],
                nuclei_masks=nuclei_masks[frame_idx],
                frame_idx=frame_idx,
                output_path=output_path,
            )
            log(f"  Saved: {output_path.name}")

        # === Summary ===
        log(f"\n{'=' * 60}")
        log("Summary:")
        log(f"  FOV: {selected_fov}")
        log(f"  Patterns detected: {len(detected_records)}")
        log(f"  Sequences extracted: {n_sequences}")
        log(f"  Plots generated: {len(frame_indices)}")
        log(f"\nOutput directory: {output_dir}")
        log(f"{'=' * 60}")
