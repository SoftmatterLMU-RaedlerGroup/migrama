"""Cell count analysis for migrama analyze."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import CellposeCounter
from ..core.cell_source import CellFovSource
from ..core.pattern import CellCropper

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRecord:
    """Analysis result for a single pattern."""

    cell: int
    fov: int
    x: int
    y: int
    w: int
    h: int
    t0: int
    t1: int


class PatternTracker:
    """Track pattern state during analysis with progressive reduction.

    Patterns are dropped when they consistently show 0 nuclei, reducing
    the number of segmentations needed as analysis progresses.
    """

    def __init__(self, n_patterns: int, zero_threshold: int = 3) -> None:
        """Initialize pattern tracking state.

        Parameters
        ----------
        n_patterns : int
            Total number of patterns to track
        zero_threshold : int
            Number of consecutive zero-count frames before dropping a pattern
        """
        self.tracked: set[int] = set(range(n_patterns))
        self.dropped_zero: set[int] = set()
        self.counts: dict[int, list[int]] = {i: [] for i in range(n_patterns)}
        self.masks: dict[int, list[np.ndarray]] = {i: [] for i in range(n_patterns)}
        self.zero_streak: dict[int, int] = {i: 0 for i in range(n_patterns)}
        self.zero_threshold = zero_threshold

    def is_tracked(self, idx: int) -> bool:
        """Check if a pattern is still being tracked."""
        return idx in self.tracked

    def get_tracked_indices(self) -> list[int]:
        """Get sorted list of indices still being tracked."""
        return sorted(self.tracked)

    def record_result(self, idx: int, count: int, mask: np.ndarray) -> None:
        """Record a segmentation result for a pattern.

        If count is 0, increment zero streak. If zero streak exceeds
        threshold, drop the pattern from tracking.
        """
        self.counts[idx].append(count)
        self.masks[idx].append(mask)

        if count == 0:
            self.zero_streak[idx] += 1
            if self.zero_streak[idx] >= self.zero_threshold:
                self._drop_zero(idx)
        else:
            self.zero_streak[idx] = 0

    def record_skipped(self, idx: int) -> None:
        """Record that a pattern was skipped (already dropped).

        Appends -1 count and empty mask to maintain frame alignment.
        """
        self.counts[idx].append(-1)
        # Use a placeholder empty mask
        self.masks[idx].append(np.zeros((1, 1), dtype=np.uint16))

    def _drop_zero(self, idx: int) -> None:
        """Mark pattern as dropped due to consecutive zero nuclei."""
        if idx in self.tracked:
            self.tracked.remove(idx)
            self.dropped_zero.add(idx)
            logger.debug(f"Dropped pattern {idx} after {self.zero_threshold} consecutive zero-count frames")

    def get_counts(self, idx: int) -> list[int]:
        """Get count history for a pattern."""
        return self.counts[idx]

    def get_masks(self, idx: int) -> list[np.ndarray]:
        """Get mask history for a pattern."""
        return self.masks[idx]


class Analyzer:
    """Analyze cell counts for patterns across frames with mask caching."""

    def __init__(
        self,
        source: CellFovSource,
        csv_path: str,
        cache_path: str,
        nuclei_channel: int = 1,
        cell_channels: list[int] | None = None,
        merge_method: str = 'none',
        n_cells: int = 4,
    ) -> None:
        """Initialize Analyzer.

        Parameters
        ----------
        source : CellFovSource
            Source of cell timelapse data (ND2 or TIFF)
        csv_path : str
            Path to patterns CSV file
        cache_path : str
            Path to output cache.ome.zarr for mask storage
        nuclei_channel : int
            Channel index for nuclei
        cell_channels : list[int] | None
            Channel indices for cell segmentation
        merge_method : str
            Channel merge method: 'add', 'multiply', or 'none'
        n_cells : int
            Target number of cells per pattern
        """
        self.source = source
        self.csv_path = Path(csv_path).resolve()
        self.cache_path = Path(cache_path).resolve()
        self.nuclei_channel = nuclei_channel
        self.cell_channels = cell_channels
        self.merge_method = merge_method
        self.n_cells = n_cells

        self.cropper = CellCropper(
            source=source,
            bboxes_csv=str(self.csv_path),
            nuclei_channel=nuclei_channel,
        )
        self.counter = CellposeCounter(
            nuclei_channel=nuclei_channel,
            cell_channels=cell_channels,
            merge_method=merge_method,
        )

    def analyze(self, output_path: str) -> list[AnalysisRecord]:
        """Run analysis, cache masks, and write CSV output.

        Uses progressive reduction: patterns with consecutive zero-count
        frames are dropped from tracking to reduce segmentation work.

        Parameters
        ----------
        output_path : str
            Output CSV file path

        Returns
        -------
        list[AnalysisRecord]
            Analysis records for each pattern
        """
        import zarr

        records: list[AnalysisRecord] = []

        # Create cache zarr store
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_store = zarr.open(str(self.cache_path), mode='w')

        for fov_idx in sorted(self.cropper.bboxes_by_fov.keys()):
            bboxes = self.cropper.get_bboxes(fov_idx)
            if not bboxes:
                continue

            n_patterns = len(bboxes)
            n_frames = self.cropper.n_frames
            logger.info(f"Analyzing FOV {fov_idx}: {n_patterns} patterns × {n_frames} frames")

            # Create FOV group in cache
            fov_group = cache_store.create_group(f"fov{fov_idx:03d}")

            # Initialize pattern tracker for progressive reduction
            tracker = PatternTracker(n_patterns, zero_threshold=3)

            for frame_idx in range(n_frames):
                # Get indices still being tracked
                tracked_indices = tracker.get_tracked_indices()

                # Log progress every 20 frames
                if frame_idx % 20 == 0 or frame_idx == n_frames - 1:
                    logger.info(f"  Frame {frame_idx + 1}/{n_frames} (tracking {len(tracked_indices)}/{n_patterns} patterns)")

                if not tracked_indices:
                    # All patterns dropped, fill remaining frames with skipped
                    for cell_idx in range(n_patterns):
                        tracker.record_skipped(cell_idx)
                    continue

                # Extract crops only for tracked patterns
                crops = []
                for cell_idx in tracked_indices:
                    crop = self.cropper.extract(fov_idx, cell_idx, frames=frame_idx)
                    crops.append(crop)

                # Count and segment tracked patterns
                result = self.counter.count_nuclei(crops)

                # Record results for tracked patterns
                for i, cell_idx in enumerate(tracked_indices):
                    tracker.record_result(cell_idx, result.counts[i], result.masks[i])

                # Record skipped for dropped patterns
                for cell_idx in range(n_patterns):
                    if not tracker.is_tracked(cell_idx) and cell_idx not in tracker.dropped_zero:
                        # Pattern was just dropped this frame, already recorded
                        pass
                    elif cell_idx in tracker.dropped_zero and len(tracker.get_counts(cell_idx)) < frame_idx + 1:
                        tracker.record_skipped(cell_idx)

            # Log dropout summary
            if tracker.dropped_zero:
                logger.info(f"  Dropped {len(tracker.dropped_zero)} patterns due to zero nuclei: {sorted(tracker.dropped_zero)}")

            # Save masks to cache and compute t0/t1
            for cell_idx, bbox in enumerate(bboxes):
                # Stack masks for this pattern: (T, H, W)
                masks_list = tracker.get_masks(cell_idx)

                # Handle variable mask sizes (from skipped frames)
                # Find the most common shape
                shapes = [m.shape for m in masks_list if m.shape != (1, 1)]
                if shapes:
                    target_shape = max(set(shapes), key=shapes.count)
                else:
                    target_shape = (64, 64)  # fallback

                # Resize placeholder masks to target shape
                normalized_masks = []
                for m in masks_list:
                    if m.shape == (1, 1):
                        normalized_masks.append(np.zeros(target_shape, dtype=np.uint16))
                    elif m.shape != target_shape:
                        # Pad or crop to target shape
                        padded = np.zeros(target_shape, dtype=m.dtype)
                        h, w = min(m.shape[0], target_shape[0]), min(m.shape[1], target_shape[1])
                        padded[:h, :w] = m[:h, :w]
                        normalized_masks.append(padded)
                    else:
                        normalized_masks.append(m)

                masks_stack = np.stack(normalized_masks)

                # Save to cache
                cell_group = fov_group.create_group(f"cell{cell_idx:03d}")
                cell_group.create_array(
                    "cell_masks",
                    data=masks_stack,
                    chunks=(1, masks_stack.shape[1], masks_stack.shape[2]),
                )
                # Store bbox metadata
                cell_group.attrs["bbox"] = [bbox.x, bbox.y, bbox.w, bbox.h]
                cell_group.attrs["fov"] = bbox.fov
                cell_group.attrs["cell"] = bbox.cell

                # Compute t0/t1 from counts (excluding -1 skipped frames)
                counts = tracker.get_counts(cell_idx)
                t0, t1 = self._find_longest_run(counts, self.n_cells)
                records.append(
                    AnalysisRecord(
                        cell=bbox.cell,
                        fov=bbox.fov,
                        x=bbox.x,
                        y=bbox.y,
                        w=bbox.w,
                        h=bbox.h,
                        t0=t0,
                        t1=t1,
                    )
                )

            logger.info(f"  Cached {len(bboxes)} pattern masks for FOV {fov_idx}")

        # Store global metadata in cache
        cache_store.attrs["nuclei_channel"] = self.nuclei_channel
        cache_store.attrs["cell_channels"] = self.cell_channels
        cache_store.attrs["merge_method"] = self.merge_method
        cache_store.attrs["n_fovs"] = len(self.cropper.bboxes_by_fov)

        logger.info(f"Saved mask cache to {self.cache_path}")

        self._write_csv(output_path, records)
        return records

    @staticmethod
    def _find_longest_run(counts: list[int], target: int) -> tuple[int, int]:
        """Find longest contiguous run of target counts.

        Skips frames with count == -1 (dropped patterns).
        """
        best_start = -1
        best_end = -1
        best_len = 0
        current_start: int | None = None

        for idx, count in enumerate(counts):
            if count == -1:
                # Skipped frame breaks the run
                if current_start is not None:
                    current_end = idx - 1
                    length = current_end - current_start + 1
                    if length > best_len:
                        best_len = length
                        best_start = current_start
                        best_end = current_end
                    current_start = None
            elif count == target:
                if current_start is None:
                    current_start = idx
            elif current_start is not None:
                current_end = idx - 1
                length = current_end - current_start + 1
                if length > best_len:
                    best_len = length
                    best_start = current_start
                    best_end = current_end
                current_start = None

        if current_start is not None:
            current_end = len(counts) - 1
            length = current_end - current_start + 1
            if length > best_len:
                best_len = length
                best_start = current_start
                best_end = current_end

        if best_len == 0:
            return -1, -1

        return best_start, best_end

    @staticmethod
    def _write_csv(output_path: str | Path, records: list[AnalysisRecord]) -> None:
        """Write analysis records to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["cell", "fov", "x", "y", "w", "h", "t0", "t1"])
            for record in records:
                writer.writerow(
                    [
                        record.cell,
                        record.fov,
                        record.x,
                        record.y,
                        record.w,
                        record.h,
                        record.t0,
                        record.t1,
                    ]
                )

        logger.info(f"Saved analysis CSV to {output_path}")
