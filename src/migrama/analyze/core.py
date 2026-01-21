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

            logger.info(f"Analyzing FOV {fov_idx} with {len(bboxes)} patterns")

            # Create FOV group in cache
            fov_group = cache_store.create_group(f"fov{fov_idx:03d}")

            counts_per_pattern = [[] for _ in bboxes]
            masks_per_pattern: list[list[np.ndarray]] = [[] for _ in bboxes]

            for frame_idx in range(self.cropper.n_frames):
                # Extract all crops for this frame
                crops = [
                    self.cropper.extract_all_channels(fov_idx, frame_idx, cell_idx)
                    for cell_idx in range(len(bboxes))
                ]

                # Count and segment all crops at once
                result = self.counter.count_nuclei(crops)

                # Distribute results to per-pattern lists
                for cell_idx, (count, mask) in enumerate(zip(result.counts, result.masks)):
                    counts_per_pattern[cell_idx].append(count)
                    masks_per_pattern[cell_idx].append(mask)

            # Save masks to cache and compute t0/t1
            for cell_idx, bbox in enumerate(bboxes):
                # Stack masks for this pattern: (T, H, W)
                masks_stack = np.stack(masks_per_pattern[cell_idx])
                
                # Save to cache
                cell_group = fov_group.create_group(f"cell{cell_idx:03d}")
                cell_group.create_dataset(
                    "cell_masks",
                    data=masks_stack,
                    chunks=(1, masks_stack.shape[1], masks_stack.shape[2]),
                    dtype=masks_stack.dtype,
                )
                # Store bbox metadata
                cell_group.attrs["bbox"] = [bbox.x, bbox.y, bbox.w, bbox.h]
                cell_group.attrs["fov"] = bbox.fov
                cell_group.attrs["cell"] = bbox.cell

                t0, t1 = self._find_longest_run(counts_per_pattern[cell_idx], self.n_cells)
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

            logger.info(f"Cached {len(bboxes)} pattern masks for FOV {fov_idx}")

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
        """Find longest contiguous run of target counts."""
        best_start = -1
        best_end = -1
        best_len = 0
        current_start: int | None = None

        for idx, count in enumerate(counts):
            if count == target:
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
