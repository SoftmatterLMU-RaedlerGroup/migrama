"""Sequence extraction with segmentation and tracking."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import CellposeSegmenter, CellTracker
from ..core.cell_source import CellFovSource
from ..core.io import create_zarr_store, write_global_metadata, write_sequence
from ..core.pattern import CellCropper

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRow:
    """Row from analysis CSV."""

    cell: int
    fov: int
    x: int
    y: int
    w: int
    h: int
    t0: int
    t1: int


class Extractor:
    """Extract sequences with segmentation and tracking."""

    def __init__(
        self,
        source: CellFovSource,
        analysis_csv: str,
        output_path: str,
        nuclei_channel: int = 1,
        cell_channels: list[int] | None = None,
        merge_method: str | None = None,
        cache_path: str | None = None,
    ) -> None:
        """Initialize extractor.

        Parameters
        ----------
        source : CellFovSource
            Source of cell timelapse data (ND2 or TIFF)
        analysis_csv : str
            Path to analysis CSV file
        output_path : str
            Output Zarr store path
        nuclei_channel : int
            Channel index for nuclei
        cell_channels : list[int]
            Channel indices for cell segmentation. For merge_method='none', uses first channel.
            For 'add' or 'multiply', merges all channels.
        merge_method : str
            Merge method: 'add', 'multiply', or 'none'
        cache_path : str | None
            Path to cache.ome.zarr with pre-computed cell masks from analyze step
        """
        self.source = source
        self.analysis_csv = Path(analysis_csv).resolve()
        self.output_path = Path(output_path).resolve()
        self.nuclei_channel = nuclei_channel
        self.cell_channels = cell_channels
        self.merge_method = merge_method
        self.cache_path = Path(cache_path).resolve() if cache_path else None

        self.cropper = CellCropper(
            source=source,
            bboxes_csv=str(self.analysis_csv),
            nuclei_channel=nuclei_channel,
        )
        self.segmenter = CellposeSegmenter()

        # Load cache if provided
        self._cache = None
        if self.cache_path and self.cache_path.exists():
            import zarr
            self._cache = zarr.open(str(self.cache_path), mode='r')
            logger.info(f"Loaded mask cache from {self.cache_path}")

    def extract(self, min_frames: int = 20) -> int:
        """Extract sequences to Zarr.

        Parameters
        ----------
        min_frames : int
            Minimum frames required to extract a sequence

        Returns
        -------
        int
            Number of sequences extracted
        """
        rows = self._load_analysis_rows(self.analysis_csv)
        # Filter valid rows upfront
        valid_rows = [
            row for row in rows
            if row.t0 >= 0 and row.t1 >= row.t0 and (row.t1 - row.t0 + 1) >= min_frames
        ]
        total = len(valid_rows)
        logger.info(f"Processing {total} sequences (from {len(rows)} rows, min_frames={min_frames})")

        root = create_zarr_store(self.output_path, overwrite=True)
        write_global_metadata(
            root,
            cells_source=f"{type(self.source).__name__}",
            nuclei_channel=self.nuclei_channel,
            cell_channels=self.cell_channels,
            merge_method=self.merge_method,
        )

        total_frames = 0
        cached_frames = 0
        segmented_frames = 0

        for idx, row in enumerate(valid_rows):
            n_frames = row.t1 - row.t0 + 1
            total_frames += n_frames
            logger.info(f"[{idx + 1}/{total}] FOV {row.fov}, cell {row.cell}: {n_frames} frames (t={row.t0}-{row.t1})")

            logger.info("  Extracting timelapse...")
            timelapse = self.cropper.extract(row.fov, row.cell, frames=(row.t0, row.t1 + 1))

            # Load from cache ONLY if --cache provided (explicit opt-in)
            cell_masks = None
            if self._cache is not None:
                cell_masks = self._load_cached_masks(row.fov, row.cell, row.t0, row.t1)
                if cell_masks is not None:
                    logger.info(f"  Loaded {len(cell_masks)} masks from cache")
                    cached_frames += len(cell_masks)

            # If no cache or cache miss, segment
            if cell_masks is None:
                if self._cache is not None:
                    logger.info(f"  Cache miss: segmenting {n_frames} frames...")
                else:
                    logger.info(f"  Segmenting {n_frames} frames...")
                segmented_frames += n_frames
                if self.merge_method is None:
                    cell_masks = self._segment_all_channels(timelapse)
                else:
                    cell_masks = self._segment_cells_merged(timelapse)

            # Track CELLS (not nuclei) - cell-first tracking
            logger.info("  Tracking cells...")
            tracker = CellTracker()
            tracking_maps = tracker.track_frames(cell_masks)
            tracked_cell_masks = [
                tracker.get_tracked_mask(mask, track_map)
                for mask, track_map in zip(cell_masks, tracking_maps, strict=False)
            ]

            # Derive nuclei by thresholding nuclear channel within each tracked cell
            logger.info("  Deriving nuclei masks...")
            tracked_nuclei_masks = []
            for frame_idx, tracked_cell_mask in enumerate(tracked_cell_masks):
                nuclear_image = timelapse[frame_idx, self.nuclei_channel]
                nuclei_mask = self._derive_nuclei(tracked_cell_mask, nuclear_image)
                tracked_nuclei_masks.append(nuclei_mask)

            channels = [f"channel_{i}" for i in range(timelapse.shape[1])]
            bbox = np.array([row.x, row.y, row.w, row.h], dtype=np.int32)

            logger.info("  Writing to zarr...")
            write_sequence(
                root,
                fov_idx=row.fov,
                cell_idx=row.cell,
                data=timelapse,
                nuclei_masks=np.stack(tracked_nuclei_masks),
                cell_masks=np.stack(tracked_cell_masks),
                channels=channels,
                t0=row.t0,
                t1=row.t1,
                bbox=bbox,
            )

        if self._cache is not None:
            logger.info(
                "Cache summary: loaded %d/%d frames from cache, segmented %d frames",
                cached_frames,
                total_frames,
                segmented_frames,
            )
        logger.info(f"Saved {total} sequences to {self.output_path}")
        return total

    def _load_cached_masks(self, fov: int, cell: int, t0: int, t1: int) -> list[np.ndarray] | None:
        """Load cell masks from cache for the given time range.

        Returns None if cache is not available or masks not found.
        """
        if self._cache is None:
            return None

        try:
            fov_key = f"fov{fov:03d}"
            cell_key = f"cell{cell:03d}"
            masks_array = self._cache[fov_key][cell_key]["cell_masks"]

            # Extract the time range [t0:t1+1]
            masks_slice = masks_array[t0:t1 + 1]
            return [masks_slice[i] for i in range(masks_slice.shape[0])]
        except KeyError:
            logger.debug(f"Cache miss: {fov_key}/{cell_key} not found")
            return None

    @staticmethod
    def _load_analysis_rows(csv_path: Path) -> list[AnalysisRow]:
        """Load analysis CSV rows."""
        rows: list[AnalysisRow] = []
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(
                    AnalysisRow(
                        cell=int(row["cell"]),
                        fov=int(row["fov"]),
                        x=int(row["x"]),
                        y=int(row["y"]),
                        w=int(row["w"]),
                        h=int(row["h"]),
                        t0=int(row["t0"]),
                        t1=int(row["t1"]),
                    )
                )
        return rows

    def _segment_channel(self, timelapse: np.ndarray, channel_idx: int) -> list[np.ndarray]:
        """Segment a single channel across frames."""
        masks = []
        for frame_idx in range(timelapse.shape[0]):
            image = timelapse[frame_idx, channel_idx]
            result = self.segmenter.segment_image(image)
            masks.append(result["masks"])
        return masks

    def _segment_all_channels(self, timelapse: np.ndarray) -> list[np.ndarray]:
        """Segment using all channels passed directly to Cellpose.

        Parameters
        ----------
        timelapse : np.ndarray
            Timelapse array with shape (T, C, H, W)

        Returns
        -------
        list[np.ndarray]
            List of cell masks (2D arrays), one per frame
        """
        masks = []
        for frame_idx in range(timelapse.shape[0]):
            frame = timelapse[frame_idx]  # (C, H, W)
            frame_hwc = np.transpose(frame, (1, 2, 0))  # (H, W, C)
            result = self.segmenter.segment_image(frame_hwc, merge_method=None)
            masks.append(result["masks"])
        return masks

    def _segment_cells_merged(self, timelapse: np.ndarray) -> list[np.ndarray]:
        """Segment cells using 2-channel approach (nuclear + merged cell channels).

        Parameters
        ----------
        timelapse : np.ndarray
            Timelapse array with shape (T, C, H, W)

        Returns
        -------
        list[np.ndarray]
            List of cell masks (2D arrays)
        """
        cell_masks = []

        for frame_idx in range(timelapse.shape[0]):
            frame_data = timelapse[frame_idx]  # Shape: (C, H, W)

            # Transpose to (H, W, C) format expected by cellpose
            if frame_data.ndim == 3:
                frame_data_hwc = np.transpose(frame_data, (1, 2, 0))
            else:
                frame_data_hwc = frame_data

            # Segment with merged channels
            result = self.segmenter.segment_image(
                frame_data_hwc,
                nuclei_channel=self.nuclei_channel,
                cell_channels=self.cell_channels,
                merge_method=self.merge_method,
            )
            cell_masks.append(result["masks"])

        return cell_masks

    def _derive_nuclei(
        self,
        tracked_cell_mask: np.ndarray,
        nuclear_image: np.ndarray,
    ) -> np.ndarray:
        """Derive nuclei by Otsu thresholding nuclear channel within each cell.

        Parameters
        ----------
        tracked_cell_mask : np.ndarray
            Cell mask with track IDs (2D array)
        nuclear_image : np.ndarray
            Nuclear channel image (2D array)

        Returns
        -------
        np.ndarray
            Nuclei mask with same track IDs as cells
        """
        from skimage.filters import threshold_otsu

        nuclei_mask = np.zeros_like(tracked_cell_mask)

        for track_id in np.unique(tracked_cell_mask):
            if track_id == 0:
                continue

            # Extract nuclear channel within this cell
            cell_region = tracked_cell_mask == track_id
            nuclear_in_cell = nuclear_image[cell_region]

            if nuclear_in_cell.size == 0:
                continue

            # Otsu threshold within this cell
            try:
                threshold = threshold_otsu(nuclear_in_cell)
                nuclei_pixels = (nuclear_image > threshold) & cell_region
                nuclei_mask[nuclei_pixels] = track_id
            except ValueError:
                # Constant intensity - no nucleus detected
                pass

        return nuclei_mask
