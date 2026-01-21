"""Sequence extraction with segmentation and tracking."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import CellposeSegmenter, CellTracker
from ..core.io import create_zarr_store, write_global_metadata, write_sequence
from ..core.cell_source import CellFovSource
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
        merge_method: str = 'none',
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

    def extract(self, min_frames: int = 1) -> int:
        """Extract sequences to OME-Zarr.

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
        sequences_written = 0

        root = create_zarr_store(self.output_path, overwrite=True)
        write_global_metadata(
            root,
            cells_source=f"{type(self.source).__name__}",
            nuclei_channel=self.nuclei_channel,
            cell_channels=self.cell_channels,
            merge_method=self.merge_method,
        )

        for row in rows:
            if row.t0 < 0 or row.t1 < row.t0:
                continue

            n_frames = row.t1 - row.t0 + 1
            if n_frames < min_frames:
                continue

            timelapse = self.cropper.extract_timelapse(
                row.fov,
                row.cell,
                start_frame=row.t0,
                end_frame=row.t1 + 1,
                channels=None,
            )

            # Segment nuclei (always needed for tracking)
            nuclei_masks = self._segment_channel(timelapse, self.nuclei_channel)

            # Try to load cell masks from cache, otherwise segment
            cell_masks = self._load_cached_masks(row.fov, row.cell, row.t0, row.t1)
            if cell_masks is None:
                # Cache miss - segment cells
                if self.merge_method == 'none':
                    cell_masks = self._segment_channel(timelapse, self.cell_channels[0])
                else:
                    cell_masks = self._segment_cells_merged(timelapse)

            tracker = CellTracker()
            tracking_maps = tracker.track_frames(nuclei_masks)
            tracked_nuclei_masks = [
                tracker.get_tracked_mask(mask, track_map)
                for mask, track_map in zip(nuclei_masks, tracking_maps, strict=False)
            ]

            tracked_cell_masks = [
                self._map_cells_to_tracks(cell_mask, tracked_nuclei_mask)
                for cell_mask, tracked_nuclei_mask in zip(cell_masks, tracked_nuclei_masks, strict=False)
            ]

            channels = [f"channel_{i}" for i in range(timelapse.shape[1])]
            bbox = np.array([row.x, row.y, row.w, row.h], dtype=np.int32)

            write_sequence(
                root,
                fov_idx=row.fov,
                cell_idx=row.cell,
                seq_idx=0,
                data=timelapse,
                nuclei_masks=np.stack(tracked_nuclei_masks),
                cell_masks=np.stack(tracked_cell_masks),
                channels=channels,
                t0=row.t0,
                t1=row.t1,
                bbox=bbox,
            )
            sequences_written += 1

        logger.info(f"Saved {sequences_written} sequences to {self.output_path}")
        return sequences_written

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

    @staticmethod
    def _map_cells_to_tracks(cell_mask: np.ndarray, tracked_nuclei_mask: np.ndarray) -> np.ndarray:
        """Assign cell labels to tracked nuclei IDs by overlap."""
        tracked_cells = np.zeros_like(cell_mask, dtype=np.int32)
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0]

        for cell_label in cell_labels:
            overlap_ids = tracked_nuclei_mask[cell_mask == cell_label]
            overlap_ids = overlap_ids[overlap_ids != 0]
            if overlap_ids.size == 0:
                continue
            unique_ids, counts = np.unique(overlap_ids, return_counts=True)
            track_id = int(unique_ids[np.argmax(counts)])
            tracked_cells[cell_mask == cell_label] = track_id

        return tracked_cells
