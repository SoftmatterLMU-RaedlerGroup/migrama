"""Convert TIFF files to OME-Zarr with segmentation and tracking."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from ..core import CellposeSegmenter, CellTracker
from ..core.io import create_zarr_store, write_global_metadata, write_sequence
from ..core.progress import ProgressEmitter

logger = logging.getLogger(__name__)


class Converter:
    """Convert TIFF files to OME-Zarr with segmentation and tracking."""

    def __init__(
        self,
        input_folder: str,
        output_path: str,
        nuclei_channel: int = 0,
        cell_channels: list[int] | None = None,
        merge_method: str = 'none',
    ) -> None:
        """Initialize converter.

        Parameters
        ----------
        input_folder : str
            Path to folder containing TIFF files
        output_path : str
            Output Zarr store path
        nuclei_channel : int
            Channel index for nuclei
        cell_channels : list[int] | None
            Channel indices for cell channels to merge. If None and merge_method != 'none', uses all channels except nuclei_channel.
        merge_method : str
            Merge method: 'add', 'multiply', or 'none'
        """
        self.input_folder = Path(input_folder).resolve()
        self.output_path = Path(output_path).resolve()
        self.nuclei_channel = nuclei_channel
        self.cell_channels = cell_channels
        self.merge_method = merge_method

        self.segmenter = CellposeSegmenter()
        self._progress = ProgressEmitter()

    @property
    def progress(self):
        """Get the progress signal for connecting callbacks."""
        return self._progress.progress

    def convert(self, min_frames: int = 1, on_file_start: Callable | None = None) -> int:
        """Convert TIFF files to OME-Zarr.

        Parameters
        ----------
        min_frames : int
            Minimum frames required to process a sequence
        on_file_start : Callable | None
            Optional callback called with (filename) before processing each file

        Returns
        -------
        int
            Number of sequences written
        """
        tiff_paths = sorted(self.input_folder.glob("*.tif*"))
        if not tiff_paths:
            raise FileNotFoundError(f"No TIFF files found in {self.input_folder}")

        sequences_written = 0

        root = create_zarr_store(self.output_path, overwrite=True)
        write_global_metadata(
            root,
            cells_source=str(self.input_folder),
            nuclei_channel=self.nuclei_channel,
            cell_channels=self.cell_channels,
            merge_method=self.merge_method,
        )

        for cell_idx, tiff_path in enumerate(tiff_paths):
            timelapse = self._load_timelapse(tiff_path)

            n_frames = timelapse.shape[0]
            if n_frames < min_frames:
                logger.info(f"Skipping {tiff_path.name}: only {n_frames} frames")
                continue

            if on_file_start:
                on_file_start(tiff_path.name)

            cell_masks = self._segment_timelapse(timelapse, tiff_path.name)

            tracker = CellTracker()
            n_frames = len(cell_masks)
            self._progress.emit("tracking", "frame", 0, n_frames)
            tracking_maps = tracker.track_frames(cell_masks)
            self._progress.emit("tracking", "frame", n_frames, n_frames)
            tracked_cell_masks = [
                tracker.get_tracked_mask(mask, track_map)
                for mask, track_map in zip(cell_masks, tracking_maps, strict=False)
            ]

            nuclei_masks = self._build_nuclei_masks(timelapse, tracked_cell_masks, tracking_maps, tiff_path.name)

            n_channels = timelapse.shape[1]
            channels = [f"channel_{i}" for i in range(n_channels)]
            dummy_bbox = np.array([-1, -1, -1, -1], dtype=np.int32)

            write_sequence(
                root,
                fov_idx=0,
                cell_idx=cell_idx,
                seq_idx=0,
                data=timelapse,
                nuclei_masks=np.stack(nuclei_masks),
                cell_masks=np.stack(tracked_cell_masks),
                channels=channels,
                t0=-1,
                t1=-1,
                bbox=dummy_bbox,
            )
            sequences_written += 1
            logger.info(f"Processed {tiff_path.name} -> cell_{cell_idx}")

        logger.info(f"Saved {sequences_written} sequences to {self.output_path}")
        return sequences_written

    def _load_timelapse(self, tiff_path: Path) -> np.ndarray:
        """Load a TIFF stack as timelapse array (t, c, y, x)."""
        with tifffile.TiffFile(tiff_path) as tif:
            data = tif.asarray()

        if data.ndim == 2:
            data = data[np.newaxis, np.newaxis, ...]
        elif data.ndim == 3:
            data = np.expand_dims(data, axis=1)
        elif data.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}, expected 2-4 dimensions")

        if data.shape[1] < 1:
            raise ValueError(f"TIFF must have at least 1 channel, got {data.shape[1]}")

        logger.debug(f"Loaded {tiff_path.name}: shape {data.shape}")
        return data

    def _segment_timelapse(self, timelapse: np.ndarray, file_name: str) -> list[np.ndarray]:
        """Segment timelapse using Cellpose with optional channel merging."""
        n_frames = timelapse.shape[0]
        self._progress.emit("segmentation", "frame", 0, n_frames)
        masks = []
        for frame_idx in range(n_frames):
            frame_data = timelapse[frame_idx]  # Shape: (C, H, W)

            # Determine cell_channels to use
            cell_channels_to_use = self.cell_channels
            if cell_channels_to_use is None and self.merge_method != 'none':
                # Use all channels except nuclei_channel
                n_channels = frame_data.shape[0]
                cell_channels_to_use = [i for i in range(n_channels) if i != self.nuclei_channel]

            # Transpose to (H, W, C) format expected by cellpose
            if frame_data.ndim == 3:
                frame_data_hwc = np.transpose(frame_data, (1, 2, 0))
            else:
                frame_data_hwc = frame_data

            # Segment with appropriate method
            result = self.segmenter.segment_image(
                frame_data_hwc,
                nuclei_channel=self.nuclei_channel if self.merge_method != 'none' else None,
                cell_channels=cell_channels_to_use if self.merge_method != 'none' else None,
                merge_method=self.merge_method,
            )
            masks.append(result["masks"])
            self._progress.emit("segmentation", "frame", frame_idx + 1, n_frames)
        return masks

    def _build_nuclei_masks(
        self,
        timelapse: np.ndarray,
        tracked_cell_masks: list[np.ndarray],
        tracking_maps: list[dict[int, int]],
        file_name: str,
    ) -> list[np.ndarray]:
        """Build nuclei masks by Otsu thresholding nuclei channel inside each cell."""
        n_frames = len(tracked_cell_masks)
        self._progress.emit("nuclei", "frame", 0, n_frames)
        nuclei_masks = []
        for frame_idx, (cell_mask, track_map) in enumerate(zip(tracked_cell_masks, tracking_maps, strict=False)):
            nuclei_mask = np.zeros_like(cell_mask, dtype=np.int32)
            nuclei_channel_img = timelapse[frame_idx, self.nuclei_channel]

            for track_id in track_map.values():
                cell_pixels = cell_mask == track_id
                if not np.any(cell_pixels):
                    continue

                cell_intensities = nuclei_channel_img[cell_pixels]
                if cell_intensities.size == 0:
                    continue

                try:
                    thresh = threshold_otsu(cell_intensities)
                except ValueError:
                    thresh = None

                if thresh is not None:
                    nuclei_pixels = cell_pixels & (nuclei_channel_img > thresh)
                    nuclei_mask[nuclei_pixels] = track_id

            nuclei_masks.append(nuclei_mask)
            self._progress.emit("nuclei", "frame", frame_idx + 1, n_frames)
        return nuclei_masks
