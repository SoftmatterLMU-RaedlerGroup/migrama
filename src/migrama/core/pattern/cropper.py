"""
Cell cropper - crops cell regions from cell data sources using bounding boxes from CSV.

This module works with CellFovSource (ND2 or per-FOV TIFFs) + CSV bounding boxes.
Input: CellFovSource + patterns.csv (from PatternDetector)
Output: cropped cell regions for analysis
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..cell_source import CellFovSource

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for a pattern."""

    cell: int
    fov: int
    x: int
    y: int
    w: int
    h: int


def load_bboxes_csv(csv_path: str | Path) -> dict[int, list[BoundingBox]]:
    """Load bounding boxes from CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to CSV file with columns: cell, fov, x, y, w, h

    Returns
    -------
    dict[int, list[BoundingBox]]
        Mapping of fov_index -> list of bounding boxes
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    bboxes_by_fov: dict[int, list[BoundingBox]] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bbox = BoundingBox(
                cell=int(row["cell"]),
                fov=int(row["fov"]),
                x=int(row["x"]),
                y=int(row["y"]),
                w=int(row["w"]),
                h=int(row["h"]),
            )
            if bbox.fov not in bboxes_by_fov:
                bboxes_by_fov[bbox.fov] = []
            bboxes_by_fov[bbox.fov].append(bbox)

    for fov in bboxes_by_fov:
        bboxes_by_fov[fov].sort(key=lambda b: b.cell)

    logger.info(f"Loaded {sum(len(v) for v in bboxes_by_fov.values())} bboxes from {len(bboxes_by_fov)} FOVs")
    return bboxes_by_fov


class CellCropper:
    """Crop cell regions from cell data sources using bounding boxes.

    This class works with CellFovSource (ND2 or per-FOV TIFFs) and uses
    pre-computed bounding boxes from a CSV file (output of PatternDetector).
    """

    def __init__(
        self,
        source: CellFovSource,
        bboxes_csv: str,
        nuclei_channel: int = 1,
    ) -> None:
        """Initialize cropper with cell source and bounding boxes.

        Parameters
        ----------
        source : CellFovSource
            Source of cell timelapse data (ND2 or TIFF)
        bboxes_csv : str
            Path to CSV file with bounding boxes (from PatternDetector)
        nuclei_channel : int
            Channel index for nuclei (default: 1)
        """
        self.source = source
        self.bboxes_csv = Path(bboxes_csv).resolve()
        self.nuclei_channel = nuclei_channel

        self.n_fovs = source.n_fovs
        self.n_frames = source.n_frames
        self.n_channels = source.n_channels
        self.height = source.height
        self.width = source.width
        self.dtype = source.dtype

        self.bboxes_by_fov = load_bboxes_csv(self.bboxes_csv)

        if self.n_channels < 2:
            raise ValueError(f"Cells source must have at least 2 channels, got {self.n_channels}")

        logger.info(
            f"Initialized CellCropper: {self.n_fovs} FOVs, {self.n_frames} frames, "
            f"{self.n_channels} channels, {sum(len(v) for v in self.bboxes_by_fov.values())} patterns"
        )

    def get_bboxes(self, fov: int) -> list[BoundingBox]:
        """Get bounding boxes for a FOV.

        Parameters
        ----------
        fov : int
            Field of view index

        Returns
        -------
        list[BoundingBox]
            Bounding boxes for this FOV
        """
        return self.bboxes_by_fov.get(fov, [])

    def n_patterns(self, fov: int) -> int:
        """Get number of patterns in a FOV."""
        return len(self.get_bboxes(fov))

    def extract(
        self,
        fov: int,
        cell: int,
        frames: int | tuple[int, int] | None = None,
        channels: list[int] | None = None,
    ) -> np.ndarray:
        """Extract cropped data for a pattern.

        Parameters
        ----------
        fov : int
            Field of view index
        cell : int
            Pattern/cell index within FOV
        frames : int | tuple[int, int] | None
            Single frame index, (start, end) range, or None for all frames
        channels : list[int] | None
            Channel indices to extract, None for all

        Returns
        -------
        np.ndarray
            Single frame: (C, H, W), multiple frames: (T, C, H, W)
        """
        bboxes = self.get_bboxes(fov)
        if cell >= len(bboxes):
            raise ValueError(f"Cell {cell} not found in FOV {fov} (has {len(bboxes)} patterns)")

        bbox = bboxes[cell]
        channels = channels or list(range(self.n_channels))

        # Determine frame range
        if frames is None:
            start, end = 0, self.n_frames
        elif isinstance(frames, int):
            # Single frame - return (C, H, W)
            stack = self.source.get_frame(fov, frames)
            return stack[channels, bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
        else:
            start, end = frames

        # Multiple frames - return (T, C, H, W)
        result = []
        for t in range(start, end):
            stack = self.source.get_frame(fov, t)
            cropped = stack[channels, bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
            result.append(cropped)

        return np.stack(result)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 uint8."""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
