"""
Pattern detector - detects micropatterns from pattern images.

This module is completely independent of cell data.
Input: PatternFovSource (ND2, TIFF, etc.)
Output: CSV with columns (cell, fov, x, y, w, h)
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .source import PatternFovSource

logger = logging.getLogger(__name__)


@dataclass
class DetectorParameters:
    """Parameters for pattern detection."""

    gaussian_blur_size: tuple[int, int] = (11, 11)
    morph_dilate_size: tuple[int, int] = (5, 5)
    edge_tolerance: int = 5
    min_area_ratio: float = 1 / 3  # Discard contours with area < median * min_area_ratio


@dataclass
class PatternRecord:
    """A single pattern detection record."""

    cell: int
    fov: int
    x: int
    y: int
    w: int
    h: int


class PatternDetector:
    """Detect micropatterns from pattern images using a PatternFovSource.

    This class works with any PatternFovSource implementation (ND2, TIFF, etc.)
    and outputs bounding box information as CSV. It has no dependency on
    cell/timelapse data.
    """

    def __init__(self, source: PatternFovSource, parameters: DetectorParameters | None = None) -> None:
        """Initialize detector with a pattern source.

        Parameters
        ----------
        source : PatternFovSource
            Source of pattern images (ND2, TIFF, etc.)
        parameters : DetectorParameters | None
            Detection parameters (uses defaults if None)
        """
        self.source = source
        self.parameters = parameters or DetectorParameters()
        self.n_fovs = source.n_fovs

        logger.info(f"Initialized PatternDetector with {self.n_fovs} FOVs")

    def _normalize_pct(self, image: np.ndarray, low: int = 10, high: int = 90) -> np.ndarray:
        """Normalize image using percentile stretch."""
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")

        nonzero = image[image > 0]
        if len(nonzero) == 0:
            return np.zeros_like(image, dtype=np.uint8)

        pct_low = np.percentile(nonzero, low)
        pct_high = np.percentile(nonzero, high)
        clipped = np.clip(image, pct_low, pct_high)
        normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _find_contours(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Find contours using thresholding."""
        blur = cv2.GaussianBlur(image, self.parameters.gaussian_blur_size, 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones(self.parameters.morph_dilate_size, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours), thresh

    def _filter_contours_by_area(self, contours: list[np.ndarray]) -> list[np.ndarray]:
        """Filter contours by area using median-based threshold.

        Removes contours with area < median_area * min_area_ratio.
        """
        if not contours:
            return []

        areas = np.array([cv2.contourArea(c) for c in contours])
        median_area = np.median(areas)
        min_area = median_area * self.parameters.min_area_ratio

        return [c for c, a in zip(contours, areas, strict=False) if a >= min_area]

    def _filter_by_edge(self, contours: list[np.ndarray], shape: tuple[int, int]) -> list[np.ndarray]:
        """Remove contours too close to image edge."""
        tol = self.parameters.edge_tolerance
        h, w = shape
        kept = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if x >= tol and y >= tol and x + bw <= w - tol and y + bh <= h - tol:
                kept.append(c)
        return kept

    def _contours_to_bboxes(self, contours: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
        """Convert contours to sorted bounding boxes."""
        bboxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            center_y = y + h // 2
            center_x = x + w // 2
            bboxes.append((x, y, w, h, center_y, center_x))

        bboxes.sort(key=lambda b: (b[4], b[5]))
        return [(x, y, w, h) for x, y, w, h, _, _ in bboxes]

    def detect_fov(self, fov_idx: int) -> list[PatternRecord]:
        """Detect patterns in a single FOV.

        Parameters
        ----------
        fov_idx : int
            Field of view index

        Returns
        -------
        list[PatternRecord]
            List of detected patterns with bounding boxes
        """
        if fov_idx < 0 or fov_idx >= self.n_fovs:
            raise ValueError(f"FOV index {fov_idx} out of range (0-{self.n_fovs - 1})")

        pattern_img = None
        for current_fov_idx, frame in self.source.iter_fovs():
            if current_fov_idx == fov_idx:
                pattern_img = frame
                break

        if pattern_img is None:
            raise ValueError(f"FOV {fov_idx} not found in source")

        normalized = self._normalize_pct(pattern_img)
        contours, _ = self._find_contours(normalized)
        # Filter edge first, then area
        contours = self._filter_by_edge(contours, normalized.shape)
        contours = self._filter_contours_by_area(contours)
        bboxes = self._contours_to_bboxes(contours)

        records = []
        for cell_idx, (x, y, w, h) in enumerate(bboxes):
            records.append(PatternRecord(cell=cell_idx, fov=fov_idx, x=x, y=y, w=w, h=h))

        logger.debug(f"FOV {fov_idx}: detected {len(records)} patterns")
        return records

    def detect_all(self) -> list[PatternRecord]:
        """Detect patterns in all FOVs.

        Returns
        -------
        list[PatternRecord]
            All detected patterns across all FOVs
        """
        all_records = []
        for fov_idx in range(self.n_fovs):
            records = self.detect_fov(fov_idx)
            all_records.extend(records)

        logger.info(f"Detected {len(all_records)} patterns across {self.n_fovs} FOVs")
        return all_records

    def save_csv(self, records: list[PatternRecord], output_path: str | Path) -> None:
        """Save pattern records to CSV file.

        Parameters
        ----------
        records : list[PatternRecord]
            Pattern records to save
        output_path : str | Path
            Output CSV file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cell", "fov", "x", "y", "w", "h"])
            for r in records:
                writer.writerow([r.cell, r.fov, r.x, r.y, r.w, r.h])

        logger.info(f"Saved {len(records)} patterns to {output_path}")

    def detect_and_save(self, output_path: str | Path) -> list[PatternRecord]:
        """Detect all patterns and save to CSV.

        Parameters
        ----------
        output_path : str | Path
            Output CSV file path

        Returns
        -------
        list[PatternRecord]
            All detected patterns
        """
        records = self.detect_all()
        self.save_csv(records, output_path)
        return records
