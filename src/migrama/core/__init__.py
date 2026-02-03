"""Migrama core: Shared utilities and interfaces for migrama modules."""

from .cell_source import CellFovSource, Nd2CellFovSource, TiffCellFovSource
from .network import CellGrapher
from .pattern import (
    BoundingBox,
    CellCropper,
    DetectorParameters,
    PatternDetector,
    PatternRecord,
    load_bboxes_csv,
)
from .segmentation import CellposeCounter, CellposeSegmenter, CountResult, WatershedCounter
from .tracking import CellTracker
from .voronoi import centroids_from_mask, generate_voronoi_labels

__all__ = [
    # Cell data sources
    "CellFovSource",
    "Nd2CellFovSource",
    "TiffCellFovSource",
    # Pattern detection (pattern.nd2 -> CSV)
    "PatternDetector",
    "DetectorParameters",
    "PatternRecord",
    # Cell cropping (cells.nd2 + CSV -> crops)
    "CellCropper",
    "BoundingBox",
    "load_bboxes_csv",
    # Segmentation
    "CellposeCounter",
    "CellposeSegmenter",
    "CountResult",
    "WatershedCounter",
    # Tracking
    "CellTracker",
    # Graph
    "CellGrapher",
    # Voronoi
    "centroids_from_mask",
    "generate_voronoi_labels",
]
