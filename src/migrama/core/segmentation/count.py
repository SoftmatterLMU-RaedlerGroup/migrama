"""
Cell counting with segmentation mask support.

Provides counters that return both counts and segmentation masks.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CountResult:
    """Result from cell counting containing both counts and masks.

    Attributes
    ----------
    counts : list[int]
        Number of cells detected in each image
    masks : list[np.ndarray]
        Segmentation mask for each image (label image where each cell has unique ID)
    """

    counts: list[int]
    masks: list[np.ndarray]


class CellposeCounter:
    """Counter that uses Cellpose to detect and count nuclei.

    Wraps CellposeSegmenter to provide a simple counting API while
    also returning segmentation masks.
    """

    def __init__(
        self,
        nuclei_channel: int | None = None,
        cell_channels: list[int] | None = None,
        merge_method: str | None = None,
    ):
        """Initialize the Cellpose counter.

        Parameters
        ----------
        nuclei_channel : int | None
            Channel index for nuclear channel. If None and merge_method != 'none',
            uses first channel.
        cell_channels : list[int] | None
            Channel indices for cell channels to merge. If None and merge_method != 'none',
            uses all channels except nuclei_channel.
        merge_method : str
            Merge method: 'add', 'multiply', or 'none'.
        """
        from .segmentation import CellposeSegmenter

        self._segmenter = CellposeSegmenter()
        self._nuclei_channel = nuclei_channel
        self._cell_channels = cell_channels
        self._merge_method = merge_method

    def count_nuclei(
        self,
        images: np.ndarray | list[np.ndarray],
        min_size: int = 15,  # noqa: ARG002 - kept for API compatibility
    ) -> CountResult:
        """Count nuclei in one or more images using Cellpose.

        Parameters
        ----------
        images : np.ndarray or list[np.ndarray]
            Single image or list of images to count nuclei in.
            Can be 2D (single channel) or 3D (multi-channel) arrays.
        min_size : int
            Deprecated - kept for API compatibility but not used.
            Cellpose 4.x handles size filtering internally.

        Returns
        -------
        CountResult
            Result containing counts and segmentation masks
        """
        # Convert single image to list
        if isinstance(images, np.ndarray) and images.ndim <= 3:
            images = [images]

        counts = []
        masks = []

        for image in images:
            if image.size == 0:
                counts.append(0)
                masks.append(np.zeros((0, 0), dtype=np.int32))
                continue

            # Use segmenter with channel configuration
            result = self._segmenter.segment_image(
                image,
                nuclei_channel=self._nuclei_channel,
                cell_channels=self._cell_channels,
                merge_method=self._merge_method,
            )
            mask = result['masks']

            # Count unique labels (excluding background 0)
            count = len(np.unique(mask)) - 1 if mask.size > 0 else 0
            counts.append(max(0, count))
            masks.append(mask)

        return CountResult(counts=counts, masks=masks)


class WatershedCounter:
    """Fast counter using classical watershed segmentation for nuclei.

    Much faster than Cellpose for simple nuclear counting tasks.
    Best suited for well-separated, roughly circular nuclei.
    """

    def __init__(self, sigma: float = 2.0):
        """Initialize the watershed counter.

        Parameters
        ----------
        sigma : float
            Gaussian blur sigma for noise reduction. Larger values smooth
            more aggressively, reducing over-segmentation but potentially
            merging close nuclei.
        """
        self.sigma = sigma

    def count_nuclei(
        self, images: np.ndarray | list[np.ndarray], min_size: int = 15
    ) -> CountResult:
        """Count nuclei in one or more images using watershed segmentation.

        Parameters
        ----------
        images : np.ndarray or list[np.ndarray]
            Single image or list of images to count nuclei in
        min_size : int
            Minimum distance between nuclei centers (used for peak detection)

        Returns
        -------
        CountResult
            Result containing counts and segmentation masks
        """
        from scipy import ndimage
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian, threshold_otsu
        from skimage.segmentation import watershed

        if isinstance(images, np.ndarray) and images.ndim == 2:
            images = [images]

        counts = []
        masks = []

        for image in images:
            if image.size == 0:
                counts.append(0)
                masks.append(np.zeros((0, 0), dtype=np.int32))
                continue

            # Normalize to 0-1 range
            img = image.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Gaussian blur to reduce noise
            blurred = gaussian(img, sigma=self.sigma)

            # Otsu threshold to get binary mask
            try:
                thresh = threshold_otsu(blurred)
                binary = blurred > thresh
            except ValueError:
                # Image is constant, no nuclei
                counts.append(0)
                masks.append(np.zeros(image.shape, dtype=np.int32))
                continue

            # Distance transform - peaks are at nuclei centers
            distance = ndimage.distance_transform_edt(binary)

            # Find local maxima as markers
            coords = peak_local_max(
                distance,
                min_distance=min_size,
                labels=binary,
            )

            if len(coords) == 0:
                counts.append(0)
                masks.append(np.zeros(image.shape, dtype=np.int32))
                continue

            # Create marker image
            markers = np.zeros(distance.shape, dtype=np.int32)
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

            # Watershed
            labels = watershed(-distance, markers, mask=binary)

            # Count unique labels (excluding background 0)
            count = len(np.unique(labels)) - 1
            counts.append(max(0, count))
            masks.append(labels)

        return CountResult(counts=counts, masks=masks)
