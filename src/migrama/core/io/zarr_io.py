"""Zarr I/O utilities for migrama.

This module provides functions for writing microscopy data in Zarr format.
"""

import shutil
from pathlib import Path

import numpy as np
import zarr

MIGRAMA_VERSION = "0.1.0"


def create_zarr_store(path: str | Path, overwrite: bool = False) -> zarr.Group:
    """Create a new Zarr store for migrama data.

    Parameters
    ----------
    path : str | Path
        Path to the zarr store directory
    overwrite : bool
        Whether to overwrite existing store

    Returns
    -------
    zarr.Group
        Root group of the zarr store
    """
    path = Path(path)
    if overwrite and path.exists():
        shutil.rmtree(path)
    return zarr.open_group(str(path), mode="w")


def write_global_metadata(
    root: zarr.Group,
    cells_source: str,
    nuclei_channel: int,
    cell_channels: list[int] | None,
    merge_method: str | None,
) -> None:
    """Write global metadata to zarr root.

    Parameters
    ----------
    root : zarr.Group
        Root group of the zarr store
    cells_source : str
        Source type name (e.g., "Nd2CellFovSource")
    nuclei_channel : int
        Channel index for nuclei
    cell_channels : list[int] | None
        Channel indices for cell segmentation
    merge_method : str | None
        Merge method: 'add' or 'multiply', or None for no merging
    """
    root.attrs["migrama"] = {
        "version": MIGRAMA_VERSION,
        "cells_source": cells_source,
        "nuclei_channel": nuclei_channel,
        "cell_channels": cell_channels,
        "merge_method": merge_method,
    }


def write_sequence(
    root: zarr.Group,
    fov_idx: int,
    cell_idx: int,
    data: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: np.ndarray,
    channels: list[str],
    t0: int,
    t1: int,
    bbox: np.ndarray,
) -> None:
    """Write a sequence to the zarr store.

    Parameters
    ----------
    root : zarr.Group
        Root group of the zarr store
    fov_idx : int
        Field of view index
    cell_idx : int
        Cell/pattern index
    data : np.ndarray
        Image data with shape (T, C, H, W)
    nuclei_masks : np.ndarray
        Nuclei segmentation masks with shape (T, H, W)
    cell_masks : np.ndarray
        Cell segmentation masks with shape (T, H, W)
    channels : list[str]
        Channel names
    t0 : int
        Start frame index
    t1 : int
        End frame index
    bbox : np.ndarray
        Bounding box [x, y, w, h]
    """
    # Create hierarchy: fov/{idx}/cell/{idx}/ for image data
    fov_group = root.require_group(f"fov/{fov_idx}")
    cell_group = fov_group.require_group(f"cell/{cell_idx}")

    # Determine chunks - optimize for frame-by-frame access
    t, c, h, w = data.shape
    data_chunks = (1, 1, min(256, h), min(256, w))
    mask_chunks = (1, min(256, h), min(256, w))

    # Write image data
    arr = cell_group.create_array(
        "data",
        shape=data.shape,
        chunks=data_chunks,
        dtype=data.dtype,
        overwrite=True,
    )
    arr[:] = data

    # Set sequence metadata
    cell_group.attrs["migrama"] = {
        "channels": channels,
        "t0": int(t0),
        "t1": int(t1),
        "bbox": [int(x) for x in bbox],
    }

    # Write masks: cell_group/mask/{nucleus, cell}
    mask_group = cell_group.require_group("mask")

    _write_label_array(mask_group, "nucleus", nuclei_masks, mask_chunks)
    _write_label_array(mask_group, "cell", cell_masks, mask_chunks)


def _write_label_array(
    parent_group: zarr.Group,
    name: str,
    data: np.ndarray,
    chunks: tuple[int, ...],
) -> None:
    """Write a label array directly under the parent group.

    Parameters
    ----------
    parent_group : zarr.Group
        Parent group (e.g., mask/)
    name : str
        Array name (e.g., "nucleus", "cell")
    data : np.ndarray
        Label data with shape (T, H, W)
    chunks : tuple
        Chunk sizes
    """
    label_data = data.astype(np.int32)
    arr = parent_group.create_array(
        name,
        shape=label_data.shape,
        chunks=chunks,
        dtype=np.int32,
        overwrite=True,
    )
    arr[:] = label_data
