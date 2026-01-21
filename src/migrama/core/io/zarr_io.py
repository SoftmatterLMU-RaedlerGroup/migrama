"""OME-Zarr I/O utilities for migrama.

This module provides functions for writing microscopy data in OME-NGFF (Zarr) format,
following the v0.4 specification for interoperability with napari, OMERO, and other tools.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr

MIGRAMA_VERSION = "0.1.0"
OME_NGFF_VERSION = "0.4"


def create_zarr_store(path: str | Path, overwrite: bool = False) -> zarr.Group:
    """Create a new OME-Zarr store for migrama data.

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
    root = zarr.open_group(str(path), mode="w")
    return root


def write_global_metadata(
    root: zarr.Group,
    cells_source: str,
    nuclei_channel: int,
    cell_channels: list[int] | None,
    merge_method: str,
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
    merge_method : str
        Merge method: 'add', 'multiply', or 'none'
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
    seq_idx: int,
    data: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: np.ndarray,
    channels: list[str],
    t0: int,
    t1: int,
    bbox: np.ndarray,
) -> None:
    """Write a sequence to the zarr store with OME-NGFF compliance.

    Parameters
    ----------
    root : zarr.Group
        Root group of the zarr store
    fov_idx : int
        Field of view index
    cell_idx : int
        Cell/pattern index
    seq_idx : int
        Sequence index
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
    # Create hierarchy: fov_{idx}/cell_{idx}/{seq_idx}/ for image data
    fov_group = root.require_group(f"fov_{fov_idx}")
    cell_group = fov_group.require_group(f"cell_{cell_idx}")
    seq_group = cell_group.require_group(str(seq_idx))

    # Determine chunks - optimize for frame-by-frame access
    t, c, h, w = data.shape
    data_chunks = (1, 1, min(256, h), min(256, w))
    mask_chunks = (1, min(256, h), min(256, w))

    # Write image data directly (no nested "0" for single resolution)
    arr = seq_group.create_array(
        "data",
        shape=data.shape,
        chunks=data_chunks,
        dtype=data.dtype,
        overwrite=True,
    )
    arr[:] = data

    # Set OME-NGFF multiscales metadata
    seq_group.attrs["multiscales"] = [
        {
            "version": OME_NGFF_VERSION,
            "name": f"fov_{fov_idx}_cell_{cell_idx}_seq_{seq_idx}",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "data"}],
        }
    ]

    # Set channel metadata (OMERO style for napari compatibility)
    seq_group.attrs["omero"] = {"channels": [{"label": ch, "active": True} for ch in channels]}

    # Set migrama-specific metadata
    seq_group.attrs["migrama"] = {
        "t0": int(t0),
        "t1": int(t1),
        "bbox": [int(x) for x in bbox],
    }

    # Write labels: cell_group/labels/{seq_idx}/{nuclei_masks, cell_masks}
    labels_group = cell_group.require_group("labels")
    seq_labels_group = labels_group.require_group(str(seq_idx))
    seq_labels_group.attrs["labels"] = ["nuclei_masks", "cell_masks"]

    _write_label_array(seq_labels_group, "nuclei_masks", nuclei_masks, mask_chunks)
    _write_label_array(seq_labels_group, "cell_masks", cell_masks, mask_chunks)


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
        Parent group (e.g., labels/{seq_idx}/)
    name : str
        Array name (e.g., "nuclei_masks", "cell_masks")
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
