"""Zarr data loader for graph analysis."""

import logging
from pathlib import Path

import yaml
import zarr

logger = logging.getLogger(__name__)


class ZarrSegmentationLoader:
    """Load segmentation masks from migrama Zarr stores."""

    def __init__(self) -> None:
        """Initialize the loader."""
        return

    def load_cell_filter_data(
        self,
        zarr_path: str,
        fov_idx: int,
        pattern_idx: int,
        yaml_path: str | None = None,
    ) -> dict[str, object]:
        """Load extracted data and segmentation masks from Zarr.

        Parameters
        ----------
        zarr_path : str
            Path to the zarr store
        fov_idx : int
            Field of view index
        pattern_idx : int
            Pattern/cell index
        yaml_path : str | None
            Optional path to YAML metadata file

        Returns
        -------
        dict
            Dictionary with keys: data, segmentation_masks, nuclei_masks,
            metadata, channels, sequence_metadata
        """
        root = zarr.open(zarr_path, mode="r")
        cell_path = f"fov/{fov_idx}/cell/{pattern_idx}"

        if cell_path not in root:
            raise ValueError(f"Sequence not found: {cell_path}")

        cell_group = root[cell_path]

        # Load image data
        data = cell_group["data"][...]

        # Load masks from mask group
        mask_group = cell_group["mask"]
        segmentation_masks = mask_group["cell"][...]

        # Load nuclei masks if present
        nuclei_masks = None
        if "nucleus" in mask_group:
            nuclei_masks = mask_group["nucleus"][...]

        # Load migrama-specific metadata
        migrama_meta = cell_group.attrs.get("migrama", {})
        channels = migrama_meta.get("channels")
        metadata = {
            "t0": migrama_meta.get("t0", -1),
            "t1": migrama_meta.get("t1", -1),
            "bbox": migrama_meta.get("bbox", None),
        }

        # Load YAML metadata if available
        if yaml_path is None:
            yaml_path = str(Path(zarr_path).with_suffix(".yaml"))

        yaml_metadata = None
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path) as handle:
                yaml_metadata = yaml.safe_load(handle)

        return {
            "data": data,
            "segmentation_masks": segmentation_masks,
            "nuclei_masks": nuclei_masks,
            "metadata": yaml_metadata,
            "channels": channels,
            "sequence_metadata": metadata,
        }

    def list_sequences(self, zarr_path: str) -> list[dict[str, int]]:
        """List available sequences in the Zarr store.

        Parameters
        ----------
        zarr_path : str
            Path to the zarr store

        Returns
        -------
        list[dict[str, int]]
            List of dictionaries with fov_idx, pattern_idx
        """
        sequences: list[dict[str, int]] = []
        root = zarr.open(zarr_path, mode="r")

        # New structure: fov/{i}/cell/{j}/
        if "fov" not in root:
            return sequences

        fov_group = root["fov"]
        for fov_idx_key in fov_group.keys():
            if not fov_idx_key.isdigit():
                continue
            fov_idx = int(fov_idx_key)

            fov_subgroup = fov_group[fov_idx_key]
            if "cell" not in fov_subgroup:
                continue

            cell_group = fov_subgroup["cell"]
            for cell_idx_key in cell_group.keys():
                if not cell_idx_key.isdigit():
                    continue
                cell_idx = int(cell_idx_key)

                sequences.append(
                    {
                        "fov_idx": fov_idx,
                        "pattern_idx": cell_idx,
                    }
                )

        return sequences

    def validate_cell_filter_output(self, zarr_path: str, yaml_path: str | None = None) -> bool:
        """Validate that the Zarr store contains valid extracted data.

        Parameters
        ----------
        zarr_path : str
            Path to the zarr store
        yaml_path : str | None
            Optional path to YAML metadata file

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        try:
            if not Path(zarr_path).exists():
                logger.error(f"Zarr store not found: {zarr_path}")
                return False

            sequences = self.list_sequences(zarr_path)
            if not sequences:
                logger.error("No sequences found in Zarr store")
                return False

            first_seq = sequences[0]
            data = self.load_cell_filter_data(
                zarr_path,
                first_seq["fov_idx"],
                first_seq["pattern_idx"],
                yaml_path,
            )["data"]

            if data.ndim != 4:
                logger.error(f"Expected 4D data, got shape {data.shape}")
                return False

            if data.shape[1] < 1:
                logger.error("Expected at least one image channel")
                return False

            return True
        except Exception as exc:
            logger.error(f"Zarr store validation failed: {exc}")
            return False
