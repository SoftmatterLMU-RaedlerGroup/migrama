"""OME-Zarr data loader for graph analysis."""

import logging
from pathlib import Path

import yaml
import zarr

logger = logging.getLogger(__name__)


class ZarrSegmentationLoader:
    """Load segmentation masks from migrama OME-Zarr stores."""

    def __init__(self) -> None:
        """Initialize the loader."""
        return

    def load_cell_filter_data(
        self,
        zarr_path: str,
        fov_idx: int,
        pattern_idx: int,
        seq_idx: int,
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
        seq_idx : int
            Sequence index
        yaml_path : str | None
            Optional path to YAML metadata file

        Returns
        -------
        dict
            Dictionary with keys: data, segmentation_masks, nuclei_masks,
            metadata, channels, sequence_metadata
        """
        root = zarr.open(zarr_path, mode="r")
        seq_path = f"fov_{fov_idx}/cell_{pattern_idx}/{seq_idx}"
        labels_path = f"fov_{fov_idx}/cell_{pattern_idx}/labels/{seq_idx}"

        if seq_path not in root:
            raise ValueError(f"Sequence not found: {seq_path}")

        seq_group = root[seq_path]

        # Load image data
        data = seq_group["data"][...]

        # Load cell masks from labels group
        labels_group = root[labels_path]
        segmentation_masks = labels_group["cell_masks"][...]

        # Load nuclei masks if present
        nuclei_masks = None
        if "nuclei_masks" in labels_group:
            nuclei_masks = labels_group["nuclei_masks"][...]

        # Load channel names from OMERO metadata
        channels = None
        if "omero" in seq_group.attrs:
            omero = seq_group.attrs["omero"]
            if "channels" in omero:
                channels = [ch.get("label", f"channel_{i}") for i, ch in enumerate(omero["channels"])]

        # Load migrama-specific metadata
        migrama_meta = seq_group.attrs.get("migrama", {})
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
            List of dictionaries with fov_idx, pattern_idx, seq_idx
        """
        sequences: list[dict[str, int]] = []
        root = zarr.open(zarr_path, mode="r")

        for fov_key in root.keys():
            if not fov_key.startswith("fov_"):
                continue
            fov_idx = int(fov_key.split("_")[1])

            for cell_key in root[fov_key].keys():
                if not cell_key.startswith("cell_"):
                    continue
                cell_idx = int(cell_key.split("_")[1])

                # Sequence indices are now numeric keys (0, 1, 2, ...)
                for seq_key in root[fov_key][cell_key].keys():
                    # Skip non-numeric keys like "labels"
                    if not seq_key.isdigit():
                        continue
                    seq_idx = int(seq_key)

                    sequences.append(
                        {
                            "fov_idx": fov_idx,
                            "pattern_idx": cell_idx,
                            "seq_idx": seq_idx,
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
                first_seq["seq_idx"],
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
