"""Tests for OME-Zarr I/O functionality."""

import numpy as np
import pytest
import zarr

from migrama.core.io.zarr_io import (
    create_zarr_store,
    write_global_metadata,
    write_sequence,
)
from migrama.graph.zarr_loader import ZarrSegmentationLoader


@pytest.fixture
def sample_data():
    """Generate sample timelapse data for testing."""
    np.random.seed(42)
    return {
        "data": np.random.randint(0, 65535, (10, 2, 64, 64), dtype=np.uint16),
        "nuclei_masks": np.random.randint(0, 5, (10, 64, 64), dtype=np.int32),
        "cell_masks": np.random.randint(0, 5, (10, 64, 64), dtype=np.int32),
        "channels": ["channel_0", "channel_1"],
        "t0": 5,
        "t1": 14,
        "bbox": np.array([10, 20, 64, 64], dtype=np.int32),
    }


class TestZarrIO:
    """Tests for zarr_io module."""

    def test_create_zarr_store(self, tmp_path):
        """Test zarr store creation."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)

        assert zarr_path.exists()
        assert isinstance(root, zarr.Group)

    def test_write_global_metadata(self, tmp_path):
        """Test writing global metadata."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)

        write_global_metadata(
            root,
            cells_source="TestSource",
            nuclei_channel=1,
            cell_channels=[0, 2],
            merge_method="add",
        )

        assert "migrama" in root.attrs
        meta = root.attrs["migrama"]
        assert meta["cells_source"] == "TestSource"
        assert meta["nuclei_channel"] == 1
        assert meta["cell_channels"] == [0, 2]
        assert meta["merge_method"] == "add"

    def test_write_sequence(self, tmp_path, sample_data):
        """Test writing a sequence with all data."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)
        write_global_metadata(root, "TestSource", 0, [1], "none")

        write_sequence(
            root,
            fov_idx=0,
            cell_idx=0,
            seq_idx=0,
            data=sample_data["data"],
            nuclei_masks=sample_data["nuclei_masks"],
            cell_masks=sample_data["cell_masks"],
            channels=sample_data["channels"],
            t0=sample_data["t0"],
            t1=sample_data["t1"],
            bbox=sample_data["bbox"],
        )

        # Verify structure: fov_0/cell_0/0/ for image, labels/0/ for masks
        assert "fov_0" in root
        assert "cell_0" in root["fov_0"]
        assert "0" in root["fov_0/cell_0"]  # Sequence index directly

        seq = root["fov_0/cell_0/0"]
        assert "data" in seq  # Image data array

        # Labels are at cell level: cell_0/labels/0/{nuclei_masks, cell_masks}
        assert "labels" in root["fov_0/cell_0"]
        labels = root["fov_0/cell_0/labels"]
        assert "0" in labels  # Sequence index
        assert "nuclei_masks" in labels["0"]
        assert "cell_masks" in labels["0"]

    def test_ome_ngff_metadata(self, tmp_path, sample_data):
        """Test OME-NGFF metadata compliance."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)
        write_global_metadata(root, "TestSource", 0, [1], "none")

        write_sequence(
            root,
            fov_idx=0,
            cell_idx=0,
            seq_idx=0,
            data=sample_data["data"],
            nuclei_masks=sample_data["nuclei_masks"],
            cell_masks=sample_data["cell_masks"],
            channels=sample_data["channels"],
            t0=sample_data["t0"],
            t1=sample_data["t1"],
            bbox=sample_data["bbox"],
        )

        seq = root["fov_0/cell_0/0"]

        # Check multiscales metadata
        assert "multiscales" in seq.attrs
        multiscales = seq.attrs["multiscales"]
        assert len(multiscales) == 1
        assert multiscales[0]["version"] == "0.4"
        assert len(multiscales[0]["axes"]) == 4

        # Check omero channel metadata
        assert "omero" in seq.attrs
        omero = seq.attrs["omero"]
        assert "channels" in omero
        assert len(omero["channels"]) == 2
        assert omero["channels"][0]["label"] == "channel_0"

        # Check migrama metadata
        assert "migrama" in seq.attrs
        migrama = seq.attrs["migrama"]
        assert migrama["t0"] == 5
        assert migrama["t1"] == 14
        assert migrama["bbox"] == [10, 20, 64, 64]

        # Check labels at cell level: cell_0/labels/0/
        labels = root["fov_0/cell_0/labels/0"]
        assert "labels" in labels.attrs
        assert "nuclei_masks" in labels.attrs["labels"]
        assert "cell_masks" in labels.attrs["labels"]

        # Check arrays exist directly under labels/0/
        assert "nuclei_masks" in labels
        assert "cell_masks" in labels


class TestZarrLoader:
    """Tests for ZarrSegmentationLoader."""

    def test_load_cell_filter_data(self, tmp_path, sample_data):
        """Test loading data from zarr store."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)
        write_global_metadata(root, "TestSource", 0, [1], "none")

        write_sequence(
            root,
            fov_idx=0,
            cell_idx=0,
            seq_idx=0,
            data=sample_data["data"],
            nuclei_masks=sample_data["nuclei_masks"],
            cell_masks=sample_data["cell_masks"],
            channels=sample_data["channels"],
            t0=sample_data["t0"],
            t1=sample_data["t1"],
            bbox=sample_data["bbox"],
        )

        loader = ZarrSegmentationLoader()
        result = loader.load_cell_filter_data(str(zarr_path), 0, 0, 0)

        np.testing.assert_array_equal(result["data"], sample_data["data"])
        np.testing.assert_array_equal(result["segmentation_masks"], sample_data["cell_masks"])
        np.testing.assert_array_equal(result["nuclei_masks"], sample_data["nuclei_masks"])
        assert result["channels"] == sample_data["channels"]
        assert result["sequence_metadata"]["t0"] == sample_data["t0"]
        assert result["sequence_metadata"]["t1"] == sample_data["t1"]
        assert result["sequence_metadata"]["bbox"] == [10, 20, 64, 64]

    def test_list_sequences(self, tmp_path, sample_data):
        """Test listing sequences in zarr store."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)
        write_global_metadata(root, "TestSource", 0, [1], "none")

        # Write multiple sequences
        for fov in range(2):
            for cell in range(3):
                write_sequence(
                    root,
                    fov_idx=fov,
                    cell_idx=cell,
                    seq_idx=0,
                    data=sample_data["data"],
                    nuclei_masks=sample_data["nuclei_masks"],
                    cell_masks=sample_data["cell_masks"],
                    channels=sample_data["channels"],
                    t0=sample_data["t0"],
                    t1=sample_data["t1"],
                    bbox=sample_data["bbox"],
                )

        loader = ZarrSegmentationLoader()
        sequences = loader.list_sequences(str(zarr_path))

        assert len(sequences) == 6
        fov_idx_set = {s["fov_idx"] for s in sequences}
        assert fov_idx_set == {0, 1}

    def test_validate_cell_filter_output(self, tmp_path, sample_data):
        """Test validation of zarr store."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)
        write_global_metadata(root, "TestSource", 0, [1], "none")

        write_sequence(
            root,
            fov_idx=0,
            cell_idx=0,
            seq_idx=0,
            data=sample_data["data"],
            nuclei_masks=sample_data["nuclei_masks"],
            cell_masks=sample_data["cell_masks"],
            channels=sample_data["channels"],
            t0=sample_data["t0"],
            t1=sample_data["t1"],
            bbox=sample_data["bbox"],
        )

        loader = ZarrSegmentationLoader()
        assert loader.validate_cell_filter_output(str(zarr_path))

    def test_validate_nonexistent_path(self, tmp_path):
        """Test validation fails for nonexistent path."""
        loader = ZarrSegmentationLoader()
        assert not loader.validate_cell_filter_output(str(tmp_path / "nonexistent.zarr"))

    def test_load_missing_sequence_raises(self, tmp_path, sample_data):
        """Test loading non-existent sequence raises ValueError."""
        zarr_path = tmp_path / "test.zarr"
        root = create_zarr_store(zarr_path)
        write_global_metadata(root, "TestSource", 0, [1], "none")

        write_sequence(
            root,
            fov_idx=0,
            cell_idx=0,
            seq_idx=0,
            data=sample_data["data"],
            nuclei_masks=sample_data["nuclei_masks"],
            cell_masks=sample_data["cell_masks"],
            channels=sample_data["channels"],
            t0=sample_data["t0"],
            t1=sample_data["t1"],
            bbox=sample_data["bbox"],
        )

        loader = ZarrSegmentationLoader()
        with pytest.raises(ValueError, match="Sequence not found"):
            loader.load_cell_filter_data(str(zarr_path), 1, 0, 0)
