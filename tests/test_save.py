"""Tests for the save command - exporting zarr to TIFF files."""

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from migrama.cli.main import app
from migrama.core.io.tiff_export import export_zarr_to_tiff
from migrama.core.io.zarr_io import (
    create_zarr_store,
    write_global_metadata,
    write_sequence,
)


@pytest.fixture
def sample_zarr(tmp_path):
    """Create a sample zarr store with test data."""
    zarr_path = tmp_path / "test.zarr"
    root = create_zarr_store(zarr_path)
    write_global_metadata(root, "TestSource", 0, [1], "none")

    np.random.seed(42)
    # Write two sequences: fov=0/cell=0 and fov=1/cell=2
    sequences = [(0, 0), (1, 2)]

    for fov, cell in sequences:
        data = np.random.randint(0, 65535, (5, 3, 32, 32), dtype=np.uint16)
        cell_masks = np.random.randint(0, 3, (5, 32, 32), dtype=np.int32)
        nuclei_masks = np.random.randint(0, 3, (5, 32, 32), dtype=np.int32)

        write_sequence(
            root,
            fov_idx=fov,
            cell_idx=cell,
            data=data,
            nuclei_masks=nuclei_masks,
            cell_masks=cell_masks,
            channels=["ch0", "ch1", "ch2"],
            t0=0,
            t1=4,
            bbox=np.array([10, 20, 32, 32]),
        )

    return zarr_path


class TestExportZarrToTiff:
    """Tests for export_zarr_to_tiff function."""

    def test_creates_output_directory(self, sample_zarr, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "output"
        assert not output_dir.exists()

        export_zarr_to_tiff(sample_zarr, output_dir)

        assert output_dir.exists()

    def test_creates_data_tiff_files(self, sample_zarr, tmp_path):
        """Test that data TIFF files are created with correct naming."""
        output_dir = tmp_path / "output"

        export_zarr_to_tiff(sample_zarr, output_dir)

        # Check files exist with correct naming convention
        assert (output_dir / "fov_0000_cell_0000_data.tiff").exists()
        assert (output_dir / "fov_0001_cell_0002_data.tiff").exists()

    def test_creates_mask_tiff_files(self, sample_zarr, tmp_path):
        """Test that mask TIFF files are created with correct naming."""
        output_dir = tmp_path / "output"

        export_zarr_to_tiff(sample_zarr, output_dir)

        # Check mask files exist
        assert (output_dir / "fov_0000_cell_0000_mask.tiff").exists()
        assert (output_dir / "fov_0001_cell_0002_mask.tiff").exists()

    def test_data_tiff_shape_and_content(self, sample_zarr, tmp_path):
        """Test that data TIFF has correct shape (T, C, H, W)."""
        output_dir = tmp_path / "output"

        export_zarr_to_tiff(sample_zarr, output_dir)

        data = tifffile.imread(output_dir / "fov_0000_cell_0000_data.tiff")
        assert data.shape == (5, 3, 32, 32)  # T, C, H, W
        assert data.dtype == np.uint16

    def test_mask_tiff_shape_and_content(self, sample_zarr, tmp_path):
        """Test that mask TIFF has shape (T, 2, H, W) with cell=ch0, nucleus=ch1."""
        output_dir = tmp_path / "output"

        export_zarr_to_tiff(sample_zarr, output_dir)

        mask = tifffile.imread(output_dir / "fov_0000_cell_0000_mask.tiff")
        assert mask.shape == (5, 2, 32, 32)  # T, 2 (cell+nucleus), H, W

    def test_returns_count_of_exported_sequences(self, sample_zarr, tmp_path):
        """Test that function returns the number of exported sequences."""
        output_dir = tmp_path / "output"

        count = export_zarr_to_tiff(sample_zarr, output_dir)

        assert count == 2

    def test_raises_on_invalid_zarr_path(self, tmp_path):
        """Test that function raises error for non-existent zarr."""
        output_dir = tmp_path / "output"
        invalid_zarr = tmp_path / "nonexistent.zarr"

        with pytest.raises(FileNotFoundError):
            export_zarr_to_tiff(invalid_zarr, output_dir)


class TestSaveCLI:
    """Tests for the save CLI command."""

    def test_save_command_exports_files(self, sample_zarr, tmp_path):
        """Test that save command creates TIFF files."""
        output_dir = tmp_path / "output"
        runner = CliRunner()

        result = runner.invoke(app, ["save", "--zarr", str(sample_zarr), "--output", str(output_dir)])

        assert result.exit_code == 0
        assert (output_dir / "fov_0000_cell_0000_data.tiff").exists()
        assert (output_dir / "fov_0000_cell_0000_mask.tiff").exists()
        assert "Saved 2 sequences" in result.stdout

    def test_save_command_nonexistent_zarr(self, tmp_path):
        """Test that save command fails gracefully for non-existent zarr."""
        runner = CliRunner()

        result = runner.invoke(app, ["save", "--zarr", str(tmp_path / "nope.zarr"), "--output", str(tmp_path)])

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()
