"""Unit tests for zarr compatibility.

Quick tests to verify zarr API usage before running expensive workflows.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr


class TestZarrCompat:
    """Test zarr array creation API compatibility."""

    def test_create_array_with_data_and_chunks(self):
        """Verify create_array works with data and chunks params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open(str(Path(tmpdir) / "test.zarr"), mode="w")
            group = store.create_group("test_group")

            # Create test data similar to masks_stack in Analyzer
            masks_stack = np.random.randint(0, 10, size=(10, 64, 64), dtype=np.uint16)

            # This should work - only data and chunks, no shape/dtype
            arr = group.create_array(
                "cell_masks",
                data=masks_stack,
                chunks=(1, masks_stack.shape[1], masks_stack.shape[2]),
            )

            assert arr.shape == (10, 64, 64)
            assert arr.dtype == np.uint16
            np.testing.assert_array_equal(arr[:], masks_stack)

    def test_create_array_without_data(self):
        """Verify create_array works with shape/dtype when no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open(str(Path(tmpdir) / "test.zarr"), mode="w")
            group = store.create_group("test_group")

            # When no data, must specify shape and dtype
            arr = group.create_array(
                "empty_masks",
                shape=(10, 64, 64),
                dtype=np.uint16,
                chunks=(1, 64, 64),
            )

            assert arr.shape == (10, 64, 64)
            assert arr.dtype == np.uint16
