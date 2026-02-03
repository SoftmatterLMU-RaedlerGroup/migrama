"""Pytest configuration and fixtures for migrama tests."""

import pytest
from pathlib import Path

from tests.data import FOURCELL_20250812


@pytest.fixture
def fourcell_dataset():
    """Provide the fourcell 20250812 dataset configuration."""
    return FOURCELL_20250812


@pytest.fixture
def tmp_output_dir(tmp_path: Path):
    """Provide a temporary output directory for test artifacts."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
