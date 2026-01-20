"""Test dataset definitions for migrama.

Each dataset is defined as a dataclass containing paths to required files
for different test scenarios.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MigramaDataset:
    """A test dataset containing paths to microscopy files."""

    name: str
    base_path: Path
    cells_nd2: Path
    patterns_nd2: Path

    def exists(self) -> bool:
        """Check if all required files exist."""
        return self.cells_nd2.exists() and self.patterns_nd2.exists()


# Dataset: fourcell 20250812
# MDCK cells on micropatterns, timelapse imaging
FOURCELL_20250812 = MigramaDataset(
    name="fourcell_20250812",
    base_path=Path("/project/ag-moonraedler/user/Tianyi.Cao/data/fourcell/20250812"),
    cells_nd2=Path("/project/ag-moonraedler/user/Tianyi.Cao/data/fourcell/20250812/20250812_MDCK_LK_timelapse.nd2"),
    patterns_nd2=Path(
        "/project/ag-moonraedler/user/Tianyi.Cao/data/fourcell/20250812/20250812_MDCK_LK_timelapse_patterns_before.nd2"
    ),
)
