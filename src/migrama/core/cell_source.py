"""
Lazy cell data source classes for unified ND2 and TIFF handling.

Provides a uniform interface for loading multi-FOV timelapse cell data
from different sources (ND2 files, per-FOV multi-page TIFFs).
"""

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile

from .io.nikon import get_nd2_channel_stack, load_nd2

logger = logging.getLogger(__name__)


class CellFovSource(ABC):
    """Abstract base class for lazy cell data sources.

    Subclasses provide a uniform interface for iterating over FOVs,
    yielding (fov_id, tcyx_array) for each field of view.
    """

    @property
    @abstractmethod
    def n_fovs(self) -> int:
        """Return the number of FOVs."""
        ...

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Return the number of time frames."""
        ...

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Return the number of channels."""
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Return the image height."""
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        """Return the image width."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Return the data type."""
        ...

    @abstractmethod
    def get_fov(self, fov_id: int) -> np.ndarray:
        """Get the full tcyx array for a specific FOV.

        Parameters
        ----------
        fov_id : int
            Field of view index

        Returns
        -------
        np.ndarray
            Array of shape (n_frames, n_channels, height, width)
        """
        ...

    @abstractmethod
    def get_frame(self, fov_id: int, frame_id: int) -> np.ndarray:
        """Get a single frame from a specific FOV.

        Parameters
        ----------
        fov_id : int
            Field of view index
        frame_id : int
            Frame/time index

        Returns
        -------
        np.ndarray
            Array of shape (n_channels, height, width)
        """
        ...

    def iter_fovs(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over FOVs in order, yielding (fov_id, tcyx_array).

        Yields
        ------
        tuple[int, np.ndarray]
            (fov_id, tcyx_array) for each field of view
        """
        for fov_id in range(self.n_fovs):
            yield fov_id, self.get_fov(fov_id)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Convenience: make the source itself iterable."""
        return self.iter_fovs()


class Nd2CellFovSource(CellFovSource):
    """Lazy ND2 source for cell timelapse data.

    Reads frames on-demand from an ND2 file without loading the entire
    dataset into memory.

    Parameters
    ----------
    nd2_path : str | Path
        Path to the ND2 file.
    """

    def __init__(self, nd2_path: str | Path) -> None:
        """Initialize the ND2 source."""
        self.nd2_path = Path(nd2_path).resolve()

        if not self.nd2_path.exists():
            raise FileNotFoundError(f"ND2 file not found: {self.nd2_path}")

        self._xarr, self._metadata = load_nd2(self.nd2_path)

        logger.info(
            f"Nd2CellFovSource: {self.nd2_path.name}, "
            f"{self._metadata.n_fovs} FOVs, {self._metadata.n_frames} frames, "
            f"{self._metadata.n_channels} channels"
        )

    @property
    def n_fovs(self) -> int:
        """Return the number of FOVs."""
        return self._metadata.n_fovs

    @property
    def n_frames(self) -> int:
        """Return the number of time frames."""
        return self._metadata.n_frames

    @property
    def n_channels(self) -> int:
        """Return the number of channels."""
        return self._metadata.n_channels

    @property
    def height(self) -> int:
        """Return the image height."""
        return self._metadata.height

    @property
    def width(self) -> int:
        """Return the image width."""
        return self._metadata.width

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return np.dtype(self._metadata.dtype)

    def get_fov(self, fov_id: int) -> np.ndarray:
        """Get the full tcyx array for a specific FOV."""
        if fov_id < 0 or fov_id >= self.n_fovs:
            raise ValueError(f"FOV {fov_id} out of range (0-{self.n_fovs - 1})")

        frames = []
        for t in range(self.n_frames):
            stack = get_nd2_channel_stack(self._xarr, fov_id, t)
            frames.append(stack)

        return np.stack(frames)

    def get_frame(self, fov_id: int, frame_id: int) -> np.ndarray:
        """Get a single frame from a specific FOV."""
        if fov_id < 0 or fov_id >= self.n_fovs:
            raise ValueError(f"FOV {fov_id} out of range (0-{self.n_fovs - 1})")
        if frame_id < 0 or frame_id >= self.n_frames:
            raise ValueError(f"Frame {frame_id} out of range (0-{self.n_frames - 1})")

        return get_nd2_channel_stack(self._xarr, fov_id, frame_id)


class TiffCellFovSource(CellFovSource):
    """Per-FOV TIFF source for cell timelapse data.

    Expects one multi-page TIFF file per FOV. The user provides a path
    containing a FOV index suffix (e.g., `./folder/xxx_0.tif`), and the
    class automatically discovers all matching files.

    Parameters
    ----------
    tiff_pattern : str | Path
        Path to a TIFF file with FOV index suffix, e.g., `./folder/cells_0.tif`.
        The FOV index is stripped to create a glob pattern for discovering
        all FOV files.
    """

    def __init__(self, tiff_pattern: str | Path) -> None:
        """Initialize the TIFF source from a pattern path."""
        tiff_path = Path(tiff_pattern).resolve()

        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

        if tiff_path.is_dir():
            raise NotADirectoryError(f"Expected a file path, got a directory: {tiff_path}")

        self._tiff_folder = tiff_path.parent
        self._pattern = self._derive_pattern(tiff_path)
        self._tiff_files = self._discover_tiff_files()
        self._n_fovs = len(self._tiff_files)

        if self._n_fovs == 0:
            raise ValueError(f"No TIFF files found matching pattern: {self._pattern}")

        self._cache: dict[int, np.ndarray] = {}

        sample_path = self._tiff_files[0][1]
        sample = tifffile.imread(str(sample_path))
        if sample.ndim == 4:
            self._tcyx = True
            self._n_frames, self._n_channels, self._height, self._width = sample.shape
        elif sample.ndim == 3:
            self._tcyx = False
            self._n_frames, self._height, self._width = sample.shape
            self._n_channels = 1
        else:
            raise ValueError(f"Unexpected TIFF shape: {sample.shape}")

        logger.info(
            f"TiffCellFovSource: {self._pattern}, {self._n_fovs} FOVs, "
            f"{self._n_frames} frames, {self._n_channels} channels"
        )

    def _derive_pattern(self, tiff_path: Path) -> str:
        """Derive a glob pattern from a specific file path.

        Strips the numeric FOV suffix from the filename to create a pattern.
        E.g., `./folder/cells_0.tif` -> `./folder/cells_*.tif`
        """
        name = tiff_path.name
        suffix = tiff_path.suffix

        if not name.endswith(suffix):
            raise ValueError(f"File does not have expected TIFF extension: {tiff_path}")

        base_without_ext = name[: -len(suffix)]

        if not base_without_ext:
            raise ValueError(f"Invalid TIFF filename: {tiff_path}")


        numeric_suffix = re.search(r"_(\d+)$", base_without_ext)
        if numeric_suffix:
            prefix = base_without_ext[: numeric_suffix.start()]
            pattern_name = f"{prefix}_*{suffix}"
        else:
            pattern_name = f"*{suffix}"

        return str(tiff_path.parent / pattern_name)

    def _discover_tiff_files(self) -> list[tuple[int, Path]]:
        """Discover TIFF files and extract FOV indices from filenames.

        Returns
        -------
        list[tuple[int, Path]]
            List of (fov_id, path) tuples, sorted by fov_id.
        """

        files = []
        pattern_str = self._pattern
        folder = self._tiff_folder
        suffix = Path(pattern_str).suffix

        for tiff_path in folder.glob(pattern_str):
            name = tiff_path.name

            base_without_ext = name[: -len(suffix)]
            match = re.search(r"_(\d+)$", base_without_ext)
            if match:
                try:
                    fov_id = int(match.group(1))
                    files.append((fov_id, tiff_path))
                except ValueError:
                    logger.warning(f"Could not parse FOV index from: {name}")
            else:
                logger.warning(f"Filename does not match pattern (missing _<num>): {name}")

        files.sort(key=lambda x: x[0])
        return files

    @property
    def n_fovs(self) -> int:
        """Return the number of FOVs."""
        return self._n_fovs

    @property
    def n_frames(self) -> int:
        """Return the number of time frames."""
        return self._n_frames

    @property
    def n_channels(self) -> int:
        """Return the number of channels."""
        return self._n_channels

    @property
    def height(self) -> int:
        """Return the image height."""
        return self._height

    @property
    def width(self) -> int:
        """Return the image width."""
        return self._width

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return self._cache[0].dtype if self._cache else tifffile.imread(str(self._tiff_files[0][1])).dtype

    def get_fov(self, fov_id: int) -> np.ndarray:
        """Get the full tcyx array for a specific FOV."""
        if fov_id < 0 or fov_id >= self.n_fovs:
            raise ValueError(f"FOV {fov_id} out of range (0-{self.n_fovs - 1})")

        if fov_id in self._cache:
            return self._cache[fov_id]

        tiff_path = None
        for fid, path in self._tiff_files:
            if fid == fov_id:
                tiff_path = path
                break

        if tiff_path is None:
            raise ValueError(f"TIFF file for FOV {fov_id} not found")

        data = tifffile.imread(str(tiff_path))

        if self._tcyx:
            self._cache[fov_id] = data
            return data
        else:
            expanded = np.expand_dims(data, axis=1)
            self._cache[fov_id] = expanded
            return expanded

    def get_frame(self, fov_id: int, frame_id: int) -> np.ndarray:
        """Get a single frame from a specific FOV."""
        if fov_id < 0 or fov_id >= self.n_fovs:
            raise ValueError(f"FOV {fov_id} out of range (0-{self.n_fovs - 1})")
        if frame_id < 0 or frame_id >= self.n_frames:
            raise ValueError(f"Frame {frame_id} out of range (0-{self.n_frames - 1})")

        # If FOV is cached, return from cache
        if fov_id in self._cache:
            return self._cache[fov_id][frame_id]

        # Otherwise read single frame from TIFF
        tiff_path = None
        for fid, path in self._tiff_files:
            if fid == fov_id:
                tiff_path = path
                break

        if tiff_path is None:
            raise ValueError(f"TIFF file for FOV {fov_id} not found")

        # Read specific frame using tifffile's page access
        with tifffile.TiffFile(str(tiff_path)) as tif:
            if self._tcyx:
                # For TCYX, frame_id indexes the T dimension
                # Each frame has C pages
                start_page = frame_id * self._n_channels
                pages = [tif.pages[start_page + c].asarray() for c in range(self._n_channels)]
                return np.stack(pages)
            else:
                # For TYX (single channel), just read the frame
                frame = tif.pages[frame_id].asarray()
                return np.expand_dims(frame, axis=0)
