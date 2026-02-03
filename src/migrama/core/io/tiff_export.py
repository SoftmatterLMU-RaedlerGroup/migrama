"""Export zarr stores to TIFF files."""

from pathlib import Path

import numpy as np
import tifffile
import zarr


def export_zarr_to_tiff(zarr_path: Path | str, output_dir: Path | str) -> int:
    """Export sequences from a zarr store to TIFF files.

    Parameters
    ----------
    zarr_path : Path | str
        Path to the input zarr store
    output_dir : Path | str
        Directory to write TIFF files to

    Returns
    -------
    int
        Number of sequences exported

    Raises
    ------
    FileNotFoundError
        If the zarr path does not exist
    """
    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open(zarr_path, mode="r")
    count = 0

    # Iterate over fov/{idx}/cell/{idx} structure
    if "fov" not in root:
        return 0

    for fov_key in sorted(root["fov"].keys()):
        fov_idx = int(fov_key)
        fov_group = root["fov"][fov_key]

        if "cell" not in fov_group:
            continue

        for cell_key in sorted(fov_group["cell"].keys()):
            cell_idx = int(cell_key)
            cell_group = fov_group["cell"][cell_key]

            # Read data array (T, C, H, W)
            data = np.asarray(cell_group["data"])

            # Read masks and stack as channels: (T, 2, H, W) with cell=0, nucleus=1
            cell_mask = np.asarray(cell_group["mask"]["cell"])
            nucleus_mask = np.asarray(cell_group["mask"]["nucleus"])
            mask = np.stack([cell_mask, nucleus_mask], axis=1)

            # Generate filenames
            base_name = f"fov_{fov_idx:04d}_cell_{cell_idx:04d}"
            data_path = output_dir / f"{base_name}_data.tiff"
            mask_path = output_dir / f"{base_name}_mask.tiff"

            # Write TIFF files
            tifffile.imwrite(data_path, data)
            tifffile.imwrite(mask_path, mask)

            count += 1

    return count
