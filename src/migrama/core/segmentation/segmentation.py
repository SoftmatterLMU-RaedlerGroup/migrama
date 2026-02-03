import logging

import numpy as np
import tifffile
from cellpose import models

# Configure logging
logger = logging.getLogger(__name__)


def merge_channels(channels: list[np.ndarray], method: str = 'multiply') -> np.ndarray:
    """Merge multiple channels into a single channel.

    Parameters
    ----------
    channels : list[np.ndarray]
        List of channel arrays to merge (each should be 2D: H, W)
    method : str
        Merge method: 'add' (sum), 'multiply' (product), or 'none' (raises error)

    Returns
    -------
    np.ndarray
        Merged channel array (H, W) as uint16 (0-65535 range)

    Raises
    ------
    ValueError
        If method is 'none' (should not call this function in that case)
    """
    if method == 'none':
        raise ValueError("merge_channels() should not be called with method='none'")

    # Normalize each channel to 0-1 range
    normalized = []
    for ch in channels:
        ch_min = ch.min()
        ch_max = ch.max()
        if ch_max > ch_min:
            normalized_ch = (ch - ch_min) / (ch_max - ch_min)
        else:
            normalized_ch = np.zeros_like(ch, dtype=np.float32)
        normalized.append(normalized_ch)

    # Merge channels
    if method == 'add':
        merged = sum(normalized)
        # Clamp to [0, 1] in case sum exceeds 1
        merged = np.clip(merged, 0, 1)
    elif method == 'multiply':
        merged = np.prod(np.stack(normalized), axis=0)
    else:
        raise ValueError(f"Unknown merge method: {method}. Must be 'add' or 'multiply'")

    # Convert to uint16 (0-65535)
    return (merged * 65535).astype(np.uint16)


class CellposeSegmenter:
    """
    A class for segmenting cells in microscopy images using Cellpose models.

    This class handles multi-channel timelapse microscopy data and applies
    Cellpose segmentation to identify individual cells.
    """

    def __init__(self):
        """
        Initialize the CellposeSegmenter.

        Note: GPU validation should be performed at application entrypoints
        before creating this instance. GPU is always enabled.
        """
        # In Cellpose 4.x, use the default model
        # GPU availability should have been validated at entrypoint
        self.model = models.CellposeModel(gpu=True)
        logger.debug("Initialized CellposeSegmenter with GPU enabled")

    def segment_image(
        self,
        image: np.ndarray,
        nuclei_channel: int | None = None,
        cell_channels: list[int] | None = None,
        merge_method: str = 'none',
    ) -> dict[str, np.ndarray]:
        """
        Segment a single image using Cellpose.

        Parameters
        ----------
        image : np.ndarray
            Input image with shape (height, width) or (height, width, channels) or (channels, height, width)
        nuclei_channel : int | None
            Channel index for nuclear channel. If None and merge_method != 'none', uses first channel.
        cell_channels : list[int] | None
            Channel indices for cell channels to merge. If None and merge_method != 'none', uses all channels except nuclei_channel.
        merge_method : str
            Merge method: 'add', 'multiply', or 'none'. If 'none', passes all channels directly to cellpose.

        Returns
        -------
        dict
            Dictionary containing:
            - 'masks': Segmentation masks (2D array)
            - 'flows': Flow fields from Cellpose
            - 'styles': Style vectors
        """
        # Handle different input formats
        if image.ndim == 2:
            # Single channel: (H, W)
            if merge_method != 'none':
                raise ValueError("Cannot merge channels on 2D image. Image must have multiple channels.")
            logger.debug(f"Segmenting single-channel image with shape {image.shape}")
            cellpose_input = image
        elif image.ndim == 3:
            # Multi-channel: determine if (H, W, C) or (C, H, W)
            # If first dim is small (< 10) and last dim is large, likely (C, H, W)
            # Otherwise assume (H, W, C)
            if image.shape[0] < 10 and image.shape[2] > image.shape[0]:
                # Likely (C, H, W) - transpose to (H, W, C)
                image = np.transpose(image, (1, 2, 0))
                logger.debug(f"Transposed image from (C, H, W) to (H, W, C): {image.shape}")

            if merge_method == 'none':
                # Pass all channels directly
                logger.debug(f"Segmenting image with all channels (shape {image.shape})")
                cellpose_input = image
            else:
                # Extract and merge channels
                # At this point, image should be in (H, W, C) format
                n_channels = image.shape[2]

                if nuclei_channel is None:
                    nuclei_channel = 0

                if cell_channels is None:
                    # Use all channels except nuclei_channel
                    cell_channels = [i for i in range(n_channels) if i != nuclei_channel]

                if not cell_channels:
                    raise ValueError(f"No cell channels specified (nuclei_channel={nuclei_channel}, total_channels={n_channels})")

                # Extract channels
                nuclei_img = image[:, :, nuclei_channel]
                cell_imgs = [image[:, :, ch] for ch in cell_channels]

                # Merge cell channels
                merged_cell = merge_channels(cell_imgs, method=merge_method)

                # Stack nuclear + merged cell into 2-channel array (H, W, 2)
                cellpose_input = np.stack([nuclei_img, merged_cell], axis=2)
                logger.debug(f"Segmenting with merged channels: nuclear={nuclei_channel}, cell={cell_channels}, method={merge_method}, shape={cellpose_input.shape}")
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

        # In Cellpose 4.x, eval returns (masks, flows, styles)
        result = self.model.eval(
            cellpose_input,
        )

        masks, flows, styles = result

        n_cells = len(np.unique(masks)) - 1  # Subtract 1 for background
        logger.debug(f"Segmentation complete: found {n_cells} cells")

        return {
            'masks': masks,
            'flows': flows,
            'styles': styles,
        }

    def segment_timelapse(
        self,
        timelapse_path: str,
        frames: int | list[int] | None = None
    ) -> list[dict[str, np.ndarray]]:
        """
        Segment a timelapse microscopy file.

        Parameters
        ----------
        timelapse_path : str
            Path to the timelapse TIFF file
        frames : int or list of int, optional
            Specific frames to process. If None, processes all frames

        Returns
        -------
        list of dict
            List of segmentation results for each frame
        """
        logger.info(f"Starting timelapse segmentation for {timelapse_path}")

        with tifffile.TiffFile(timelapse_path) as tif:
            data = tif.asarray()

        if data.ndim == 3:
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D data, got shape {data.shape}")

        n_frames = data.shape[0]
        logger.debug(f"Timelapse data shape: {data.shape}, processing {n_frames} frames")

        if frames is None:
            frames = list(range(n_frames))
        elif isinstance(frames, int):
            frames = [frames]

        logger.debug(f"Processing frames: {frames}")

        results = []
        for frame_idx in frames:
            logger.debug(f"Processing frame {frame_idx}")
            frame_data = data[frame_idx]

            if frame_data.ndim == 3 and frame_data.shape[0] <= 3:
                frame_data = np.transpose(frame_data, (1, 2, 0))

            result = self.segment_image(frame_data)
            result['frame'] = frame_idx
            results.append(result)

        logger.info(f"Timelapse segmentation complete: processed {len(results)} frames")
        return results
