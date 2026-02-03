"""Unified CLI for all migrama modules."""

import logging
import sys
from pathlib import Path

import typer

# Create main app
app = typer.Typer(help="Migrama: A comprehensive toolkit for micropatterned timelapse microscopy analysis")


def parse_fov_string(fov_string: str) -> list[int] | None:
    """Parse FOV string like '1,3-5,8' into list [1,3,4,5,8].

    Parameters
    ----------
    fov_string : str
        FOV specification: 'all' for all FOVs, or comma-separated values/ranges

    Returns
    -------
    list[int] | None
        Sorted list of unique FOV indices, or None if 'all'
    """
    if fov_string.strip().lower() == "all":
        return None

    fovs = []
    for part in fov_string.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            fovs.extend(range(int(start), int(end) + 1))
        else:
            fovs.append(int(part))
    return sorted(set(fovs))


def create_cell_source(path: str):
    """Auto-detect and create the appropriate cell source based on file extension.

    Parameters
    ----------
    path : str
        Path to cells file (.nd2 or .tif/.tiff)

    Returns
    -------
    CellFovSource
        Either Nd2CellFovSource or TiffCellFovSource
    """
    from ..core.cell_source import Nd2CellFovSource, TiffCellFovSource

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".nd2":
        return Nd2CellFovSource(path)
    elif suffix in (".tif", ".tiff"):
        return TiffCellFovSource(path)
    else:
        raise typer.BadParameter(
            f"Unsupported file format: {suffix}. Expected .nd2 or .tif/.tiff"
        )


@app.command()
def pattern(
    patterns: str = typer.Option(
        ..., "--patterns", "-p", help="Path to patterns file (.nd2) or folder of TIFFs"
    ),
    output: str = typer.Option(..., "--output", "-o", help="Output CSV file path"),
    fovs: str = typer.Option(..., "--fovs", help="FOVs to process: 'all' or ranges like '1,3-5,8' (required)"),
    plot: str | None = typer.Option(None, "--plot", help="Output folder for bbox overlay plots (one PNG per FOV)"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Detect micropatterns and save bounding boxes to CSV.

    Accepts either an ND2 file or a folder of pre-averaged TIFFs (auto-detected).
    Output CSV format: cell,fov,x,y,w,h

    Use --plot to generate visualization of detected bboxes overlaid on
    the pattern images (one PNG per FOV).
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    patterns_path = Path(patterns)
    if not patterns_path.exists():
        typer.echo(f"Error: Path does not exist: {patterns}", err=True)
        raise typer.Exit(1)

    # Auto-detect source type based on path
    if patterns_path.is_dir():
        from ..core.pattern.source import TiffPatternFovSource

        source = TiffPatternFovSource(patterns_path)
    else:
        from ..core.pattern.source import Nd2PatternFovSource

        source = Nd2PatternFovSource(patterns)

    from ..core.pattern import PatternDetector

    detector = PatternDetector(source=source)

    # Parse --fovs (required): 'all' or ranges like '1,3-5,8'
    try:
        fov_filter = parse_fov_string(fovs)
    except ValueError:
        typer.echo(f"Error: Invalid --fovs format: {fovs}. Expected 'all' or ranges like '1,3-5,8'", err=True)
        raise typer.Exit(1) from None

    if fov_filter is not None and len(fov_filter) == 1:
        records = detector.detect_fov(fov_filter[0])
        typer.echo(f"Detected {len(records)} patterns in FOV {fov_filter[0]}")
    elif fov_filter is not None:
        records = detector.detect_all(fov_filter=fov_filter)
        typer.echo(f"Detected {len(records)} patterns across {len(fov_filter)} FOVs")
    else:
        records = detector.detect_all()
        typer.echo(f"Detected {len(records)} patterns across {detector.n_fovs} FOVs")

    detector.save_csv(records, output)
    typer.echo(f"Saved to: {output}")

    # Generate bbox overlay plots if requested
    if plot is not None:
        from ..utils.plot import plot_pattern_bboxes

        plot_dir = Path(plot)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Group records by FOV
        records_by_fov: dict[int, list] = {}
        for r in records:
            records_by_fov.setdefault(r.fov, []).append(r)

        # Generate plot for each FOV
        for fov_idx, frame in source.iter_fovs():
            if fov_idx not in records_by_fov:
                continue

            out_path = plot_dir / f"fov_{fov_idx:03d}.png"
            plot_pattern_bboxes(frame, records_by_fov[fov_idx], fov_idx, out_path)

        typer.echo(f"Saved {len(records_by_fov)} plots to: {plot_dir}")


@app.command()
def average(
    cells: str = typer.Option(..., "--cells", "-c", help="Path to cells ND2 file"),
    cell_channel: int = typer.Option(0, "--cc", help="Channel index for cell bodies (phase contrast)"),
    t0: int | None = typer.Option(None, "--t0", help="Start frame index (inclusive, supports negative)"),
    t1: int | None = typer.Option(None, "--t1", help="End frame index (exclusive, supports negative)"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory for averaged TIFFs"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Average phase contrast time-lapse frames to generate pattern images when no dedicated patterns.nd2 was recorded."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

    from ..core.pattern import PatternAverager
    from ..core.progress import ProgressEvent

    averager = PatternAverager(
        cells_path=cells,
        cell_channel=cell_channel,
        t0=t0,
        t1=t1,
        output_dir=output_dir,
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    )
    progress.start()

    tasks: dict[str, TaskID] = {}

    def handle_progress(event: ProgressEvent) -> None:
        if event.state not in tasks:
            tasks[event.state] = progress.add_task(
                f"{event.state} ({event.iterator})",
                total=event.total or 1,
            )
        task_id = tasks[event.state]
        progress.update(task_id, completed=event.current)

    averager.progress.connect(handle_progress)

    try:
        output_paths = averager.run()
        progress.stop()
        typer.echo(f"Averaged {len(output_paths)} FOVs to {output_dir}")
    except Exception:
        progress.stop()
        raise


@app.command()
def analyze(
    cells: str = typer.Option(
        ..., "--cells", "-c", help="Path to cells file (.nd2 or .tif/.tiff)"
    ),
    csv: str = typer.Option(..., "--csv", help="Path to patterns CSV file"),
    cache: str | None = typer.Option(None, "--cache", help="Output cache.ome.zarr path for cell mask storage (optional)"),
    output: str = typer.Option(..., "--output", "-o", help="Output CSV file path"),
    nc: int | None = typer.Option(None, "--nc", help="Channel index for nucleus"),
    cc: str | None = typer.Option(None, "--cc", help="Comma-separated cell channel indices"),
    merge_method: str | None = typer.Option(None, "--merge-method", help="Channel merge method: 'add' or 'multiply'"),
    n_cells: int = typer.Option(..., "--n-cells", help="Target number of cells per pattern (required)"),
    allowed_gap: int = typer.Option(6, "--allowed-gap", help="Maximum consecutive non-target frames to bridge over"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Analyze cell counts using all-channel segmentation.

    Segments cells using all channels (Cellpose all-channel mode), counts
    cells per frame, and caches masks for the extract step.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    # Validate channel args required when merge_method is specified
    if merge_method is not None:
        if nc is None:
            typer.echo("Error: --nc is required when --merge-method is not 'none'", err=True)
            raise typer.Exit(1)
        if cc is None:
            typer.echo("Error: --cc is required when --merge-method is not 'none'", err=True)
            raise typer.Exit(1)

    # Parse cc (optional, metadata only)
    cell_channels_list: list[int] | None = None
    if cc is not None:
        try:
            cell_channels_list = [int(x.strip()) for x in cc.split(",")]
        except ValueError:
            typer.echo(f"Error: Invalid --cc format: {cc}. Expected comma-separated integers (e.g., '0' or '1,2')", err=True)
            raise typer.Exit(1) from None

    from ..analyze import Analyzer

    source = create_cell_source(cells)

    analyzer = Analyzer(
        source=source,
        csv_path=csv,
        cache_path=cache,
        nuclei_channel=nc,
        cell_channels=cell_channels_list,
        merge_method=merge_method,
        n_cells=n_cells,
        allowed_gap=allowed_gap,
    )
    records = analyzer.analyze(output)
    typer.echo(f"Saved {len(records)} records to {output}")
    if cache:
        typer.echo(f"Cached masks to {cache}")


@app.command()
def extract(
    cells: str = typer.Option(
        ..., "--cells", "-c", help="Path to cells file (.nd2 or .tif/.tiff)"
    ),
    csv: str = typer.Option(..., "--csv", help="Path to analysis CSV file"),
    output: str = typer.Option(..., "--output", "-o", help="Output Zarr store path"),
    nc: int | None = typer.Option(None, "--nc", help="Channel index for nucleus"),
    cc: str | None = typer.Option(None, "--cc", help="Comma-separated cell channel indices"),
    merge_method: str | None = typer.Option(None, "--merge-method", help="Channel merge method: 'add' or 'multiply'"),
    cache: str | None = typer.Option(None, "--cache", help="Path to cache.ome.zarr with pre-computed cell masks (explicit opt-in)"),
    min_frames: int = typer.Option(20, "--min-frames", help="Minimum frames per sequence"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Extract sequences with cell-first tracking.

    Cell-first workflow: segment cells (all channels) → track cells across
    frames → derive nuclei by Otsu thresholding within each tracked cell.

    Without --cache, cells are re-segmented from the source data.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    # Validate merge_method
    if merge_method is not None and merge_method not in ('add', 'multiply'):
        typer.echo(f"Error: Invalid --merge-method: {merge_method}. Must be 'add' or 'multiply'", err=True)
        raise typer.Exit(1)

    # Validate channel args required when merge_method is specified
    if merge_method is not None:
        if nc is None:
            typer.echo("Error: --nc is required when --merge-method is not 'none'", err=True)
            raise typer.Exit(1)
        if cc is None:
            typer.echo("Error: --cc is required when --merge-method is not 'none'", err=True)
            raise typer.Exit(1)

    # Parse cc (optional, metadata only)
    cell_channels_list: list[int] | None = None
    if cc is not None:
        try:
            cell_channels_list = [int(x.strip()) for x in cc.split(",")]
        except ValueError:
            typer.echo(f"Error: Invalid --cc format: {cc}. Expected comma-separated integers (e.g., '0' or '1,2')", err=True)
            raise typer.Exit(1) from None

    from ..extract import Extractor

    source = create_cell_source(cells)

    extractor = Extractor(
        source=source,
        analysis_csv=csv,
        output_path=output,
        nuclei_channel=nc,
        cell_channels=cell_channels_list,
        merge_method=merge_method,
        cache_path=cache,
    )
    sequences = extractor.extract(min_frames=min_frames)
    typer.echo(f"Saved {sequences} sequences to {output}")


@app.command()
def convert(
    input_folder: str = typer.Option(..., "--input", "-i", help="Path to folder with TIFF files"),
    output: str = typer.Option("./converted.zarr", "--output", "-o", help="Output Zarr store path"),
    nc: int | None = typer.Option(None, "--nc", help="Channel index for nucleus"),
    cc: str | None = typer.Option(None, "--cc", help="Comma-separated cell channel indices"),
    merge_method: str | None = typer.Option(None, "--merge-method", help="Channel merge method: 'add' or 'multiply'"),
    min_frames: int = typer.Option(20, "--min-frames", help="Minimum frames per sequence"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Convert TIFF files to Zarr with segmentation and tracking."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    # Validate merge_method
    if merge_method is not None and merge_method not in ('add', 'multiply'):
        typer.echo(f"Error: Invalid --merge-method: {merge_method}. Must be 'add' or 'multiply'", err=True)
        raise typer.Exit(1)

    # Validate channel args required when merge_method is specified
    if merge_method is not None:
        if nc is None:
            typer.echo("Error: --nc is required when --merge-method is not 'none'", err=True)
            raise typer.Exit(1)
        if cc is None:
            typer.echo("Error: --cc is required when --merge-method is not 'none'", err=True)
            raise typer.Exit(1)

    # Parse cc if provided
    cell_channels_list: list[int] | None = None
    if cc is not None:
        try:
            cell_channels_list = [int(x.strip()) for x in cc.split(",")]
        except ValueError:
            typer.echo(f"Error: Invalid --cc format: {cc}. Expected comma-separated integers (e.g., '1,2')", err=True)
            raise typer.Exit(1) from None

    from ..convert import Converter
    from ..core.progress import ProgressEvent

    converter = Converter(
        input_folder=input_folder,
        output_path=output,
        nuclei_channel=nc,
        cell_channels=cell_channels_list,
        merge_method=merge_method,
    )

    from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    )
    progress.start()

    tasks: dict[str, TaskID] = {}

    def handle_progress(event: ProgressEvent) -> None:
        if event.state not in tasks:
            tasks[event.state] = progress.add_task(
                f"{event.state} ({event.iterator})",
                total=event.total or 1,
            )
        task_id = tasks[event.state]
        progress.update(task_id, completed=event.current)

    def on_file_start(filename: str) -> None:
        typer.echo(f"\n{filename}")

    converter.progress.connect(handle_progress)

    try:
        sequences = converter.convert(min_frames=min_frames, on_file_start=on_file_start)
        progress.stop()
        typer.echo(f"Saved {sequences} sequences to {output}")
    except Exception:
        progress.stop()
        raise


# Graph command disabled - being redesigned
# @app.command()
# def graph(
#     input: str = typer.Option(..., "--input", "-i", help="Path to H5 file with segmentation data"),
#     output: str = typer.Option(..., "--output", "-o", help="Output directory for analysis results"),
#     fov: int = typer.Option(..., "--fov", help="FOV index"),
#     pattern: int = typer.Option(..., "--pattern", help="Pattern index"),
#     sequence: int = typer.Option(..., "--sequence", help="Sequence index"),
#     start_frame: int | None = typer.Option(None, "--start-frame", "-s", help="Starting frame"),
#     end_frame: int | None = typer.Option(None, "--end-frame", "-e", help="Ending frame (exclusive)"),
#     search_radius: float = typer.Option(100.0, "--search-radius", help="Max search radius for tracking"),
#     debug: bool = typer.Option(False, "--debug"),
# ):
#     """Create region adjacency graphs and analyze T1 transitions."""
#     import os
#     import tempfile
#
#     import numpy as np
#     import yaml as yaml_module
#
#     log_level = logging.DEBUG if debug else logging.INFO
#     logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")
#
#     if not Path(input).exists():
#         typer.echo(f"Error: Input file does not exist: {input}", err=True)
#         raise typer.Exit(1)
#
#     from ..graph.h5_loader import H5SegmentationLoader
#     from ..graph.pipeline import analyze_cell_filter_data
#
#     loader = H5SegmentationLoader()
#
#     typer.echo(f"Loading sequence: FOV {fov}, Pattern {pattern}, Sequence {sequence}")
#     loaded_data = loader.load_cell_filter_data(input, fov, pattern, sequence, None)
#     data = cast(np.ndarray, loaded_data["data"])
#     segmentation_masks = cast(np.ndarray, loaded_data["segmentation_masks"])
#
#     # Create temporary NPY file for pipeline compatibility
#     with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
#         combined_data = np.concatenate([data, segmentation_masks[:, np.newaxis, :, :]], axis=1)
#         np.save(tmp.name, combined_data)
#         tmp_npy_path = tmp.name
#
#     tmp_yaml_path = None
#     if loaded_data["metadata"]:
#         tmp_yaml_path = tmp_npy_path.replace(".npy", ".yaml")
#         with open(tmp_yaml_path, "w") as f:
#             yaml_module.dump(loaded_data["metadata"], f)
#
#     try:
#         results = analyze_cell_filter_data(
#             npy_path=tmp_npy_path,
#             yaml_path=tmp_yaml_path,
#             output_dir=output,
#             start_frame=start_frame,
#             end_frame=end_frame,
#             tracking_params={"search_radius": search_radius},
#         )
#
#         typer.echo(f"\nAnalysis complete: {results['total_frames']} frames, {results['t1_events_detected']} T1 events")
#         for file_type, path in results["output_files"].items():
#             typer.echo(f"  {file_type}: {path}")
#
#     finally:
#         os.unlink(tmp_npy_path)
#         if tmp_yaml_path and os.path.exists(tmp_yaml_path):
#             os.unlink(tmp_yaml_path)


@app.command()
def tension(
    mask: str = typer.Option(..., "--mask", help="Path to segmentation mask .npy file"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save the VMSI model (pickle)"),
    is_labelled: bool = typer.Option(True, "--labelled", help="Mask already labelled"),
    optimiser: str = typer.Option("nlopt", "--optimiser", help="Optimiser for VMSI (nlopt or matlab)"),
    verbose: bool = typer.Option(False, "--verbose"),
):
    """Run TensionMap VMSI analysis on a segmentation mask."""
    import pickle

    import numpy as np

    from ..tension.integration import run_tensionmap_analysis

    data = np.load(mask, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data_dict = data.item()
        if isinstance(data_dict, dict) and "segmentation_mask" in data_dict:
            mask_arr = data_dict["segmentation_mask"]
        else:
            mask_arr = data
    elif isinstance(data, dict) and "segmentation_mask" in data:
        mask_arr = data["segmentation_mask"]
    else:
        mask_arr = data

    mask_arr = np.asarray(mask_arr)

    model = run_tensionmap_analysis(
        mask_arr,
        is_labelled=is_labelled,
        optimiser=optimiser,
        verbose=verbose,
    )

    if output:
        with open(output, "wb") as f:
            pickle.dump(model, f)
        typer.echo(f"VMSI model saved to {output}")
    else:
        typer.echo("VMSI analysis completed (model not saved)")


@app.command()
def graph(
    input: str = typer.Option(..., "--input", "-i", help="Path to Zarr store with extracted data"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for plots"),
    fov: int = typer.Option(..., "--fov", help="FOV index"),
    pattern: int = typer.Option(..., "--pattern", help="Pattern/cell index"),
    start_frame: int | None = typer.Option(None, "--start-frame", "-s", help="Starting frame (default: 0)"),
    end_frame: int | None = typer.Option(None, "--end-frame", "-e", help="Ending frame (exclusive, default: all)"),
    plot: bool = typer.Option(False, "--plot", help="Generate boundary visualization plots"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Visualize cell boundaries (doublets, triplets, quartets) from extracted Zarr data."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    if not Path(input).exists():
        typer.echo(f"Error: Input file does not exist: {input}", err=True)
        raise typer.Exit(1)

    from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

    from ..graph.adjacency import BoundaryPixelTracker
    from ..graph.zarr_loader import ZarrSegmentationLoader

    loader = ZarrSegmentationLoader()
    tracker = BoundaryPixelTracker()

    typer.echo(f"Loading sequence: FOV {fov}, Pattern {pattern}")
    loaded_data = loader.load_cell_filter_data(input, fov, pattern)
    segmentation_masks = np.asarray(loaded_data["segmentation_masks"])
    nuclei_masks = np.asarray(loaded_data["nuclei_masks"]) if loaded_data["nuclei_masks"] is not None else None

    if segmentation_masks.ndim != 3:
        typer.echo(f"Error: Expected 3D segmentation masks, got shape {segmentation_masks.shape}", err=True)
        raise typer.Exit(1)

    n_frames = segmentation_masks.shape[0]
    start = start_frame if start_frame is not None else 0
    end = end_frame if end_frame is not None else n_frames

    if start < 0 or end > n_frames or start >= end:
        typer.echo(f"Error: Invalid frame range [{start}, {end}) for {n_frames} frames", err=True)
        raise typer.Exit(1)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Processing frames {start} to {end - 1} ({end - start} frames)")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    )
    progress.start()

    n_total = end - start
    plot_task: TaskID | None = None
    if plot:
        plot_task = progress.add_task("Generating plots", total=n_total)

    try:
        for frame_idx in range(start, end):
            mask = segmentation_masks[frame_idx]
            nuclei_mask_frame = nuclei_masks[frame_idx] if nuclei_masks is not None else None
            boundaries = tracker.extract_boundaries(mask)

            if plot and plot_task is not None:
                fig, _ = tracker.plot_4panel_figure(mask, nuclei_mask_frame, boundaries, frame_idx)
                out_path = output_dir / f"frame_{frame_idx:04d}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                progress.update(plot_task, completed=frame_idx - start + 1)

        progress.stop()
        typer.echo(f"Done. Output saved to {output_dir}")
    except Exception:
        progress.stop()
        raise


@app.command()
def info(  # noqa: C901
    input: str = typer.Option(..., "--input", "-i", help="Path to Zarr store"),
    plot: str | None = typer.Option(None, "--plot", "-p", help="Plot a dataset slice: 'path,(dim0,dim1,...)'"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save plot to PNG file"),
):
    """Print Zarr store structure or plot a dataset slice."""
    import matplotlib.pyplot as plt
    import numpy as np
    import zarr

    path = Path(input)
    if not path.exists():
        typer.echo(f"Error: Path not found: {input}", err=True)
        raise typer.Exit(1)

    def print_zarr_tree(group: zarr.Group, prefix: str = "") -> None:
        for key in sorted(group.keys()):
            item = group[key]
            if isinstance(item, zarr.Array):
                typer.echo(f"{prefix}{key}: array {item.shape} {item.dtype}")
            else:
                typer.echo(f"{prefix}{key}/ (group)")
                for k, v in item.attrs.items():
                    typer.echo(f"{prefix}  attr {k}: {v}")
                print_zarr_tree(item, prefix + "  ")

    root = zarr.open(path, mode="r")

    if plot is not None:
        if "," not in plot or "(" not in plot or ")" not in plot:
            typer.echo("Error: Invalid --plot format. Expected 'path,(dim0,dim1,...)'", err=True)
            raise typer.Exit(1)

        path_part, slice_part = plot.split(",", 1)
        path_part = path_part.strip()
        slice_part = slice_part.strip()

        if not slice_part.startswith("(") or not slice_part.endswith(")"):
            typer.echo("Error: Invalid --plot format. Expected 'path,(dim0,dim1,...)'", err=True)
            raise typer.Exit(1)

        slice_str = slice_part[1:-1]
        try:
            indices = [int(x.strip()) for x in slice_str.split(",") if x.strip() != ""]
        except ValueError:
            typer.echo("Error: Slice indices must be integers", err=True)
            raise typer.Exit(1) from None

        if path_part not in root:
            typer.echo(f"Error: Dataset not found: {path_part}", err=True)
            raise typer.Exit(1)

        obj = root[path_part]
        if not isinstance(obj, zarr.Array):
            typer.echo(f"Error: Not an array: {path_part}", err=True)
            raise typer.Exit(1)

        data = obj[...]

        try:
            sliced = data[tuple(indices)]
        except IndexError as e:
            typer.echo(f"Error: Invalid slice indices for array shape {data.shape}: {e}", err=True)
            raise typer.Exit(1) from None

        if sliced.ndim != 2:
            typer.echo(f"Error: Sliced result has {sliced.ndim} dimensions, expected 2", err=True)
            raise typer.Exit(1)

        plt.figure(figsize=(8, 6))
        plt.imshow(np.asarray(sliced), cmap="viridis")
        plt.colorbar()
        plt.title(path_part)

        if output:
            plt.savefig(output)
            typer.echo(f"Saved plot to: {output}")
        else:
            typer.echo("Error: --output is required when using --plot", err=True)
            raise typer.Exit(1)

        plt.close()
    else:
        typer.echo(f"Zarr Structure: {path}")
        typer.echo("-" * 60)
        for k, v in root.attrs.items():
            typer.echo(f"attr {k}: {v}")
        print_zarr_tree(root)



@app.command()
def save(
    zarr_path: str = typer.Option(..., "--zarr", "-z", help="Path to input Zarr store"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for TIFF files"),
):
    """Export zarr sequences to TIFF files.

    Creates two files per sequence:
    - fov_XXXX_cell_XXXX_data.tiff: Image data (T, C, H, W)
    - fov_XXXX_cell_XXXX_mask.tiff: Masks (T, 2, H, W) where ch0=cell, ch1=nucleus
    """
    from pathlib import Path

    from ..core.io.tiff_export import export_zarr_to_tiff

    zarr_p = Path(zarr_path)
    if not zarr_p.exists():
        typer.echo(f"Error: Zarr store not found: {zarr_path}", err=True)
        raise typer.Exit(1)

    count = export_zarr_to_tiff(zarr_p, Path(output))
    typer.echo(f"Saved {count} sequences to {output}")

@app.command()
def viewer():
    """Launch the interactive Zarr viewer."""
    from PySide6.QtWidgets import QApplication

    from ..viewer.ui.main_window import MainWindow

    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec())


def main():
    """Main entry point for migrama CLI."""
    app()


if __name__ == "__main__":
    main()
