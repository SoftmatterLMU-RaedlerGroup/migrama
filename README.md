# Migrama: Cell Migration Automated Analysis

A toolkit for automated analysis of cell migration in timelapse microscopy images, focusing on cell tracking and pattern recognition in micropatterned cellular arrays.

## Installation

```bash
git clone <repository-url>
cd migrama
uv sync
```

## Commands

```bash
uv run migrama --help          # Show all commands
uv run migrama pattern --help  # Detect micropatterns, save bounding boxes to CSV
uv run migrama average --help  # Average time-lapse frames for pattern detection
uv run migrama analyze --help  # Analyze cell counts, find valid frame ranges
uv run migrama extract --help  # Extract sequences with cell-first tracking
uv run migrama save --help     # Export Zarr sequences to TIFF files
uv run migrama convert --help  # Convert TIFF folders to Zarr
uv run migrama graph --help    # Visualize cell boundaries from extracted data
uv run migrama tension --help  # Run TensionMap VMSI analysis
uv run migrama info --help     # Inspect Zarr store structure
uv run migrama viewer          # Launch interactive viewer
```

## Example

See [`examples/run_pipeline.sh`](examples/run_pipeline.sh) for a complete pipeline example.

## Documentation

Full documentation: https://cell-lisca.readthedocs.io/

## License

MIT License
