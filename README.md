# Migrama: Cell Migration Automated Analysis

A toolkit for automated analysis of cell migration in timelapse microscopy images, focusing on cell tracking and pattern recognition in micropatterned cellular arrays.

## Installation

```bash
git clone <repository-url>
cd migrama
uv sync
```

## Pipeline Commands

The main analysis pipeline runs in four stages:

```bash
# 1. Detect micropatterns and save bounding boxes
uv run migrama pattern -p patterns.nd2 --fovs "all" -o patterns.csv

# 2. Analyze cell counts and find valid frame ranges
uv run migrama analyze -c cells.nd2 --csv patterns.csv --cache cache.zarr -o analysis.csv --nc 1 --n-cells 4

# 3. Extract sequences with cell-first tracking
uv run migrama extract -c cells.nd2 --csv analysis.csv --cache cache.zarr -o extracted.zarr --nc 1

# 4. Export Zarr sequences to TIFF files
uv run migrama save --zarr extracted.zarr --output ./tiffs/

# 5. Launch interactive viewer
uv run migrama viewer
```

## Utility Commands

```bash
# Visualize cell boundaries from extracted data
uv run migrama graph -i extracted.zarr --fov 0 --pattern 0 -o ./output --plot

# Average time-lapse frames (useful for noisy pattern images)
uv run migrama average -c cells.nd2 -o ./averaged

# Convert TIFF folder to Zarr format
uv run migrama convert -i tiff_folder/ -o converted.zarr --nc 0 --cell-channels 1,2

# Inspect Zarr store structure
uv run migrama info -i extracted.zarr

# Run TensionMap VMSI analysis
uv run migrama tension --mask mask.npy
```

## Example

See [`examples/run_pipeline.sh`](examples/run_pipeline.sh) for a complete pipeline example.

## Documentation

Full documentation: https://cell-lisca.readthedocs.io/

## License

MIT License
