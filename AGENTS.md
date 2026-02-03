# AGENTS.md - Development Guidelines for migrama Repository

This document contains development guidelines for contributors to the migrama repository. For usage instructions, see the documentation in the docs/ directory.

## Build/Lint/Test Commands

This is a monolithic Python package. Run commands from the repository root.

### General Commands
- **Install**: `uv sync` or `uv pip install -e .`
- **Run CLI**: `uv run migrama <command> ...`
- **Lint**: `ruff check --fix .`
- **Type check**: No explicit typecheck command configured

### Testing
- **Run all tests**: `uv run pytest tests/ -v`
- **Run single test**: `uv run pytest tests/test_specific.py::test_function -v`
- **Test files**: Located in `tests/` directory (test_analyze.py, test_extract.py, test_pattern.py, etc.)
- **AI agents**: Do NOT run full test suite automatically - tests are slow. Prefer targeted manual testing of changed functionality. Only run specific tests when explicitly requested.

### Module Entry Points
- **migrama pattern**: `migrama pattern -p patterns.nd2 --fovs "all" -o patterns.csv`
- **migrama average**: `migrama average -c cells.nd2 --output-dir ./averaged`
- **migrama analyze**: `migrama analyze -c cells.nd2 --csv patterns.csv --cache ./cache.zarr -o analysis.csv --nc 1 --n-cells 4`
- **migrama extract**: `migrama extract -c cells.nd2 --csv analysis.csv --cache ./cache.zarr -o extracted.zarr --nc 1`
- **migrama convert**: `migrama convert -i tiff_folder/ -o converted.zarr --nc 0 --cell-channels 1,2`
- **migrama info**: `migrama info -i extracted.zarr`
- **migrama graph**: `migrama graph -i extracted.zarr --fov 0 --pattern 0 -o ./output --plot`
- **migrama tension**: `migrama tension --mask xxx.npy`
- **migrama viewer**: `migrama viewer`

## Code Style Guidelines

### Python Version & Formatting
- **Python**: >=3.11 required
- **Type hints**: Use modern union syntax (`int | str` instead of `Union[int, str]`)
- **Docstrings**: Google-style with sections (Parameters, Returns, etc.)
- **Line length**: Not strictly enforced, aim for readability

### Import Organization
```python
# Standard library
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models

# Local imports
from migrama.core import PatternDetector, CellCropper
```

### Class Structure
- Use section comments: `# Constructor`, `# Private Methods`, `# Public Methods`
- Exception handling with try/except blocks and logging
- Use f-strings for formatting
- Prefer `raise ValueError("message")` over generic exceptions

### Naming Conventions
- **Classes**: PascalCase (e.g., `CellposeCounter`, `PatternDetector`)
- **Functions/variables**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: prefix with underscore (`_method_name`)

### Documentation
- **Location**: All documentation files must be placed in the `docs/` directory
- **Formats**: Use Markdown (.md) for documentation
- **Structure**: Follow the existing documentation structure in docs/

### Dependencies
- **migrama**: numpy, zarr, pydantic>=2.0.0, typer, cellpose>4, torch, torchvision, matplotlib, opencv-python, scikit-image, scipy, tifffile, networkx, btrack, nd2, xarray, dask, pyyaml>=6.0.2, PySide6, rich

### Testing
- **Test Location**: Tests are in `tests/` directory
- **Test Guidelines**: Use assertions for validation, not specific numeric results for data-dependent tests
- **Data Fixtures**: Tests use synthetic data generated in `tests/conftest.py` and `tests/data.py`

## Pipeline Architecture

The migrama pipeline processes micropatterned timelapse microscopy data through four stages:

### Stage 1: Pattern Detection (`migrama pattern`)
- **Input**: Pattern image file (.nd2 or .tif/.tiff, single-frame, single-channel)
- **Output**: `patterns.csv` with columns: `cell,fov,x,y,w,h`
  - `cell`: pattern index within FOV
  - `fov`: field of view index
  - `x,y,w,h`: bounding box coordinates

### Stage 2: Cell Analysis (`migrama analyze`)
- **Input**: Cells file (.nd2 or .tif/.tiff) + `patterns.csv`
- **Output**:
  - `analysis.csv` with columns: `cell,fov,x,y,w,h,t0,t1`
    - `t0,t1`: longest contiguous frame range where target cell count is maintained
  - `cache.zarr`: cached cell masks for the extract step

### Stage 3: Sequence Extraction (`migrama extract`)
- **Input**: Cells file (.nd2 or .tif/.tiff) + `analysis.csv` + optional `cache.zarr`
- **Output**: `extracted.zarr` containing:
  - Cropped timelapse sequences
  - Tracked cell masks (cell-first tracking)
  - Derived nuclei masks (Otsu threshold within tracked cells)

### Stage 4: Graph Analysis (`migrama graph`)
- **Input**: `extracted.zarr` (tracked segmentation layer)
- **Output**: Boundary visualization plots (doublets, triplets, quartets)

## Core Classes

### Pattern Detection
- `PatternDetector`: Detects patterns from ND2 files, outputs CSV
- `DetectorParameters`: Configuration for detection algorithm

### Cell Cropping
- `CellCropper`: Loads cells.nd2 + CSV, provides cropping methods
- `BoundingBox`: Dataclass for bounding box coordinates
- `load_bboxes_csv()`: Utility to load CSV into dict[fov, list[BoundingBox]]

### Segmentation & Tracking
- `CellposeCounter`: Counts cells in images
- `CellposeSegmenter`: Segments cells using Cellpose
- `CellTracker`: Tracks cells across frames

### Graph Analysis
- `CellGrapher`: Builds and analyzes region adjacency graphs
