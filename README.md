# Migrama: Cell Migration Automated Analysis

A comprehensive toolkit for automated analysis of cell migration in timelapse microscopy images, focusing on cell tracking, pattern recognition, and topological transitions in cellular arrays.

## Overview

Migrama is a monolithic Python package that provides specialized tools for processing and analyzing microscopy data of cells grown on micropatterns. The suite consists of eight integrated modules that work together to provide a complete analysis pipeline:

1. **migrama.pattern**: Pattern detection and annotation for micropatterned microscopy images
2. **migrama.analyze**: Cell counting and analysis with segmentation
3. **migrama.extract**: Extract cropped sequences based on cell count criteria
4. **migrama.convert**: Convert raw TIFF folders into HDF5 for downstream analysis
5. **migrama.graph**: Boundary and junction visualization from extracted segmentation masks
6. **migrama.tension**: Integration with TensionMap for stress tensor inference
7. **migrama.viewer**: Interactive visualization and frame selection for microscopy data
8. **migrama.core**: Shared utilities and interfaces used across all modules

## Features

- **Pattern Detection**: Automatically identify and annotate micropatterns in microscopy images (migrama.pattern)
- **Cell Segmentation**: Advanced cell segmentation using Cellpose (migrama.analyze)
- **Boundary Visualization**: Inspect cell-cell boundaries and junctions from extracted masks (migrama.graph)
- **Tension Inference**: Calculate cellular stress tensors using VMSI (migrama.tension)
- **Interactive Visualization**: User-friendly interface for data exploration (migrama.viewer)
- **Data Export**: Multiple output formats for downstream analysis
- **Data Conversion**: Convert TIFF stacks into analysis-ready HDF5 (migrama.convert)

## Installation

### Prerequisites

- Python 3.11 or higher
- Either [uv](https://github.com/astral-sh/uv) package manager (recommended) or conda

### Option 1: Installation with uv (Recommended)

1. Clone the repository:

```bash
git clone <repository-url>
cd migrama
```

2. Install the package using uv:

```bash
uv sync
```

The package will be installed in editable mode, and you can run it with:

```bash
uv run migrama --help
```

### Option 2: Installation with pip

1. Clone the repository:

```bash
git clone <repository-url>
cd migrama
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
## Quick Start

### Verification

To verify installation, run:

```bash
migrama --help
```

This will show all available subcommands:

- `migrama average` - Average time-lapse frames for pattern detection
- `migrama pattern` - Pattern detection and annotation
- `migrama analyze` - Cell counting and analysis  
- `migrama extract` - Extract cropped sequences
- `migrama convert` - Convert TIFF folders to HDF5
- `migrama graph` - Boundary and junction visualization
- `migrama tension` - Tension map analysis
- `migrama info` - Inspect H5 file structure or plot a dataset slice
- `migrama viewer` - Interactive viewer

## Documentation

Full documentation is available at: https://cell-lisca.readthedocs.io/

The documentation includes:
- **Installation Guide**: Detailed setup instructions
- **Quick Start**: Step-by-step tutorial
- **Module Documentation**: Detailed API reference for all modules
- **Examples**: Practical examples and workflows
- **Contributing Guide**: How to contribute to the project

To build the documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
# Open _build/html/index.html in your browser
```

### Basic Workflow

1. **Optional Conversion**: Convert TIFF folders to HDF5

```bash
migrama convert \
  --input /path/to/tiff_folder \
  --output ./converted.h5 \
  --nc 0
```

2. **Pattern Detection**: Detect patterns and save bounding boxes to CSV

```bash
migrama pattern \
  --patterns /path/to/patterns.nd2 \
  --output ./patterns.csv
```

The output CSV has columns: `cell,fov,x,y,w,h`

3. **Cell Analysis**: Analyze cell counts and identify valid frame ranges

```bash
migrama analyze \
  --cells /path/to/cells.nd2 \
  --csv ./patterns.csv \
  --output ./analysis.csv \
  --nc 1 \
  --n-cells 4
```

The output CSV adds columns: `t0,t1` (valid frame range for each pattern).

4. **Data Extraction**: Extract cropped timelapse sequences with tracking

```bash
migrama extract \
  --cells /path/to/cells.nd2 \
  --csv ./analysis.csv \
  --output ./extracted.h5 \
  --nc 1 \
  --min-frames 20
```

5. **Boundary & Junction Visualization**: Inspect doublets, triplets, and quartets

```bash
migrama graph \
  --input ./extracted.h5 \
  --output ./analysis \
  --fov 0 \
  --pattern 0 \
  --sequence 0 \
  --plot
```

6. **Visualization**: Interactively view and select frames

```bash
migrama viewer
# Then open files through the application's file menu
```

## Project Structure

```
migrama/
├── src/                 # Source code directory
│   └── migrama/         # Main package directory
│       ├── __init__.py  # Package initialization
│       ├── cli/         # Command-line interface
│       │   ├── __init__.py
│       │   └── main.py  # Unified CLI entry point
│       ├── core/        # Shared utilities and interfaces
│       │   ├── __init__.py
│       │   ├── io/      # I/O operations
│       │   ├── models/  # Data models
│       │   ├── interfaces/ # Pipeline interfaces
│       │   ├── segmentation/ # Cell segmentation
│       │   ├── pattern/ # Pattern detection utilities
│       │   ├── tracking/ # Cell tracking
│       │   └── network/ # Graph operations
│       ├── pattern/     # Pattern detection module
│       │   └── __init__.py
│       ├── analyze/     # Cell counting and analysis
│       │   └── __init__.py
│       ├── extract/     # Data extraction module
│       │   └── __init__.py
│       ├── convert/     # TIFF-to-H5 conversion
│       │   └── __init__.py
│       ├── graph/       # Boundary and junction visualization
│       │   └── __init__.py
│       ├── tension/     # Tension map analysis
│       │   ├── __init__.py
│       │   ├── cli.py   # Tension CLI
│       │   └── integration.py
│       └── viewer/      # Interactive data viewer
│           ├── __init__.py
│           └── ui/      # User interface components
├── docs/                # Documentation
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

## Dependencies

### Core Dependencies

- Python >=3.11
- numpy
- matplotlib
- pyyaml

### Package-Specific Dependencies

**Segmentation & Analysis**:

- torch
- matplotlib
- opencv-python
- cellpose>4
- nd2
- xarray
- dask
- scikit-image
- scipy
- tifffile
- networkx
- pyyaml

**Tracking**:

- numpy
- matplotlib
- scikit-image
- networkx
- scipy
- pyyaml
- opencv-python
- btrack

**Viewer**:

- PySide6
- numpy
- matplotlib
- pyyaml>=6.0.2

## Development

### Running Tests

There are currently no tests in this repository. When tests are added, they will
live in `tests/` and can be run with `python -m pytest tests/ -v`.

### Code Style

This project follows the development guidelines outlined in [AGENTS.md](AGENTS.md), including:

- Google-style docstrings
- Type hints with modern union syntax
- Organized imports (standard, third-party, local)
- Consistent naming conventions

## License

MIT License - see individual package LICENSE files for details.

## Citation

If you use this software in your research, please cite:

```
cell-lisca: Cell Migration Studies with Live Cell Imaging of Single Cell Arrays
Tianyi Cao
```

## Support

For questions, issues, or contributions, please refer to the individual package documentation or contact the maintainer at ctyjackcao@outlook.com.
