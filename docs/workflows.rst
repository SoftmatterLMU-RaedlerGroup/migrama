Workflows and Examples
======================

This section provides detailed workflows and examples for common analysis tasks using Migrama.

Complete Analysis Pipeline
--------------------------

This workflow demonstrates the full Migrama pipeline using cell-first tracking.

.. code-block:: bash

   #!/bin/bash
   # Complete Migrama analysis pipeline

   # Configuration
   PATTERNS_FILE="data/patterns.nd2"
   CELLS_FILE="data/cells.nd2"
   OUTPUT_DIR="results"
   NUCLEI_CHANNEL=1
   N_CELLS=4
   MIN_FRAMES=20

   mkdir -p $OUTPUT_DIR

   # Step 1: Pattern detection
   echo "Step 1: Detecting patterns..."
   migrama pattern \
     --patterns $PATTERNS_FILE \
     --fovs "all" \
     --output $OUTPUT_DIR/patterns.csv \
     --plot $OUTPUT_DIR/pattern_plots/

   # Step 2: Cell count analysis
   echo "Step 2: Analyzing cell counts..."
   migrama analyze \
     --cells $CELLS_FILE \
     --csv $OUTPUT_DIR/patterns.csv \
     --cache $OUTPUT_DIR/cache.zarr \
     --output $OUTPUT_DIR/analysis.csv \
     --nc $NUCLEI_CHANNEL \
     --n-cells $N_CELLS

   # Step 3: Data extraction with cell-first tracking
   echo "Step 3: Extracting sequences..."
   migrama extract \
     --cells $CELLS_FILE \
     --csv $OUTPUT_DIR/analysis.csv \
     --cache $OUTPUT_DIR/cache.zarr \
     --output $OUTPUT_DIR/extracted.zarr \
     --nc $NUCLEI_CHANNEL \
     --min-frames $MIN_FRAMES

   # Step 4: Boundary visualization (first sequence)
   echo "Step 4: Visualizing boundaries..."
   migrama graph \
     --input $OUTPUT_DIR/extracted.zarr \
     --output $OUTPUT_DIR/boundaries \
     --fov 0 --pattern 0 --sequence 0 \
     --plot

   # Step 5: Export to TIFF (optional)
   echo "Step 5: Exporting to TIFF..."
   migrama save \
     --zarr $OUTPUT_DIR/extracted.zarr \
     --output $OUTPUT_DIR/tiffs/

   echo "Pipeline complete!"

Python API Workflow
-------------------

.. code-block:: python

   #!/usr/bin/env python
   """Complete Migrama workflow using Python API."""

   from pathlib import Path
   from migrama.core.pattern import PatternDetector
   from migrama.core.pattern.source import Nd2PatternFovSource
   from migrama.core.cell_source import Nd2CellFovSource
   from migrama.analyze import Analyzer
   from migrama.extract import Extractor

   # Configuration
   config = {
       "patterns_file": "data/patterns.nd2",
       "cells_file": "data/cells.nd2",
       "output_dir": Path("results"),
       "nuclei_channel": 1,
       "n_cells": 4,
       "min_frames": 20,
   }

   config["output_dir"].mkdir(parents=True, exist_ok=True)

   # Step 1: Pattern detection
   print("Step 1: Detecting patterns...")
   pattern_source = Nd2PatternFovSource(config["patterns_file"])
   detector = PatternDetector(source=pattern_source)
   records = detector.detect_all()
   detector.save_csv(records, config["output_dir"] / "patterns.csv")
   print(f"Detected {len(records)} patterns")

   # Step 2: Cell count analysis
   print("Step 2: Analyzing cell counts...")
   cell_source = Nd2CellFovSource(config["cells_file"])
   analyzer = Analyzer(
       source=cell_source,
       csv_path=str(config["output_dir"] / "patterns.csv"),
       cache_path=str(config["output_dir"] / "cache.zarr"),
       nuclei_channel=config["nuclei_channel"],
       n_cells=config["n_cells"],
   )
   analysis_records = analyzer.analyze(str(config["output_dir"] / "analysis.csv"))
   print(f"Analyzed {len(analysis_records)} patterns")

   # Step 3: Data extraction
   print("Step 3: Extracting sequences...")
   extractor = Extractor(
       source=cell_source,
       analysis_csv=str(config["output_dir"] / "analysis.csv"),
       output_path=str(config["output_dir"] / "extracted.zarr"),
       nuclei_channel=config["nuclei_channel"],
       cache_path=str(config["output_dir"] / "cache.zarr"),
   )
   n_sequences = extractor.extract(min_frames=config["min_frames"])
   print(f"Extracted {n_sequences} sequences")

   print("Workflow complete!")

Processing Specific FOVs
------------------------

Process only specific fields of view for testing or batch processing:

.. code-block:: bash

   # Process single FOV
   migrama pattern --patterns data/patterns.nd2 --fovs "0" -o patterns_fov0.csv

   # Process range of FOVs
   migrama pattern --patterns data/patterns.nd2 --fovs "0-4" -o patterns_fov0-4.csv

   # Process specific FOVs
   migrama pattern --patterns data/patterns.nd2 --fovs "0,2,5,8-10" -o patterns_selected.csv

   # Process all FOVs
   migrama pattern --patterns data/patterns.nd2 --fovs "all" -o patterns_all.csv

Batch Processing Multiple Datasets
----------------------------------

.. code-block:: python

   import subprocess
   from pathlib import Path

   def process_dataset(name, patterns, cells, n_cells):
       """Process a single dataset."""
       output_dir = Path(f"results/{name}")
       output_dir.mkdir(parents=True, exist_ok=True)

       # Pattern detection
       subprocess.run([
           "migrama", "pattern",
           "--patterns", patterns,
           "--fovs", "all",
           "--output", str(output_dir / "patterns.csv"),
           "--plot", str(output_dir / "plots"),
       ], check=True)

       # Analysis
       subprocess.run([
           "migrama", "analyze",
           "--cells", cells,
           "--csv", str(output_dir / "patterns.csv"),
           "--cache", str(output_dir / "cache.zarr"),
           "--output", str(output_dir / "analysis.csv"),
           "--nc", "1",
           "--n-cells", str(n_cells),
       ], check=True)

       # Extraction
       subprocess.run([
           "migrama", "extract",
           "--cells", cells,
           "--csv", str(output_dir / "analysis.csv"),
           "--cache", str(output_dir / "cache.zarr"),
           "--output", str(output_dir / "extracted.zarr"),
           "--nc", "1",
           "--min-frames", "20",
       ], check=True)

       return output_dir

   # Process multiple datasets
   datasets = [
       {"name": "exp1", "patterns": "data/exp1_patterns.nd2", "cells": "data/exp1_cells.nd2", "n_cells": 4},
       {"name": "exp2", "patterns": "data/exp2_patterns.nd2", "cells": "data/exp2_cells.nd2", "n_cells": 6},
   ]

   for ds in datasets:
       print(f"Processing {ds['name']}...")
       process_dataset(**ds)

Inspecting Results
------------------

Use ``migrama info`` to inspect Zarr store structure:

.. code-block:: bash

   # Print structure
   migrama info --input results/extracted.zarr

   # Plot a dataset slice
   migrama info --input results/extracted.zarr \
     --plot "fov000/cell000/seq000/data,(0,0)" \
     --output frame0_channel0.png

Working with TIFF Files
-----------------------

If your data is in TIFF format instead of ND2:

.. code-block:: bash

   # Pattern detection from averaged TIFFs
   migrama pattern \
     --patterns /path/to/tiff_folder/ \
     --avg \
     --fovs "all" \
     --output patterns.csv

   # Analysis with TIFF input
   migrama analyze \
     --cells /path/to/cells_folder/ \
     --tiff \
     --csv patterns.csv \
     --cache cache.zarr \
     --output analysis.csv \
     --nc 1 \
     --n-cells 4

   # Extraction with TIFF input
   migrama extract \
     --cells /path/to/cells_folder/ \
     --tiff \
     --csv analysis.csv \
     --cache cache.zarr \
     --output extracted.zarr \
     --nc 1

Converting TIFF to Zarr
-----------------------

Convert raw TIFF folders to Zarr format:

.. code-block:: bash

   migrama convert \
     --input /path/to/tiff_folder \
     --output converted.zarr \
     --nc 0 \
     --min-frames 10

Tension Map Analysis
--------------------

Run VMSI tension analysis on segmentation masks:

.. code-block:: bash

   migrama tension \
     --mask segmentation.npy \
     --output tension_model.pkl \
     --optimiser nlopt \
     --verbose

Tips and Best Practices
-----------------------

1. **Start small**: Test with ``--fovs "0"`` before processing all FOVs
2. **Check plots**: Use ``--plot`` to verify pattern detection
3. **Cache masks**: Use ``--cache`` to avoid re-segmentation
4. **GPU acceleration**: Install PyTorch with CUDA for faster Cellpose
5. **Memory management**: Process FOVs in batches for large datasets

Troubleshooting
---------------

**No patterns detected**
   - Check that the patterns file is correctly formatted
   - Try adjusting detection parameters

**No valid frame ranges (t0=-1, t1=-1)**
   - Adjust ``--n-cells`` to match your data
   - Check cell segmentation quality

**Out of memory**
   - Process fewer FOVs at once with ``--fovs``
   - Reduce frame range if possible

**Poor tracking**
   - Ensure cells are well-separated
   - Check that segmentation masks are accurate
