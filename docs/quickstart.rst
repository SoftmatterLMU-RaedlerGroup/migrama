Quick Start Guide
=================

This guide walks you through a typical Migrama analysis workflow from raw microscopy data to cell tracking.

Cell-First Tracking
-------------------

Migrama uses **cell-first tracking**:

1. Segment cells using all channels (Cellpose all-channel mode)
2. Track cells across frames
3. Derive nuclei by Otsu thresholding within each tracked cell

This approach provides more robust tracking than nucleus-first methods.

Basic Workflow
--------------

1. **Pattern Detection**: Extract bounding boxes for all micropatterns
2. **Cell Count Analysis**: Count cells and find valid frame ranges
3. **Data Extraction**: Extract sequences with cell-first tracking
4. **Visualization**: Inspect boundaries and junctions

Step 1: Pattern Detection
-------------------------

Detect and extract pattern bounding boxes from your microscopy data:

.. code-block:: bash

   migrama pattern \
     --patterns /path/to/patterns.nd2 \
     --fovs "all" \
     --output ./patterns.csv \
     --plot ./pattern_plots/

Options:

- ``--fovs`` (required): ``"all"`` or ranges like ``"0,2-5,8"``
- ``--plot``: Generate bbox overlay visualizations (one PNG per FOV)

Output CSV columns: ``cell,fov,x,y,w,h``

Step 2: Cell Count Analysis
---------------------------

Analyze cell counts for all patterns across the timelapse:

.. code-block:: bash

   migrama analyze \
     --cells /path/to/cells.nd2 \
     --csv ./patterns.csv \
     --cache ./cache.zarr \
     --output ./analysis.csv \
     --nc 1 \
     --n-cells 4

Options:

- ``--n-cells`` (required): Target number of cells per pattern
- ``--nc``: Nuclear channel index (stored for extract step)
- ``--cache``: Output path for segmentation mask cache

This segments cells using all channels and finds the longest contiguous
run of frames where the cell count matches ``--n-cells``.

Output CSV adds columns: ``t0,t1`` (valid frame range for each pattern)

Step 3: Data Extraction with Tracking
-------------------------------------

Extract sequences with cell-first tracking:

.. code-block:: bash

   migrama extract \
     --cells /path/to/cells.nd2 \
     --csv ./analysis.csv \
     --cache ./cache.zarr \
     --output ./extracted.zarr \
     --nc 1 \
     --min-frames 20

Options:

- ``--cache``: Load pre-computed masks (optional, re-segments without it)
- ``--nc``: Nuclear channel for deriving nuclei within tracked cells
- ``--min-frames``: Minimum frames required per sequence

The output Zarr store contains:

- ``data``: Image data (T, C, H, W)
- ``cell_masks``: Tracked cell masks (T, H, W)
- ``nuclei_masks``: Derived nuclei masks (T, H, W)

Step 4: Boundary Visualization
------------------------------

Inspect cell boundaries (doublets, triplets, quartets):

.. code-block:: bash

   migrama graph \
     --input ./extracted.zarr \
     --output ./analysis \
     --fov 0 \
     --pattern 0 \
     --sequence 0 \
     --plot

Step 5: Interactive Visualization
---------------------------------

Launch the interactive viewer:

.. code-block:: bash

   migrama viewer

Use the GUI to navigate frames and inspect data.

Step 6: Export to TIFF (Optional)
---------------------------------

Export Zarr sequences to TIFF files for use with other tools:

.. code-block:: bash

   migrama save \
     --zarr ./extracted.zarr \
     --output ./tiff_exports/

Output files per sequence:

- ``fov_XXXX_cell_XXXX_data.tiff``: Image data (T, C, H, W)
- ``fov_XXXX_cell_XXXX_mask.tiff``: Masks (T, 2, H, W) where ch0=cell, ch1=nucleus

Complete Example
----------------

.. code-block:: bash

   # 1. Detect patterns in all FOVs
   migrama pattern \
     --patterns data/patterns.nd2 \
     --fovs "all" \
     --output results/patterns.csv \
     --plot results/pattern_plots/

   # 2. Analyze cell counts (4 cells per pattern)
   migrama analyze \
     --cells data/cells.nd2 \
     --csv results/patterns.csv \
     --cache results/cache.zarr \
     --output results/analysis.csv \
     --nc 1 \
     --n-cells 4

   # 3. Extract sequences with cell-first tracking
   migrama extract \
     --cells data/cells.nd2 \
     --csv results/analysis.csv \
     --cache results/cache.zarr \
     --output results/extracted.zarr \
     --nc 1 \
     --min-frames 20

   # 4. Visualize boundaries
   migrama graph \
     --input results/extracted.zarr \
     --output results/boundaries \
     --fov 0 --pattern 0 --sequence 0 \
     --plot

   # 5. Export to TIFF (optional)
   migrama save \
     --zarr results/extracted.zarr \
     --output results/tiffs/

Tips and Best Practices
-----------------------

1. **Start with a single FOV**: Use ``--fovs "0"`` to test parameters
2. **Check pattern detection**: Use ``--plot`` to verify bounding boxes
3. **GPU acceleration**: Ensure PyTorch with CUDA is installed for faster segmentation
4. **Memory management**: Process large datasets with specific FOV ranges
5. **Quality control**: Use ``migrama info`` to inspect Zarr structure

Common Issues
-------------

1. **Pattern detection fails**: Check that patterns file has correct format
2. **No valid frame ranges**: Adjust ``--n-cells`` to match your data
3. **Memory errors**: Process fewer FOVs at once with ``--fovs``
4. **Poor tracking**: Ensure cells are well-separated in the images

Next Steps
----------

- Explore the :doc:`workflows` section for advanced pipelines
- Check the :doc:`modules/index` for module documentation
- See the :doc:`examples` page for specific use cases
