#!/bin/bash
# Example pipeline for processing micropatterned timelapse data

# Step 1: Detect patterns
uv run migrama pattern --patterns ~/data/20250812/20250812_MDCK_LK_timelapse_patterns_before.nd2 --fovs "0" -o ~/results/20250812/patterns.csv --plot ~/plots/20250812/patterns

# Step 2: Analyze cells and find valid time intervals
uv run migrama analyze --cells ~/data/20250812/20250812_MDCK_LK_timelapse.nd2 --csv ~/results/20250812/patterns.csv --cache ~/results/20250812/cache.zarr -o ~/results/20250812/analysis.csv --n-cells 4

# Step 3: Extract sequences with tracking
uv run migrama extract --cells ~/data/20250812/20250812_MDCK_LK_timelapse.nd2 --csv ~/results/20250812/analysis.csv --cache ~/results/20250812/cache.zarr -o ~/results/20250812/extracted.zarr
