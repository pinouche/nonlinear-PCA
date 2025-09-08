#!/bin/bash

# Description: This script runs the read_results.py script to analyze experimental results.
# Usage: ./run_read_results.sh

DATASET="spheres"

# Specify whether partial contribution was used during the run.
# Valid options: "true" or "false"
PARTIAL_CONTRIB="true"

# --- Script Execution ---
echo "Running analysis for dataset: $DATASET with partial contribution: $PARTIAL_CONTRIB"

# Execute the Python script with the configured parameters
python es_pca/read_results.py --dataset "$DATASET" --partial_contrib "$PARTIAL_CONTRIB"

echo "Analysis script finished."