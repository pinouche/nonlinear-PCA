#!/bin/bash

# Description: This script runs the read_results.py script to analyze experimental results.
# You can pass arguments instead of editing the file.
#
# Usage examples:
#   ./run_read_results.sh -d ionosphere -p true -k 2
#   ./run_read_results.sh --dataset circles --partial_contrib false --num_components 3

set -euo pipefail

# Resolve repository root (directory where this script resides)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (can be overridden via CLI flags)
DATASET="spheres"
# Valid options: "true" or "false"
PARTIAL_CONTRIB="true"
# Number of PCA components (k) to read/plot results for
NUM_COMPONENTS=2

print_help() {
  cat <<EOF
Usage: $0 [options]

Options:
  -d, --dataset <name>           Dataset name (e.g., ionosphere, circles)
  -p, --partial_contrib <bool>   Whether partial contribution was used: true|false
  -k, --num_components <int>     PCA dimensionality k to analyze (e.g., 2)
  -h, --help                     Show this help and exit

Examples:
  $0 -d ionosphere -p true -k 2
  $0 --dataset circles --partial_contrib false --num_components 3
EOF
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset)
      DATASET=${2:-}
      shift 2
      ;;
    -p|--partial_contrib)
      PARTIAL_CONTRIB=${2:-}
      shift 2
      ;;
    -k|--num_components)
      NUM_COMPONENTS=${2:-}
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use -h or --help for usage." >&2
      exit 1
      ;;
  esac
done

# Basic validation
if [[ "$PARTIAL_CONTRIB" != "true" && "$PARTIAL_CONTRIB" != "false" ]]; then
  echo "Error: --partial_contrib must be 'true' or 'false' (got '$PARTIAL_CONTRIB')." >&2
  exit 1
fi

if ! [[ "$NUM_COMPONENTS" =~ ^[0-9]+$ ]]; then
  echo "Error: --num_components must be a positive integer (got '$NUM_COMPONENTS')." >&2
  exit 1
fi

# --- Script Execution ---
echo "Running analysis for dataset: $DATASET with partial contribution: $PARTIAL_CONTRIB and k=$NUM_COMPONENTS"
echo "Using read_results at: $SCRIPT_DIR/es_pca/read_results.py"

# Execute the Python script with the configured parameters
# Use short -k to avoid any long-option mismatch; Python accepts both.
# Ensure Python resolves this repository first for module imports
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" \
python "$SCRIPT_DIR/es_pca/read_results.py" \
  --dataset "$DATASET" \
  --partial_contrib "$PARTIAL_CONTRIB" \
  -k "$NUM_COMPONENTS"

echo "Analysis script finished."