#!/bin/bash
# Multi-height training data collection script for truck LDW/FCW project.
# Iterates over multiple camera heights and Carla towns to collect diverse training data.
#
# Usage:
#   bash tools/dashcam/collect_multi_height.sh
#
# Prerequisites:
#   - Carla server running on localhost:2000
#   - openpilot venv activated

set -euo pipefail

HEIGHTS=(1.2 1.8 2.0 2.4 2.8)
TOWNS=(Town04_Opt Town03_Opt Town06_Opt)
FRAMES_PER_RUN=2000
OUTPUT_BASE="data/training"

echo "=== Multi-Height Training Data Collection ==="
echo "Heights: ${HEIGHTS[*]}"
echo "Towns: ${TOWNS[*]}"
echo "Frames per run: $FRAMES_PER_RUN"
echo "Output base: $OUTPUT_BASE"
echo ""

for town in "${TOWNS[@]}"; do
  for height in "${HEIGHTS[@]}"; do
    output_dir="${OUTPUT_BASE}/${town}_h${height}"

    # Skip if already collected enough data
    if [ -d "$output_dir" ]; then
      count=$(find "$output_dir" -name "*.npz" 2>/dev/null | wc -l)
      if [ "$count" -ge "$FRAMES_PER_RUN" ]; then
        echo "=== Skipping: $output_dir already has $count frames ==="
        continue
      fi
    fi

    echo "=== Collecting: town=$town height=$height ==="
    python tools/dashcam/run.py \
      --town "$town" \
      --camera-height "$height" \
      --perfect-cam \
      --road-only \
      --record "$output_dir" \
      --record-only \
      --max-frames "$FRAMES_PER_RUN" \
      --fast

    echo "=== Done: $output_dir ==="
    echo ""
  done
done

echo "=== All collections complete ==="
# Print summary
total=0
for town in "${TOWNS[@]}"; do
  for height in "${HEIGHTS[@]}"; do
    output_dir="${OUTPUT_BASE}/${town}_h${height}"
    if [ -d "$output_dir" ]; then
      count=$(find "$output_dir" -name "*.npz" 2>/dev/null | wc -l)
      total=$((total + count))
      echo "  $output_dir: $count frames"
    fi
  done
done
echo "  Total: $total frames"
