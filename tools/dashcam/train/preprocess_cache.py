#!/usr/bin/env python3
"""Precompute warped YUV420 images for dual-camera training.

For each NPZ frame in a data directory:
  - Warp road_rgb → YUV420 (6, 128, 256) uint8  [road camera, medmodel]
  - Warp wide_rgb → YUV420 (6, 128, 256) uint8  [wide camera, sbigmodel]
  - Extract all GT labels

Saves one cache NPZ per frame in <data_dir>_cache/.
Cache files are ~18x smaller than originals (no raw RGB).

Usage:
  python tools/dashcam/train/preprocess_cache.py data/dual_camera_train/Town04_001
  # Output: data/dual_camera_train/Town04_001_cache/
  # Use --workers N to control parallelism (default: all cores)
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.tools.dashcam.train.dataset import extract_targets, rgb_to_modeld_input


def _process_one(args: tuple) -> str | None:
  """Process a single NPZ file: warp + YUV convert + extract labels → cache NPZ.

  Returns path of written cache file, or None on error.
  """
  npz_path, cache_dir, fcam_intrinsics, ecam_intrinsics = args

  out_path = cache_dir / Path(npz_path).name
  if out_path.exists():
    return str(out_path)  # already cached

  try:
    data = dict(np.load(npz_path, allow_pickle=True))
    rpyCalib = data['rpyCalib'].astype(np.float64)

    # Road camera: faithful modeld pipeline (NV12 → warp Y/UV → loadyuv 6ch)
    M_road = compute_warp_matrix(rpyCalib, fcam_intrinsics, bigmodel_frame=False)
    road_yuv = rgb_to_modeld_input(data['road_rgb'], M_road)  # (6, 128, 256) uint8

    # Wide camera: faithful modeld pipeline
    M_wide = compute_warp_matrix(rpyCalib, ecam_intrinsics, bigmodel_frame=True)
    wide_yuv = rgb_to_modeld_input(data['wide_rgb'], M_wide)  # (6, 128, 256) uint8

    # Extract GT labels
    targets = extract_targets(data)

    # Save cache
    np.savez_compressed(str(out_path), road_yuv=road_yuv, wide_yuv=wide_yuv, **targets)
    return str(out_path)
  except Exception as e:
    print(f"  ERROR {npz_path}: {e}")
    return None


def preprocess_dir(data_dir: str, workers: int | None = None) -> str:
  """Preprocess all NPZ files in data_dir, save cache in <data_dir>_cache/.

  Args:
    data_dir: path to directory with raw NPZ files (road_rgb + wide_rgb)
    workers: number of parallel workers (default: all CPU cores)

  Returns:
    cache_dir path
  """
  data_path = Path(data_dir)
  cache_path = data_path.parent / (data_path.name + '_cache')
  cache_path.mkdir(parents=True, exist_ok=True)

  npz_files = sorted(data_path.glob('*.npz'))
  if not npz_files:
    raise ValueError(f"No NPZ files found in {data_dir}")

  # Get camera intrinsics
  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  fcam_intrinsics = dc.fcam.intrinsics
  ecam_intrinsics = dc.ecam.intrinsics

  n_workers = workers if workers is not None else mp.cpu_count()
  print(f"Preprocessing {len(npz_files)} frames with {n_workers} workers...")
  print(f"  Input:  {data_path}")
  print(f"  Output: {cache_path}")

  # Build args list
  args_list = [(str(p), cache_path, fcam_intrinsics, ecam_intrinsics) for p in npz_files]

  done = 0
  errors = 0
  with mp.Pool(n_workers) as pool:
    for result in pool.imap_unordered(_process_one, args_list, chunksize=16):
      done += 1
      if result is None:
        errors += 1
      if done % 500 == 0 or done == len(npz_files):
        print(f"  {done}/{len(npz_files)} ({errors} errors)")

  print(f"Done. Cache: {cache_path} ({done - errors}/{len(npz_files)} frames)")
  return str(cache_path)


def main():
  parser = argparse.ArgumentParser(description='Precompute YUV cache for dual-camera training')
  parser.add_argument('data_dirs', nargs='+', help='Data directories with raw NPZ files')
  parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: all cores)')
  args = parser.parse_args()

  for data_dir in args.data_dirs:
    cache_dir = preprocess_dir(data_dir, args.workers)
    print(f"Cache ready: {cache_dir}\n")


if __name__ == '__main__':
  main()
