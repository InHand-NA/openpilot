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
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.tools.dashcam.train.dataset import extract_targets, rgb_to_modeld_input


def _load_clip_info(npz_path: str) -> dict:
  """Load clip_info.json from parent.parent of npz_path, or return {}."""
  clip_info_path = Path(npz_path).parent.parent / 'clip_info.json'
  if clip_info_path.exists():
    try:
      with open(clip_info_path) as f:
        return json.load(f)
    except Exception:
      pass
  return {}


def _get_rpyCalib(npz_path: str, data: dict) -> np.ndarray:
  """Get calibration RPY from clip_info.json (method A) or from npz data (fallback).

  Method A (multi-height annotated data): reads clip_info.json from the session
  directory (parent of the height subdirectory). npz_path is like:
    session_annotated/H1/000001.npz → parent.parent = session_annotated/

  Fallback (legacy dual-camera data): reads 'rpyCalib' field directly from npz.
  """
  clip_info = _load_clip_info(npz_path)
  if clip_info:
    cam = clip_info.get('camera', {})
    pitch_rad = math.radians(cam.get('pitch_deg', 0.0))
    yaw_rad = math.radians(cam.get('yaw_deg', 0.0))
    return np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)
  # Legacy: rpyCalib stored directly in NPZ
  return data['rpyCalib'].astype(np.float64)


def _get_source_rgb(npz_path: str, clip_info: dict) -> tuple[np.ndarray, np.ndarray] | None:
  """Load road_rgb/wide_rgb from the original session dir recorded in clip_info.

  Annotated NPZs omit RGB to keep file size small. clip_info.json stores
  source_session_dir (relative to annotations/ dir) so we can find the original
  frame by height tag + filename.

  source_session_dir = '..' means the session dir is the parent of annotations/.
  Relative paths are resolved against the annotations/ directory (parent.parent of npz).

  Returns (road_rgb, wide_rgb) or None if source not found.
  """
  source_session_dir = clip_info.get('source_session_dir')
  if not source_session_dir:
    return None
  p = Path(npz_path)
  # p = annotations/H1/000001.npz  →  annotations/ = p.parent.parent
  # Resolve source_session_dir relative to annotations/ so '..' -> session_dir
  clip_info_dir = p.parent.parent
  source_dir = (clip_info_dir / source_session_dir).resolve()
  tag = p.parent.name
  source_path = source_dir / tag / p.name
  if not source_path.exists():
    return None
  src = np.load(str(source_path), allow_pickle=True)
  road_rgb = src.get('road_rgb')
  wide_rgb = src.get('wide_rgb')
  if road_rgb is None or wide_rgb is None:
    return None
  return road_rgb, wide_rgb


def _process_one(args: tuple) -> str | None:
  """Process a single NPZ file: warp + YUV convert + extract labels → cache NPZ.

  Supports both legacy dual-camera NPZs (with rpyCalib field) and
  multi-height annotated NPZs (with clip_info.json in parent.parent).

  Returns path of written cache file, or None on error.
  """
  npz_path, cache_dir, fcam_intrinsics, ecam_intrinsics = args

  out_path = cache_dir / Path(npz_path).name
  if out_path.exists():
    return str(out_path)  # already cached

  try:
    data = dict(np.load(npz_path, allow_pickle=True))
    clip_info = _load_clip_info(npz_path)
    rpyCalib = _get_rpyCalib(npz_path, data)

    # Resolve road_rgb/wide_rgb: prefer inline (legacy), fall back to source_session_dir
    # (annotated NPZs omit RGB to keep file size small; source_session_dir is in clip_info)
    road_rgb = data.get('road_rgb')
    wide_rgb = data.get('wide_rgb')
    if road_rgb is None or wide_rgb is None:
      src = _get_source_rgb(npz_path, clip_info)
      if src is None:
        raise ValueError("road_rgb/wide_rgb not found in NPZ or source_session_dir")
      road_rgb, wide_rgb = src

    # Road camera: faithful modeld pipeline (NV12 → warp Y/UV → loadyuv 6ch)
    M_road = compute_warp_matrix(rpyCalib, fcam_intrinsics, bigmodel_frame=False)
    road_yuv = rgb_to_modeld_input(road_rgb, M_road)  # (6, 128, 256) uint8

    # Wide camera: faithful modeld pipeline
    M_wide = compute_warp_matrix(rpyCalib, ecam_intrinsics, bigmodel_frame=True)
    wide_yuv = rgb_to_modeld_input(wide_rgb, M_wide)  # (6, 128, 256) uint8

    # Extract GT labels (works for both legacy and annotated NPZ formats)
    targets = extract_targets(data)

    # Preserve camera_height for HeightConditionedHead (future use)
    camera_height = data.get('camera_height', np.float32(1.22))
    targets['camera_height'] = np.float32(camera_height)

    # Save cache
    np.savez_compressed(str(out_path), road_yuv=road_yuv, wide_yuv=wide_yuv, **targets)
    return str(out_path)
  except Exception as e:
    print(f"  ERROR {npz_path}: {e}")
    return None


def preprocess_dir(data_dir: str, workers: int | None = None, use_gpu: bool = False) -> str:
  """Preprocess all NPZ files in data_dir, save cache in <data_dir>_cache/.

  Args:
    data_dir: path to directory with raw NPZ files (road_rgb + wide_rgb)
    workers: number of parallel workers (default: all CPU cores; ignored with --gpu)
    use_gpu: use GPU OpenCL pipeline for pixel-identical preprocessing (single-process)

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

  print(f"Preprocessing {len(npz_files)} frames  {'[GPU OpenCL]' if use_gpu else f'[CPU {workers or mp.cpu_count()} workers]'}")
  print(f"  Input:  {data_path}")
  print(f"  Output: {cache_path}")

  if use_gpu:
    _preprocess_dir_gpu(npz_files, cache_path, fcam_intrinsics, ecam_intrinsics)
  else:
    args_list = [(str(p), cache_path, fcam_intrinsics, ecam_intrinsics) for p in npz_files]
    n_workers = workers if workers is not None else mp.cpu_count()
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


def _preprocess_dir_gpu(npz_files: list, cache_path: Path,
                         fcam_intrinsics: np.ndarray, ecam_intrinsics: np.ndarray) -> None:
  """Single-process GPU preprocessing loop using ModeldInputPreprocessorCL.

  Runs in-process (no multiprocessing) to reuse GPU buffers across frames.
  OpenCL context is not fork-safe, so a subprocess approach is avoided.
  """
  import time

  from openpilot.tools.dashcam.modeld_preprocess_cl import ModeldInputPreprocessorCL
  with ModeldInputPreprocessorCL() as prep_road, ModeldInputPreprocessorCL() as prep_wide:
    done = errors = 0
    t_start = time.monotonic()

    for npz_path in npz_files:
      out_path = cache_path / npz_path.name
      if out_path.exists():
        done += 1
        continue

      try:
        data = dict(np.load(str(npz_path), allow_pickle=True))
        clip_info = _load_clip_info(str(npz_path))
        rpyCalib = _get_rpyCalib(str(npz_path), data)

        road_rgb = data.get('road_rgb')
        wide_rgb = data.get('wide_rgb')
        if road_rgb is None or wide_rgb is None:
          src = _get_source_rgb(str(npz_path), clip_info)
          if src is None:
            raise ValueError("road_rgb/wide_rgb not found in NPZ or source_session_dir")
          road_rgb, wide_rgb = src

        M_road = compute_warp_matrix(rpyCalib, fcam_intrinsics, bigmodel_frame=False)
        M_wide = compute_warp_matrix(rpyCalib, ecam_intrinsics, bigmodel_frame=True)

        road_yuv = prep_road.process(road_rgb, M_road)
        wide_yuv = prep_wide.process(wide_rgb, M_wide)

        targets = extract_targets(data)
        targets['camera_height'] = np.float32(data.get('camera_height', np.float32(1.22)))

        np.savez_compressed(str(out_path), road_yuv=road_yuv, wide_yuv=wide_yuv, **targets)
      except Exception as e:
        print(f"  ERROR {npz_path}: {e}")
        errors += 1

      done += 1
      if done % 500 == 0 or done == len(npz_files):
        elapsed = time.monotonic() - t_start
        fps = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{len(npz_files)} ({errors} errors) | {fps:.1f} fps")

  print(f"Done. Cache: {cache_path} ({done - errors}/{len(npz_files)} frames)")


def main():
  parser = argparse.ArgumentParser(description='Precompute YUV cache for dual-camera training')
  parser.add_argument('data_dirs', nargs='+', help='Data directories with raw NPZ files')
  parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: all cores)')
  parser.add_argument('--gpu', action='store_true',
                      help='Use GPU OpenCL preprocessing (requires pyopencl; pixel-identical to modeld)')
  args = parser.parse_args()

  for data_dir in args.data_dirs:
    cache_dir = preprocess_dir(data_dir, args.workers, use_gpu=args.gpu)
    print(f"Cache ready: {cache_dir}\n")


if __name__ == '__main__':
  main()
