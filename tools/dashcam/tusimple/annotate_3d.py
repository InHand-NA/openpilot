#!/usr/bin/env python3
"""TuSimple Phase 2: 3D annotation pipeline.

Runs modeld inference on H0 reference stereo frames (narrow + wide) to produce
canonical 3D lane/edge/lead annotations, then height-transforms them for H1~H6.

Key differences from annotate_multi_height.py:
  - Inference source: H0 (reference stereo) instead of H1
  - Output directory: session_dir/3d_labels/ instead of annotations/
  - label_source: 'pretrained_h0'
  - Prefetch: only H0 frames (inference depends solely on H0)
  - Height transform baseline: H0_HEIGHT (1.22m)

Output per annotated height directory:
  <session_dir>/3d_labels/<tag>/<frame_id>.json

Usage:
  python tools/dashcam/tusimple/annotate_3d.py \\
      data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \\
      --onnx selfdrive/modeld/models/driving_vision.onnx

  # Only annotate specific heights
  python tools/dashcam/tusimple/annotate_3d.py ... --heights H1 H6

  # CPU preprocessing fallback
  python tools/dashcam/tusimple/annotate_3d.py ... --no-gpu-preprocess
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# DEV must be set before importing tinygrad
os.environ.setdefault('PYOPENCL_CTX', '')
if 'DEV' not in os.environ:
  os.environ['DEV'] = 'CUDA'

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.annotate_multi_height import (
  decode_model_output,
  transform_annotation,
  _load_or_compile_model,
  _run_inference,
)
from openpilot.tools.dashcam.train.dataset import rgb_to_modeld_input
from openpilot.tools.dashcam.tusimple.config import H0_HEIGHT, HEIGHT_DEFS

TEMPORAL_SKIP = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4

PREFETCH_AHEAD = 8


def _write_annotation(
  out_dir: Path,
  frame_id: str,
  meta: dict,
  canonical: dict,
) -> None:
  """Write annotation JSON to out_dir."""
  record: dict = {
    'frame_id':        frame_id,
    'camera_height':   float(meta.get('camera_height', H0_HEIGHT)),
    'v_ego':           float(meta.get('v_ego', 0.0)),
    'world_pose':      [float(x) for x in meta.get('world_pose', [0.0] * 6)],
    'label_source':    'pretrained_h0',
  }
  for k, v in canonical.items():
    record[k] = v.tolist() if isinstance(v, np.ndarray) else v
  with open(out_dir / f'{frame_id}.json', 'w') as f:
    json.dump(record, f)


def _load_metadata(meta_path: Path) -> dict[str, dict]:
  """Load metadata.jsonl → {frame_id_str: meta_dict}."""
  result: dict[str, dict] = {}
  if not meta_path.exists():
    return result
  with open(meta_path) as f:
    for line in f:
      line = line.strip()
      if line:
        m = json.loads(line)
        result[f"{m['frame']:06d}"] = m
  return result


def _load_h0_frame(h0_dir: Path, frame_id: str, meta: dict[str, dict]) -> dict | None:
  """Load one H0 frame: road + wide PNG images + metadata."""
  road_path = h0_dir / f'road_{frame_id}.png'
  wide_path = h0_dir / f'wide_{frame_id}.png'
  if not road_path.exists() or not wide_path.exists():
    return None
  road_bgr = cv2.imread(str(road_path))
  wide_bgr = cv2.imread(str(wide_path))
  if road_bgr is None or wide_bgr is None:
    return None
  m = meta.get(frame_id, {})
  return {
    'road_rgb':      cv2.cvtColor(road_bgr, cv2.COLOR_BGR2RGB),
    'wide_rgb':      cv2.cvtColor(wide_bgr, cv2.COLOR_BGR2RGB),
    'camera_height': float(m.get('camera_height', H0_HEIGHT)),
    'v_ego':         float(m.get('v_ego', 0.0)),
    'world_pose':    m.get('world_pose', [0.0] * 6),
  }


def annotate_session(
  session_dir: Path,
  output_dir: Path,
  onnx_path: str,
  heights_to_annotate: list[str] | None,
  use_gpu_preprocess: bool = True,
) -> dict:
  """Annotate all H0 frames in session_dir, write 3D labels to output_dir.

  Returns stats dict with counts per height.
  """
  clip_info_path = session_dir / 'clip_info.json'
  if not clip_info_path.exists():
    raise FileNotFoundError(f"clip_info.json not found in {session_dir}")
  with open(clip_info_path) as f:
    clip_info = json.load(f)

  pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
  yaw_rad = math.radians(clip_info['camera']['yaw_deg'])
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  save_every: int = clip_info.get('save_every', 1)
  if save_every not in (1, 4):
    raise ValueError(f"Unsupported save_every={save_every} (must be 1 or 4)")
  buf_depth = TEMPORAL_SKIP // save_every + 1

  # Determine which heights to annotate
  available_heights = {tag: h for tag, h in clip_info.get('heights', HEIGHT_DEFS).items()}
  if heights_to_annotate:
    heights = {tag: h for tag, h in available_heights.items() if tag in heights_to_annotate}
  else:
    heights = available_heights

  if not heights:
    raise ValueError(f"No matching heights to annotate")

  # H0 source directory — used for inference
  h0_dir = session_dir / 'H0'
  if not h0_dir.exists():
    raise FileNotFoundError(f"H0 directory not found: {h0_dir}")
  frame_ids = sorted(p.stem.replace('road_', '') for p in h0_dir.glob('road_*.png'))
  if not frame_ids:
    raise ValueError(f"No frames found in {h0_dir}")

  dev = os.environ.get('DEV', 'CUDA')
  print(f"Session: {session_dir.name}")
  print(f"  Frames: {len(frame_ids)}  Heights to annotate: {list(heights.keys())}")
  print(f"  pitch={clip_info['camera']['pitch_deg']:.1f} deg  yaw={clip_info['camera']['yaw_deg']:.1f} deg")
  print(f"  Inference source: H0 (height={H0_HEIGHT}m)")
  print(f"  DEV={dev}")

  # Create output directories
  output_dir.mkdir(parents=True, exist_ok=True)
  for tag in heights:
    (output_dir / tag).mkdir(exist_ok=True)

  # Write clip_info.json to output dir
  clip_info_out = dict(clip_info)
  clip_info_out['source_session_dir'] = '..'
  clip_info_out['annotation_source'] = 'H0'
  with open(output_dir / 'clip_info.json', 'w') as f:
    json.dump(clip_info_out, f, indent=2)

  # Camera intrinsics and warp matrices
  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
  warp_wide = compute_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True)

  # Load tinygrad model
  print(f"  Loading model from {onnx_path}...")
  vision_run, input_shapes, output_slices = _load_or_compile_model(onnx_path)

  # GPU OpenCL preprocessor (fallback to CPU numpy)
  cl_prep_road = None
  cl_prep_wide = None
  if use_gpu_preprocess:
    from openpilot.tools.dashcam.modeld_preprocess_cl import ModeldInputPreprocessorCL
    print("  Using GPU OpenCL preprocessor (modeld_preprocess_cl)...")
    cl_prep_road = ModeldInputPreprocessorCL()
    cl_prep_wide = ModeldInputPreprocessorCL()

  def _preprocess(frame_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess road + wide RGB -> (6, 128, 256) uint8 YUV each."""
    if cl_prep_road is not None:
      road_yuv = cl_prep_road.process(frame_data['road_rgb'], warp_road).copy()
      wide_yuv = cl_prep_wide.process(frame_data['wide_rgb'], warp_wide).copy()
    else:
      road_yuv = rgb_to_modeld_input(frame_data['road_rgb'], warp_road)
      wide_yuv = rgb_to_modeld_input(frame_data['wide_rgb'], warp_wide)
    return road_yuv, wide_yuv

  stats = {tag: {'total': 0} for tag in heights}
  stats['H0_inferred'] = 0

  # Circular buffers for temporal context
  road_buf: deque[np.ndarray] = deque(maxlen=buf_depth)
  wide_buf: deque[np.ndarray] = deque(maxlen=buf_depth)

  # Load H0 metadata
  h0_meta = _load_metadata(h0_dir / 'metadata.jsonl')

  # Thread pools for I/O
  read_pool = ThreadPoolExecutor(max_workers=4)
  write_pool = ThreadPoolExecutor(max_workers=len(heights))

  def _submit_prefetch(fid: str) -> Future:
    return read_pool.submit(_load_h0_frame, h0_dir, fid, h0_meta)

  # Pre-fill prefetch pipeline
  prefetch_queue: deque[Future] = deque()
  for fid in frame_ids[:PREFETCH_AHEAD]:
    prefetch_queue.append(_submit_prefetch(fid))

  t_start = time.monotonic()

  try:
    for i, frame_id in enumerate(frame_ids):
      # Submit read for frame PREFETCH_AHEAD steps ahead
      next_idx = i + PREFETCH_AHEAD
      if next_idx < len(frame_ids):
        prefetch_queue.append(_submit_prefetch(frame_ids[next_idx]))

      h0_future = prefetch_queue.popleft()
      h0_data = h0_future.result()
      if h0_data is None:
        continue

      # Idempotency: skip if all heights already annotated
      all_exist = all(
        (output_dir / tag / f'{frame_id}.json').exists()
        for tag in heights
      )
      if all_exist:
        # Still need to feed temporal buffer for subsequent frames
        road_yuv, wide_yuv = _preprocess(h0_data)
        road_buf.append(road_yuv)
        wide_buf.append(wide_yuv)
        stats['H0_inferred'] += 1
        continue

      # Preprocess H0 frame
      road_yuv, wide_yuv = _preprocess(h0_data)
      road_buf.append(road_yuv)
      wide_buf.append(wide_yuv)

      # Temporal context
      road_yuv_prev = road_buf[0]
      wide_yuv_prev = wide_buf[0]

      # Tinygrad inference
      parsed = _run_inference(vision_run, input_shapes, output_slices,
                              road_yuv, wide_yuv, road_yuv_prev, wide_yuv_prev)
      canonical = decode_model_output(parsed)
      stats['H0_inferred'] += 1

      # Per-frame metadata (shared across all heights — same timestamp/pose)
      frame_meta = {
        'camera_height': h0_data['camera_height'],
        'v_ego': h0_data['v_ego'],
        'world_pose': h0_data['world_pose'],
      }

      for tag, h_k in heights.items():
        if tag == 'H1' and abs(h_k - H0_HEIGHT) < 0.01:
          # H1 is same height as H0 — use canonical directly
          anno = canonical
        else:
          anno = transform_annotation(canonical, h1=H0_HEIGHT, h_k=h_k)

        # Override camera_height in metadata for this specific height
        tag_meta = dict(frame_meta)
        tag_meta['camera_height'] = h_k

        stats[tag]['total'] += 1

        write_pool.submit(
          _write_annotation,
          output_dir / tag, frame_id, tag_meta, anno,
        )

      # Progress log
      n_done = stats['H0_inferred']
      if n_done % 500 == 0 or n_done == len(frame_ids):
        elapsed = time.monotonic() - t_start
        fps = n_done / elapsed if elapsed > 0 else 0
        eta = (len(frame_ids) - n_done) / fps if fps > 0 else 0
        print(f"  {n_done}/{len(frame_ids)} | {fps:.1f} fps | ETA {eta:.0f}s")

  finally:
    write_pool.shutdown(wait=True)
    read_pool.shutdown(wait=False)
    if cl_prep_road is not None:
      cl_prep_road.close()
      cl_prep_wide.close()

  elapsed = time.monotonic() - t_start
  print(f"\n  Done in {elapsed:.1f}s ({stats['H0_inferred'] / max(elapsed, 0.001):.1f} fps)")
  for tag in heights:
    print(f"  {tag}: {stats[tag]['total']} frames annotated")

  return stats


def main():
  parser = argparse.ArgumentParser(description='TuSimple Phase 2: 3D annotation (H0 inference + height transform)')
  parser.add_argument('session_dir', help='Session directory (contains H0/, H1/... and clip_info.json)')
  parser.add_argument('--onnx', default='selfdrive/modeld/models/driving_vision.onnx',
                      help='Path to driving_vision.onnx')
  parser.add_argument('--output', default=None,
                      help='Output directory (default: <session_dir>/3d_labels/)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='Heights to annotate (default: all in clip_info.json)')
  parser.add_argument('--no-gpu-preprocess', action='store_true',
                      help='Disable GPU OpenCL preprocessing, fall back to CPU')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"ERROR: session directory not found: {session_dir}", file=sys.stderr)
    sys.exit(1)

  if args.output:
    output_dir = Path(args.output).resolve()
  else:
    output_dir = session_dir / '3d_labels'

  try:
    annotate_session(
      session_dir=session_dir,
      output_dir=output_dir,
      onnx_path=args.onnx,
      heights_to_annotate=args.heights,
      use_gpu_preprocess=not args.no_gpu_preprocess,
    )
    print(f"\n3D labels saved to: {output_dir}")
  except KeyboardInterrupt:
    print("\n[Interrupted]")
    sys.exit(0)


if __name__ == '__main__':
  main()
