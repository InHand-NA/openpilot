#!/usr/bin/env python3
"""TuSimple Phase 2: 3D annotation pipeline.

Runs modeld inference on H0 reference stereo frames (narrow + wide) to produce
canonical 3D lane/edge/lead annotations, then height-transforms them for H1~H9.

支持两种采集模式:
  paired: 每个 main 帧自带 _prev.png → 直接加载，无需 buffer
  dense (legacy): 连续等间距帧 → circular buffer 取 prev

Output: <session_dir>/3d_labels/<tag>/<frame_id>.json

Usage:
  python tools/dashcam/tusimple/annotate_3d.py \\
      data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \\
      --onnx selfdrive/modeld/models/driving_vision.onnx
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

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


def _write_annotation(out_dir: Path, frame_id: str, meta: dict, canonical: dict) -> None:
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


def _list_main_frames(h0_dir: Path) -> list[str]:
  """List main frame IDs (exclude _prev.png)."""
  pattern = re.compile(r'^road_(\d+)\.png$')
  return sorted(m.group(1) for f in h0_dir.iterdir() if (m := pattern.match(f.name)))


def _load_h0_image(h0_dir: Path, name: str) -> np.ndarray | None:
  """Load a single H0 PNG → RGB. Returns None if not found."""
  path = h0_dir / name
  if not path.exists():
    return None
  bgr = cv2.imread(str(path))
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def annotate_session(
  session_dir: Path,
  output_dir: Path,
  onnx_path: str,
  heights_to_annotate: list[str] | None,
  use_gpu_preprocess: bool = True,
  force: bool = False,
) -> dict:
  """Annotate all H0 frames in session_dir, write 3D labels to output_dir."""
  clip_info_path = session_dir / 'clip_info.json'
  if not clip_info_path.exists():
    raise FileNotFoundError(f"clip_info.json not found in {session_dir}")
  with open(clip_info_path) as f:
    clip_info = json.load(f)

  pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
  yaw_rad = math.radians(clip_info['camera']['yaw_deg'])
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  save_mode = clip_info.get('save_mode', 'dense')

  # Determine which heights to annotate
  available_heights = {tag: h for tag, h in clip_info.get('heights', HEIGHT_DEFS).items()}
  if heights_to_annotate:
    heights = {tag: h for tag, h in available_heights.items() if tag in heights_to_annotate}
  else:
    heights = available_heights
  if not heights:
    raise ValueError("No matching heights to annotate")

  h0_dir = session_dir / 'H0'
  if not h0_dir.exists():
    raise FileNotFoundError(f"H0 directory not found: {h0_dir}")
  frame_ids = _list_main_frames(h0_dir)
  if not frame_ids:
    raise ValueError(f"No frames found in {h0_dir}")

  dev = os.environ.get('DEV', 'CUDA')
  print(f"Session: {session_dir.name}")
  print(f"  Frames: {len(frame_ids)}  Heights: {list(heights.keys())}  mode={save_mode}")
  print(f"  pitch={clip_info['camera']['pitch_deg']:.1f} deg  yaw={clip_info['camera']['yaw_deg']:.1f} deg  DEV={dev}")

  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / 'H0').mkdir(exist_ok=True)
  for tag in heights:
    (output_dir / tag).mkdir(exist_ok=True)

  clip_info_out = dict(clip_info)
  clip_info_out['source_session_dir'] = '..'
  clip_info_out['annotation_source'] = 'H0'
  with open(output_dir / 'clip_info.json', 'w') as f:
    json.dump(clip_info_out, f, indent=2)

  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
  warp_wide = compute_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True)

  print(f"  Loading model from {onnx_path}...")
  vision_run, input_shapes, output_slices = _load_or_compile_model(onnx_path)

  cl_prep_road = None
  cl_prep_wide = None
  if use_gpu_preprocess:
    from openpilot.tools.dashcam.modeld_preprocess_cl import ModeldInputPreprocessorCL
    print("  Using GPU OpenCL preprocessor...")
    cl_prep_road = ModeldInputPreprocessorCL()
    cl_prep_wide = ModeldInputPreprocessorCL()

  def _preprocess_rgb(road_rgb: np.ndarray, wide_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cl_prep_road is not None:
      return cl_prep_road.process(road_rgb, warp_road).copy(), cl_prep_wide.process(wide_rgb, warp_wide).copy()
    return rgb_to_modeld_input(road_rgb, warp_road), rgb_to_modeld_input(wide_rgb, warp_wide)

  stats = {tag: {'total': 0} for tag in heights}
  stats['H0_inferred'] = 0

  h0_meta = _load_metadata(h0_dir / 'metadata.jsonl')
  write_pool = ThreadPoolExecutor(max_workers=len(heights))

  # Dense mode: need circular buffer for temporal context
  if save_mode != 'paired':
    save_every: int = clip_info.get('save_every', 4)
    buf_depth = TEMPORAL_SKIP // save_every + 1
  road_buf: deque[np.ndarray] = deque(maxlen=buf_depth if save_mode != 'paired' else 1)
  wide_buf: deque[np.ndarray] = deque(maxlen=buf_depth if save_mode != 'paired' else 1)

  t_start = time.monotonic()

  try:
    for i, frame_id in enumerate(frame_ids):
      # Idempotency
      all_exist = not force and all(
        (output_dir / tag / f'{frame_id}.json').exists() for tag in heights
      )

      if save_mode == 'paired':
        # ── Paired mode: load _prev.png directly ──
        road_main = _load_h0_image(h0_dir, f'road_{frame_id}.png')
        wide_main = _load_h0_image(h0_dir, f'wide_{frame_id}.png')
        if road_main is None or wide_main is None:
          continue

        if all_exist:
          stats['H0_inferred'] += 1
          continue

        road_prev = _load_h0_image(h0_dir, f'road_{frame_id}_prev.png')
        wide_prev = _load_h0_image(h0_dir, f'wide_{frame_id}_prev.png')
        # Fallback: if no prev, use main as prev (first frame or missing)
        if road_prev is None:
          road_prev = road_main
        if wide_prev is None:
          wide_prev = wide_main

        road_yuv, wide_yuv = _preprocess_rgb(road_main, wide_main)
        road_yuv_prev, wide_yuv_prev = _preprocess_rgb(road_prev, wide_prev)

      else:
        # ── Dense mode (legacy): circular buffer ──
        road_main = _load_h0_image(h0_dir, f'road_{frame_id}.png')
        wide_main = _load_h0_image(h0_dir, f'wide_{frame_id}.png')
        if road_main is None or wide_main is None:
          continue

        road_yuv, wide_yuv = _preprocess_rgb(road_main, wide_main)
        road_buf.append(road_yuv)
        wide_buf.append(wide_yuv)

        if all_exist:
          stats['H0_inferred'] += 1
          continue

        road_yuv_prev = road_buf[0]
        wide_yuv_prev = wide_buf[0]

      # Inference
      parsed = _run_inference(vision_run, input_shapes, output_slices,
                              road_yuv, wide_yuv, road_yuv_prev, wide_yuv_prev)
      canonical = decode_model_output(parsed)
      stats['H0_inferred'] += 1

      m = h0_meta.get(frame_id, {})
      frame_meta = {
        'camera_height': float(m.get('camera_height', H0_HEIGHT)),
        'v_ego': float(m.get('v_ego', 0.0)),
        'world_pose': m.get('world_pose', [0.0] * 6),
      }

      # Write H0 canonical annotation (untransformed, at reference height)
      h0_meta = dict(frame_meta)
      h0_meta['camera_height'] = H0_HEIGHT
      write_pool.submit(_write_annotation, output_dir / 'H0', frame_id, h0_meta, canonical)

      for tag, h_k in heights.items():
        if abs(h_k - H0_HEIGHT) < 0.01:
          anno = canonical
        else:
          anno = transform_annotation(canonical, h1=H0_HEIGHT, h_k=h_k)
        tag_meta = dict(frame_meta)
        tag_meta['camera_height'] = h_k
        stats[tag]['total'] += 1
        write_pool.submit(_write_annotation, output_dir / tag, frame_id, tag_meta, anno)

      n_done = stats['H0_inferred']
      if n_done % 500 == 0 or n_done == len(frame_ids):
        elapsed = time.monotonic() - t_start
        fps = n_done / elapsed if elapsed > 0 else 0
        eta = (len(frame_ids) - n_done) / fps if fps > 0 else 0
        print(f"  {n_done}/{len(frame_ids)} | {fps:.1f} fps | ETA {eta:.0f}s")

  finally:
    write_pool.shutdown(wait=True)
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
  parser.add_argument('--force', action='store_true',
                      help='Force re-annotate existing frames')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"ERROR: session directory not found: {session_dir}", file=sys.stderr)
    sys.exit(1)

  output_dir = Path(args.output).resolve() if args.output else session_dir / '3d_labels'

  try:
    annotate_session(
      session_dir=session_dir,
      output_dir=output_dir,
      onnx_path=args.onnx,
      heights_to_annotate=args.heights,
      use_gpu_preprocess=not args.no_gpu_preprocess,
      force=args.force,
    )
    print(f"\n3D labels saved to: {output_dir}")
  except KeyboardInterrupt:
    print("\n[Interrupted]")
    sys.exit(0)


if __name__ == '__main__':
  main()
