#!/usr/bin/env python3
"""Offline multi-height annotation pipeline.

Uses tinygrad (TinyJit pkl) for inference, matching the custom_modeld.py pipeline.
ONNX model is auto-compiled to a tinygrad pkl on first run.

Output per annotated height directory:
  <output_dir>/<tag>/<frame_id>.json  ← labels + metadata (JSON)

The JSON contains scalar metadata (camera_height, v_ego, world_pose) and all
model output arrays serialised as nested Python lists (lane_lines, road_edges,
lead, pose, road_transform, lane_lines_prob, lead_prob, wide_from_device_euler).
Preprocessed model input images are NOT saved here; the training pipeline reads
original RGB frames from source_session_dir and preprocesses on the fly.

Usage:
  python tools/dashcam/annotate_multi_height.py \\
      data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/ \\
      --onnx selfdrive/modeld/models/driving_vision.onnx \\
      --output data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/annotations/

  # Only annotate specific heights
  python tools/dashcam/annotate_multi_height.py ... --heights H2 H3 H6

  # CPU preprocessing fallback (default uses GPU OpenCL)
  python tools/dashcam/annotate_multi_height.py ... --no-gpu-preprocess

  # Select tinygrad device (default: CUDA)
  DEV=CPU python tools/dashcam/annotate_multi_height.py ...
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# DEV must be set before importing tinygrad (same requirement as custom_modeld.py)
os.environ.setdefault('PYOPENCL_CTX', '')
if 'DEV' not in os.environ:
  os.environ['DEV'] = 'CUDA'

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.train.dataset import rgb_to_modeld_input

TEMPORAL_SKIP = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4

X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)  # (33,) forward distances

# Number of frames to prefetch ahead of the current processing position.
PREFETCH_AHEAD = 8


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def decode_model_output(parsed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Decode tinygrad output slices to canonical annotation format.

  `parsed` is {key: np.ndarray of shape (1, N)} produced by slicing the
  flat model output with output_slices from metadata — identical to
  custom_modeld.py's decode_outputs() input convention.

  MDN layout (non-interleaved): [all_means | all_log_sigma]
    lane_lines  (528,): 264 means + 264 log_sigma, means reshape (4, 33, 2) = [y, z]
    road_edges  (264,): 132 means + 132 log_sigma, means reshape (2, 33, 2) = [y, z]
    lead        (144,): 72  means + 72  log_sigma, means reshape (3, 6, 4)  = [x, y, v, a]
    pose        (12,):  6   means + 6   log_sigma, means = [trans(3), rot(3)]
    road_transform (12,): 6 means + 6 log_sigma, means[:3] = [tx, ty, tz]
    lane_lines_prob (8,):  (4, 2) raw logits, prob = sigmoid([:, 1])
    lead_prob   (3,):  raw logits → sigmoid
  """
  def _get(key: str) -> np.ndarray:
    return parsed[key][0]  # (1, N) → (N,)

  # lane_lines: (528,) → means (4, 33, 2) [y, z] → prepend x → (4, 33, 3)
  ll_raw = _get('lane_lines')
  ll_means = ll_raw[:264].reshape(4, 33, 2)
  x_col = np.broadcast_to(X_IDXS[None, :, None], (4, 33, 1))
  lane_lines = np.concatenate([x_col, ll_means], axis=-1).astype(np.float32)  # (4, 33, 3)

  # lane_lines_prob: (8,) → (4, 2) logits → sigmoid([:, 1]) → (4,)
  ll_prob_raw = _get('lane_lines_prob').reshape(4, 2)
  lane_lines_prob = _sigmoid(ll_prob_raw[:, 1]).astype(np.float32)

  # road_edges: (264,) → means (2, 33, 2) → prepend x → (2, 33, 3)
  re_raw = _get('road_edges')
  re_means = re_raw[:132].reshape(2, 33, 2)
  x_col2 = np.broadcast_to(X_IDXS[None, :, None], (2, 33, 1))
  road_edges = np.concatenate([x_col2, re_means], axis=-1).astype(np.float32)  # (2, 33, 3)

  # road_edges_prob: model doesn't output per-edge probability → all 1.0
  road_edges_prob = np.ones(2, dtype=np.float32)

  # lead: (144,) → means (3, 6, 4) [x, y, v, a]
  ld_raw = _get('lead')
  lead = ld_raw[:72].reshape(3, 6, 4).astype(np.float32)

  # lead_prob: (3,) logits → sigmoid
  lead_prob = _sigmoid(_get('lead_prob')).astype(np.float32)

  # pose: (12,) means = first 6
  pose = _get('pose')[:6].astype(np.float32)

  # road_transform: (12,) means = first 6
  road_transform = _get('road_transform')[:6].astype(np.float32)

  # wide_from_device_euler: (6,) means = first 3
  wfde = _get('wide_from_device_euler')[:3].astype(np.float32)

  return {
    'lane_lines':             lane_lines,        # (4, 33, 3) float32
    'lane_lines_prob':        lane_lines_prob,   # (4,) float32
    'road_edges':             road_edges,        # (2, 33, 3) float32
    'road_edges_prob':        road_edges_prob,   # (2,) float32
    'lead':                   lead,              # (3, 6, 4) float32
    'lead_prob':              lead_prob,         # (3,) float32
    'pose':                   pose,              # (6,) float32
    'road_transform':         road_transform,    # (6,) float32
    'wide_from_device_euler': wfde,              # (3,) float32
  }


def transform_annotation(canonical: dict[str, np.ndarray], h1: float, h_k: float) -> dict[str, np.ndarray]:
  """Shift z_height components from camera height h1 to h_k.

  Only the z (height) components of lane_lines, road_edges, and road_transform
  are modified. All other fields are copied unchanged.

  Valid for X > ~10m (near-field blind zone grows with height).
  """
  delta_h = h_k - h1
  out = {k: v.copy() for k, v in canonical.items()}
  out['lane_lines'][:, :, 2] += delta_h        # (4, 33, 3), z = col 2
  out['road_edges'][:, :, 2] += delta_h        # (2, 33, 3), z = col 2
  out['road_transform'][2] += delta_h           # (6,), tz = index 2
  return out


def _load_or_compile_model(onnx_path: str) -> tuple[object, dict, dict]:
  """Load tinygrad TinyJit and metadata for the given ONNX model.

  Auto-compiles ONNX → tinygrad pkl on first run (same logic as run.py).

  Returns:
    (vision_run, input_shapes, output_slices)
  """
  dev = os.environ.get('DEV', 'CUDA').lower()
  base = os.path.splitext(onnx_path)[0]
  pkl_path = f"{base}_tinygrad_{dev}.pkl"
  metadata_path = f"{base}_metadata.pkl"

  needs_compile = (
    not os.path.exists(pkl_path) or
    not os.path.exists(metadata_path) or
    os.path.getmtime(onnx_path) > os.path.getmtime(pkl_path)
  )
  if needs_compile:
    print(f"  Compiling ONNX → tinygrad pkl (DEV={dev.upper()})...")
    from openpilot.tools.dashcam.train.compile_tinygrad import compile_model, generate_metadata
    generate_metadata(onnx_path, metadata_path)
    compile_model(onnx_path, pkl_path)
    print(f"  Compiled: {pkl_path}")
  else:
    print(f"  Using cached tinygrad pkl: {pkl_path}")

  with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)
  input_shapes = metadata['input_shapes']
  output_slices = metadata['output_slices']
  output_size = metadata['output_shapes']['outputs'][1]

  with open(pkl_path, 'rb') as f:
    vision_run = pickle.load(f)

  print(f"  Model loaded: inputs={list(input_shapes.keys())}  output_size={output_size}")
  return vision_run, input_shapes, output_slices


def _run_inference(vision_run, input_shapes: dict, output_slices: dict,
                   road_yuv: np.ndarray, wide_yuv: np.ndarray,
                   road_yuv_prev: np.ndarray, wide_yuv_prev: np.ndarray) -> dict[str, np.ndarray]:
  """Run one tinygrad inference step (same pattern as custom_modeld.py:CustomModelState.run).

  Stacks [prev, curr] along channel axis to form (12, 128, 256) temporal input,
  wraps as uint8 Tensor, runs TinyJit, slices output.

  Returns parsed dict {key: np.ndarray (1, N)}.
  """
  img_np = np.concatenate([road_yuv_prev, road_yuv], axis=0)[np.newaxis]      # (1, 12, 128, 256)
  big_img_np = np.concatenate([wide_yuv_prev, wide_yuv], axis=0)[np.newaxis]  # (1, 12, 128, 256)

  img_t = Tensor(img_np.reshape(input_shapes['img']), dtype=dtypes.uint8).realize()
  big_img_t = Tensor(big_img_np.reshape(input_shapes['big_img']), dtype=dtypes.uint8).realize()

  raw_output = vision_run(img=img_t, big_img=big_img_t).contiguous().realize().uop.base.buffer.numpy()
  return {k: raw_output[np.newaxis, v] for k, v in output_slices.items()}


def _write_annotation(
  out_dir: Path,
  frame_id: str,
  src_data: dict,
  canonical: dict,
  passes: bool,
) -> None:
  """Write annotation JSON to out_dir.

  JSON contains scalar metadata (camera_height, v_ego, world_pose) and all
  label arrays serialised as nested Python lists.
  """
  record: dict = {
    'frame_id':        frame_id,
    'camera_height':   float(src_data.get('camera_height', 1.22)),
    'v_ego':           float(src_data.get('v_ego', 0.0)),
    'world_pose':      [float(x) for x in src_data.get('world_pose', np.zeros(6))],
    'label_source':    'pretrained_h1',
    'll_quality_pass': bool(passes),
  }
  for k, v in canonical.items():
    record[k] = v.tolist() if isinstance(v, np.ndarray) else v
  with open(out_dir / f'{frame_id}.json', 'w') as f:
    json.dump(record, f)


def annotate_session(
  session_dir: Path,
  output_dir: Path,
  onnx_path: str,
  heights_to_annotate: list[str] | None,
  min_ll_prob: float,
  use_gpu_preprocess: bool = True,
) -> dict:
  """Annotate all frames in session_dir, write JSON labels to output_dir.

  Returns stats dict with counts per height.
  """
  # Load clip_info.json
  clip_info_path = session_dir / 'clip_info.json'
  if not clip_info_path.exists():
    raise FileNotFoundError(f"clip_info.json not found in {session_dir}")
  with open(clip_info_path) as f:
    clip_info = json.load(f)

  pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
  yaw_rad = math.radians(clip_info['camera']['yaw_deg'])
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  # save_every: how many Carla ticks between consecutive saved frames.
  # Affects the temporal buffer depth needed to feed the model's context window.
  #   save_every=1 → frames are 1 tick apart → need 5-frame buffer (buf[0] = 4 ticks back)
  #   save_every=4 → frames are 4 ticks apart → need 2-frame buffer (buf[0] = 4 ticks back)
  # Formula: buf_depth = TEMPORAL_SKIP // save_every + 1
  save_every: int = clip_info.get('save_every', 1)
  if save_every not in (1, 4):
    raise ValueError(f"Unsupported save_every={save_every} in clip_info.json (must be 1 or 4)")
  buf_depth = TEMPORAL_SKIP // save_every + 1  # 5 for save_every=1, 2 for save_every=4

  # Determine which heights to annotate
  available_heights = {tag: h for tag, h in clip_info['heights'].items()}
  if heights_to_annotate:
    heights = {tag: h for tag, h in available_heights.items() if tag in heights_to_annotate}
  else:
    heights = available_heights

  if 'H1' not in available_heights:
    raise ValueError(f"H1 not found in session heights: {list(available_heights.keys())}")
  h1_height = available_heights['H1']

  # H1 source directory — used for inference
  h1_dir = session_dir / 'H1'
  if not h1_dir.exists():
    raise FileNotFoundError(f"H1 directory not found: {h1_dir}")
  frame_ids = sorted(p.stem.replace('road_', '') for p in h1_dir.glob('road_*.png'))
  if not frame_ids:
    raise ValueError(f"No frames found in {h1_dir}")

  dev = os.environ.get('DEV', 'CUDA')
  print(f"Session: {session_dir.name}")
  print(f"  Frames: {len(frame_ids)}  Heights to annotate: {list(heights.keys())}")
  print(f"  pitch={clip_info['camera']['pitch_deg']:.1f}°  yaw={clip_info['camera']['yaw_deg']:.1f}°")
  print(f"  min_ll_prob={min_ll_prob}  DEV={dev}")

  # Create output directories
  output_dir.mkdir(parents=True, exist_ok=True)
  for tag in heights:
    (output_dir / tag).mkdir(exist_ok=True)
  # Write clip_info.json; keep source_session_dir for viz scripts that load original RGB
  clip_info_out = dict(clip_info)
  clip_info_out['source_session_dir'] = '..'
  with open(output_dir / 'clip_info.json', 'w') as f:
    json.dump(clip_info_out, f, indent=2)

  # Camera intrinsics and warp matrices (same pitch/yaw applies to all height slots)
  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
  warp_wide = compute_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True)

  # Load tinygrad model (auto-compile ONNX if needed)
  print(f"  Loading model from {onnx_path}...")
  vision_run, input_shapes, output_slices = _load_or_compile_model(onnx_path)

  # GPU OpenCL preprocessor (pixel-identical to modeld; fallback to CPU numpy)
  cl_prep_road = None
  cl_prep_wide = None
  if use_gpu_preprocess:
    from openpilot.tools.dashcam.modeld_preprocess_cl import ModeldInputPreprocessorCL
    print("  Using GPU OpenCL preprocessor (modeld_preprocess_cl)...")
    cl_prep_road = ModeldInputPreprocessorCL()
    cl_prep_wide = ModeldInputPreprocessorCL()

  def _preprocess(frame_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess road + wide RGB → (6, 128, 256) uint8 YUV each."""
    if cl_prep_road is not None:
      road_yuv = cl_prep_road.process(frame_data['road_rgb'], warp_road).copy()
      wide_yuv = cl_prep_wide.process(frame_data['wide_rgb'], warp_wide).copy()
    else:
      road_yuv = rgb_to_modeld_input(frame_data['road_rgb'], warp_road)
      wide_yuv = rgb_to_modeld_input(frame_data['wide_rgb'], warp_wide)
    return road_yuv, wide_yuv

  stats = {tag: {'total': 0, 'pass': 0, 'fail': 0} for tag in heights}
  stats['H1_inferred'] = 0

  # Circular buffers for temporal context.
  # Model input = [frame(t − TEMPORAL_SKIP ticks), frame(t)] → 12 channels total.
  # buf_depth is chosen so that buf[0] is always exactly TEMPORAL_SKIP real ticks behind buf[-1]:
  #   save_every=1: buf_depth=5, buf[0] is 4 list-steps back = 4 ticks back ✓
  #   save_every=4: buf_depth=2, buf[0] is 1 list-step  back = 4 ticks back ✓
  road_buf: deque[np.ndarray] = deque(maxlen=buf_depth)
  wide_buf: deque[np.ndarray] = deque(maxlen=buf_depth)

  # Load metadata.jsonl for each height (eager, O(N) once per height)
  def _load_metadata(height_dir: Path) -> dict[str, dict]:
    meta_path = height_dir / 'metadata.jsonl'
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

  def _load_frame(frame_id: str, height_dir: Path, meta: dict[str, dict]) -> dict | None:
    """Load one raw frame: PNG images + metadata."""
    road_path = height_dir / f'road_{frame_id}.png'
    wide_path  = height_dir / f'wide_{frame_id}.png'
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
      'camera_height': np.float32(m.get('camera_height', 1.22)),
      'v_ego':         np.float32(m.get('v_ego', 0.0)),
      'world_pose':    np.array(m.get('world_pose', [0.0]*6), dtype=np.float32),
    }

  # Load per-height metadata caches
  meta_cache: dict[str, dict[str, dict]] = {}
  for tag_t in list(heights.keys()) + (['H1'] if 'H1' not in heights else []):
    meta_cache[tag_t] = _load_metadata(session_dir / tag_t)

  # Separate read and write thread pools so prefetch reads and async writes
  # never compete for workers.
  n_heights = len(heights)
  read_pool  = ThreadPoolExecutor(max_workers=n_heights)
  write_pool = ThreadPoolExecutor(max_workers=n_heights)

  PrefetchSlot = dict[str, Future]

  def _submit_prefetch(fid: str) -> PrefetchSlot:
    slot: PrefetchSlot = {
      'H1': read_pool.submit(_load_frame, fid, h1_dir, meta_cache['H1'])
    }
    for tag in heights:
      if tag != 'H1':
        slot[tag] = read_pool.submit(_load_frame, fid, session_dir / tag, meta_cache[tag])
    return slot

  # Pre-fill the prefetch pipeline before the main loop
  prefetch_queue: deque[PrefetchSlot] = deque()
  for fid in frame_ids[:PREFETCH_AHEAD]:
    prefetch_queue.append(_submit_prefetch(fid))

  import time
  t_start = time.monotonic()

  try:
    for i, frame_id in enumerate(frame_ids):
      # Submit read for the frame PREFETCH_AHEAD steps ahead
      next_idx = i + PREFETCH_AHEAD
      if next_idx < len(frame_ids):
        prefetch_queue.append(_submit_prefetch(frame_ids[next_idx]))

      slot = prefetch_queue.popleft()
      h1_data = slot['H1'].result()
      if h1_data is None:
        continue

      # Preprocess H1 frame → (6, 128, 256) uint8 YUV for inference
      road_yuv, wide_yuv = _preprocess(h1_data)

      road_buf.append(road_yuv)
      wide_buf.append(wide_yuv)

      # Oldest frame in buffer = t-TEMPORAL_SKIP (or t when not enough history)
      road_yuv_prev = road_buf[0]
      wide_yuv_prev = wide_buf[0]

      # tinygrad inference (matching custom_modeld.py:CustomModelState.run)
      parsed = _run_inference(vision_run, input_shapes, output_slices,
                              road_yuv, wide_yuv, road_yuv_prev, wide_yuv_prev)
      canonical_h1 = decode_model_output(parsed)
      stats['H1_inferred'] += 1

      # Quality filter: check inner lane lines L0(idx=1) and R0(idx=2)
      ll_prob = canonical_h1['lane_lines_prob']
      passes = bool(ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob)

      for tag, h_k in heights.items():
        # Get source frame metadata for this height
        if tag == 'H1':
          src_data = h1_data
          canonical_hk = canonical_h1
        else:
          src_data = slot[tag].result()
          if src_data is None:
            continue
          canonical_hk = transform_annotation(canonical_h1, h1=h1_height, h_k=h_k)

        stats[tag]['total'] += 1
        stats[tag]['pass' if passes else 'fail'] += 1

        write_pool.submit(
          _write_annotation,
          output_dir / tag, frame_id, src_data, canonical_hk, passes,
        )

      # Progress log
      n_done = stats['H1_inferred']
      if n_done % 500 == 0 or n_done == len(frame_ids):
        elapsed = time.monotonic() - t_start
        fps = n_done / elapsed if elapsed > 0 else 0
        eta = (len(frame_ids) - n_done) / fps if fps > 0 else 0
        pass_rate = stats[list(heights.keys())[0]]['pass'] / max(n_done, 1) * 100
        print(f"  {n_done}/{len(frame_ids)} | {fps:.1f} fps | ETA {eta:.0f}s | PASS {pass_rate:.1f}%")

  finally:
    write_pool.shutdown(wait=True)
    read_pool.shutdown(wait=False)
    if cl_prep_road is not None:
      cl_prep_road.close()
      cl_prep_wide.close()

  elapsed = time.monotonic() - t_start
  print(f"\n  Done in {elapsed:.1f}s ({stats['H1_inferred'] / elapsed:.1f} fps)")
  for tag in heights:
    s = stats[tag]
    pass_pct = s['pass'] / max(s['total'], 1) * 100
    print(f"  {tag}: {s['total']} total, {s['pass']} PASS ({pass_pct:.1f}%), {s['fail']} FAIL")

  return stats


def main():
  parser = argparse.ArgumentParser(description='Offline multi-height annotation (tinygrad inference)')
  parser.add_argument('session_dir', help='Session directory (contains H1/, H2/, ... and clip_info.json)')
  parser.add_argument('--onnx', default='selfdrive/modeld/models/driving_vision.onnx',
                      help='Path to driving_vision.onnx (auto-compiled to tinygrad pkl on first run)')
  parser.add_argument('--output', default=None,
                      help='Output directory (default: <session_dir>/annotations/)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='Heights to annotate (default: all in clip_info.json)')
  parser.add_argument('--min-ll-prob', type=float, default=0.1,
                      help='Min lane line probability for quality filter (default: 0.1)')
  parser.add_argument('--no-gpu-preprocess', action='store_true',
                      help='Disable GPU OpenCL preprocessing, fall back to CPU (slower)')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"ERROR: session directory not found: {session_dir}", file=sys.stderr)
    sys.exit(1)

  if args.output:
    output_dir = Path(args.output).resolve()
  else:
    output_dir = session_dir / 'annotations'

  try:
    annotate_session(
      session_dir=session_dir,
      output_dir=output_dir,
      onnx_path=args.onnx,
      heights_to_annotate=args.heights,
      min_ll_prob=args.min_ll_prob,
      use_gpu_preprocess=not args.no_gpu_preprocess,
    )
    print(f"\nAnnotated data saved to: {output_dir}")
  except KeyboardInterrupt:
    print("\n[Interrupted]")
    sys.exit(0)


if __name__ == '__main__':
  main()
