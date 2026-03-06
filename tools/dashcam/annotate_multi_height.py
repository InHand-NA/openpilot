#!/usr/bin/env python3
"""Offline multi-height annotation pipeline.

Reads H1 raw frames, runs PretrainedVisionModel, decodes outputs to canonical format,
applies z_height transform for each other height, writes annotated NPZs.

Usage:
  python tools/dashcam/annotate_multi_height.py \\
      data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/ \\
      --onnx selfdrive/modeld/models/driving_vision.onnx \\
      --output data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/annotations/

  # Only annotate specific heights
  python tools/dashcam/annotate_multi_height.py ... --heights H2 H3 H6

  # CPU preprocessing fallback (default uses GPU OpenCL)
  python tools/dashcam/annotate_multi_height.py ... --device cuda --no-gpu-preprocess
"""

import argparse
import json
import math
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.train.dataset import rgb_to_modeld_input

TEMPORAL_SKIP = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4


X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)  # (33,) forward distances

# Default H1 height (meters)
H1_HEIGHT = 1.22

# Number of frames to prefetch ahead of the current processing position.
# Each prefetch slot holds futures for H1 + all non-H1 height files.
PREFETCH_AHEAD = 8


def decode_model_output(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
  """Decode PretrainedVisionModel dict output to canonical annotation format.

  MDN layout (non-interleaved): [all_means | all_log_sigma]
    lane_lines  (528,): 264 means + 264 log_sigma, means reshape (4, 33, 2) = [y, z]
    road_edges  (264,): 132 means + 132 log_sigma, means reshape (2, 33, 2) = [y, z]
    lead        (144,): 72 means  + 72  log_sigma, means reshape (3, 6, 4)  = [x, y, v, a]
    pose        (12,):  6 means   + 6   log_sigma, means = [trans(3), rot(3)]
    road_transform (12,): 6 means + 6 log_sigma, means[:3] = [tx, ty, tz]
    lane_lines_prob (8,):  (4, 2) raw logits, prob = sigmoid([:, 1])
    lead_prob   (3,):  raw logits

  Returns canonical dict compatible with extract_targets() and visualizer.
  """
  def _np(t: torch.Tensor) -> np.ndarray:
    return t[0].cpu().numpy()  # (1, N) → (N,)

  # lane_lines: (528,) → means (4, 33, 2) [y, z] → prepend x → (4, 33, 3)
  ll_raw = _np(outputs['lane_lines'])
  ll_means = ll_raw[:264].reshape(4, 33, 2)   # [y_lat, z_height]
  x_col = np.broadcast_to(X_IDXS[None, :, None], (4, 33, 1))
  lane_lines = np.concatenate([x_col, ll_means], axis=-1).astype(np.float32)  # (4, 33, 3)

  # lane_lines_prob: (8,) → (4, 2) logits → sigmoid([:, 1]) → (4,)
  ll_prob_raw = _np(outputs['lane_lines_prob']).reshape(4, 2)
  lane_lines_prob = _sigmoid(ll_prob_raw[:, 1]).astype(np.float32)

  # road_edges: (264,) → means (2, 33, 2) → prepend x → (2, 33, 3)
  re_raw = _np(outputs['road_edges'])
  re_means = re_raw[:132].reshape(2, 33, 2)
  x_col2 = np.broadcast_to(X_IDXS[None, :, None], (2, 33, 1))
  road_edges = np.concatenate([x_col2, re_means], axis=-1).astype(np.float32)  # (2, 33, 3)

  # road_edges_prob: model doesn't output per-edge probability → all 1.0
  road_edges_prob = np.ones(2, dtype=np.float32)

  # lead: (144,) → means (3, 6, 4) [x, y, v, a]
  ld_raw = _np(outputs['lead'])
  lead = ld_raw[:72].reshape(3, 6, 4).astype(np.float32)

  # lead_prob: (3,) logits → sigmoid → (3,)
  lead_prob = _sigmoid(_np(outputs['lead_prob'])).astype(np.float32)

  # pose: (12,) means = first 6
  pose = _np(outputs['pose'])[:6].astype(np.float32)

  # road_transform: (12,) means = first 6, means[:3] = [tx, ty, tz]
  road_transform = _np(outputs['road_transform'])[:6].astype(np.float32)

  # wide_from_device_euler: (6,) means = first 3
  wfde = _np(outputs['wide_from_device_euler'])[:3].astype(np.float32)

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


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


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


def annotate_session(
  session_dir: Path,
  output_dir: Path,
  onnx_path: str,
  heights_to_annotate: list[str] | None,
  min_ll_prob: float,
  device: str,
  use_gpu_preprocess: bool = False,
) -> dict:
  """Annotate all H1 frames in session_dir, write to output_dir.

  Returns stats dict with counts per height.
  """
  from openpilot.tools.dashcam.train.pretrained_model import PretrainedVisionModel

  # Load clip_info.json
  clip_info_path = session_dir / 'clip_info.json'
  if not clip_info_path.exists():
    raise FileNotFoundError(f"clip_info.json not found in {session_dir}")
  with open(clip_info_path) as f:
    clip_info = json.load(f)

  pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
  yaw_rad = math.radians(clip_info['camera']['yaw_deg'])
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  # Determine which heights to annotate
  available_heights = {tag: h for tag, h in clip_info['heights'].items()}
  if heights_to_annotate:
    heights = {tag: h for tag, h in available_heights.items() if tag in heights_to_annotate}
  else:
    heights = available_heights

  if 'H1' not in available_heights:
    raise ValueError(f"H1 not found in session heights: {list(available_heights.keys())}")
  h1_height = available_heights['H1']

  # H1 source directory
  h1_dir = session_dir / 'H1'
  if not h1_dir.exists():
    raise FileNotFoundError(f"H1 directory not found: {h1_dir}")
  frame_files = sorted(h1_dir.glob('*.npz'))
  if not frame_files:
    raise ValueError(f"No NPZ files in {h1_dir}")

  print(f"Session: {session_dir.name}")
  print(f"  Frames: {len(frame_files)}  Heights to annotate: {list(heights.keys())}")
  print(f"  pitch={clip_info['camera']['pitch_deg']:.1f}°  yaw={clip_info['camera']['yaw_deg']:.1f}°")
  print(f"  min_ll_prob={min_ll_prob}  device={device}")

  # Create output directories
  output_dir.mkdir(parents=True, exist_ok=True)
  for tag in heights:
    (output_dir / tag).mkdir(exist_ok=True)
  # Write clip_info.json with source_session_dir so preprocess_cache.py can locate
  # the original RGB frames (annotated NPZs store only labels, not road_rgb/wide_rgb).
  clip_info_out = dict(clip_info)
  clip_info_out['source_session_dir'] = str(session_dir)
  with open(output_dir / 'clip_info.json', 'w') as f:
    json.dump(clip_info_out, f, indent=2)

  # Camera intrinsics
  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  fcam_intrinsics = dc.fcam.intrinsics
  ecam_intrinsics = dc.ecam.intrinsics

  # Warp matrices for H1
  warp_road = compute_warp_matrix(rpyCalib, fcam_intrinsics, bigmodel_frame=False)
  warp_wide = compute_warp_matrix(rpyCalib, ecam_intrinsics, bigmodel_frame=True)

  # Load model
  print(f"  Loading model from {onnx_path}...")
  model = PretrainedVisionModel(onnx_path, freeze=True)
  model.eval().to(device)
  print(f"  Model loaded ({model.n_total_params() / 1e6:.1f}M params)")

  stats = {tag: {'total': 0, 'pass': 0, 'fail': 0} for tag in heights}
  stats['H1_inferred'] = 0

  # Circular buffers for temporal context: store last TEMPORAL_SKIP+1 preprocessed frames.
  # Model input = [prev_frame(t-TEMPORAL_SKIP), curr_frame(t)] → 12ch total.
  road_buf: deque[np.ndarray] = deque(maxlen=TEMPORAL_SKIP + 1)
  wide_buf: deque[np.ndarray] = deque(maxlen=TEMPORAL_SKIP + 1)

  # Optionally use GPU OpenCL pipeline (pixel-identical to modeld)
  cl_prep_road = None
  cl_prep_wide = None
  if use_gpu_preprocess:
    from openpilot.tools.dashcam.modeld_preprocess_cl import ModeldInputPreprocessorCL
    print("  Using GPU OpenCL preprocessor (modeld_preprocess_cl)...")
    cl_prep_road = ModeldInputPreprocessorCL()
    cl_prep_wide = ModeldInputPreprocessorCL()

  import time
  t_start = time.monotonic()

  # Separate read and write thread pools so prefetch reads and async writes
  # never compete for workers.
  #
  # read_pool: one worker per height (H1~H6 = up to 6). Each _submit_prefetch()
  #   submits len(heights) futures — at most 6 concurrent reads, one per height dir.
  #
  # write_pool: one worker per height. Each frame writes up to len(heights) NPZs.
  #   np.savez (uncompressed) is used because annotated NPZs are intermediate files;
  #   preprocess_cache.py converts RGB→YUV and discards the raw images.
  #
  # PREFETCH_AHEAD frames: GPU processes frame i while read_pool has already started
  # reading frame i+PREFETCH_AHEAD. With 6 readers and ~50ms/file, all 6 height files
  # for one frame finish in ~50ms — well within the PREFETCH_AHEAD × GPU_time budget.
  n_heights = len(heights)
  read_pool  = ThreadPoolExecutor(max_workers=n_heights)
  write_pool = ThreadPoolExecutor(max_workers=n_heights)

  # Prefetch slot: {tag: Future[dict|None]}
  # H1 future loads from h1_dir; non-H1 futures load from session_dir/tag/.
  PrefetchSlot = dict[str, Future]

  def _load_npz(path: str) -> dict | None:
    p = Path(path)
    return dict(np.load(path, allow_pickle=True)) if p.exists() else None

  def _write_npz(path: str, data: dict) -> None:
    np.savez(path, **data)

  def _submit_prefetch(fp: Path) -> PrefetchSlot:
    slot: PrefetchSlot = {'H1': read_pool.submit(_load_npz, str(fp))}
    for tag in heights:
      if tag != 'H1':
        slot[tag] = read_pool.submit(_load_npz, str(session_dir / tag / fp.name))
    return slot

  # Pre-fill the prefetch pipeline before the main loop
  prefetch_queue: deque[PrefetchSlot] = deque()
  for fp in frame_files[:PREFETCH_AHEAD]:
    prefetch_queue.append(_submit_prefetch(fp))

  try:
    for i, frame_path in enumerate(frame_files):
      # Submit read for the frame PREFETCH_AHEAD steps ahead while processing frame i
      next_idx = i + PREFETCH_AHEAD
      if next_idx < len(frame_files):
        prefetch_queue.append(_submit_prefetch(frame_files[next_idx]))

      # Collect current frame's data — .result() returns immediately if already done
      slot = prefetch_queue.popleft()
      h1_data = slot['H1'].result()
      if h1_data is None:
        continue

      # Preprocess H1 frame to model input (BGR channel order, matching Carla output)
      if cl_prep_road is not None:
        road_yuv = cl_prep_road.process(h1_data['road_rgb'], warp_road)  # (6, 128, 256)
        wide_yuv = cl_prep_wide.process(h1_data['wide_rgb'], warp_wide)
      else:
        road_yuv = rgb_to_modeld_input(h1_data['road_rgb'], warp_road)
        wide_yuv = rgb_to_modeld_input(h1_data['wide_rgb'], warp_wide)

      road_buf.append(road_yuv)
      wide_buf.append(wide_yuv)

      # Use frame from TEMPORAL_SKIP steps ago as previous frame (matching modeld's context)
      road_yuv_prev = road_buf[0]   # oldest in buffer (t-TEMPORAL_SKIP or t if not enough history)
      wide_yuv_prev = wide_buf[0]

      road_tensor = torch.from_numpy(np.concatenate([road_yuv_prev, road_yuv], axis=0)).unsqueeze(0).to(device)
      wide_tensor = torch.from_numpy(np.concatenate([wide_yuv_prev, wide_yuv], axis=0)).unsqueeze(0).to(device)

      with torch.no_grad():
        outputs_h1 = model(road_tensor, wide_tensor)

      canonical_h1 = decode_model_output(outputs_h1)
      stats['H1_inferred'] += 1

      # Quality filter: check inner lane lines L0(idx=1) and R0(idx=2)
      ll_prob = canonical_h1['lane_lines_prob']
      passes = bool(ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob)

      for tag, h_k in heights.items():
        stats[tag]['total'] += 1
        if passes:
          stats[tag]['pass'] += 1
        else:
          stats[tag]['fail'] += 1

        # Generate annotation for height h_k
        if tag == 'H1':
          canonical_hk = canonical_h1
        else:
          canonical_hk = transform_annotation(canonical_h1, h1=h1_height, h_k=h_k)

        # Get prefetched src_data (H1 reuses h1_data to avoid a redundant read)
        if tag == 'H1':
          src_data = h1_data
        else:
          src_data = slot[tag].result()
          if src_data is None:
            continue  # height frame missing (partial collection)

        # Merge: small metadata + canonical annotation (no road_rgb/wide_rgb).
        # RGB is omitted to keep annotated NPZs tiny (<1KB vs 14MB), eliminating
        # disk write pressure. preprocess_cache.py reads RGB from source_session_dir
        # recorded in clip_info.json.
        out_data = {}
        for k in ('camera_height', 'v_ego', 'world_pose'):
          if k in src_data:
            out_data[k] = src_data[k]
        out_data.update(canonical_hk)
        out_data['label_source'] = np.bytes_(b'pretrained_h1')
        out_data['ll_quality_pass'] = np.bool_(passes)

        # Async write (uncompressed — intermediate files only)
        write_pool.submit(_write_npz, str(output_dir / tag / frame_path.name), out_data)

      # Progress log
      n_done = stats['H1_inferred']
      if n_done % 500 == 0 or n_done == len(frame_files):
        elapsed = time.monotonic() - t_start
        fps = n_done / elapsed if elapsed > 0 else 0
        eta = (len(frame_files) - n_done) / fps if fps > 0 else 0
        pass_rate = stats[list(heights.keys())[0]]['pass'] / max(n_done, 1) * 100
        print(f"  {n_done}/{len(frame_files)} | {fps:.1f} fps | ETA {eta:.0f}s | PASS {pass_rate:.1f}%")

  finally:
    write_pool.shutdown(wait=True)  # flush all pending writes before returning
    read_pool.shutdown(wait=False)  # reads are already done; cancel any remaining prefetch
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
  parser = argparse.ArgumentParser(description='Offline multi-height annotation')
  parser.add_argument('session_dir', help='Session directory (contains H1/, H2/, ... and clip_info.json)')
  parser.add_argument('--onnx', default='selfdrive/modeld/models/driving_vision.onnx',
                      help='Path to driving_vision.onnx (default: selfdrive/modeld/models/driving_vision.onnx)')
  parser.add_argument('--output', default=None,
                      help='Output directory (default: <session_dir>/annotations/)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='Heights to annotate (default: all in clip_info.json)')
  parser.add_argument('--min-ll-prob', type=float, default=0.1,
                      help='Min lane line probability for quality filter (default: 0.1)')
  parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                      help='Inference device (default: cpu)')
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
      device=args.device,
      use_gpu_preprocess=not args.no_gpu_preprocess,
    )
    print(f"\nAnnotated data saved to: {output_dir}")
  except KeyboardInterrupt:
    print("\n[Interrupted]")
    sys.exit(0)


if __name__ == '__main__':
  main()
