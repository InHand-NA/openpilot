#!/usr/bin/env python3
"""Offline multi-height annotation pipeline.

Reads H1 raw frames, runs PretrainedVisionModel, decodes outputs to canonical format,
applies z_height transform for each other height, writes annotated NPZs.

Usage:
  python tools/dashcam/annotate_multi_height.py \\
      data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/ \\
      --onnx selfdrive/modeld/models/driving_vision.onnx \\
      --output data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0_annotated/

  # Only annotate specific heights
  python tools/dashcam/annotate_multi_height.py ... --heights H2 H3 H6

  # GPU inference
  python tools/dashcam/annotate_multi_height.py ... --device cuda --batch-size 8
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.train.dataset import rgb_to_modeld_input


X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)  # (33,) forward distances

# Default H1 height (meters)
H1_HEIGHT = 1.22


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
  batch_size: int,
  device: str,
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
  # Copy clip_info.json so preprocess_cache.py can find it via parent.parent
  shutil.copy(clip_info_path, output_dir / 'clip_info.json')

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

  import time
  t_start = time.monotonic()

  for frame_path in frame_files:
    h1_data = dict(np.load(frame_path, allow_pickle=True))

    # Preprocess H1 frame to model input
    road_yuv = rgb_to_modeld_input(h1_data['road_rgb'], warp_road)  # (6, 128, 256)
    wide_yuv = rgb_to_modeld_input(h1_data['wide_rgb'], warp_wide)

    # Stack two identical frames (model expects temporal pair)
    road_tensor = torch.from_numpy(np.concatenate([road_yuv, road_yuv], axis=0)).unsqueeze(0).to(device)
    wide_tensor = torch.from_numpy(np.concatenate([wide_yuv, wide_yuv], axis=0)).unsqueeze(0).to(device)

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

      # Load source frame for this height (for road_rgb, wide_rgb, meta)
      src_path = session_dir / tag / frame_path.name
      if not src_path.exists():
        continue  # height frame missing (partial collection)
      src_data = dict(np.load(src_path, allow_pickle=True))

      # Merge: source meta + canonical annotation
      out_data = {}
      for k in ('road_rgb', 'wide_rgb', 'camera_height', 'v_ego', 'world_pose'):
        if k in src_data:
          out_data[k] = src_data[k]
      out_data.update(canonical_hk)
      out_data['label_source'] = np.bytes_(b'pretrained_h1')
      out_data['ll_quality_pass'] = np.bool_(passes)

      out_path = output_dir / tag / frame_path.name
      np.savez_compressed(str(out_path), **out_data)

    # Progress log
    n_done = stats['H1_inferred']
    if n_done % 500 == 0 or n_done == len(frame_files):
      elapsed = time.monotonic() - t_start
      fps = n_done / elapsed if elapsed > 0 else 0
      eta = (len(frame_files) - n_done) / fps if fps > 0 else 0
      pass_rate = stats[list(heights.keys())[0]]['pass'] / max(n_done, 1) * 100
      print(f"  {n_done}/{len(frame_files)} | {fps:.1f} fps | ETA {eta:.0f}s | PASS {pass_rate:.1f}%")

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
                      help='Output directory (default: <session_dir>_annotated)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='Heights to annotate (default: all in clip_info.json)')
  parser.add_argument('--min-ll-prob', type=float, default=0.1,
                      help='Min lane line probability for quality filter (default: 0.5)')
  parser.add_argument('--batch-size', type=int, default=1,
                      help='Inference batch size (default: 1; model is fixed batch=1, this is unused but kept for CLI compatibility)')
  parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                      help='Inference device (default: cpu)')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"ERROR: session directory not found: {session_dir}", file=sys.stderr)
    sys.exit(1)

  if args.output:
    output_dir = Path(args.output).resolve()
  else:
    output_dir = session_dir.parent / (session_dir.name + '_annotated')

  try:
    annotate_session(
      session_dir=session_dir,
      output_dir=output_dir,
      onnx_path=args.onnx,
      heights_to_annotate=args.heights,
      min_ll_prob=args.min_ll_prob,
      batch_size=args.batch_size,
      device=args.device,
    )
    print(f"\nAnnotated data saved to: {output_dir}")
  except KeyboardInterrupt:
    print("\n[Interrupted]")
    sys.exit(0)


if __name__ == '__main__':
  main()
