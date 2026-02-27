#!/usr/bin/env python3
"""Evaluate pretrained model accuracy against GT labels.

Loads the exported .pt model (TorchScript traced) and compares predictions
against ground truth labels from the dataset.

Modes:
  1. Metrics mode (default): batch evaluate all samples, report MAE/RMSE/accuracy.
  2. Visualize mode (--visualize): interactive per-frame side-by-side comparison
     of GT vs model predictions on the warped road camera image.

Usage:
  # Metrics only
  python3 tools/dashcam/eval_pretrained.py \
    --data-dir data/dual_camera_train/Town04_003 --max-dist 80

  # Interactive visualization
  python3 tools/dashcam/eval_pretrained.py \
    --data-dir data/dual_camera_train/Town04_003 --visualize

  # Visualize starting from frame 500
  python3 tools/dashcam/eval_pretrained.py \
    --data-dir data/dual_camera_train/Town04_003 --visualize --start 500

  # Use custom model path
  python3 tools/dashcam/eval_pretrained.py \
    --model checkpoints/my_model.pt --data-dir data/dual_camera_train/Town04_003
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, SBIGMODEL_INPUT_SIZE, get_warp_matrix, medmodel_intrinsics
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.train.dataset import (
  DualCameraDrivingDataset,
  extract_targets,
  rgb_to_yuv420_6ch,
  warp_image,
)
from openpilot.tools.dashcam.train.pretrained_model import ONNX_OUTPUT_SLICES, PretrainedVisionModel
from openpilot.tools.dashcam.visualizer import (
  _build_transform,
  _draw_polygon_alpha,
  _get_path_length_idx,
  _map_line_to_polygon,
  MAX_DRAW_DISTANCE,
  MIN_DRAW_DISTANCE,
  project_points_to_image,
)

X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)  # (33,)
MW, MH = MEDMODEL_INPUT_SIZE  # 512, 256
TEMPORAL_SKIP = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4


# ============================================================
# Metrics evaluation functions
# ============================================================

def extract_mdn_mean(raw: np.ndarray) -> np.ndarray:
  n = raw.shape[-1] // 2
  return raw[..., :n]


def _dist_mask(n_lines: int, max_dist: float | None) -> np.ndarray:
  if max_dist is None:
    return np.ones((1, 1, 33, 1), dtype=np.float32)
  return (X_IDXS <= max_dist).astype(np.float32).reshape(1, 1, 33, 1)


def eval_lane_lines(pred_raw, gt, gt_prob, gt_valid, max_dist=None):
  pred_mean = extract_mdn_mean(pred_raw).reshape(-1, 4, 33, 2)
  mask = (gt_prob > 0.5)[..., np.newaxis] * gt_valid
  mask = mask[..., np.newaxis] * _dist_mask(4, max_dist)
  if mask.sum() == 0:
    return {'mae': float('nan'), 'rmse': float('nan'), 'n_valid': 0}
  err = np.abs(pred_mean - gt) * mask
  sq_err = (pred_mean - gt) ** 2 * mask
  n = mask.sum()
  return {'mae': float(err.sum() / n), 'rmse': float(np.sqrt(sq_err.sum() / n)), 'n_valid': int(n)}


def eval_road_edges(pred_raw, gt, gt_prob, gt_valid, max_dist=None):
  pred_mean = extract_mdn_mean(pred_raw).reshape(-1, 2, 33, 2)
  mask = (gt_prob > 0.5)[..., np.newaxis] * gt_valid
  mask = mask[..., np.newaxis] * _dist_mask(2, max_dist)
  if mask.sum() == 0:
    return {'mae': float('nan'), 'rmse': float('nan'), 'n_valid': 0}
  err = np.abs(pred_mean - gt) * mask
  sq_err = (pred_mean - gt) ** 2 * mask
  n = mask.sum()
  return {'mae': float(err.sum() / n), 'rmse': float(np.sqrt(sq_err.sum() / n)), 'n_valid': int(n)}


def eval_lead(pred_raw, gt, gt_prob):
  pred_mean = extract_mdn_mean(pred_raw).reshape(-1, 3, 6, 4)
  mask = (gt_prob > 0.5)[:, :, np.newaxis, np.newaxis]
  if mask.sum() == 0:
    return {'mae': float('nan'), 'rmse': float('nan'), 'n_valid': 0}
  err = np.abs(pred_mean - gt) * mask
  sq_err = (pred_mean - gt) ** 2 * mask
  n = mask.sum() * 6 * 4
  return {'mae': float(err.sum() / n), 'rmse': float(np.sqrt(sq_err.sum() / n)), 'n_valid': int(mask.sum())}


def eval_mdn_simple(pred_raw, gt, name):
  pred_mean = extract_mdn_mean(pred_raw)
  err = np.abs(pred_mean - gt)
  sq_err = (pred_mean - gt) ** 2
  return {
    'mae': float(err.mean()), 'rmse': float(np.sqrt(sq_err.mean())),
    'per_dim_mae': [float(err[:, i].mean()) for i in range(gt.shape[-1])],
  }


def eval_prob(pred_raw, gt, name):
  if name == 'lane_lines_prob':
    pred_logits = pred_raw.reshape(-1, 4, 2)
    pred_prob = 1.0 / (1.0 + np.exp(-pred_logits[:, :, 1]))
    gt_binary = (gt > 0.5).astype(np.float32)
  elif name == 'lead_prob':
    pred_prob = 1.0 / (1.0 + np.exp(-pred_raw))
    gt_binary = (gt > 0.5).astype(np.float32)
  else:
    return {}
  pred_binary = (pred_prob > 0.5).astype(np.float32)
  return {
    'accuracy': float((pred_binary == gt_binary).mean()),
    'prob_mae': float(np.abs(pred_prob - gt).mean()),
    'gt_positive_rate': float(gt_binary.mean()),
    'pred_positive_rate': float(pred_binary.mean()),
  }


# ============================================================
# Visualization functions
# ============================================================

def _pred_to_lane_xyz(pred_raw: np.ndarray) -> np.ndarray:
  """Convert model lane_lines prediction (528,) → (4, 33, 3) with X_IDXS as x."""
  mean = extract_mdn_mean(pred_raw)  # (264,)
  yz = mean.reshape(4, 33, 2)  # (4, 33, 2) = y, z
  xyz = np.zeros((4, 33, 3), dtype=np.float32)
  xyz[:, :, 0] = X_IDXS[np.newaxis, :]  # x from X_IDXS
  xyz[:, :, 1:] = yz
  return xyz


def _pred_to_edge_xyz(pred_raw: np.ndarray) -> np.ndarray:
  """Convert model road_edges prediction (264,) → (2, 33, 3) with X_IDXS as x."""
  mean = extract_mdn_mean(pred_raw)  # (132,)
  yz = mean.reshape(2, 33, 2)
  xyz = np.zeros((2, 33, 3), dtype=np.float32)
  xyz[:, :, 0] = X_IDXS[np.newaxis, :]
  xyz[:, :, 1:] = yz
  return xyz


def _pred_to_lead(pred_raw: np.ndarray) -> np.ndarray:
  """Convert model lead prediction (144,) → (3, 6, 4) mean values."""
  return extract_mdn_mean(pred_raw).reshape(3, 6, 4)


def _pred_lane_probs(pred_raw: np.ndarray) -> np.ndarray:
  """Convert lane_lines_prob (8,) logits → (4,) probabilities."""
  logits = pred_raw.reshape(4, 2)
  return 1.0 / (1.0 + np.exp(-logits[:, 1]))


def _pred_lead_probs(pred_raw: np.ndarray) -> np.ndarray:
  """Convert lead_prob (3,) logits → (3,) probabilities."""
  return 1.0 / (1.0 + np.exp(-pred_raw))


def draw_lanes(img, lane_lines, lane_probs, transform, color, alpha_scale=1.0):
  """Draw lane lines on image.

  Args:
    lane_lines: (4, 33, 3) xyz in calibrated space
    lane_probs: (4,) probabilities
    color: BGR tuple
    alpha_scale: multiply alpha for lighter/heavier drawing
  """
  max_distance = np.clip(X_IDXS[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(X_IDXS, max_distance)
  for i in range(lane_lines.shape[0]):
    prob = float(lane_probs[i])
    if prob < 0.01:
      continue
    polygon = _map_line_to_polygon(lane_lines[i], 0.025 * prob, 0.0, max_idx, max_distance, transform)
    if len(polygon) >= 3:
      _draw_polygon_alpha(img, polygon, color, float(np.clip(prob * alpha_scale, 0.0, 0.7)))


def draw_road_edges(img, road_edges, road_probs, transform, color, alpha_scale=1.0):
  """Draw road edges on image."""
  max_distance = np.clip(X_IDXS[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(X_IDXS, max_distance)
  for i in range(road_edges.shape[0]):
    prob = float(road_probs[i])
    if prob < 0.01:
      continue
    polygon = _map_line_to_polygon(road_edges[i], 0.025, 0.0, max_idx, max_distance, transform)
    if len(polygon) >= 3:
      _draw_polygon_alpha(img, polygon, color, float(np.clip(prob * alpha_scale, 0.0, 0.7)))


def draw_lead(img, lead, lead_prob, K, rpyCalib, camera_height, color):
  """Draw lead vehicle marker."""
  if lead_prob[0] < 0.3:
    return
  x_dist, y_offset, v_abs, accel = (float(v) for v in lead[0, 0])
  if x_dist < 1.0 or x_dist > 200.0:
    return
  pts = project_points_to_image(
    np.array([x_dist]), np.array([y_offset]), np.array([camera_height]), K, rpyCalib)
  if np.isnan(pts[0]).any():
    return
  x, y = float(pts[0, 0]), float(pts[0, 1])
  sz = np.clip((25 * 30) / (x_dist / 3 + 30), 15.0, 30.0)
  x = np.clip(x, sz, MW - sz)
  y = min(y, MH - sz * 0.6)
  tri = np.array([
    [x, y - sz * 0.4], [x - sz * 0.8, y + sz * 0.4], [x + sz * 0.8, y + sz * 0.4],
  ], dtype=np.int32)
  cv2.fillPoly(img, [tri], color)
  cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
  label = f"{x_dist:.1f}m v={v_abs:.1f}"
  cv2.putText(img, label, (int(x + sz), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def make_info_bar(width, texts, bg_color=(30, 30, 30)):
  """Create a multi-line info bar. texts: list of (text, color) tuples."""
  h = 16 * len(texts) + 4
  bar = np.full((h, width, 3), bg_color, dtype=np.uint8)
  for i, (text, color) in enumerate(texts):
    cv2.putText(bar, text, (4, 14 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
  return bar


def render_comparison(npz_data, prev_npz_data, model, device, camera_height):
  """Render side-by-side GT vs Pred visualization for one frame.

  Returns: BGR image (H, W*2, 3)
  """
  dc = DEVICE_CAMERAS[("pc", "unknown")]
  fcam_intrinsics = dc.fcam.intrinsics
  ecam_intrinsics = dc.ecam.intrinsics
  rpyCalib = npz_data['rpyCalib'].astype(np.float64)
  prev_rpyCalib = prev_npz_data['rpyCalib'].astype(np.float64)

  # Warp road image for display
  warped = warp_image(npz_data['road_rgb'], rpyCalib, fcam_intrinsics)
  gt_img = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
  pred_img = gt_img.copy()

  # Prepare model input: road + wide, prev + curr
  road_prev = rgb_to_yuv420_6ch(warp_image(prev_npz_data['road_rgb'], prev_rpyCalib, fcam_intrinsics))
  road_curr = rgb_to_yuv420_6ch(warped)
  road = np.concatenate([road_prev, road_curr], axis=0)  # (12, 128, 256)

  M_wide = get_warp_matrix(rpyCalib, ecam_intrinsics, bigmodel_frame=True)
  wide_curr = rgb_to_yuv420_6ch(cv2.warpPerspective(
    npz_data['wide_rgb'], M_wide, SBIGMODEL_INPUT_SIZE, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))
  M_wide_prev = get_warp_matrix(prev_rpyCalib, ecam_intrinsics, bigmodel_frame=True)
  wide_prev = rgb_to_yuv420_6ch(cv2.warpPerspective(
    prev_npz_data['wide_rgb'], M_wide_prev, SBIGMODEL_INPUT_SIZE, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))
  wide = np.concatenate([wide_prev, wide_curr], axis=0)

  # Run model
  img_t = torch.from_numpy(road[np.newaxis]).to(device)
  big_img_t = torch.from_numpy(wide[np.newaxis]).to(device)
  with torch.no_grad():
    preds = model(img_t, big_img_t)
  preds_np = {k: v.cpu().numpy()[0] for k, v in preds.items()}

  # Projection setup: warped space uses medmodel_intrinsics with rpyCalib=[0,0,0]
  K = medmodel_intrinsics
  rpy_zero = np.zeros(3)
  transform = _build_transform(K, rpy_zero)

  # --- GT side (left) ---
  gt_lanes = npz_data['lane_lines']       # (4, 33, 3)
  gt_lane_probs = npz_data['lane_lines_prob']  # (4,)
  gt_edges = npz_data['road_edges']       # (2, 33, 3)
  gt_edge_probs = npz_data['road_edges_prob']  # (2,)
  gt_lead = npz_data['lead']              # (3, 6, 4)
  gt_lead_prob = npz_data['lead_prob']    # (3,)

  draw_lanes(gt_img, gt_lanes, gt_lane_probs, transform, (0, 255, 0))       # green
  draw_road_edges(gt_img, gt_edges, gt_edge_probs, transform, (0, 165, 255))  # orange
  draw_lead(gt_img, gt_lead, gt_lead_prob, K, rpy_zero, camera_height, (0, 200, 255))  # yellow

  # --- Pred side (right) ---
  pred_lanes = _pred_to_lane_xyz(preds_np['lane_lines'])
  pred_lane_probs = _pred_lane_probs(preds_np['lane_lines_prob'])
  pred_edges = _pred_to_edge_xyz(preds_np['road_edges'])
  # road_edges_prob not predicted by openpilot model — use 1.0
  pred_edge_probs = np.ones(2, dtype=np.float32)
  pred_lead = _pred_to_lead(preds_np['lead'])
  pred_lead_prob = _pred_lead_probs(preds_np['lead_prob'])

  draw_lanes(pred_img, pred_lanes, pred_lane_probs, transform, (255, 200, 0))    # cyan
  draw_road_edges(pred_img, pred_edges, pred_edge_probs, transform, (255, 0, 255))  # magenta
  draw_lead(pred_img, pred_lead, pred_lead_prob, K, rpy_zero, camera_height, (0, 200, 255))

  # Labels
  white = (255, 255, 255)
  cv2.putText(gt_img, "GT", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2, cv2.LINE_AA)
  cv2.putText(pred_img, "PRED", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2, cv2.LINE_AA)

  # Concat side by side
  combined = np.hstack([gt_img, pred_img])

  # Info bar
  gt_pose = npz_data['pose']
  pred_pose = extract_mdn_mean(preds_np['pose'])
  gt_rt = npz_data['road_transform']
  pred_rt = extract_mdn_mean(preds_np['road_transform'])
  v_ego = float(npz_data['v_ego'])

  info = make_info_bar(combined.shape[1], [
    (f"v_ego={v_ego:.1f}m/s  h={camera_height:.2f}m  "
     f"rpyCalib=[{np.degrees(rpyCalib[0]):.1f}, {np.degrees(rpyCalib[1]):.1f}, {np.degrees(rpyCalib[2]):.1f}]",
     (200, 200, 200)),
    (f"GT  pose: v=[{gt_pose[0]:.2f},{gt_pose[1]:.2f},{gt_pose[2]:.2f}] w=[{gt_pose[3]:.3f},{gt_pose[4]:.3f},{gt_pose[5]:.3f}]  "
     f"rt_z={gt_rt[2]:.2f}",
     (0, 255, 0)),
    (f"PRD pose: v=[{pred_pose[0]:.2f},{pred_pose[1]:.2f},{pred_pose[2]:.2f}] w=[{pred_pose[3]:.3f},{pred_pose[4]:.3f},{pred_pose[5]:.3f}]  "
     f"rt_z={pred_rt[2]:.2f}",
     (255, 200, 0)),
    (f"GT lanes=[{gt_lane_probs[0]:.2f},{gt_lane_probs[1]:.2f},{gt_lane_probs[2]:.2f},{gt_lane_probs[3]:.2f}]  "
     f"PRD lanes=[{pred_lane_probs[0]:.2f},{pred_lane_probs[1]:.2f},{pred_lane_probs[2]:.2f},{pred_lane_probs[3]:.2f}]  "
     f"GT lead_p={gt_lead_prob[0]:.2f}  PRD lead_p={pred_lead_prob[0]:.2f}",
     (200, 200, 200)),
  ])

  return np.vstack([info, combined])


def run_visualize(args):
  """Interactive per-frame visualization mode."""
  print(f"Dataset: {args.data_dir}")
  print(f"Model:   {args.model}")
  print(f"Device:  {args.device}")

  # Load NPZ file list
  data_path = Path(args.data_dir)
  npz_files = sorted(data_path.glob('*.npz'))
  if not npz_files:
    print(f"ERROR: No NPZ files found in {args.data_dir}")
    return
  print(f"Samples: {len(npz_files)}")

  # Load model
  print("Loading pretrained model...")
  model = PretrainedVisionModel.from_traced(args.model, freeze=True, device=args.device)
  model.eval()
  print(f"  Parameters: {model.n_total_params():,}")

  # Window setup
  win_name = 'eval_pretrained: GT vs PRED'
  cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win_name, MW * 2, MH + 80)

  idx = max(0, min(args.start, len(npz_files) - 1))
  cached_img = None
  cached_idx = -1

  print(f"\nControls: Left/Right = prev/next, PgUp/PgDn = ±100, Home/End = first/last, q/ESC = quit\n")

  while True:
    if idx != cached_idx:
      npz_data = dict(np.load(npz_files[idx], allow_pickle=True))
      prev_idx = max(0, idx - TEMPORAL_SKIP)
      prev_data = dict(np.load(npz_files[prev_idx], allow_pickle=True))
      camera_height = float(npz_data.get('camera_height', 1.13))

      t0 = time.monotonic()
      cached_img = render_comparison(npz_data, prev_data, model, args.device, camera_height)
      dt = time.monotonic() - t0

      fname = npz_files[idx].name
      print(f"\r[{idx+1}/{len(npz_files)}] {fname}  ({dt:.2f}s)", end="", flush=True)
      cached_idx = idx

    cv2.imshow(win_name, cached_img)
    key = cv2.waitKey(0) & 0xFFFF

    if key == ord('q') or key == 27:  # q / ESC
      break
    elif key == 65363 or key == ord('d'):  # Right
      idx = min(idx + 1, len(npz_files) - 1)
    elif key == 65361 or key == ord('a'):  # Left
      idx = max(idx - 1, 0)
    elif key == 65366:  # PgDn
      idx = min(idx + 100, len(npz_files) - 1)
    elif key == 65365:  # PgUp
      idx = max(idx - 100, 0)
    elif key == 65360:  # Home
      idx = 0
    elif key == 65367:  # End
      idx = len(npz_files) - 1

  print()
  cv2.destroyAllWindows()


# ============================================================
# Metrics mode
# ============================================================

def run_metrics(args):
  """Batch evaluation mode: compute metrics over all samples."""
  if args.max_dist is not None:
    n_pts = int((X_IDXS <= args.max_dist).sum())
    last_x = X_IDXS[n_pts - 1] if n_pts > 0 else 0
    print(f"Max dist: {args.max_dist}m  -> using {n_pts}/33 points (last x={last_x:.1f}m)")
  else:
    print("Max dist: None (all 33 points, up to 192m)")

  print(f"Dataset: {args.data_dir}")
  print(f"Model:   {args.model}")
  print(f"Device:  {args.device}")

  dataset = DualCameraDrivingDataset(data_dirs=[args.data_dir])
  print(f"Samples: {len(dataset)}")

  loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

  print("Loading pretrained model...")
  model = PretrainedVisionModel.from_traced(args.model, freeze=True, device=args.device)
  model.eval()
  print(f"  Parameters: {model.n_total_params():,}")

  all_preds: dict[str, list[np.ndarray]] = {k: [] for k in ONNX_OUTPUT_SLICES}
  all_targets: dict[str, list[np.ndarray]] = {}
  target_keys = [
    'lane_lines', 'lane_lines_prob', 'lane_lines_valid',
    'road_edges', 'road_edges_prob', 'road_edges_valid',
    'lead', 'lead_prob', 'pose', 'road_transform', 'wide_from_device_euler',
  ]
  for k in target_keys:
    all_targets[k] = []

  t0 = time.monotonic()
  n_total = len(loader)

  for i, (img, big_img, targets) in enumerate(loader):
    img = img.to(args.device)
    big_img = big_img.to(args.device)
    with torch.no_grad():
      preds = model(img, big_img)
    for name in ONNX_OUTPUT_SLICES:
      all_preds[name].append(preds[name].cpu().numpy())
    for k in target_keys:
      all_targets[k].append(targets[k].numpy())
    if (i + 1) % 2000 == 0 or i == 0:
      elapsed = time.monotonic() - t0
      fps = (i + 1) / elapsed
      eta = (n_total - i - 1) / fps
      print(f"  [{i+1:>6}/{n_total}]  {fps:.1f} samples/s  ETA: {eta:.0f}s")

  elapsed = time.monotonic() - t0
  print(f"\nProcessed {n_total} samples in {elapsed:.1f}s ({n_total/elapsed:.1f} samples/s)\n")

  for k in all_preds:
    all_preds[k] = np.concatenate(all_preds[k], axis=0)
  for k in all_targets:
    all_targets[k] = np.concatenate(all_targets[k], axis=0)

  print("=" * 75)
  print(f"{'Component':<30} {'MAE':>10} {'RMSE':>10} {'Extra':>20}")
  print("-" * 75)

  r = eval_lane_lines(all_preds['lane_lines'], all_targets['lane_lines'],
                      all_targets['lane_lines_prob'], all_targets['lane_lines_valid'], max_dist=args.max_dist)
  dist_note = f" (≤{args.max_dist}m)" if args.max_dist else ""
  print(f"{'lane_lines' + dist_note:<30} {r['mae']:>10.4f} {r['rmse']:>10.4f} {'valid=' + str(r['n_valid']):>20}")

  r = eval_prob(all_preds['lane_lines_prob'], all_targets['lane_lines_prob'], 'lane_lines_prob')
  acc_str = f"acc={r['accuracy']:.4f}"
  print(f"{'lane_lines_prob':<30} {'':>10} {'':>10} {acc_str:>20}")

  r = eval_road_edges(all_preds['road_edges'], all_targets['road_edges'],
                      all_targets['road_edges_prob'], all_targets['road_edges_valid'], max_dist=args.max_dist)
  print(f"{'road_edges' + dist_note:<30} {r['mae']:>10.4f} {r['rmse']:>10.4f} {'valid=' + str(r['n_valid']):>20}")

  r = eval_lead(all_preds['lead'], all_targets['lead'], all_targets['lead_prob'])
  print(f"{'lead':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f} {'valid=' + str(r['n_valid']):>20}")

  r = eval_prob(all_preds['lead_prob'], all_targets['lead_prob'], 'lead_prob')
  acc_str = f"acc={r['accuracy']:.4f}"
  print(f"{'lead_prob':<30} {'':>10} {'':>10} {acc_str:>20}")

  r = eval_mdn_simple(all_preds['pose'], all_targets['pose'], 'pose')
  print(f"{'pose':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f}")
  for lbl, v in zip(['tx', 'ty', 'tz', 'rx', 'ry', 'rz'], r['per_dim_mae']):
    print(f"{'  ' + lbl:<30} {v:>10.4f}")

  r = eval_mdn_simple(all_preds['road_transform'], all_targets['road_transform'], 'road_transform')
  print(f"{'road_transform':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f}")
  for lbl, v in zip(['tx', 'ty', 'tz', 'rx', 'ry', 'rz'], r['per_dim_mae']):
    print(f"{'  ' + lbl:<30} {v:>10.4f}")

  r = eval_mdn_simple(all_preds['wide_from_device_euler'], all_targets['wide_from_device_euler'], 'wide_from_device_euler')
  print(f"{'wide_from_device_euler':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f}")
  for lbl, v in zip(['roll', 'pitch', 'yaw'], r['per_dim_mae']):
    print(f"{'  ' + lbl:<30} {v:>10.4f}")

  print("=" * 75)
  print("Done.")


def main():
  parser = argparse.ArgumentParser(description='Evaluate pretrained model against GT')
  parser.add_argument('--data-dir', default='data/dual_camera_train/Town04_003',
                      help='Directory with dual-camera NPZ files')
  parser.add_argument('--model', default='checkpoints/pretrained_openpilot.pt',
                      help='Path to exported .pt model (TorchScript traced)')
  parser.add_argument('--num-workers', type=int, default=8,
                      help='DataLoader workers for preprocessing (metrics mode)')
  parser.add_argument('--max-dist', type=float, default=None,
                      help='Max longitudinal distance (m) for lane/edge evaluation')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  parser.add_argument('--visualize', action='store_true',
                      help='Interactive per-frame visualization (GT vs Pred side by side)')
  parser.add_argument('--start', type=int, default=0,
                      help='Starting frame index for visualization mode')
  args = parser.parse_args()

  if args.visualize:
    run_visualize(args)
  else:
    run_metrics(args)


if __name__ == '__main__':
  main()
