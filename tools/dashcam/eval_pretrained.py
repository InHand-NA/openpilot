#!/usr/bin/env python3
"""Evaluate pretrained ONNX model accuracy against GT labels.

Loads the full dataset, runs the PretrainedVisionModel on every sample,
and compares model predictions (MDN mean) against ground truth.

Metrics per component:
  - MAE: mean absolute error (predicted mean vs GT)
  - RMSE: root mean squared error
  - For probability heads: accuracy and AUC

Usage:
  python3 tools/dashcam/eval_pretrained.py \
    --data-dir data/dual_camera_train/Town04_003 \
    --onnx selfdrive/modeld/models/driving_vision.onnx
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.train.dataset import DualCameraDrivingDataset
from openpilot.tools.dashcam.train.pretrained_model import ONNX_OUTPUT_SLICES, PretrainedVisionModel

# Longitudinal distance (m) of each of the 33 lane/edge points
X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)  # (33,)


# MDN outputs: first half = mu, second half = log_sigma
# BCE outputs: raw logits
OUTPUT_SPECS = {
  'pose':                   ('mdn', 6),    # 12 → 6 mu + 6 sigma
  'wide_from_device_euler': ('mdn', 3),    # 6  → 3 mu + 3 sigma
  'road_transform':         ('mdn', 6),    # 12 → 6 mu + 6 sigma
  'lane_lines':             ('mdn', 264),  # 528 → 264 mu + 264 sigma
  'lane_lines_prob':        ('bce', 8),    # 8 logits → 4 pairs of [1-p, p]
  'road_edges':             ('mdn', 132),  # 264 → 132 mu + 132 sigma
  'lead':                   ('mdn', 72),   # 144 → 72 mu + 72 sigma
  'lead_prob':              ('bce', 3),    # 3 logits
}


def extract_mdn_mean(raw: np.ndarray) -> np.ndarray:
  """Extract mean from MDN output (first half of last dim)."""
  n = raw.shape[-1] // 2
  return raw[..., :n]


def _dist_mask(n_lines: int, max_dist: float | None) -> np.ndarray:
  """Build a (1, 1, 33, 1) boolean mask for points within max_dist meters."""
  if max_dist is None:
    return np.ones((1, 1, 33, 1), dtype=np.float32)
  return (X_IDXS <= max_dist).astype(np.float32).reshape(1, 1, 33, 1)


def eval_lane_lines(pred_raw: np.ndarray, gt: np.ndarray, gt_prob: np.ndarray,
                    gt_valid: np.ndarray, max_dist: float | None = None) -> dict:
  """Evaluate lane lines: pred (B, 528) → (B, 4, 33, 2) mean vs GT (B, 4, 33, 2).

  Args:
    max_dist: if set, only evaluate points where X_IDXS <= max_dist (meters).
  """
  pred_mean = extract_mdn_mean(pred_raw).reshape(-1, 4, 33, 2)
  # Mask: lane prob > 0.5 AND per-point valid
  mask = (gt_prob > 0.5)[..., np.newaxis] * gt_valid  # (B, 4, 33)
  mask = mask[..., np.newaxis]  # (B, 4, 33, 1) → broadcast to (B, 4, 33, 2)
  mask = mask * _dist_mask(4, max_dist)

  if mask.sum() == 0:
    return {'mae': float('nan'), 'rmse': float('nan'), 'n_valid': 0}

  err = np.abs(pred_mean - gt) * mask
  sq_err = (pred_mean - gt) ** 2 * mask
  n = mask.sum()
  return {
    'mae': float(err.sum() / n),
    'rmse': float(np.sqrt(sq_err.sum() / n)),
    'n_valid': int(n),
  }


def eval_road_edges(pred_raw: np.ndarray, gt: np.ndarray, gt_prob: np.ndarray,
                    gt_valid: np.ndarray, max_dist: float | None = None) -> dict:
  """Evaluate road edges: pred (B, 264) → (B, 2, 33, 2) mean vs GT (B, 2, 33, 2).

  Args:
    max_dist: if set, only evaluate points where X_IDXS <= max_dist (meters).
  """
  pred_mean = extract_mdn_mean(pred_raw).reshape(-1, 2, 33, 2)
  mask = (gt_prob > 0.5)[..., np.newaxis] * gt_valid
  mask = mask[..., np.newaxis]
  mask = mask * _dist_mask(2, max_dist)

  if mask.sum() == 0:
    return {'mae': float('nan'), 'rmse': float('nan'), 'n_valid': 0}

  err = np.abs(pred_mean - gt) * mask
  sq_err = (pred_mean - gt) ** 2 * mask
  n = mask.sum()
  return {
    'mae': float(err.sum() / n),
    'rmse': float(np.sqrt(sq_err.sum() / n)),
    'n_valid': int(n),
  }


def eval_lead(pred_raw: np.ndarray, gt: np.ndarray, gt_prob: np.ndarray) -> dict:
  """Evaluate lead: pred (B, 144) → (B, 3, 6, 4) mean vs GT (B, 3, 6, 4)."""
  pred_mean = extract_mdn_mean(pred_raw).reshape(-1, 3, 6, 4)
  mask = (gt_prob > 0.5)  # (B, 3)
  mask = mask[:, :, np.newaxis, np.newaxis]  # (B, 3, 1, 1)

  if mask.sum() == 0:
    return {'mae': float('nan'), 'rmse': float('nan'), 'n_valid': 0}

  err = np.abs(pred_mean - gt) * mask
  sq_err = (pred_mean - gt) ** 2 * mask
  n = mask.sum() * 6 * 4  # expand per time-step and per-dim
  return {
    'mae': float(err.sum() / n),
    'rmse': float(np.sqrt(sq_err.sum() / n)),
    'n_valid': int(mask.sum()),
  }


def eval_mdn_simple(pred_raw: np.ndarray, gt: np.ndarray, name: str) -> dict:
  """Evaluate simple MDN head (pose, road_transform, wide_from_device_euler)."""
  pred_mean = extract_mdn_mean(pred_raw)
  err = np.abs(pred_mean - gt)
  sq_err = (pred_mean - gt) ** 2
  return {
    'mae': float(err.mean()),
    'rmse': float(np.sqrt(sq_err.mean())),
    'per_dim_mae': [float(err[:, i].mean()) for i in range(gt.shape[-1])],
  }


def eval_prob(pred_raw: np.ndarray, gt: np.ndarray, name: str) -> dict:
  """Evaluate probability head (BCE logits)."""
  if name == 'lane_lines_prob':
    # pred (B, 8) → (B, 4, 2) logits, gt (B, 4) probs
    pred_logits = pred_raw.reshape(-1, 4, 2)
    pred_prob = 1.0 / (1.0 + np.exp(-pred_logits[:, :, 1]))  # sigmoid of second logit
    gt_binary = (gt > 0.5).astype(np.float32)
  elif name == 'lead_prob':
    # pred (B, 3) logits, gt (B, 3) probs
    pred_prob = 1.0 / (1.0 + np.exp(-pred_raw))
    gt_binary = (gt > 0.5).astype(np.float32)
  else:
    return {}

  pred_binary = (pred_prob > 0.5).astype(np.float32)
  accuracy = float((pred_binary == gt_binary).mean())
  mae = float(np.abs(pred_prob - gt).mean())

  return {
    'accuracy': accuracy,
    'prob_mae': mae,
    'gt_positive_rate': float(gt_binary.mean()),
    'pred_positive_rate': float(pred_binary.mean()),
  }


def main():
  parser = argparse.ArgumentParser(description='Evaluate pretrained model against GT')
  parser.add_argument('--data-dir', default='data/dual_camera_train/Town04_003',
                      help='Directory with dual-camera NPZ files')
  parser.add_argument('--onnx', default='selfdrive/modeld/models/driving_vision.onnx',
                      help='Path to driving_vision.onnx')
  parser.add_argument('--num-workers', type=int, default=8,
                      help='DataLoader workers for preprocessing')
  parser.add_argument('--max-dist', type=float, default=None,
                      help='Max longitudinal distance (m) for lane/edge evaluation (default: all 192m)')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  args = parser.parse_args()

  # Compute how many of the 33 points fall within max_dist
  if args.max_dist is not None:
    n_pts = int((X_IDXS <= args.max_dist).sum())
    last_x = X_IDXS[n_pts - 1] if n_pts > 0 else 0
    print(f"Max dist: {args.max_dist}m  → using {n_pts}/33 points (last x={last_x:.1f}m)")
  else:
    print("Max dist: None (all 33 points, up to 192m)")

  print(f"Dataset: {args.data_dir}")
  print(f"Model:   {args.onnx}")
  print(f"Device:  {args.device}")

  # Load dataset
  dataset = DualCameraDrivingDataset(data_dirs=[args.data_dir])
  print(f"Samples: {len(dataset)}")

  loader = DataLoader(
    dataset,
    batch_size=1,  # model is fixed to batch=1
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
  )

  # Load model
  print("Loading pretrained model...")
  model = PretrainedVisionModel(args.onnx, freeze=True)
  model.eval()
  model = model.to(args.device)
  print(f"  Parameters: {model.n_total_params():,}")

  # Accumulators
  all_preds: dict[str, list[np.ndarray]] = {k: [] for k in ONNX_OUTPUT_SLICES}
  all_targets: dict[str, list[np.ndarray]] = {}

  target_keys = [
    'lane_lines', 'lane_lines_prob', 'lane_lines_valid',
    'road_edges', 'road_edges_prob', 'road_edges_valid',
    'lead', 'lead_prob',
    'pose', 'road_transform', 'wide_from_device_euler',
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

  # Stack all
  for k in all_preds:
    all_preds[k] = np.concatenate(all_preds[k], axis=0)
  for k in all_targets:
    all_targets[k] = np.concatenate(all_targets[k], axis=0)

  # --- Evaluate each component ---
  print("=" * 75)
  print(f"{'Component':<30} {'MAE':>10} {'RMSE':>10} {'Extra':>20}")
  print("-" * 75)

  # Lane lines
  r = eval_lane_lines(
    all_preds['lane_lines'], all_targets['lane_lines'],
    all_targets['lane_lines_prob'], all_targets['lane_lines_valid'],
    max_dist=args.max_dist,
  )
  dist_note = f" (≤{args.max_dist}m)" if args.max_dist else ""
  print(f"{'lane_lines' + dist_note:<30} {r['mae']:>10.4f} {r['rmse']:>10.4f} {'valid=' + str(r['n_valid']):>20}")

  # Lane lines prob
  r = eval_prob(all_preds['lane_lines_prob'], all_targets['lane_lines_prob'], 'lane_lines_prob')
  acc_str = f"acc={r['accuracy']:.4f}"
  print(f"{'lane_lines_prob':<30} {'':>10} {'':>10} {acc_str:>20}")

  # Road edges
  r = eval_road_edges(
    all_preds['road_edges'], all_targets['road_edges'],
    all_targets['road_edges_prob'], all_targets['road_edges_valid'],
    max_dist=args.max_dist,
  )
  print(f"{'road_edges' + dist_note:<30} {r['mae']:>10.4f} {r['rmse']:>10.4f} {'valid=' + str(r['n_valid']):>20}")

  # Lead
  r = eval_lead(all_preds['lead'], all_targets['lead'], all_targets['lead_prob'])
  print(f"{'lead':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f} {'valid=' + str(r['n_valid']):>20}")

  # Lead prob
  r = eval_prob(all_preds['lead_prob'], all_targets['lead_prob'], 'lead_prob')
  acc_str = f"acc={r['accuracy']:.4f}"
  print(f"{'lead_prob':<30} {'':>10} {'':>10} {acc_str:>20}")

  # Pose
  r = eval_mdn_simple(all_preds['pose'], all_targets['pose'], 'pose')
  print(f"{'pose':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f}")
  labels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
  for j, (lbl, v) in enumerate(zip(labels, r['per_dim_mae'])):
    print(f"{'  ' + lbl:<30} {v:>10.4f}")

  # Road transform
  r = eval_mdn_simple(all_preds['road_transform'], all_targets['road_transform'], 'road_transform')
  print(f"{'road_transform':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f}")
  labels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
  for j, (lbl, v) in enumerate(zip(labels, r['per_dim_mae'])):
    print(f"{'  ' + lbl:<30} {v:>10.4f}")

  # Wide from device euler
  r = eval_mdn_simple(all_preds['wide_from_device_euler'], all_targets['wide_from_device_euler'], 'wide_from_device_euler')
  print(f"{'wide_from_device_euler':<30} {r['mae']:>10.4f} {r['rmse']:>10.4f}")
  labels = ['roll', 'pitch', 'yaw']
  for j, (lbl, v) in enumerate(zip(labels, r['per_dim_mae'])):
    print(f"{'  ' + lbl:<30} {v:>10.4f}")

  print("=" * 75)
  print("Done.")


if __name__ == '__main__':
  main()
