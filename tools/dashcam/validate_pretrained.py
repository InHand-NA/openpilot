#!/usr/bin/env python3
"""Validate pretrained_model.py against the original ONNX model.

Runs N samples through both:
  1. Original ONNX (via onnxruntime, reference baseline)
  2. PretrainedVisionModel (onnx2torch, float32)

Reports per-component statistics:
  - Mean absolute error (MAE)
  - Max absolute error
  - Pearson correlation coefficient

Expected: MAE < 1.0, correlation > 0.999 for all components
(difference is due to float16 vs float32 computation)

Usage:
  python3 tools/dashcam/validate_pretrained.py \\
    --onnx selfdrive/modeld/models/driving_vision.onnx \\
    --data-dir data/dual_camera_exp_002 \\
    --n-samples 10
"""

import argparse
import cv2
from pathlib import Path

import numpy as np
import torch

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import SBIGMODEL_INPUT_SIZE, get_warp_matrix
from openpilot.tools.dashcam.train.dataset import rgb_to_yuv420_6ch, warp_image
from openpilot.tools.dashcam.train.pretrained_model import ONNX_OUTPUT_SLICES, PretrainedVisionModel


def load_onnx_session(onnx_path: str):
  """Load ONNX model via onnxruntime for reference."""
  try:
    import onnxruntime as ort
  except ImportError as e:
    raise ImportError("onnxruntime is required: pip install onnxruntime") from e
  return ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])


def preprocess_sample(npz_data: dict, fcam_intrinsics: np.ndarray, ecam_intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Preprocess road + wide RGB to dual-camera YUV420 uint8.

  Returns:
    img:     (1, 12, 128, 256) uint8 — road camera (prev=zeros + curr)
    big_img: (1, 12, 128, 256) uint8 — wide camera (prev=zeros + curr)
  """
  rpyCalib = npz_data['rpyCalib'].astype(np.float64)

  # Road camera: medmodel warp
  road_warped = warp_image(npz_data['road_rgb'], rpyCalib, fcam_intrinsics)
  road_yuv = rgb_to_yuv420_6ch(road_warped)

  # Wide camera: sbigmodel warp
  M_wide = get_warp_matrix(rpyCalib, ecam_intrinsics, bigmodel_frame=True)
  wide_warped = cv2.warpPerspective(
    npz_data['wide_rgb'], M_wide, SBIGMODEL_INPUT_SIZE,
    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
  )
  wide_yuv = rgb_to_yuv420_6ch(wide_warped)

  # Use zeros for "previous" frame (temporal pairing not critical for validation)
  prev_zeros = np.zeros_like(road_yuv)
  img = np.concatenate([prev_zeros, road_yuv], axis=0)[np.newaxis]       # (1, 12, 128, 256)
  big_img = np.concatenate([prev_zeros, wide_yuv], axis=0)[np.newaxis]   # (1, 12, 128, 256)

  return img, big_img


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
  """Pearson r between two flat arrays."""
  a = a.flatten().astype(np.float64)
  b = b.flatten().astype(np.float64)
  if a.std() < 1e-10 or b.std() < 1e-10:
    return float('nan')
  return float(np.corrcoef(a, b)[0, 1])


def main():
  parser = argparse.ArgumentParser(description='Validate pretrained model against ONNX')
  parser.add_argument('--onnx', default='selfdrive/modeld/models/driving_vision.onnx',
                      help='Path to driving_vision.onnx')
  parser.add_argument('--data-dir', default='data/dual_camera_exp_002',
                      help='Directory with dual-camera NPZ files')
  parser.add_argument('--n-samples', type=int, default=10,
                      help='Number of samples to validate')
  args = parser.parse_args()

  print(f"Loading ONNX model: {args.onnx}")
  ort_session = load_onnx_session(args.onnx)

  print("Loading PyTorch model (PretrainedVisionModel)...")
  pt_model = PretrainedVisionModel(args.onnx, freeze=True)
  pt_model.eval()
  print(f"  Total params: {pt_model.n_total_params():,}")

  # Camera intrinsics
  dc = DEVICE_CAMERAS[("pc", "unknown")]
  fcam_intrinsics = dc.fcam.intrinsics
  ecam_intrinsics = dc.ecam.intrinsics

  # Load NPZ files
  data_path = Path(args.data_dir)
  npz_files = sorted(data_path.glob('*.npz'))[:args.n_samples]
  if not npz_files:
    print(f"ERROR: No NPZ files found in {args.data_dir}")
    return

  print(f"\nRunning {len(npz_files)} samples...\n")

  # Collect outputs for all samples (single pass)
  ort_all: dict[str, list[np.ndarray]] = {k: [] for k in ONNX_OUTPUT_SLICES}
  pt_all: dict[str, list[np.ndarray]] = {k: [] for k in ONNX_OUTPUT_SLICES}

  for i, npz_path in enumerate(npz_files):
    npz_data = dict(np.load(npz_path, allow_pickle=True))
    img_np, big_img_np = preprocess_sample(npz_data, fcam_intrinsics, ecam_intrinsics)

    # --- ONNX (reference) ---
    ort_flat = ort_session.run(None, {'img': img_np, 'big_img': big_img_np})[0].astype(np.float32)  # (1, 1576)

    # --- PyTorch ---
    img_t = torch.from_numpy(img_np)
    big_img_t = torch.from_numpy(big_img_np)
    with torch.no_grad():
      pt_out_dict = pt_model(img_t, big_img_t)

    for name, sl in ONNX_OUTPUT_SLICES.items():
      ort_all[name].append(ort_flat[:, sl])
      pt_all[name].append(pt_out_dict[name].numpy())

    if (i + 1) % 5 == 0:
      print(f"  Processed {i+1}/{len(npz_files)} samples")

  # Report statistics
  print("\n" + "=" * 70)
  print(f"{'Component':<30} {'MAE':>8} {'MaxErr':>8} {'Corr':>8} {'Dims':>6}")
  print("-" * 70)

  all_ok = True
  for name, sl in ONNX_OUTPUT_SLICES.items():
    ort_vals = np.concatenate(ort_all[name], axis=0)  # (N, dims)
    pt_vals = np.concatenate(pt_all[name], axis=0)    # (N, dims)
    err = np.abs(pt_vals - ort_vals)
    mae = float(err.mean())
    max_err = float(err.max())
    dims = sl.stop - sl.start
    corr = pearson_correlation(ort_vals, pt_vals)

    ok = mae < 1.0 and (np.isnan(corr) or corr > 0.999)
    status = "OK" if ok else "WARN"
    if not ok:
      all_ok = False
    corr_str = f"{corr:.6f}" if not np.isnan(corr) else "    nan"
    print(f"{name:<30} {mae:>8.4f} {max_err:>8.4f} {corr_str:>8} {dims:>6}  {status}")

  print("=" * 70)
  if all_ok:
    print("\nRESULT: PyTorch model matches ONNX. Ready for fine-tuning!")
  else:
    print("\nRESULT: Some components show larger-than-expected differences.")
    print("  This may be acceptable for float16->float32 precision differences.")


if __name__ == '__main__':
  main()
