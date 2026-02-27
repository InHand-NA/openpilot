#!/usr/bin/env python3
"""Export openpilot ONNX model to PyTorch .pt checkpoint.

Converts driving_vision.onnx via onnx2torch and saves as .pt.
Two formats available:
  1. state_dict (default): saves weights only, needs onnx2torch to reload
  2. --traced: saves TorchScript traced model, no onnx2torch needed to reload

Usage:
  # State dict (requires onnx2torch to load)
  python3 tools/dashcam/train/export_pretrained_pt.py

  # TorchScript traced (standalone, no onnx2torch needed)
  python3 tools/dashcam/train/export_pretrained_pt.py --traced

  # Custom paths
  python3 tools/dashcam/train/export_pretrained_pt.py --traced \
    --onnx selfdrive/modeld/models/driving_vision.onnx \
    --output checkpoints/pretrained_openpilot.pt
"""

import argparse
import os

import torch

from openpilot.tools.dashcam.train.pretrained_model import ONNX_OUTPUT_SLICES, PretrainedVisionModel


def main():
  parser = argparse.ArgumentParser(description='Export ONNX model to PyTorch .pt')
  parser.add_argument('--onnx', default='selfdrive/modeld/models/driving_vision.onnx',
                      help='Path to driving_vision.onnx')
  parser.add_argument('--output', default='checkpoints/pretrained_openpilot.pt',
                      help='Output .pt file path')
  parser.add_argument('--traced', action='store_true',
                      help='Export as TorchScript traced model (standalone, no onnx2torch needed)')
  args = parser.parse_args()

  print(f"Loading ONNX: {args.onnx}")
  model = PretrainedVisionModel(args.onnx, freeze=False)
  model.eval()
  n_params = model.n_total_params()
  print(f"  Parameters: {n_params:,}")

  # Verify forward pass
  dummy_img = torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8)
  dummy_big = torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8)
  with torch.no_grad():
    out = model(dummy_img, dummy_big)
  total_dims = sum(v.shape[-1] for v in out.values())
  print(f"  Output: {len(out)} heads, {total_dims} total dims")

  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

  if args.traced:
    # TorchScript trace of the raw backbone (flat 1576-dim output)
    # Slicing into named outputs is done by PretrainedVisionModel.forward()
    print("Tracing backbone...")
    traced = torch.jit.trace(model._backbone, (dummy_img, dummy_big))

    # Verify traced output matches
    with torch.no_grad():
      traced_out = traced(dummy_img, dummy_big).float()
      orig_out = model._backbone(dummy_img, dummy_big).float()
    max_diff = (traced_out - orig_out).abs().max().item()
    print(f"  Trace vs original max diff: {max_diff:.2e}")

    output_slices = {name: (sl.start, sl.stop) for name, sl in ONNX_OUTPUT_SLICES.items()}
    torch.jit.save(traced, args.output, _extra_files={
      'output_slices': str(output_slices),
      'n_params': str(n_params),
    })
  else:
    # State dict checkpoint
    output_slices = {name: (sl.start, sl.stop) for name, sl in ONNX_OUTPUT_SLICES.items()}
    torch.save({
      'model_state_dict': model.state_dict(),
      'n_params': n_params,
      'output_slices': output_slices,
    }, args.output)

  size_mb = os.path.getsize(args.output) / (1024 * 1024)
  fmt = "TorchScript traced" if args.traced else "state_dict"
  print(f"\nSaved ({fmt}): {args.output} ({size_mb:.1f} MB)")


if __name__ == '__main__':
  main()
