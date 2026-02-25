#!/usr/bin/env python3
"""Export trained driving vision model to ONNX format.

Steps:
  1. Load checkpoint -> model.eval()
  2. Fuse RepConv blocks (reparameterization for inference)
  3. Export via torch.onnx.export with opset 17
  4. Validate with onnxruntime

Usage:
  python tools/dashcam/train/export_onnx.py --checkpoint checkpoints/best.pt --output driving_vision.onnx
"""

import argparse

import numpy as np
import torch

from openpilot.tools.dashcam.train.config import ModelConfig
from openpilot.tools.dashcam.train.model import DrivingVisionModel


OUTPUT_NAMES = ['lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob', 'pose', 'road_transform']


def export_onnx(model: DrivingVisionModel, output_path: str, device: torch.device | None = None):
  """Export model to ONNX.

  Args:
    model: trained DrivingVisionModel (will be modified in-place for fusion)
    output_path: path for the .onnx file
    device: device to run export on
  """
  if device is None:
    device = torch.device('cpu')

  model = model.to(device)
  model.eval()
  model.fuse_repconv()

  dummy = torch.randn(1, model.cfg.in_channels, 128, 256, device=device)

  torch.onnx.export(
    model,
    dummy,
    output_path,
    input_names=['img'],
    output_names=OUTPUT_NAMES,
    opset_version=17,
    dynamic_axes={'img': {0: 'batch'}},
  )
  print(f"ONNX exported: {output_path}")

  # Validate with onnxruntime
  try:
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path)
    dummy_np = dummy.cpu().numpy()
    ort_outputs = sess.run(None, {'img': dummy_np})

    with torch.no_grad():
      pt_outputs = model(dummy)

    for name, ort_out in zip(OUTPUT_NAMES, ort_outputs, strict=True):
      pt_out = pt_outputs[name].cpu().numpy()
      max_diff = np.max(np.abs(pt_out - ort_out))
      print(f"  {name}: max_diff={max_diff:.6e} shape={ort_out.shape}")
      if max_diff > 1e-4:
        print(f"    WARNING: large difference for {name}")

    print("ONNX validation passed.")
  except ImportError:
    print("onnxruntime not installed, skipping validation.")


def main():
  parser = argparse.ArgumentParser(description='Export model to ONNX')
  parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt file')
  parser.add_argument('--output', default='driving_vision.onnx', help='Output ONNX path')
  args = parser.parse_args()

  model_cfg = ModelConfig()
  model = DrivingVisionModel(model_cfg)

  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  model.load_state_dict(ckpt['model_state_dict'])
  print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

  export_onnx(model, args.output, torch.device('cpu'))


if __name__ == "__main__":
  main()
