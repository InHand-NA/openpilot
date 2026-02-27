#!/usr/bin/env python3
"""Export TorchScript traced .pt model to ONNX format.

Converts the pretrained openpilot model (TorchScript traced) to ONNX,
with optional fp16 conversion.

Usage:
  # Default (fp32)
  python3 tools/dashcam/train/export_pretrained_onnx.py

  # With fp16 conversion
  python3 tools/dashcam/train/export_pretrained_onnx.py --fp16

  # Custom paths
  python3 tools/dashcam/train/export_pretrained_onnx.py \
    --input checkpoints/pretrained_openpilot.pt \
    --output checkpoints/pretrained_openpilot.onnx
"""

import argparse
import os

import numpy as np
import onnx
import torch


def main():
  parser = argparse.ArgumentParser(description='Export TorchScript .pt to ONNX')
  parser.add_argument('--input', default='checkpoints/pretrained_openpilot.pt',
                      help='Input TorchScript traced .pt file')
  parser.add_argument('--output', default='checkpoints/pretrained_openpilot.onnx',
                      help='Output ONNX file path')
  parser.add_argument('--fp16', action='store_true',
                      help='Convert to fp16 after export')
  parser.add_argument('--opset', type=int, default=17,
                      help='ONNX opset version')
  args = parser.parse_args()

  print(f"Loading TorchScript: {args.input}")
  model = torch.jit.load(args.input, map_location='cpu')
  model.eval()

  n_params = sum(p.numel() for p in model.parameters())
  print(f"  Parameters: {n_params:,}")

  # Dummy inputs matching the model's expected format
  dummy_img = torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8)
  dummy_big = torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8)

  # Verify forward pass
  with torch.no_grad():
    out = model(dummy_img, dummy_big)
  print(f"  Output shape: {out.shape}, dtype: {out.dtype}")

  # Export to ONNX
  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

  print(f"Exporting to ONNX (opset {args.opset})...")
  torch.onnx.export(
    model,
    (dummy_img, dummy_big),
    args.output,
    opset_version=args.opset,
    input_names=['img', 'big_img'],
    output_names=['outputs'],
    dynamic_axes=None,  # fixed batch=1
    dynamo=False,  # use legacy exporter (ScriptModule compatible)
  )

  # Validate with onnxruntime
  onnx_model = onnx.load(args.output)
  onnx.checker.check_model(onnx_model)
  print("  ONNX checker: OK")

  import onnxruntime as ort
  sess = ort.InferenceSession(args.output)
  ort_out = sess.run(None, {
    'img': dummy_img.numpy(),
    'big_img': dummy_big.numpy(),
  })[0]
  pt_out = out.float().numpy()
  max_diff = np.abs(ort_out.astype(np.float32) - pt_out).max()
  print(f"  ORT vs PyTorch max diff: {max_diff:.2e}")

  # Optional fp16 conversion
  if args.fp16:
    from onnx import TensorProto
    fp16_model = onnx.load(args.output)
    for initializer in fp16_model.graph.initializer:
      if initializer.data_type == TensorProto.FLOAT:
        data = np.frombuffer(initializer.raw_data, dtype=np.float32).astype(np.float16)
        initializer.raw_data = data.tobytes()
        initializer.data_type = TensorProto.FLOAT16
    for node in fp16_model.graph.node:
      for attr in node.attribute:
        if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
          data = np.frombuffer(attr.t.raw_data, dtype=np.float32).astype(np.float16)
          attr.t.raw_data = data.tobytes()
          attr.t.data_type = TensorProto.FLOAT16
    fp16_path = args.output.replace('.onnx', '_fp16.onnx') if not args.output.endswith('_fp16.onnx') else args.output
    onnx.save(fp16_model, fp16_path)
    fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"  Saved fp16: {fp16_path} ({fp16_size:.1f} MB)")

  size_mb = os.path.getsize(args.output) / (1024 * 1024)
  print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")


if __name__ == '__main__':
  main()
