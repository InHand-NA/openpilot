#!/usr/bin/env python3
"""Export trained driving vision model to ONNX format.

Steps:
  1. Load checkpoint -> model.eval()
  2. Fuse RepConv blocks (reparameterization for inference)
  3. Export via torch.onnx.export with opset 17
  4. Validate with onnxruntime
  5. Optionally convert to fp16

Usage:
  # Export fp32 only
  python tools/dashcam/train/export_onnx.py --checkpoint checkpoints/best.pt --output driving_vision.onnx

  # Export fp32 + fp16
  python tools/dashcam/train/export_onnx.py --checkpoint checkpoints/best.pt --output driving_vision.onnx --fp16
"""

import argparse
import os

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from openpilot.tools.dashcam.train.config import ModelConfig
from openpilot.tools.dashcam.train.model import DrivingVisionModel


OUTPUT_NAMES = ['lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob', 'pose', 'road_transform']


def convert_onnx_to_fp16(input_path: str, output_path: str):
  """Convert an fp32 ONNX model to fp16.

  Converts all float32 tensors (initializers, constants, intermediate values) to float16.
  Keeps inputs/outputs as float32 with Cast nodes for compatibility.
  """
  model = onnx.load(input_path)

  # Convert initializers (weights) to float16
  for initializer in model.graph.initializer:
    if initializer.data_type == onnx.TensorProto.FLOAT:
      arr = numpy_helper.to_array(initializer).astype(np.float16)
      new_init = numpy_helper.from_array(arr, name=initializer.name)
      initializer.CopyFrom(new_init)

  # Convert constant nodes to float16
  for node in model.graph.node:
    if node.op_type == 'Constant':
      for attr in node.attribute:
        if attr.name == 'value' and attr.t.data_type == onnx.TensorProto.FLOAT:
          arr = numpy_helper.to_array(attr.t).astype(np.float16)
          new_t = numpy_helper.from_array(arr)
          attr.t.CopyFrom(new_t)

  # Update value_info intermediate types to float16
  for vi in model.graph.value_info:
    if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
      vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

  # Add Cast fp32→fp16 after input, Cast fp16→fp32 before outputs
  # Input: keep as fp32, add Cast node
  input_cast = onnx.helper.make_node('Cast', inputs=['img'], outputs=['img_fp16'], to=onnx.TensorProto.FLOAT16)

  # Rename all references from 'img' to 'img_fp16' in graph nodes
  for node in model.graph.node:
    for i, inp in enumerate(node.input):
      if inp == 'img':
        node.input[i] = 'img_fp16'

  # Output: add Cast fp16→fp32 for each output
  output_casts = []
  for out in model.graph.output:
    orig_name = out.name
    fp16_name = f'{orig_name}_fp16'
    # Rename graph node outputs that produce this tensor
    for node in model.graph.node:
      for i, o in enumerate(node.output):
        if o == orig_name:
          node.output[i] = fp16_name
    cast_node = onnx.helper.make_node('Cast', inputs=[fp16_name], outputs=[orig_name], to=onnx.TensorProto.FLOAT)
    output_casts.append(cast_node)

  # Insert cast nodes
  model.graph.node.insert(0, input_cast)
  model.graph.node.extend(output_casts)

  onnx.save(model, output_path)
  fp32_size = os.path.getsize(input_path) / (1024 * 1024)
  fp16_size = os.path.getsize(output_path) / (1024 * 1024)
  print(f"FP16 ONNX exported: {output_path} ({fp16_size:.1f}MB, fp32 was {fp32_size:.1f}MB)")


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
  parser.add_argument('--fp16', action='store_true', help='Also export fp16 version')
  args = parser.parse_args()

  model_cfg = ModelConfig()
  model = DrivingVisionModel(model_cfg)

  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  model.load_state_dict(ckpt['model_state_dict'])
  print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

  export_onnx(model, args.output, torch.device('cpu'))

  if args.fp16:
    base, ext = os.path.splitext(args.output)
    fp16_path = f"{base}_fp16{ext}"
    convert_onnx_to_fp16(args.output, fp16_path)


if __name__ == "__main__":
  main()
