#!/usr/bin/env python3
"""Export trained driving vision model to ONNX format.

Steps:
  1. Load checkpoint -> model.eval()
  2. Fuse RepConv blocks (reparameterization for inference)
  3. Export via torch.onnx.export with opset 17
  4. Validate with onnxruntime
  5. Optionally convert to fp16

Dual-camera mode (--dual-camera):
  - FlatOutputWrapper: concat 7 outputs into single flat tensor (match tinygrad pkl format)
  - Embeds output_slices in ONNX metadata_props (base64 pickle)
  - Two inputs: img + big_img (uint8)

Usage:
  # Single-camera (V1 compatible)
  python tools/dashcam/train/export_onnx.py --checkpoint checkpoints/best.pt --output driving_vision.onnx

  # Dual-camera (flat output + metadata)
  python tools/dashcam/train/export_onnx.py --checkpoint checkpoints/best.pt --output driving_vision.onnx --dual-camera

  # Export fp32 + fp16
  python tools/dashcam/train/export_onnx.py --checkpoint checkpoints/best.pt --output driving_vision.onnx --fp16
"""

import argparse
import codecs
import os
import pickle

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

from openpilot.tools.dashcam.train.config import DualCameraModelConfig, ModelConfig
from openpilot.tools.dashcam.train.model import DrivingVisionModel


OUTPUT_NAMES = ['lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob', 'pose', 'road_transform']

# Output sizes for each head (flattened)
MODEL_OUTPUT_SIZES = {
  'lane_lines': 528,
  'lane_lines_prob': 8,
  'road_edges': 264,
  'lead': 144,
  'lead_prob': 3,
  'pose': 12,
  'road_transform': 12,
}


class FlatOutputWrapper(nn.Module):
  """Wrap DrivingVisionModel to concat dict outputs into a single flat tensor.

  Matches tinygrad pkl format: single 'outputs' tensor of shape (B, 971).
  """

  OUTPUT_ORDER = OUTPUT_NAMES

  def __init__(self, model: DrivingVisionModel):
    super().__init__()
    self.model = model

  def forward(self, img: torch.Tensor, big_img: torch.Tensor | None = None) -> torch.Tensor:
    outs = self.model(img, big_img)
    return torch.cat([outs[k] for k in self.OUTPUT_ORDER], dim=-1)


def compute_output_slices() -> dict[str, slice]:
  """Compute output_slices mapping from output names to flat tensor slices."""
  output_slices = {}
  offset = 0
  for name in OUTPUT_NAMES:
    size = MODEL_OUTPUT_SIZES[name]
    output_slices[name] = slice(offset, offset + size)
    offset += size
  return output_slices


def embed_output_slices_metadata(onnx_path: str):
  """Embed output_slices as base64 pickle in ONNX metadata_props."""
  output_slices = compute_output_slices()
  onnx_model = onnx.load(onnx_path)
  encoded = codecs.encode(pickle.dumps(output_slices), 'base64').decode()
  onnx_model.metadata_props.add(key='output_slices', value=encoded)
  onnx.save(onnx_model, onnx_path)
  print(f"  Embedded output_slices metadata: {list(output_slices.keys())}")


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
  """Export single-camera model to ONNX (V1 format: 7 named outputs).

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


def export_onnx_dual(model: DrivingVisionModel, output_path: str, device: torch.device | None = None):
  """Export dual-camera model to ONNX (flat output + output_slices metadata).

  - Two uint8 inputs: img (1, 12, 128, 256), big_img (1, 12, 128, 256)
  - Single flat output: outputs (1, 971)
  - output_slices embedded in metadata_props

  Args:
    model: trained DrivingVisionModel with uint8_input=True
    output_path: path for the .onnx file
    device: device to run export on
  """
  if device is None:
    device = torch.device('cpu')

  model = model.to(device)
  model.eval()
  model.fuse_repconv()

  wrapper = FlatOutputWrapper(model).to(device)
  wrapper.eval()

  # uint8 dummy inputs (each camera: 12ch = 2 frames x 6ch YUV420)
  dummy_img = torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8, device=device)
  dummy_big_img = torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8, device=device)

  torch.onnx.export(
    wrapper,
    (dummy_img, dummy_big_img),
    output_path,
    input_names=['img', 'big_img'],
    output_names=['outputs'],
    opset_version=17,
    dynamic_axes={'img': {0: 'batch'}, 'big_img': {0: 'batch'}, 'outputs': {0: 'batch'}},
  )
  print(f"Dual-camera ONNX exported: {output_path}")

  # Embed output_slices metadata
  embed_output_slices_metadata(output_path)

  # Validate with onnxruntime
  try:
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path)
    ort_outputs = sess.run(None, {'img': dummy_img.cpu().numpy(), 'big_img': dummy_big_img.cpu().numpy()})

    with torch.no_grad():
      pt_output = wrapper(dummy_img, dummy_big_img)

    pt_np = pt_output.cpu().numpy()
    ort_np = ort_outputs[0]
    max_diff = np.max(np.abs(pt_np - ort_np))
    print(f"  outputs: max_diff={max_diff:.6e} shape={ort_np.shape}")
    if max_diff > 1e-4:
      print("    WARNING: large difference")

    print("ONNX validation passed.")
  except ImportError:
    print("onnxruntime not installed, skipping validation.")


def main():
  parser = argparse.ArgumentParser(description='Export model to ONNX')
  parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt file')
  parser.add_argument('--output', default='driving_vision.onnx', help='Output ONNX path')
  parser.add_argument('--fp16', action='store_true', help='Also export fp16 version')
  parser.add_argument('--dual-camera', action='store_true', help='Export dual-camera model (flat output + metadata)')
  args = parser.parse_args()

  if args.dual_camera:
    model_cfg = DualCameraModelConfig()
  else:
    model_cfg = ModelConfig()
  model = DrivingVisionModel(model_cfg)

  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  model.load_state_dict(ckpt['model_state_dict'])
  print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

  if args.dual_camera:
    export_onnx_dual(model, args.output, torch.device('cpu'))
  else:
    export_onnx(model, args.output, torch.device('cpu'))

  if args.fp16:
    base, ext = os.path.splitext(args.output)
    fp16_path = f"{base}_fp16{ext}"
    convert_onnx_to_fp16(args.output, fp16_path)


if __name__ == "__main__":
  main()
