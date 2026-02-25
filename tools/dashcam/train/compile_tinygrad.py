#!/usr/bin/env python3
"""Compile ONNX model to tinygrad TinyJit pkl + generate metadata.pkl.

Converts a flat-output ONNX model (with embedded output_slices) into a tinygrad
TinyJit pkl file for inference, matching the openpilot modeld pipeline.

Usage:
  python tools/dashcam/train/compile_tinygrad.py checkpoints/driving_vision.onnx
  # Output: checkpoints/driving_vision_tinygrad_cuda.pkl
  #         checkpoints/driving_vision_metadata.pkl

  # Specify device
  DEV=CUDA python tools/dashcam/train/compile_tinygrad.py checkpoints/driving_vision.onnx
"""

import argparse
import codecs
import os
import pickle
import sys

import numpy as np
import onnx


def get_metadata_value_by_name(model: onnx.ModelProto, name: str) -> str | None:
  for prop in model.metadata_props:
    if prop.key == name:
      return prop.value
  return None


def generate_metadata(onnx_path: str, metadata_path: str):
  """Extract metadata from ONNX and save as pkl (reuses get_model_metadata.py logic)."""
  model = onnx.load(onnx_path)
  output_slices_raw = get_metadata_value_by_name(model, 'output_slices')
  assert output_slices_raw is not None, "output_slices not found in ONNX metadata"

  # Replace dynamic dims (0) with 1 for batch dimension (match openpilot convention)
  def _fix_shape(dims):
    return tuple(max(d.dim_value, 1) for d in dims)

  metadata = {
    'output_slices': pickle.loads(codecs.decode(output_slices_raw.encode(), 'base64')),
    'input_shapes': {inp.name: _fix_shape(inp.type.tensor_type.shape.dim) for inp in model.graph.input},
    'output_shapes': {out.name: _fix_shape(out.type.tensor_type.shape.dim) for out in model.graph.output},
  }
  with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
  print(f"Metadata saved: {metadata_path}")
  print(f"  input_shapes: {metadata['input_shapes']}")
  print(f"  output_shapes: {metadata['output_shapes']}")
  print(f"  output_slices: {list(metadata['output_slices'].keys())}")


def compile_model(onnx_path: str, output_pkl_path: str):
  """Compile ONNX to tinygrad TinyJit pkl (matching compile3.py pattern)."""
  from tinygrad import Device, Tensor, TinyJit
  from tinygrad.dtype import dtypes
  from tinygrad.nn.onnx import OnnxRunner

  print(f"Loading ONNX: {onnx_path}")
  run_onnx = OnnxRunner(onnx_path)

  # Read input shapes and types
  input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
  input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}
  print(f"  Inputs: {input_shapes}")
  print(f"  Types: {input_types}")

  # Generate dummy inputs (uint8 → randint, float32 → randn)
  inputs = {}
  for k, shp in sorted(input_shapes.items()):
    if input_types[k] is dtypes.uint8:
      inputs[k] = Tensor(np.random.randint(0, 256, shp, dtype=np.uint8), device='NPY')
    else:
      inputs[k] = Tensor(Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize().numpy(), device='NPY')

  # img inputs go to default device (GPU)
  inputs = {k: Tensor(v.numpy(), device=Device.DEFAULT).realize() if 'img' in k else v for k, v in inputs.items()}

  # TinyJit wrapper (single flat output)
  run_onnx_jit = TinyJit(
    lambda **kwargs: next(iter(run_onnx({k: v.to(Device.DEFAULT) for k, v in kwargs.items()}).values())).cast('float32'),
    prune=True,
  )

  # Run 3 times to capture JIT graph
  print("Capturing JIT graph (3 runs)...")
  test_val = None
  for i in range(3):
    ret = run_onnx_jit(**inputs).numpy()
    print(f"  Run {i}: output shape={ret.shape}, sum={ret.sum():.4f}")
    if i == 1:
      test_val = np.copy(ret)
  np.testing.assert_equal(test_val, ret, err_msg="JIT validation failed: run 1 != run 2")
  print("JIT validation passed.")

  with open(output_pkl_path, 'wb') as f:
    pickle.dump(run_onnx_jit, f)
  pkl_size = os.path.getsize(output_pkl_path) / (1024 * 1024)
  print(f"TinyJit pkl saved: {output_pkl_path} ({pkl_size:.1f}MB)")


def main():
  parser = argparse.ArgumentParser(description='Compile ONNX to tinygrad pkl')
  parser.add_argument('onnx_path', help='Path to ONNX model with output_slices metadata')
  parser.add_argument('--output', default='', help='Output pkl path (default: auto-generated)')
  args = parser.parse_args()

  onnx_path = args.onnx_path
  if not os.path.exists(onnx_path):
    print(f"Error: {onnx_path} not found")
    sys.exit(1)

  # Auto-generate output paths
  base = os.path.splitext(onnx_path)[0]
  dev = os.environ.get('DEV', 'CUDA').lower()
  pkl_path = args.output if args.output else f"{base}_tinygrad_{dev}.pkl"
  metadata_path = f"{base}_metadata.pkl"

  # Step 1: Generate metadata
  generate_metadata(onnx_path, metadata_path)

  # Step 2: Compile to tinygrad pkl
  compile_model(onnx_path, pkl_path)

  print("\nDone. Files:")
  print(f"  pkl:      {pkl_path}")
  print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
  main()
