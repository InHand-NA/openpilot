import os
import pickle
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pathlib import Path

from openpilot.common.basedir import BASEDIR
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.constants import ModelConstants

from openpilot.tools.dashcam.carla_world import W, H

VISION_PKL_PATH = Path(BASEDIR) / 'selfdrive/modeld/models/driving_vision_tinygrad.pkl'
VISION_METADATA_PATH = Path(BASEDIR) / 'selfdrive/modeld/models/driving_vision_metadata.pkl'

NV12_SIZE = W * H * 3 // 2


class RGBToNV12Converter:
  """OpenCL-accelerated RGB to NV12 conversion, reuses tools/sim/rgb_to_nv12.cl."""

  def __init__(self):
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)

    cl_arg = (f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} "
              f"-DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG ")

    kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(cl_arg)
      self.krnl = prg.rgb_to_nv12

    self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4

  def convert(self, rgb):
    """RGB (H,W,3) uint8 -> NV12 (H*W*3//2,) uint8 numpy array."""
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (self.Wdiv4, self.Hdiv4), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), NV12_SIZE)
    return np.ascontiguousarray(yuv).astype(np.uint8)


class VisionModel:
  """Vision model: OpenCL preprocessing + tinygrad inference + output parsing."""

  def __init__(self):
    # Set device before importing tinygrad
    if 'DEV' not in os.environ:
      os.environ['DEV'] = 'CPU'
    from tinygrad.tensor import Tensor
    from tinygrad.dtype import dtypes
    self._Tensor = Tensor
    self._dtypes = dtypes

    # Load metadata
    with open(VISION_METADATA_PATH, 'rb') as f:
      vision_metadata = pickle.load(f)
      self.input_shapes = vision_metadata['input_shapes']
      self.input_names = list(self.input_shapes.keys())
      self.output_slices = vision_metadata['output_slices']
      self.output_size = vision_metadata['output_shapes']['outputs'][1]

    # Initialize OpenCL context and DrivingModelFrames
    from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext, DrivingModelFrame
    self.cl_context = CLContext()
    temporal_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    self.frames = {name: DrivingModelFrame(self.cl_context, temporal_skip) for name in self.input_names}

    # RGB to NV12 converter (separate pyopencl context)
    self.rgb_converter = RGBToNV12Converter()

    # Parser
    self.parser = Parser()

    # Load vision model
    print("Loading vision model...")
    with open(VISION_PKL_PATH, 'rb') as f:
      self.vision_run = pickle.load(f)
    print("Vision model loaded")

  def preprocess(self, road_rgb, wide_rgb, intrinsics, rpyCalib):
    """Preprocess camera images through full OpenCL pipeline.

    Returns dict of {input_name: Tensor} ready for inference.
    """
    # Map input names to camera images
    # 'big' in name -> wide camera (SBIG model params), else -> narrow camera (MED params)
    images = {}
    for name in self.input_names:
      if 'big' in name:
        images[name] = wide_rgb
      else:
        images[name] = road_rgb

    # Compute warp matrices
    warp_matrices = {}
    for name in self.input_names:
      bigmodel = 'big' in name
      warp_matrices[name] = get_warp_matrix(rpyCalib, intrinsics.ecam.intrinsics if bigmodel else intrinsics.fcam.intrinsics, bigmodel).astype(np.float32)

    # Process each input
    vision_inputs = {}
    for name in self.input_names:
      rgb = images[name]
      if rgb is None:
        # If wide camera not available, use narrow for both
        rgb = images[self.input_names[0]]

      # RGB -> NV12 (pyopencl context, result on CPU)
      nv12 = self.rgb_converter.convert(rgb)

      # NV12 -> model input via DrivingModelFrame (CLContext's OpenCL context)
      projection = warp_matrices[name].flatten()
      cl_mem = self.frames[name].prepare_from_yuv(
        self.cl_context, nv12, W, H, W, W * H, projection)

      # Read back from CL and create Tensor
      frame_data = self.frames[name].buffer_from_cl(cl_mem).reshape(self.input_shapes[name])
      vision_inputs[name] = self._Tensor(frame_data, dtype=self._dtypes.uint8).realize()

    return vision_inputs

  def run(self, vision_inputs):
    """Run vision network inference and parse outputs."""
    raw_output = self.vision_run(**vision_inputs).contiguous().realize().uop.base.buffer.numpy()
    sliced = {k: raw_output[np.newaxis, v] for k, v in self.output_slices.items()}
    parsed = self.parser.parse_vision_outputs(sliced)
    return parsed
