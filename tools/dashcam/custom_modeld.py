#!/usr/bin/env python3
"""Custom modeld subprocess for self-trained driving vision models.

Image preprocessing + inference engine fully match openpilot modeld:
  - DrivingModelFrame (OpenCL): warp + loadyuv + temporal frame management
  - get_warp_matrix: medmodel (road) / sbigmodel (wide) warp matrices
  - Input format: uint8 (matching openpilot)
  - tinygrad TinyJit pkl inference engine

Differences from openpilot modeld:
  - Self-trained model (971-dim vision-only), no policy network
  - 7 output branches (lane/edge/lead/pose/rt), no plan/desire/meta/hidden_state

Usage:
  CUSTOM_MODEL_PKL=checkpoints/driving_vision_tinygrad_cuda.pkl \
  CUSTOM_MODEL_METADATA=checkpoints/driving_vision_metadata.pkl \
  python -m openpilot.tools.dashcam.custom_modeld
"""

import os
import pickle
import time

import numpy as np

# Environment variables must be set before importing tinygrad/openpilot modules
os.environ.setdefault('NOBOARD', '1')
os.environ.setdefault('SIMULATION', '1')
os.environ.setdefault('PYOPENCL_CTX', '')

if 'DEV' not in os.environ:
  os.environ['DEV'] = 'CUDA'

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

from cereal import messaging
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.models.commonmodel_pyx import DrivingModelFrame, CLContext
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import fill_xyzt, fill_xyvat


def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


class CustomModelState:
  """Model state for custom vision-only model (simplified openpilot ModelState).

  Matches openpilot ModelState pipeline:
    - DrivingModelFrame (OpenCL warp + loadyuv + temporal)
    - CLContext shared
    - tinygrad TinyJit pkl inference
    - Tensor(frame_input, dtype=dtypes.uint8).realize()
    - .contiguous().realize().uop.base.buffer.numpy() output extraction

  Simplified:
    - No policy network (vision only)
    - No desire/traffic_convention/features_buffer inputs
  """

  def __init__(self, pkl_path: str, metadata_path: str):
    self.cl_context = CLContext()
    temporal_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4

    # Load metadata
    with open(metadata_path, 'rb') as f:
      metadata = pickle.load(f)
      self.input_shapes = metadata['input_shapes']
      self.input_names = list(self.input_shapes.keys())
      self.output_slices = metadata['output_slices']
      self.output_size = metadata['output_shapes']['outputs'][1]

    # DrivingModelFrame: one per img input (same as openpilot modeld)
    self.frames = {name: DrivingModelFrame(self.cl_context, temporal_skip) for name in self.input_names}

    # Load tinygrad TinyJit (same as openpilot modeld.py:241)
    with open(pkl_path, 'rb') as f:
      self.vision_run = pickle.load(f)

    self.vision_inputs: dict[str, Tensor] = {}
    self.vision_output = np.zeros(self.output_size, dtype=np.float32)

    print(f"[custom_modeld] Model loaded: {pkl_path}")
    print(f"[custom_modeld] Inputs: {self.input_shapes}")
    print(f"[custom_modeld] Output size: {self.output_size}")

  def run(self, bufs, transforms):
    """Execute one frame of inference (simplified ModelState.run).

    Args:
      bufs: {name: VisionBuf} vision buffer for each input
      transforms: {name: np.ndarray (9,)} warp matrix for each input

    Returns:
      dict[str, np.ndarray] parsed outputs, or None if not enough frames
    """
    # 1. OpenCL preprocessing (same as openpilot modeld.py:270)
    imgs_cl = {name: self.frames[name].prepare(bufs[name], transforms[name].flatten()) for name in self.input_names}

    # 2. CL → Tensor (same as openpilot modeld.py:280-282)
    for key in imgs_cl:
      frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.input_shapes[key])
      self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8).realize()

    # 3. tinygrad inference (same as openpilot modeld.py:289)
    self.vision_output = self.vision_run(**self.vision_inputs).contiguous().realize().uop.base.buffer.numpy()

    # 4. Slice and parse (same as openpilot modeld.py:248-251)
    parsed = {k: self.vision_output[np.newaxis, v] for k, v in self.output_slices.items()}
    return parsed


def decode_outputs(parsed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Decode raw network outputs (MDN / BCE) from flat slices."""
  p: dict[str, np.ndarray] = {}

  # Lane lines: (1, 528) → (1, 4, 33, 4), MDN: mu=[:,:,:2], std=exp([:,:,2:])
  ll = parsed['lane_lines'].reshape(1, 4, 33, 4)
  p['lane_mu'] = ll[:, :, :, :2]
  p['lane_std'] = np.exp(ll[:, :, :, 2:])

  # Lane line probs: (1, 8) → (1, 4, 2), sigmoid col 1
  ll_prob = parsed['lane_lines_prob'].reshape(1, 4, 2)
  p['lane_prob'] = _sigmoid(ll_prob[:, :, 1])

  # Road edges: (1, 264) → (1, 2, 33, 4)
  re = parsed['road_edges'].reshape(1, 2, 33, 4)
  p['edge_mu'] = re[:, :, :, :2]
  p['edge_std'] = np.exp(re[:, :, :, 2:])

  # Lead: (1, 144) → (1, 3, 6, 8), MDN: mu=[:,:,:4], std=exp([:,:,4:])
  ld = parsed['lead'].reshape(1, 3, 6, 8)
  p['lead_mu'] = ld[:, :, :, :4]
  p['lead_std'] = np.exp(ld[:, :, :, 4:])

  # Lead prob: (1, 3) → sigmoid
  p['lead_prob'] = _sigmoid(parsed['lead_prob'])

  # Pose: (1, 12) → mu=[:6], std=exp([6:])
  pose = parsed['pose'][0]
  p['pose_mu'] = pose[:6]
  p['pose_std'] = np.exp(pose[6:])

  # Road transform: (1, 12)
  rt = parsed['road_transform'][0]
  p['rt_mu'] = rt[:6]
  p['rt_std'] = np.exp(rt[6:])

  return p


def build_model_v2_msg(parsed: dict[str, np.ndarray]) -> messaging.log.Event:
  """Build cereal modelV2 message from parsed flat outputs."""
  p = decode_outputs(parsed)

  msg = messaging.new_message('modelV2', valid=True)
  mv2 = msg.modelV2

  X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)
  LINE_T: list[float] = []

  # Lane lines
  mv2.init('laneLines', 4)
  for i in range(4):
    fill_xyzt(mv2.laneLines[i], LINE_T, X_IDXS, p['lane_mu'][0, i, :, 0], p['lane_mu'][0, i, :, 1])
  mv2.laneLineStds = p['lane_std'][0, :, 0, 0].tolist()
  mv2.laneLineProbs = p['lane_prob'][0].tolist()

  # Road edges
  mv2.init('roadEdges', 2)
  for i in range(2):
    fill_xyzt(mv2.roadEdges[i], LINE_T, X_IDXS, p['edge_mu'][0, i, :, 0], p['edge_mu'][0, i, :, 1])
  mv2.roadEdgeStds = p['edge_std'][0, :, 0, 0].tolist()

  # Leads
  mv2.init('leadsV3', 3)
  for i in range(3):
    fill_xyvat(
      mv2.leadsV3[i],
      ModelConstants.LEAD_T_IDXS,
      p['lead_mu'][0, i, :, 0],
      p['lead_mu'][0, i, :, 1],
      p['lead_mu'][0, i, :, 2],
      p['lead_mu'][0, i, :, 3],
      p['lead_std'][0, i, :, 0],
      p['lead_std'][0, i, :, 1],
      p['lead_std'][0, i, :, 2],
      p['lead_std'][0, i, :, 3],
    )
    mv2.leadsV3[i].prob = float(p['lead_prob'][0, i])
    mv2.leadsV3[i].probTime = ModelConstants.LEAD_T_OFFSETS[i]

  return msg


def build_camera_odometry_msg(parsed: dict[str, np.ndarray], frame_id: int, timestamp_eof: int, live_calib_seen: bool) -> messaging.log.Event:
  """Build cameraOdometry message for calibrationd."""
  msg = messaging.new_message('cameraOdometry', valid=live_calib_seen)
  odom = msg.cameraOdometry
  odom.frameId = frame_id
  odom.timestampEof = timestamp_eof

  pose = parsed['pose'][0]  # (12,): first 6=mu, last 6=log_std
  odom.trans = pose[:3].tolist()
  odom.rot = pose[3:6].tolist()
  odom.transStd = np.exp(pose[6:9]).tolist()
  odom.rotStd = np.exp(pose[9:12]).tolist()

  if 'wide_from_device_euler' in parsed:
    wfde = parsed['wide_from_device_euler'][0]  # (6,): 3 mu + 3 log_sigma
    odom.wideFromDeviceEuler = wfde[:3].tolist()
    odom.wideFromDeviceEulerStd = np.exp(wfde[3:6]).tolist()
  else:
    odom.wideFromDeviceEuler = [0.0, 0.0, 0.0]
    odom.wideFromDeviceEulerStd = [0.0, 0.0, 0.0]

  rt = parsed['road_transform'][0]  # (12,)
  odom.roadTransformTrans = rt[:3].tolist()
  odom.roadTransformTransStd = np.exp(rt[6:9]).tolist()

  return msg


def main():
  """custom_modeld main loop (mirrors selfdrive/modeld/modeld.py:main)."""
  pkl_path = os.environ['CUSTOM_MODEL_PKL']
  metadata_path = os.environ['CUSTOM_MODEL_METADATA']

  print(f"[custom_modeld] Starting with pkl={pkl_path}")
  model = CustomModelState(pkl_path, metadata_path)

  # VisionIPC connection (same as openpilot modeld)
  while True:
    available = VisionIpcClient.available_streams("camerad", block=False)
    if available:
      use_extra = VisionStreamType.VISION_STREAM_WIDE_ROAD in available and VisionStreamType.VISION_STREAM_ROAD in available
      main_wide = VisionStreamType.VISION_STREAM_ROAD not in available
      break
    time.sleep(0.1)

  main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide else VisionStreamType.VISION_STREAM_ROAD
  vipc_main = VisionIpcClient("camerad", main_stream, True, model.cl_context)
  vipc_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, model.cl_context) if use_extra else None

  while not vipc_main.connect(False):
    time.sleep(0.1)
  if vipc_extra is not None:
    while not vipc_extra.connect(False):
      time.sleep(0.1)

  print(f"[custom_modeld] VisionIPC connected: main_wide={main_wide}, use_extra={use_extra}")

  pm = PubMaster(["modelV2", "cameraOdometry"])
  sm = SubMaster(["liveCalibration"])

  dc = DEVICE_CAMERAS[("pc", "unknown")]

  live_calib_seen = False
  transform_main = np.zeros((3, 3), dtype=np.float32)
  transform_extra = np.zeros((3, 3), dtype=np.float32)

  run_count = 0
  last_log_time = time.monotonic()

  while True:
    buf_main = vipc_main.recv()
    if buf_main is None:
      continue

    buf_extra = None
    if vipc_extra is not None:
      buf_extra = vipc_extra.recv()

    # Build bufs/transforms mapping (same as openpilot modeld.py:460)
    bufs = {}
    transforms = {}
    for name in model.input_names:
      if 'big' in name:
        bufs[name] = buf_extra if buf_extra is not None else buf_main
        transforms[name] = transform_extra.flatten()
      else:
        bufs[name] = buf_main
        transforms[name] = transform_main.flatten()

    sm.update(0)

    if sm.updated["liveCalibration"]:
      rpyCalib = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      intrinsics = dc.ecam.intrinsics if main_wide else dc.fcam.intrinsics
      transform_main = get_warp_matrix(rpyCalib, intrinsics, bigmodel_frame=False).astype(np.float32)
      transform_extra = get_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True).astype(np.float32)
      live_calib_seen = True

    t0 = time.monotonic()
    output = model.run(bufs, transforms)
    exec_time = time.monotonic() - t0

    model_msg = build_model_v2_msg(output)
    model_msg.modelV2.modelExecutionTime = exec_time
    model_msg.modelV2.frameId = vipc_main.frame_id
    model_msg.modelV2.timestampEof = vipc_main.timestamp_eof
    pm.send('modelV2', model_msg)

    odom_msg = build_camera_odometry_msg(output, vipc_main.frame_id, vipc_main.timestamp_eof, live_calib_seen)
    pm.send('cameraOdometry', odom_msg)

    run_count += 1
    now = time.monotonic()
    if now - last_log_time >= 10.0:
      print(f"[custom_modeld] runs={run_count} exec={exec_time * 1000:.1f}ms calib={'yes' if live_calib_seen else 'no'}")
      last_log_time = now


if __name__ == "__main__":
  main()
