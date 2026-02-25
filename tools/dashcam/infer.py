"""In-process inference with custom-trained ONNX model.

Runs onnxruntime directly on RGB frames, bypassing VisionIPC / modeld.
Produces cereal modelV2 messages for the dashcam visualizer.

Pipeline per frame:
  RGB (1208, 1928, 3)
    -> warp_image()          -> (256, 512, 3) uint8
    -> rgb_to_yuv420_6ch()   -> (6, 128, 256) uint8
    -> concat(prev, curr)    -> (12, 128, 256)
    -> float32 / 128 - 1     -> (1, 12, 128, 256)
    -> onnxruntime inference  -> 7 output tensors
    -> MDN/BCE decode
    -> cereal modelV2 message
"""

import numpy as np

from cereal import messaging
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import fill_xyzt, fill_xyvat
from openpilot.tools.dashcam.train.dataset import rgb_to_yuv420_6ch, warp_image

OUTPUT_NAMES = ['lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob', 'pose', 'road_transform']


def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


class CustomModelInference:
  """In-process ONNX inference for a custom-trained single-camera driving vision model.

  First call to process_frame() returns None (needs two frames for temporal pair).
  Subsequent calls return a cereal log message with modelV2 filled.
  """

  def __init__(self, onnx_path: str, camera_intrinsics: np.ndarray):
    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    self.session = ort.InferenceSession(onnx_path, providers=providers)
    self.camera_intrinsics = camera_intrinsics
    self.prev_yuv: np.ndarray | None = None

    active = self.session.get_providers()
    print(f"[CustomModel] Loaded: {onnx_path}")
    print(f"[CustomModel] Providers: {active}")

  def process_frame(self, rgb: np.ndarray, rpyCalib: np.ndarray):
    """Process one RGB frame, return cereal message or None (first frame).

    Args:
      rgb: (H, W, 3) uint8 raw camera image
      rpyCalib: (3,) calibration euler angles [roll, pitch, yaw] in radians

    Returns:
      cereal log message with modelV2 filled, or None on first frame
    """
    warped = warp_image(rgb, rpyCalib, self.camera_intrinsics)
    curr_yuv = rgb_to_yuv420_6ch(warped)

    if self.prev_yuv is None:
      self.prev_yuv = curr_yuv
      return None

    combined = np.concatenate([self.prev_yuv, curr_yuv], axis=0)
    inp = (combined.astype(np.float32) / 128.0 - 1.0)[None]
    self.prev_yuv = curr_yuv

    outputs = self.session.run(OUTPUT_NAMES, {'img': inp})
    raw = {name: out[0] for name, out in zip(OUTPUT_NAMES, outputs, strict=True)}
    parsed = self._decode_outputs(raw)
    return self._build_model_msg(parsed)

  @staticmethod
  def _decode_outputs(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Decode raw network outputs (MDN / BCE)."""
    p: dict[str, np.ndarray] = {}

    # Lane lines: (528,) -> (4, 33, 4), MDN: mu=[:,:,:2], std=exp([:,:,2:])
    ll = raw['lane_lines'].reshape(4, 33, 4)
    p['lane_mu'] = ll[:, :, :2]
    p['lane_std'] = np.exp(ll[:, :, 2:])

    # Lane line probs: (8,) -> (4, 2), sigmoid col 1
    ll_prob = raw['lane_lines_prob'].reshape(4, 2)
    p['lane_prob'] = _sigmoid(ll_prob[:, 1])

    # Road edges: (264,) -> (2, 33, 4), MDN same
    re = raw['road_edges'].reshape(2, 33, 4)
    p['edge_mu'] = re[:, :, :2]
    p['edge_std'] = np.exp(re[:, :, 2:])

    # Lead: (144,) -> (3, 6, 8), MDN: mu=[:,:,:4], std=exp([:,:,4:])
    ld = raw['lead'].reshape(3, 6, 8)
    p['lead_mu'] = ld[:, :, :4]
    p['lead_std'] = np.exp(ld[:, :, 4:])

    # Lead prob: (3,) -> sigmoid
    p['lead_prob'] = _sigmoid(raw['lead_prob'])

    # Pose: (12,) -> MDN: mu=[:6], std=exp([6:])
    pose = raw['pose']
    p['pose_mu'] = pose[:6]
    p['pose_std'] = np.exp(pose[6:])

    # Road transform: (12,) -> MDN same
    rt = raw['road_transform']
    p['rt_mu'] = rt[:6]
    p['rt_std'] = np.exp(rt[6:])

    return p

  @staticmethod
  def _build_model_msg(p: dict[str, np.ndarray]):
    """Build cereal modelV2 message from decoded outputs."""
    msg = messaging.new_message('modelV2', valid=True)
    mv2 = msg.modelV2

    X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)
    LINE_T: list[float] = []

    # Lane lines: x=X_IDXS (forward distance), y=mu[:,0] (lateral), z=mu[:,1] (height)
    mv2.init('laneLines', 4)
    for i in range(4):
      fill_xyzt(mv2.laneLines[i], LINE_T, X_IDXS, p['lane_mu'][i, :, 0], p['lane_mu'][i, :, 1])
    mv2.laneLineStds = p['lane_std'][:, 0, 0].tolist()
    mv2.laneLineProbs = p['lane_prob'].tolist()

    # Road edges
    mv2.init('roadEdges', 2)
    for i in range(2):
      fill_xyzt(mv2.roadEdges[i], LINE_T, X_IDXS, p['edge_mu'][i, :, 0], p['edge_mu'][i, :, 1])
    mv2.roadEdgeStds = p['edge_std'][:, 0, 0].tolist()

    # Leads: x=distance, y=offset, v=speed, a=acceleration
    mv2.init('leadsV3', 3)
    for i in range(3):
      fill_xyvat(mv2.leadsV3[i], ModelConstants.LEAD_T_IDXS,
                 p['lead_mu'][i, :, 0], p['lead_mu'][i, :, 1],
                 p['lead_mu'][i, :, 2], p['lead_mu'][i, :, 3],
                 p['lead_std'][i, :, 0], p['lead_std'][i, :, 1],
                 p['lead_std'][i, :, 2], p['lead_std'][i, :, 3])
      mv2.leadsV3[i].prob = float(p['lead_prob'][i])
      mv2.leadsV3[i].probTime = ModelConstants.LEAD_T_OFFSETS[i]

    return msg
