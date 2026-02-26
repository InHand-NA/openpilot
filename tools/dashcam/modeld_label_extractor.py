"""Extract training labels from openpilot modelV2 + cameraOdometry messages.

Converts cereal modelV2 and cameraOdometry messages into NPZ-compatible label
dicts, aligned with the format expected by extract_targets() in dataset.py.

Field mapping:
  modelV2.laneLines[i].x/y/z     → lane_lines[i] (33, 3)
  modelV2.laneLineProbs[i]        → lane_lines_prob[i]
  modelV2.roadEdges[i].x/y/z     → road_edges[i] (33, 3)
  modelV2.roadEdgeStds            → road_edges_prob (all 1.0, modeld has no prob)
  modelV2.leadsV3[i].x/y/v/a     → lead[i] (6, 4)
  modelV2.leadsV3[i].prob         → lead_prob[i]
  cameraOdometry.trans+rot        → pose (6,)
  cameraOdometry.roadTransformTrans → road_transform (6,) = [trans[:3], 0, 0, 0]
"""

import numpy as np


class ModeldLabelExtractor:
  """Extract training labels from modelV2 + cameraOdometry cereal messages."""

  def extract(self, model_msg, cam_odom_msg=None) -> dict[str, np.ndarray] | None:
    """Extract labels compatible with DataRecorder / DualCameraDataRecorder NPZ format.

    Args:
      model_msg: cereal modelV2 message
      cam_odom_msg: cereal cameraOdometry message (optional, for pose/road_transform)

    Returns:
      dict of label arrays, or None if model_msg is invalid
    """
    if model_msg is None:
      return None

    mv2 = model_msg.modelV2 if hasattr(model_msg, 'modelV2') else model_msg
    labels: dict[str, np.ndarray] = {}

    # Lane lines: (4, 33, 3) — x, y, z
    lane_lines = []
    for i in range(4):
      ll = mv2.laneLines[i]
      x = np.array(ll.x, dtype=np.float32)
      y = np.array(ll.y, dtype=np.float32)
      z = np.array(ll.z, dtype=np.float32)
      lane_lines.append(np.column_stack([x, y, z]))
    labels['lane_lines'] = np.stack(lane_lines, axis=0).astype(np.float32)  # (4, 33, 3)

    # Lane line probs: (4,)
    labels['lane_lines_prob'] = np.array(mv2.laneLineProbs, dtype=np.float32)

    # Road edges: (2, 33, 3) — x, y, z
    road_edges = []
    for i in range(2):
      re = mv2.roadEdges[i]
      x = np.array(re.x, dtype=np.float32)
      y = np.array(re.y, dtype=np.float32)
      z = np.array(re.z, dtype=np.float32)
      road_edges.append(np.column_stack([x, y, z]))
    labels['road_edges'] = np.stack(road_edges, axis=0).astype(np.float32)  # (2, 33, 3)

    # Road edges prob: modeld doesn't output per-edge probabilities, use 1.0
    labels['road_edges_prob'] = np.ones(2, dtype=np.float32)

    # Lead vehicles: (3, 6, 4) — x, y, v, a at 6 time steps
    lead = np.zeros((3, 6, 4), dtype=np.float32)
    lead_prob = np.zeros(3, dtype=np.float32)
    for i in range(min(3, len(mv2.leadsV3))):
      ld = mv2.leadsV3[i]
      x = np.array(ld.x, dtype=np.float32)
      y = np.array(ld.y, dtype=np.float32)
      v = np.array(ld.v, dtype=np.float32)
      a = np.array(ld.a, dtype=np.float32)
      lead[i] = np.column_stack([x, y, v, a])
      lead_prob[i] = float(ld.prob)
    labels['lead'] = lead
    labels['lead_prob'] = lead_prob

    # Pose and road_transform from cameraOdometry
    if cam_odom_msg is not None:
      odom = cam_odom_msg.cameraOdometry if hasattr(cam_odom_msg, 'cameraOdometry') else cam_odom_msg
      trans = np.array(odom.trans, dtype=np.float32)
      rot = np.array(odom.rot, dtype=np.float32)
      labels['pose'] = np.concatenate([trans, rot])  # (6,)

      rt_trans = np.array(odom.roadTransformTrans, dtype=np.float32)
      # road_transform: first 3 dims from roadTransformTrans, last 3 zero-padded
      labels['road_transform'] = np.concatenate([rt_trans[:3], np.zeros(3, dtype=np.float32)])  # (6,)
      # wide_from_device_euler: euler angles of wide camera relative to device frame
      if hasattr(odom, 'wideFromDeviceEuler'):
        labels['wide_from_device_euler'] = np.array(odom.wideFromDeviceEuler, dtype=np.float32)  # (3,)
      else:
        labels['wide_from_device_euler'] = np.zeros(3, dtype=np.float32)
    else:
      labels['pose'] = np.zeros(6, dtype=np.float32)
      labels['road_transform'] = np.zeros(6, dtype=np.float32)
      labels['wide_from_device_euler'] = np.zeros(3, dtype=np.float32)

    return labels
