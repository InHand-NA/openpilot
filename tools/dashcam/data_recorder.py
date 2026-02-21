"""Training data recorder: saves per-frame RGB + GT labels as compressed .npz files.

Output format per frame:
  frame_rgb: [H, W, 3] uint8
  lane_lines: [4, 33, 3] float32
  lane_lines_prob: [4] float32
  road_edges: [2, 33, 3] float32
  road_edges_prob: [2] float32
  lead: [3, 6, 4] float32
  lead_prob: [3] float32
  pose: [6] float32
  road_transform: [6] float32
  camera_height: float
  camera_pitch: float (rad)
  camera_yaw: float (rad)
  rpyCalib: [3] float32 (per-frame [roll, -pitch, -yaw] including vehicle tilt)
  world_pose: [6] float32 (Carla world coords: [x, y, z, roll_deg, pitch_deg, yaw_deg])
  v_ego: float (m/s)
  town: str
"""

import json
import os
import time

import numpy as np


class DataRecorder:
  """Record training data frames to disk as .npz files."""

  def __init__(self, output_dir, town, camera_height, camera_pitch=0.0, camera_yaw=0.0,
               skip_frames=1, metadata=None):
    """
    Args:
      output_dir: directory to save .npz files.
      town: Carla town name (stored as metadata).
      camera_height: camera height in meters.
      camera_pitch: camera pitch in radians.
      camera_yaw: camera yaw in radians.
      skip_frames: save every N-th frame (1 = save all).
      metadata: dict of clip-level metadata to save as clip_info.json.
    """
    self.output_dir = output_dir
    self.town = town
    self.camera_height = camera_height
    self.camera_pitch = camera_pitch
    self.camera_yaw = camera_yaw
    self.skip_frames = max(1, skip_frames)

    self.frame_idx = 0
    self.saved_count = 0
    self.skipped_count = 0
    self.start_time = time.monotonic()

    os.makedirs(output_dir, exist_ok=True)
    if metadata is not None:
      info_path = os.path.join(output_dir, "clip_info.json")
      with open(info_path, "w") as f:
        json.dump(metadata, f, indent=2)
      print("[DataRecorder] Saved clip_info.json")
    print(f"[DataRecorder] Saving to {output_dir} (skip={skip_frames})")

  def record(self, rgb, lane_gt, lead_gt, pose_gt, road_transform_gt, road_edges_gt=None):
    """Record one frame of training data.

    Args:
      rgb: [H, W, 3] uint8 RGB image.
      lane_gt: tuple of (lane_lines, lane_probs) from LaneGroundTruth.
                lane_lines: list of 4 arrays [33, 3], lane_probs: list of 4 floats.
                Can be (None, None) if GT unavailable (saved as zeros with prob=0).
      lead_gt: tuple of (lead_data, lead_probs) from LeadGroundTruth.
                lead_data: [3, 6, 4], lead_probs: [3].
      pose_gt: [6] float32 pose increments.
      road_transform_gt: [6] float32 road transform.
      road_edges_gt: tuple of (edges_list, edge_probs) from LaneGroundTruth.get_road_edges().
                      edges_list: list of 2 arrays [33, 3], edge_probs: list of 2 floats.
                      Can be None or (None, None) if GT unavailable.

    Returns:
      True if frame was saved, False if skipped (skip_frames only).
    """
    self.frame_idx += 1

    if self.frame_idx % self.skip_frames != 0:
      self.skipped_count += 1
      return False

    # Unpack lane GT (None → zeros with prob=0, preserving temporal continuity)
    lane_lines, lane_probs = self._unpack_lane_gt(lane_gt)

    # Unpack lead GT
    lead_data, lead_probs = lead_gt

    # Road edges GT
    road_edges, road_edge_probs = self._unpack_road_edges(road_edges_gt)

    # Save compressed npz
    filename = f"{self.frame_idx:06d}.npz"
    filepath = os.path.join(self.output_dir, filename)

    np.savez_compressed(filepath,
      frame_rgb=rgb,
      lane_lines=lane_lines,
      lane_lines_prob=lane_probs,
      road_edges=road_edges,
      road_edges_prob=road_edge_probs,
      lead=lead_data,
      lead_prob=lead_probs,
      pose=pose_gt,
      road_transform=road_transform_gt,
      camera_height=np.float32(self.camera_height),
      camera_pitch=np.float32(self.camera_pitch),
      camera_yaw=np.float32(self.camera_yaw),
      v_ego=np.float32(0.0),
      town=np.array(self.town),
    )

    self.saved_count += 1

    if self.saved_count % 100 == 0:
      elapsed = time.monotonic() - self.start_time
      fps = self.saved_count / elapsed if elapsed > 0 else 0
      print(f"[DataRecorder] Saved {self.saved_count} frames ({fps:.1f} frames/s)")

    return True

  def record_with_vego(self, rgb, lane_gt, lead_gt, pose_gt, road_transform_gt, v_ego,
                       road_edges_gt=None, rpyCalib=None, world_pose=None):
    """Record one frame with explicit v_ego, per-frame rpyCalib, and optional world_pose.

    Same as record() but includes v_ego, rpyCalib and world_pose in the saved data.
    world_pose: [6] float32 Carla world coords [x, y, z, roll_deg, pitch_deg, yaw_deg].
    """
    self.frame_idx += 1

    if self.frame_idx % self.skip_frames != 0:
      self.skipped_count += 1
      return False

    lane_lines, lane_probs = self._unpack_lane_gt(lane_gt)

    lead_data, lead_probs = lead_gt

    road_edges, road_edge_probs = self._unpack_road_edges(road_edges_gt)

    filename = f"{self.frame_idx:06d}.npz"
    filepath = os.path.join(self.output_dir, filename)

    save_dict = dict(
      frame_rgb=rgb,
      lane_lines=lane_lines,
      lane_lines_prob=lane_probs,
      road_edges=road_edges,
      road_edges_prob=road_edge_probs,
      lead=lead_data,
      lead_prob=lead_probs,
      pose=pose_gt,
      road_transform=road_transform_gt,
      camera_height=np.float32(self.camera_height),
      camera_pitch=np.float32(self.camera_pitch),
      camera_yaw=np.float32(self.camera_yaw),
      rpyCalib=np.asarray(rpyCalib, dtype=np.float32),
      v_ego=np.float32(v_ego),
      town=np.array(self.town),
    )
    if world_pose is not None:
      save_dict['world_pose'] = np.asarray(world_pose, dtype=np.float32)
    np.savez_compressed(filepath, **save_dict)

    self.saved_count += 1

    if self.saved_count % 100 == 0:
      elapsed = time.monotonic() - self.start_time
      fps = self.saved_count / elapsed if elapsed > 0 else 0
      print(f"[DataRecorder] Saved {self.saved_count} frames ({fps:.1f} frames/s)")

    return True

  @staticmethod
  def _unpack_lane_gt(lane_gt):
    """Unpack lane GT into arrays, falling back to zeros with prob=0 if unavailable."""
    lane_lines_list, lane_probs_list = lane_gt
    if lane_lines_list is not None:
      lane_lines = np.stack(lane_lines_list, axis=0).astype(np.float32)  # [4, 33, 3]
      lane_probs = np.array(lane_probs_list, dtype=np.float32)           # [4]
    else:
      lane_lines = np.zeros((4, 33, 3), dtype=np.float32)
      lane_probs = np.zeros(4, dtype=np.float32)
    return lane_lines, lane_probs

  @staticmethod
  def _unpack_road_edges(road_edges_gt):
    """Unpack road edges GT into arrays, falling back to zeros if unavailable."""
    if road_edges_gt is not None and road_edges_gt[0] is not None:
      road_edges = np.stack(road_edges_gt[0], axis=0).astype(np.float32)  # [2, 33, 3]
      road_edge_probs = np.array(road_edges_gt[1], dtype=np.float32)      # [2]
    else:
      road_edges = np.zeros((2, 33, 3), dtype=np.float32)
      road_edge_probs = np.zeros(2, dtype=np.float32)
    return road_edges, road_edge_probs

  def close(self):
    """Print recording statistics."""
    elapsed = time.monotonic() - self.start_time
    fps = self.saved_count / elapsed if elapsed > 0 else 0
    print("\n=== DataRecorder Summary ===")
    print(f"  Output dir: {self.output_dir}")
    print(f"  Total frames seen: {self.frame_idx}")
    print(f"  Frames saved: {self.saved_count}")
    print(f"  Frames skipped: {self.skipped_count}")
    print(f"  Elapsed: {elapsed:.1f}s ({fps:.1f} frames/s)")
