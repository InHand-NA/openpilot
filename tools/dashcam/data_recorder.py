"""Training data recorder: saves per-frame RGB + GT labels as compressed .npz files.

Output format per frame:
  frame_rgb: [H, W, 3] uint8
  lane_lines: [4, 33, 3] float32
  lane_lines_prob: [4] float32
  road_edges: [2, 33, 3] float32
  lead: [3, 6, 4] float32
  lead_prob: [3] float32
  pose: [6] float32
  road_transform: [6] float32
  camera_height: float
  camera_pitch: float (rad)
  camera_yaw: float (rad)
  v_ego: float (m/s)
  town: str
"""

import os
import time

import numpy as np


class DataRecorder:
  """Record training data frames to disk as .npz files."""

  def __init__(self, output_dir, town, camera_height, camera_pitch=0.0, camera_yaw=0.0, skip_frames=1):
    """
    Args:
      output_dir: directory to save .npz files.
      town: Carla town name (stored as metadata).
      camera_height: camera height in meters.
      camera_pitch: camera pitch in radians.
      camera_yaw: camera yaw in radians.
      skip_frames: save every N-th frame (1 = save all).
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
    print(f"[DataRecorder] Saving to {output_dir} (skip={skip_frames})")

  def record(self, rgb, lane_gt, lead_gt, pose_gt, road_transform_gt):
    """Record one frame of training data.

    Args:
      rgb: [H, W, 3] uint8 RGB image.
      lane_gt: tuple of (lane_lines, lane_probs) from LaneGroundTruth.
                lane_lines: list of 4 arrays [33, 3], lane_probs: list of 4 floats.
                Can be (None, None) if GT unavailable (e.g., junction).
      lead_gt: tuple of (lead_data, lead_probs) from LeadGroundTruth.
                lead_data: [3, 6, 4], lead_probs: [3].
      pose_gt: [6] float32 pose increments.
      road_transform_gt: [6] float32 road transform.

    Returns:
      True if frame was saved, False if skipped.
    """
    self.frame_idx += 1

    if self.frame_idx % self.skip_frames != 0:
      self.skipped_count += 1
      return False

    # Unpack lane GT
    lane_lines_list, lane_probs_list = lane_gt
    if lane_lines_list is None:
      # Junction or invalid -- skip this frame
      self.skipped_count += 1
      return False

    lane_lines = np.stack(lane_lines_list, axis=0).astype(np.float32)  # [4, 33, 3]
    lane_probs = np.array(lane_probs_list, dtype=np.float32)  # [4]

    # Unpack lead GT
    lead_data, lead_probs = lead_gt

    # Road edges placeholder (Phase 1: zeros)
    road_edges = np.zeros((2, 33, 3), dtype=np.float32)

    # Save compressed npz
    filename = f"{self.saved_count:06d}.npz"
    filepath = os.path.join(self.output_dir, filename)

    np.savez_compressed(filepath,
      frame_rgb=rgb,
      lane_lines=lane_lines,
      lane_lines_prob=lane_probs,
      road_edges=road_edges,
      lead=lead_data,
      lead_prob=lead_probs,
      pose=pose_gt,
      road_transform=road_transform_gt,
      camera_height=np.float32(self.camera_height),
      camera_pitch=np.float32(self.camera_pitch),
      camera_yaw=np.float32(self.camera_yaw),
      v_ego=np.float32(0.0),  # updated below if provided via meta
      town=np.array(self.town),
    )

    self.saved_count += 1

    if self.saved_count % 100 == 0:
      elapsed = time.monotonic() - self.start_time
      fps = self.saved_count / elapsed if elapsed > 0 else 0
      print(f"[DataRecorder] Saved {self.saved_count} frames ({fps:.1f} frames/s)")

    return True

  def record_with_vego(self, rgb, lane_gt, lead_gt, pose_gt, road_transform_gt, v_ego):
    """Record one frame with explicit v_ego metadata.

    Same as record() but includes v_ego in the saved data.
    """
    self.frame_idx += 1

    if self.frame_idx % self.skip_frames != 0:
      self.skipped_count += 1
      return False

    lane_lines_list, lane_probs_list = lane_gt
    if lane_lines_list is None:
      self.skipped_count += 1
      return False

    lane_lines = np.stack(lane_lines_list, axis=0).astype(np.float32)
    lane_probs = np.array(lane_probs_list, dtype=np.float32)

    lead_data, lead_probs = lead_gt

    road_edges = np.zeros((2, 33, 3), dtype=np.float32)

    filename = f"{self.saved_count:06d}.npz"
    filepath = os.path.join(self.output_dir, filename)

    np.savez_compressed(filepath,
      frame_rgb=rgb,
      lane_lines=lane_lines,
      lane_lines_prob=lane_probs,
      road_edges=road_edges,
      lead=lead_data,
      lead_prob=lead_probs,
      pose=pose_gt,
      road_transform=road_transform_gt,
      camera_height=np.float32(self.camera_height),
      camera_pitch=np.float32(self.camera_pitch),
      camera_yaw=np.float32(self.camera_yaw),
      v_ego=np.float32(v_ego),
      town=np.array(self.town),
    )

    self.saved_count += 1

    if self.saved_count % 100 == 0:
      elapsed = time.monotonic() - self.start_time
      fps = self.saved_count / elapsed if elapsed > 0 else 0
      print(f"[DataRecorder] Saved {self.saved_count} frames ({fps:.1f} frames/s)")

    return True

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
