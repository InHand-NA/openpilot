"""Dual-camera training data recorder: saves per-frame road_rgb + wide_rgb + labels as .npz.

Output format per frame:
  road_rgb: [H, W, 3] uint8 — narrow road camera
  wide_rgb: [H, W, 3] uint8 — wide road camera
  label_source: str ('modeld' | 'carla_gt')
  lane_lines: [4, 33, 3] float32
  lane_lines_prob: [4] float32
  road_edges: [2, 33, 3] float32
  road_edges_prob: [2] float32
  lead: [3, 6, 4] float32
  lead_prob: [3] float32
  pose: [6] float32
  road_transform: [6] float32
  rpyCalib: [3] float32 (per-frame calibration angles)
  camera_height: float
  v_ego: float (m/s)
  town: str
  world_pose: [6] float32 (optional, Carla world coords)
"""

import json
import os
import time

import numpy as np


class DualCameraDataRecorder:
  """Record dual-camera training data frames to disk as .npz files."""

  def __init__(self, output_dir: str, town: str, camera_height: float, label_source: str = 'modeld', skip_frames: int = 1, metadata: dict | None = None):
    """
    Args:
      output_dir: directory to save .npz files
      town: Carla town name (stored as metadata)
      camera_height: camera height in meters
      label_source: 'modeld' or 'carla_gt'
      skip_frames: save every N-th frame (1 = save all)
      metadata: dict of clip-level metadata to save as clip_info.json
    """
    self.output_dir = output_dir
    self.town = town
    self.camera_height = camera_height
    self.label_source = label_source
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
      print("[DualRecorder] Saved clip_info.json")
    print(f"[DualRecorder] Saving to {output_dir} (skip={skip_frames}, label_source={label_source})")

  def record(
    self, road_rgb: np.ndarray, wide_rgb: np.ndarray, labels: dict[str, np.ndarray], rpyCalib: np.ndarray, v_ego: float, world_pose: np.ndarray | None = None
  ) -> bool:
    """Record one frame of dual-camera training data.

    Args:
      road_rgb: [H, W, 3] uint8 narrow road camera image
      wide_rgb: [H, W, 3] uint8 wide road camera image
      labels: dict from ModeldLabelExtractor.extract() or GT extractor
      rpyCalib: [3] per-frame calibration angles [roll, -pitch, -yaw]
      v_ego: vehicle speed in m/s
      world_pose: [6] optional Carla world coords [x, y, z, roll, pitch, yaw]

    Returns:
      True if frame was saved, False if skipped
    """
    self.frame_idx += 1

    if self.frame_idx % self.skip_frames != 0:
      self.skipped_count += 1
      return False

    filename = f"{self.frame_idx:06d}.npz"
    filepath = os.path.join(self.output_dir, filename)

    save_dict = dict(
      road_rgb=road_rgb,
      wide_rgb=wide_rgb,
      label_source=np.array(self.label_source),
      lane_lines=labels['lane_lines'],
      lane_lines_prob=labels['lane_lines_prob'],
      road_edges=labels['road_edges'],
      road_edges_prob=labels['road_edges_prob'],
      lead=labels['lead'],
      lead_prob=labels['lead_prob'],
      pose=labels['pose'],
      road_transform=labels['road_transform'],
      rpyCalib=np.asarray(rpyCalib, dtype=np.float32),
      camera_height=np.float32(self.camera_height),
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
      print(f"[DualRecorder] Saved {self.saved_count} frames ({fps:.1f} frames/s)")

    return True

  def close(self):
    """Print recording statistics."""
    elapsed = time.monotonic() - self.start_time
    fps = self.saved_count / elapsed if elapsed > 0 else 0
    print("\n=== DualCameraDataRecorder Summary ===")
    print(f"  Output dir: {self.output_dir}")
    print(f"  Label source: {self.label_source}")
    print(f"  Total frames seen: {self.frame_idx}")
    print(f"  Frames saved: {self.saved_count}")
    print(f"  Frames skipped: {self.skipped_count}")
    print(f"  Elapsed: {elapsed:.1f}s ({fps:.1f} frames/s)")
