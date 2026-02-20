"""Ground truth pose (ego motion) and road transform extraction from Carla.

Computes ego velocity and angular velocity from consecutive Carla frames,
matching openpilot's cameraOdometry message semantics:
  pose[0:3] = translational velocity (m/s) in calibrated frame [forward, right, down]
  pose[3:6] = angular velocity (rad/s) in calibrated frame [roll, pitch, yaw]

road_transform[2] = camera height above road surface (meters).
"""

from math import cos, radians, sin

import numpy as np


class PoseGroundTruth:
  """Extract pose and road_transform ground truth from consecutive Carla frames."""

  def __init__(self, camera_offset_x=0.8, camera_height=1.13):
    self.camera_offset_x = camera_offset_x
    self.camera_height = camera_height
    self.prev_transform = None

  def update(self, vehicle_transform, dt=0.05):
    """Compute inter-frame pose change and road_transform.

    Args:
      vehicle_transform: carla.Transform of the ego vehicle this frame.
      dt: time delta between frames in seconds.

    Returns:
      pose: [6] float32 -- translational velocity (m/s) + angular velocity (rad/s)
      road_transform: [6] float32 -- [0,0,camera_height, 0,0,0]
    """
    road_transform = np.zeros(6, dtype=np.float32)
    road_transform[2] = self.camera_height

    if self.prev_transform is None:
      self.prev_transform = vehicle_transform
      pose = np.zeros(6, dtype=np.float32)
      return pose, road_transform

    # Translation: world-frame displacement, yaw-only rotation into calibrated frame
    dx = vehicle_transform.location.x - self.prev_transform.location.x
    dy = vehicle_transform.location.y - self.prev_transform.location.y
    dz = vehicle_transform.location.z - self.prev_transform.location.z

    # Use previous frame's yaw only (gravity-aligned calibrated frame)
    yaw = radians(self.prev_transform.rotation.yaw)
    cy, sy = cos(yaw), sin(yaw)

    # Rz(yaw)^T @ [dx, dy, dz]
    cal_x = cy * dx + sy * dy
    cal_y = -sy * dx + cy * dy
    cal_z = dz  # world z-up preserved

    # Convert to openpilot convention (z-down) and displacement → velocity
    trans = np.array([cal_x, cal_y, -cal_z], dtype=np.float32) / dt

    # Rotation increments (in radians)
    d_roll = radians(vehicle_transform.rotation.roll - self.prev_transform.rotation.roll)
    d_pitch = radians(vehicle_transform.rotation.pitch - self.prev_transform.rotation.pitch)
    d_yaw = radians(vehicle_transform.rotation.yaw - self.prev_transform.rotation.yaw)

    # Handle yaw wraparound (-180 to 180)
    if d_yaw > np.pi:
      d_yaw -= 2 * np.pi
    elif d_yaw < -np.pi:
      d_yaw += 2 * np.pi

    # Convert angle increments → angular velocity (rad/s)
    rot = np.array([d_roll, -d_pitch, -d_yaw], dtype=np.float32) / dt

    pose = np.concatenate([trans, rot])

    self.prev_transform = vehicle_transform
    return pose, road_transform
