"""Ground truth pose (ego motion) and road transform extraction from Carla.

Computes frame-to-frame ego motion increments and road transform
(camera height as z-component) from Carla vehicle transforms.
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
      pose: [6] float32 -- translation(x,y,z) + rotation(roll,pitch,yaw) increments
      road_transform: [6] float32 -- [0,0,camera_height, 0,0,0]
    """
    road_transform = np.zeros(6, dtype=np.float32)
    road_transform[2] = self.camera_height

    if self.prev_transform is None:
      self.prev_transform = vehicle_transform
      pose = np.zeros(6, dtype=np.float32)
      return pose, road_transform

    # Translation: world-frame displacement, then rotate into previous frame's body coords
    dx = vehicle_transform.location.x - self.prev_transform.location.x
    dy = vehicle_transform.location.y - self.prev_transform.location.y
    dz = vehicle_transform.location.z - self.prev_transform.location.z

    # Use previous frame's rotation to express displacement in ego body frame
    yaw = radians(self.prev_transform.rotation.yaw)
    pitch = radians(self.prev_transform.rotation.pitch)
    roll = radians(self.prev_transform.rotation.roll)

    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)
    cr, sr = cos(roll), sin(roll)

    # R^T @ [dx, dy, dz] (body frame: x=forward, y=left in UE4, z=up in UE4)
    lx = (cy * cp) * dx + (sy * cp) * dy + (-sp) * dz
    ly = (cy * sp * sr - sy * cr) * dx + (sy * sp * sr + cy * cr) * dy + (cp * sr) * dz
    lz = (cy * sp * cr + sy * sr) * dx + (sy * sp * cr - cy * sr) * dy + (cp * cr) * dz

    # Convert to openpilot convention (z-down)
    trans = np.array([lx, ly, -lz], dtype=np.float32)

    # Rotation increments (in radians)
    d_roll = radians(vehicle_transform.rotation.roll - self.prev_transform.rotation.roll)
    d_pitch = radians(vehicle_transform.rotation.pitch - self.prev_transform.rotation.pitch)
    d_yaw = radians(vehicle_transform.rotation.yaw - self.prev_transform.rotation.yaw)

    # Handle yaw wraparound (-180 to 180)
    if d_yaw > np.pi:
      d_yaw -= 2 * np.pi
    elif d_yaw < -np.pi:
      d_yaw += 2 * np.pi

    rot = np.array([d_roll, -d_pitch, -d_yaw], dtype=np.float32)

    pose = np.concatenate([trans, rot])

    self.prev_transform = vehicle_transform
    return pose, road_transform
