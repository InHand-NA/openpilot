"""Ground truth lane line extraction from Carla map API.

Extracts lane boundaries from Carla waypoints, transforms to openpilot calibrated
frame (x=forward, y=right, z=down), and interpolates at ModelConstants.X_IDXS distances.
"""

from math import cos, radians, sin

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants


class LaneGroundTruth:
  """Extract ego lane boundary ground truth from Carla map.

  Only extracts the two ego lane boundaries (indices 1 and 2):
    [1] near-left: ego lane's left boundary
    [2] near-right: ego lane's right boundary
  """

  def __init__(self, carla_map, camera_offset_x=0.8, camera_height=1.13):
    self.map = carla_map
    self.camera_offset_x = camera_offset_x
    self.camera_height = camera_height
    self.x_idxs = np.array(ModelConstants.X_IDXS)

  def get_lane_lines(self, vehicle_transform):
    """Extract GT lane lines in calibrated frame at X_IDXS distances.

    Args:
      vehicle_transform: carla.Transform of the ego vehicle.

    Returns:
      gt_lines: list of 4 arrays, each 33x3 (x, y, z) in calibrated frame.
      gt_probs: list of 4 floats (1.0 if lane exists, 0.0 otherwise).
    """
    import carla

    loc = vehicle_transform.location
    ego_wp = self.map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
      return None, None

    # Skip frame if any junction exists within 0-100m ahead
    if self._has_junction_ahead(ego_wp, max_dist=100.0, step=2.0):
      return None, None

    # Sample ego lane waypoints only (no adjacent lanes needed)
    ego_wps = self._sample_waypoints(ego_wp, max_dist=100.0, step=2.0)

    # Compute ego lane boundaries in calibrated frame
    ego_left_boundary = self._compute_boundary(ego_wps, vehicle_transform, side='left')
    ego_right_boundary = self._compute_boundary(ego_wps, vehicle_transform, side='right')

    gt_lines = [np.zeros((33, 3), dtype=np.float32) for _ in range(4)]
    gt_probs = [0.0, 0.0, 0.0, 0.0]

    # [1] near-left: ego lane left boundary
    interp = self._interpolate_at_x_idxs(ego_left_boundary)
    if interp is not None:
      gt_lines[1] = interp
      gt_probs[1] = 1.0

    # [2] near-right: ego lane right boundary
    interp = self._interpolate_at_x_idxs(ego_right_boundary)
    if interp is not None:
      gt_lines[2] = interp
      gt_probs[2] = 1.0

    return gt_lines, gt_probs

  def _has_junction_ahead(self, start_wp, max_dist=100.0, step=2.0):
    """Check if there's a junction within max_dist meters ahead."""
    wp = start_wp
    total_dist = 0.0
    while total_dist < max_dist:
      if wp.is_junction:
        return True
      next_wps = wp.next(step)
      if not next_wps:
        break
      wp = next_wps[0]
      total_dist += step
    return False

  def _sample_waypoints(self, start_wp, max_dist=200.0, step=2.0, back_dist=10.0):
    """Sample waypoints along lane, collecting (transform, lane_width).

    Samples backwards first (up to back_dist) to ensure coverage at the
    camera position (X=0 in calibrated frame), then forwards up to max_dist.
    """
    # Sample backwards to cover area behind/at the camera
    back_wps = []
    wp = start_wp
    dist = 0.0
    while dist < back_dist:
      prev = wp.previous(step)
      if not prev:
        break
      wp = prev[0]
      dist += step
      back_wps.append((wp.transform, wp.lane_width))

    # Assemble: backwards (reversed to maintain spatial order) + forward
    wps = list(reversed(back_wps))

    # Sample forwards
    wp = start_wp
    total_dist = 0.0
    while total_dist < max_dist:
      wps.append((wp.transform, wp.lane_width))
      next_wps = wp.next(step)
      if not next_wps:
        break
      wp = next_wps[0]
      total_dist += step
    return wps

  def _compute_boundary(self, waypoints, vehicle_transform, side='left'):
    """Compute lane boundary points in calibrated frame.

    Args:
      waypoints: list of (carla.Transform, lane_width).
      vehicle_transform: ego vehicle carla.Transform.
      side: 'left' or 'right' boundary of the lane.

    Returns:
      Nx3 array of (x, y, z) in calibrated frame.
    """
    points = []
    for wp_transform, lane_width in waypoints:
      right_vec = wp_transform.get_right_vector()
      half_w = lane_width / 2.0
      center = wp_transform.location

      if side == 'left':
        bx = center.x - right_vec.x * half_w
        by = center.y - right_vec.y * half_w
        bz = center.z - right_vec.z * half_w
      else:
        bx = center.x + right_vec.x * half_w
        by = center.y + right_vec.y * half_w
        bz = center.z + right_vec.z * half_w

      cal = self._world_to_calibrated(bx, by, bz, vehicle_transform)
      points.append(cal)

    return np.array(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)

  def _world_to_calibrated(self, wx, wy, wz, vehicle_transform):
    """Convert world point to calibrated frame (x=forward, y=right, z=down).

    Steps:
      1. Compute relative position in world frame.
      2. Apply inverse vehicle rotation (R^T) to get body frame.
      3. Subtract camera offset in body frame.
      4. Flip z axis (Carla z-up -> openpilot z-down).
    """
    dx = wx - vehicle_transform.location.x
    dy = wy - vehicle_transform.location.y
    dz = wz - vehicle_transform.location.z

    # Vehicle rotation (UE4 convention: yaw=Z, pitch=Y, roll=X)
    yaw = radians(vehicle_transform.rotation.yaw)
    pitch = radians(vehicle_transform.rotation.pitch)
    roll = radians(vehicle_transform.rotation.roll)

    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)
    cr, sr = cos(roll), sin(roll)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll), local = R^T @ [dx, dy, dz]
    lx = (cy * cp) * dx + (sy * cp) * dy + (-sp) * dz
    ly = (cy * sp * sr - sy * cr) * dx + (sy * sp * sr + cy * cr) * dy + (cp * sr) * dz
    lz = (cy * sp * cr + sy * sr) * dx + (sy * sp * cr - cy * sr) * dy + (cp * cr) * dz

    # Subtract camera offset in body frame
    lx -= self.camera_offset_x
    lz -= self.camera_height

    # Convert z-up (UE4/Carla) to z-down (openpilot calibrated)
    return (lx, ly, -lz)

  def _interpolate_at_x_idxs(self, boundary_points):
    """Interpolate boundary points at X_IDXS distances.

    Args:
      boundary_points: Nx3 array, (x, y, z) in calibrated frame.

    Returns:
      33x3 array interpolated at X_IDXS, or None if insufficient data.
    """
    if boundary_points.shape[0] < 2:
      return None

    # Sort by x (forward distance).
    # Keep points with x > -back_margin so backward-sampled points
    # participate in interpolation at X_IDXS[0]=0.
    order = np.argsort(boundary_points[:, 0])
    pts = boundary_points[order]
    mask = pts[:, 0] > -5.0
    pts = pts[mask]
    if pts.shape[0] < 2:
      return None

    xs = pts[:, 0]
    result = np.zeros((33, 3), dtype=np.float32)
    result[:, 0] = self.x_idxs

    # Interpolate y and z at X_IDXS.
    # Out-of-range points keep NaN so downstream code can distinguish them
    # from real data. Replacing NaN with 0 would create (x, 0, 0) points
    # that project to the image center (cx, cy) — the vanishing point artifact.
    result[:, 1] = np.interp(self.x_idxs, xs, pts[:, 1], left=np.nan, right=np.nan)
    result[:, 2] = np.interp(self.x_idxs, xs, pts[:, 2], left=np.nan, right=np.nan)

    return result
