"""Ground truth lane line extraction from Carla map API.

Extracts lane boundaries from Carla waypoints, transforms to openpilot calibrated
frame (x=forward, y=right, z=down), and interpolates at ModelConstants.X_IDXS distances.
"""

from math import cos, radians, sin

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants


class LaneGroundTruth:
  """Extract lane boundary and road edge ground truth from Carla map.

  Lane lines (openpilot format):
    [0] far-left: left adjacent lane's left boundary
    [1] near-left: ego lane's left boundary
    [2] near-right: ego lane's right boundary
    [3] far-right: right adjacent lane's right boundary

  Road edges:
    [0] left edge: outermost same-direction lane's left boundary
    [1] right edge: outermost same-direction lane's right boundary
  """

  def __init__(self, carla_map, camera_offset_x=0.8, camera_height=1.13, road_edge_offset=0.5):
    self.map = carla_map
    self.camera_offset_x = camera_offset_x
    self.camera_height = camera_height
    self.road_edge_offset = road_edge_offset
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
    #if self._has_junction_ahead(ego_wp, max_dist=100.0, step=2.0):
    #  return None, None

    # Sample ego lane waypoints
    ego_wps = self._sample_waypoints(ego_wp, max_dist=200.0, step=2.0)

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

    # [0] far-left: left adjacent lane's left boundary
    left_lane_wp = ego_wp.get_left_lane()
    if (left_lane_wp is not None
        and left_lane_wp.lane_type == carla.LaneType.Driving
        and self._is_same_direction(ego_wp, left_lane_wp)):
      left_wps = self._sample_waypoints(left_lane_wp, max_dist=200.0, step=2.0)
      far_left_boundary = self._compute_boundary(left_wps, vehicle_transform, side='left')
      interp = self._interpolate_at_x_idxs(far_left_boundary)
      if interp is not None:
        gt_lines[0] = interp
        gt_probs[0] = 1.0

    # [3] far-right: right adjacent lane's right boundary
    right_lane_wp = ego_wp.get_right_lane()
    if (right_lane_wp is not None
        and right_lane_wp.lane_type == carla.LaneType.Driving
        and self._is_same_direction(ego_wp, right_lane_wp)):
      right_wps = self._sample_waypoints(right_lane_wp, max_dist=200.0, step=2.0)
      far_right_boundary = self._compute_boundary(right_wps, vehicle_transform, side='right')
      interp = self._interpolate_at_x_idxs(far_right_boundary)
      if interp is not None:
        gt_lines[3] = interp
        gt_probs[3] = 1.0

    return gt_lines, gt_probs

  def get_road_edges(self, vehicle_transform):
    """Extract GT road edges in calibrated frame at X_IDXS distances.

    Traverses outward from ego lane through all road-surface lane types
    (Driving, Shoulder, Parking, Biking, etc.) and returns the outer
    boundary of the outermost road-surface lane as the road edge.

    Args:
      vehicle_transform: carla.Transform of the ego vehicle.

    Returns:
      edges_list: list of 2 arrays, each 33x3 (x, y, z) in calibrated frame.
      edge_probs: list of 2 floats (1.0 if edge exists, 0.0 otherwise).
      Returns (None, None) if ego waypoint not found or junction ahead.
    """
    import carla

    road_surface_types = {
      carla.LaneType.Driving,
      carla.LaneType.Shoulder,
      carla.LaneType.Parking,
      carla.LaneType.Biking,
      carla.LaneType.Entry,
      carla.LaneType.Exit,
      carla.LaneType.OnRamp,
      carla.LaneType.OffRamp,
      carla.LaneType.Restricted,
    }

    loc = vehicle_transform.location
    ego_wp = self.map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
      return None, None

    #if self._has_junction_ahead(ego_wp, max_dist=100.0, step=2.0):
    #  return None, None

    edges_list = [np.zeros((33, 3), dtype=np.float32) for _ in range(2)]
    edge_probs = [0.0, 0.0]

    # [0] left road edge: traverse left through all road-surface lanes
    wp = ego_wp
    while True:
      left = wp.get_left_lane()
      if left is None or left.lane_type not in road_surface_types:
        break
      # Only check direction for Driving lanes; Shoulder etc. have unreliable forward vectors
      if left.lane_type == carla.LaneType.Driving and not self._is_same_direction(wp, left):
        break
      wp = left
    left_edge_wps = self._sample_waypoints(wp, max_dist=200.0, step=2.0)
    left_edge = self._compute_boundary(left_edge_wps, vehicle_transform, side='left')
    interp = self._interpolate_at_x_idxs(left_edge)
    if interp is not None:
      interp[:, 1] += self.road_edge_offset  # shift rightward (toward ego)
      edges_list[0] = interp
      edge_probs[0] = 1.0

    # [1] right road edge: traverse right through all road-surface lanes
    wp = ego_wp
    while True:
      right = wp.get_right_lane()
      if right is None or right.lane_type not in road_surface_types:
        break
      if right.lane_type == carla.LaneType.Driving and not self._is_same_direction(wp, right):
        break
      wp = right
    right_edge_wps = self._sample_waypoints(wp, max_dist=200.0, step=2.0)
    right_edge = self._compute_boundary(right_edge_wps, vehicle_transform, side='right')
    interp = self._interpolate_at_x_idxs(right_edge)
    if interp is not None:
      interp[:, 1] -= self.road_edge_offset  # shift leftward (toward ego)
      edges_list[1] = interp
      edge_probs[1] = 1.0

    return edges_list, edge_probs

  @staticmethod
  def filter_road_edges(lane_gt, road_edges_gt):
    """Filter road edges that are not outside the outermost lane lines.

    In calibrated frame (y+ = right):
      - Left road edge y should be < outermost left lane line y (more left)
      - Right road edge y should be > outermost right lane line y (more right)
    If violated, the road edge is zeroed out with prob=0.

    Args:
      lane_gt: (gt_lines, gt_probs) from get_lane_lines(). Can be (None, None).
      road_edges_gt: (edges_list, edge_probs) from get_road_edges(). Can be (None, None).

    Returns:
      Filtered (edges_list, edge_probs).
    """
    if road_edges_gt is None or road_edges_gt[0] is None:
      return road_edges_gt
    if lane_gt is None or lane_gt[0] is None:
      return road_edges_gt

    gt_lines, gt_probs = lane_gt
    edges_list, edge_probs = road_edges_gt

    # Left side: outermost left lane = [0] if exists, else [1]
    if edge_probs[0] > 0:
      left_lane_idx = 0 if gt_probs[0] > 0 else (1 if gt_probs[1] > 0 else -1)
      if left_lane_idx >= 0:
        edge_y = edges_list[0][:, 1]
        lane_y = gt_lines[left_lane_idx][:, 1]
        valid = ~np.isnan(edge_y) & ~np.isnan(lane_y)
        if np.any(valid) and np.mean(edge_y[valid]) >= np.mean(lane_y[valid]):
          edges_list[0] = np.zeros((33, 3), dtype=np.float32)
          edge_probs[0] = 0.0

    # Right side: outermost right lane = [3] if exists, else [2]
    if edge_probs[1] > 0:
      right_lane_idx = 3 if gt_probs[3] > 0 else (2 if gt_probs[2] > 0 else -1)
      if right_lane_idx >= 0:
        edge_y = edges_list[1][:, 1]
        lane_y = gt_lines[right_lane_idx][:, 1]
        valid = ~np.isnan(edge_y) & ~np.isnan(lane_y)
        if np.any(valid) and np.mean(edge_y[valid]) <= np.mean(lane_y[valid]):
          edges_list[1] = np.zeros((33, 3), dtype=np.float32)
          edge_probs[1] = 0.0

    return edges_list, edge_probs

  def _is_same_direction(self, wp_a, wp_b):
    """Check if two waypoints have the same driving direction (forward vector dot product > 0)."""
    fwd_a = wp_a.transform.get_forward_vector()
    fwd_b = wp_b.transform.get_forward_vector()
    return (fwd_a.x * fwd_b.x + fwd_a.y * fwd_b.y + fwd_a.z * fwd_b.z) > 0

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
    """Convert world point to gravity-aligned calibrated frame (x=forward, y=right, z=down).

    Steps:
      1. Compute camera world position using full vehicle rotation (camera is physically on the tilted car).
      2. Compute delta from camera to target point in world frame.
      3. Apply yaw-only rotation (gravity-aligned: no pitch/roll).
      4. Flip z axis (Carla z-up -> openpilot z-down).
    """
    yaw = radians(vehicle_transform.rotation.yaw)
    pitch = radians(vehicle_transform.rotation.pitch)
    roll = radians(vehicle_transform.rotation.roll)

    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)
    cr, sr = cos(roll), sin(roll)

    # Camera world position: vehicle_pos + R_full @ [cam_x, 0, cam_h]
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    cam_x = self.camera_offset_x
    cam_h = self.camera_height
    cam_wx = vehicle_transform.location.x + (cy * cp) * cam_x + (cy * sp * cr + sy * sr) * cam_h
    cam_wy = vehicle_transform.location.y + (sy * cp) * cam_x + (sy * sp * cr - cy * sr) * cam_h
    cam_wz = vehicle_transform.location.z + (-sp) * cam_x + (cp * cr) * cam_h

    # Delta from camera to target in world frame
    dx = wx - cam_wx
    dy = wy - cam_wy
    dz = wz - cam_wz

    # Yaw-only rotation: Rz(yaw)^T @ [dx, dy, dz]
    cal_x = cy * dx + sy * dy
    cal_y = -sy * dx + cy * dy
    cal_z = dz  # world z-up preserved

    # Convert z-up (UE4/Carla) to z-down (openpilot calibrated)
    return (cal_x, cal_y, -cal_z)

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

    # Truncate beyond hill crest (not visible to camera)
    self._truncate_at_crest(result)

    return result

  def _truncate_at_crest(self, points, descent_threshold=1.0):
    """Truncate interpolated points beyond the hill crest.

    On uphills, the road surface beyond the crest is occluded and not visible
    to the camera. Points past the crest are set to NaN.

    In calibrated frame (z-down), the crest is where z reaches its minimum
    (highest 3D point). Only truncates when there is a significant elevation
    change (>= descent_threshold) both before and after the crest.

    Args:
      points: 33x3 array (x, y, z) in calibrated frame (z-down).
      descent_threshold: minimum elevation change (meters) to detect a real crest.
    """
    z = points[:, 2]
    valid = ~np.isnan(z)
    if np.sum(valid) < 3:
      return

    valid_idx = np.where(valid)[0]
    valid_z = z[valid_idx]

    # Running minimum of z (tracks the highest 3D point seen along the path)
    running_min = np.minimum.accumulate(valid_z)
    descent = valid_z - running_min

    # First point that has descended significantly below the crest
    beyond = descent >= descent_threshold
    if not np.any(beyond):
      return

    first_beyond = np.argmax(beyond)
    crest_local = np.argmin(valid_z[:first_beyond + 1])

    # Only truncate if road actually rose before the crest
    z_max_before = np.max(valid_z[:crest_local + 1]) if crest_local > 0 else valid_z[0]
    if z_max_before - valid_z[crest_local] < descent_threshold:
      return

    # NaN out everything after the crest
    crest_global = valid_idx[crest_local]
    points[crest_global + 1:, 1] = np.nan
    points[crest_global + 1:, 2] = np.nan
