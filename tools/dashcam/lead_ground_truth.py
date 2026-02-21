"""Ground truth lead vehicle extraction from Carla world.

Extracts front vehicles' 3D positions, absolute speeds and accelerations,
outputting in openpilot lead format: [3, 6, 4] (3 selections x 6 time steps x 4 dims).

Feature semantics match openpilot model output (LeadDataV3):
  [sel, t, 0] x  — forward distance relative to ego (meters)
  [sel, t, 1] y  — lateral offset relative to ego (meters, right positive)
  [sel, t, 2] v  — lead absolute forward speed (m/s, NOT relative)
  [sel, t, 3] a  — lead forward acceleration (m/s²)
"""

from math import cos, radians, sin

import numpy as np


# Time indices for lead prediction (seconds into future)
LEAD_T_IDXS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# Time offsets for three lead selections: current / 2s-ahead / 4s-ahead closest vehicle
LEAD_T_OFFSETS = [0.0, 2.0, 4.0]


class LeadGroundTruth:
  """Extract lead vehicle ground truth from Carla world actors."""

  # Validity constraints matching openpilot downstream processing
  MIN_FORWARD_DIST = 2.0    # meters — closer vehicles are mostly occluded
  MAX_FORWARD_DIST = 200.0  # meters — beyond model's useful range
  MAX_LATERAL_DIST = 4.0    # meters — coarse lateral filter for candidate collection
  EGO_LANE_HALF_WIDTH = 2.0 # meters — fine lateral filter for lead selection per time offset
  MIN_HEIGHT = -1.0         # meters (cal_z, z-down) — allow uphill vehicles slightly above camera
  MAX_HEIGHT = 5.0          # meters (cal_z, z-down) — reject overpass vehicles far above
  ACCEL_CLIP_MIN = -10.0    # m/s² — matches long_mpc.py clipping
  ACCEL_CLIP_MAX = 5.0      # m/s² — matches long_mpc.py clipping

  def __init__(self, carla_world, ego_vehicle, camera_offset_x=0.8, camera_height=1.13,
               image_w=1928, image_h=1208, focal_length=2648.0):
    self.carla_world = carla_world
    self.ego_vehicle = ego_vehicle
    self.camera_offset_x = camera_offset_x
    self.camera_height = camera_height
    # Camera intrinsics for image visibility check
    self.image_w = image_w
    self.image_h = image_h
    self.focal_length = focal_length
    self.cx = image_w / 2.0
    self.cy = image_h / 2.0

  def get_lead_vehicles(self, vehicle_transform, ego_velocity, road_edges=None):
    """Extract up to 3 lead vehicle GT in calibrated frame.

    Args:
      vehicle_transform: carla.Transform of the ego vehicle.
      ego_velocity: scalar speed of ego vehicle in m/s.
      road_edges: (edges_list, edge_probs) from LaneGroundTruth, used to reject
                  vehicles separated from ego by a road edge (e.g. highway median).

    Returns:
      lead_data: [3, 6, 4] float32 -- 3 selections x 6 time points x 4 dims (x, y, v_abs, a)
      lead_probs: [3] float32 -- existence probability per selection (0 or 1)
    """
    lead_data = np.zeros((3, 6, 4), dtype=np.float32)
    lead_probs = np.zeros(3, dtype=np.float32)

    # Build road edge polylines for separation check
    edge_polylines = self._build_edge_polylines(road_edges)

    # Get all vehicles in the world, exclude ego
    actors = self.carla_world.get_actors().filter('vehicle.*')
    ego_id = self.ego_vehicle.id

    # Collect forward vehicles with their calibrated-frame positions
    forward_vehicles = []
    for actor in actors:
      if actor.id == ego_id:
        continue

      npc_transform = actor.get_transform()
      npc_velocity = actor.get_velocity()
      npc_accel = actor.get_acceleration()

      # Transform NPC position to ego calibrated frame
      cal_x, cal_y, cal_z = self._world_to_calibrated(
        npc_transform.location.x, npc_transform.location.y, npc_transform.location.z,
        vehicle_transform)

      # 1. Forward distance constraint
      if cal_x < self.MIN_FORWARD_DIST or cal_x > self.MAX_FORWARD_DIST:
        continue

      # 2. Lateral distance — exclude vehicles in adjacent lanes
      if abs(cal_y) > self.MAX_LATERAL_DIST:
        continue

      # 3. Height constraint — exclude vehicles on overpasses or underpasses
      if cal_z < self.MIN_HEIGHT or cal_z > self.MAX_HEIGHT:
        continue

      # 4. Road edge separation — exclude vehicles on the other side of a road edge
      if self._separated_by_edge(cal_x, cal_y, edge_polylines):
        continue

      # 5. Image visibility — vehicle center must project within camera FOV
      #    Simplified projection (calibrated ≈ device frame):
      #    view_x = cal_y, view_y = cal_z, view_z = cal_x
      u = self.focal_length * (cal_y / cal_x) + self.cx
      v = self.focal_length * (cal_z / cal_x) + self.cy
      if u < 0 or u >= self.image_w or v < 0 or v >= self.image_h:
        continue

      # Compute NPC velocity in ego calibrated frame
      vx, vy, vz = self._velocity_to_calibrated(
        npc_velocity.x, npc_velocity.y, npc_velocity.z,
        vehicle_transform)

      # Compute NPC acceleration in ego calibrated frame, clipped to valid range
      ax, ay, az = self._velocity_to_calibrated(
        npc_accel.x, npc_accel.y, npc_accel.z,
        vehicle_transform)
      ax = float(np.clip(ax, self.ACCEL_CLIP_MIN, self.ACCEL_CLIP_MAX))

      # Relative velocity (along forward axis) — used for position prediction
      v_rel = vx - ego_velocity

      forward_vehicles.append({
        'x': cal_x, 'y': cal_y, 'z': cal_z,
        'v_rel': v_rel, 'vx': vx, 'vy': vy, 'ax': ax,
        'dist': cal_x,
      })

    if not forward_vehicles:
      return lead_data, lead_probs

    # Sort by forward distance
    forward_vehicles.sort(key=lambda v: v['dist'])

    # For each selection offset, find the closest vehicle that will be in ego lane at that time.
    # This captures both same-lane leads (t=0) and cut-in vehicles from adjacent lanes (t=2s/4s).
    for sel_idx, t_offset in enumerate(LEAD_T_OFFSETS):
      best = None
      best_future_x = float('inf')
      for v in forward_vehicles:
        future_x = v['x'] + v['v_rel'] * t_offset
        future_y = v['y'] + v['vy'] * t_offset
        if future_x > 0 and abs(future_y) < self.EGO_LANE_HALF_WIDTH and future_x < best_future_x:
          best = v
          best_future_x = future_x

      if best is None:
        continue

      lead_probs[sel_idx] = 1.0

      # Fill 6 time-step predictions using constant acceleration model.
      # Trajectory always covers LEAD_T_IDXS from current time (t=0),
      # regardless of which t_offset was used for vehicle selection.
      for t_idx, t in enumerate(LEAD_T_IDXS):
        pred_x = best['x'] + best['v_rel'] * t + 0.5 * best['ax'] * t ** 2
        pred_y = best['y'] + best['vy'] * t
        pred_v = max(0.0, best['vx'] + best['ax'] * t)  # absolute speed, non-negative
        lead_data[sel_idx, t_idx, 0] = pred_x       # x: forward distance (relative to ego)
        lead_data[sel_idx, t_idx, 1] = pred_y       # y: lateral offset
        lead_data[sel_idx, t_idx, 2] = pred_v       # v: absolute forward speed
        lead_data[sel_idx, t_idx, 3] = best['ax']   # a: forward acceleration

    return lead_data, lead_probs

  def _world_to_calibrated(self, wx, wy, wz, vehicle_transform):
    """Convert world point to gravity-aligned calibrated frame (x=forward, y=right, z=down).

    Same coordinate transform as LaneGroundTruth._world_to_calibrated.
    """
    yaw = radians(vehicle_transform.rotation.yaw)
    pitch = radians(vehicle_transform.rotation.pitch)
    roll = radians(vehicle_transform.rotation.roll)

    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)
    cr, sr = cos(roll), sin(roll)

    # Camera world position: vehicle_pos + R_full @ [cam_x, 0, cam_h]
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
    cal_z = dz

    return (cal_x, cal_y, -cal_z)

  def _velocity_to_calibrated(self, vx, vy, vz, vehicle_transform):
    """Convert world-frame velocity vector to gravity-aligned calibrated frame.

    Only applies yaw rotation (no pitch/roll, no translation offset).
    """
    yaw = radians(vehicle_transform.rotation.yaw)
    cy, sy = cos(yaw), sin(yaw)

    # Rz(yaw)^T @ [vx, vy, vz]
    cal_vx = cy * vx + sy * vy
    cal_vy = -sy * vx + cy * vy
    cal_vz = vz

    return (cal_vx, cal_vy, -cal_vz)

  @staticmethod
  def _build_edge_polylines(road_edges):
    """Build (x, y) polylines for each valid road edge."""
    if road_edges is None or road_edges[0] is None:
      return []
    edges_list, edge_probs = road_edges
    polylines = []
    for edge, prob in zip(edges_list, edge_probs, strict=True):
      if prob < 0.5:
        continue
      valid = ~np.isnan(edge[:, 0]) & ~np.isnan(edge[:, 1])
      if np.sum(valid) < 2:
        continue
      ex = edge[valid, 0]
      ey = edge[valid, 1]
      order = np.argsort(ex)
      polylines.append((ex[order], ey[order]))
    return polylines

  @staticmethod
  def _separated_by_edge(cal_x, cal_y, edge_polylines):
    """Check if the line segment from ego (0,0) to NPC (cal_x, cal_y) crosses any road edge polyline."""
    for ex, ey in edge_polylines:
      for i in range(len(ex) - 1):
        # Cross product test for segment intersection
        # Segment A: (0, 0) → (cal_x, cal_y)
        # Segment B: (ex[i], ey[i]) → (ex[i+1], ey[i+1])
        bx0, by0 = float(ex[i]), float(ey[i])
        bx1, by1 = float(ex[i + 1]), float(ey[i + 1])
        d1 = (bx1 - bx0) * (0.0 - by0) - (by1 - by0) * (0.0 - bx0)
        d2 = (bx1 - bx0) * (cal_y - by0) - (by1 - by0) * (cal_x - bx0)
        d3 = cal_x * (by0 - 0.0) - cal_y * (bx0 - 0.0)
        d4 = cal_x * (by1 - 0.0) - cal_y * (bx1 - 0.0)
        if d1 * d2 < 0 and d3 * d4 < 0:
          return True
    return False

