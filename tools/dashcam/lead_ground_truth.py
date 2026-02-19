"""Ground truth lead vehicle extraction from Carla world.

Extracts front vehicles' 3D positions and velocities relative to the ego vehicle,
outputting in openpilot lead format: [3, 6, 4] (3 selections x 6 time steps x 4 dims).
"""

from math import cos, radians, sin

import numpy as np


# Time indices for lead prediction (seconds into future)
LEAD_T_IDXS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# Time offsets for three lead selections: current / 2s-ahead / 4s-ahead closest vehicle
LEAD_T_OFFSETS = [0.0, 2.0, 4.0]


class LeadGroundTruth:
  """Extract lead vehicle ground truth from Carla world actors."""

  def __init__(self, carla_world, ego_vehicle, camera_offset_x=0.8, camera_height=1.13):
    self.carla_world = carla_world
    self.ego_vehicle = ego_vehicle
    self.camera_offset_x = camera_offset_x
    self.camera_height = camera_height

  def get_lead_vehicles(self, vehicle_transform, ego_velocity):
    """Extract up to 3 lead vehicle GT in calibrated frame.

    Args:
      vehicle_transform: carla.Transform of the ego vehicle.
      ego_velocity: scalar speed of ego vehicle in m/s.

    Returns:
      lead_data: [3, 6, 4] float32 -- 3 selections x 6 time points x 4 dims (x, y, v_rel, a)
      lead_probs: [3] float32 -- existence probability per selection (0 or 1)
    """
    lead_data = np.zeros((3, 6, 4), dtype=np.float32)
    lead_probs = np.zeros(3, dtype=np.float32)

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

      # Transform NPC position to ego calibrated frame
      cal_x, cal_y, cal_z = self._world_to_calibrated(
        npc_transform.location.x, npc_transform.location.y, npc_transform.location.z,
        vehicle_transform)

      # Only consider vehicles ahead (x > 0) and within reasonable range
      if cal_x <= 0.0 or cal_x > 200.0:
        continue

      # Compute NPC velocity in ego calibrated frame
      vx, vy, vz = self._velocity_to_calibrated(
        npc_velocity.x, npc_velocity.y, npc_velocity.z,
        vehicle_transform)

      # Relative velocity (along forward axis)
      v_rel = vx - ego_velocity

      forward_vehicles.append({
        'x': cal_x, 'y': cal_y, 'z': cal_z,
        'v_rel': v_rel, 'vx': vx,
        'dist': cal_x,
      })

    if not forward_vehicles:
      return lead_data, lead_probs

    # Sort by forward distance
    forward_vehicles.sort(key=lambda v: v['dist'])

    # For each selection offset, find the closest vehicle at that future time
    for sel_idx, t_offset in enumerate(LEAD_T_OFFSETS):
      # Find the closest vehicle that would be ahead at t_offset
      best = None
      for v in forward_vehicles:
        # Predict position at t_offset using constant velocity
        future_x = v['x'] + v['v_rel'] * t_offset
        if future_x > 0:
          best = v
          break

      if best is None:
        continue

      lead_probs[sel_idx] = 1.0

      # Fill 6 time-step predictions using constant velocity model
      for t_idx, t in enumerate(LEAD_T_IDXS):
        total_t = t_offset + t
        pred_x = best['x'] + best['v_rel'] * total_t
        pred_y = best['y']  # assume lateral position stays constant
        lead_data[sel_idx, t_idx, 0] = pred_x  # x (forward distance)
        lead_data[sel_idx, t_idx, 1] = pred_y  # y (lateral offset)
        lead_data[sel_idx, t_idx, 2] = best['v_rel']  # relative velocity
        lead_data[sel_idx, t_idx, 3] = 0.0  # acceleration (assume constant velocity)

    return lead_data, lead_probs

  def _world_to_calibrated(self, wx, wy, wz, vehicle_transform):
    """Convert world point to calibrated frame (x=forward, y=right, z=down).

    Same coordinate transform as LaneGroundTruth._world_to_calibrated.
    """
    dx = wx - vehicle_transform.location.x
    dy = wy - vehicle_transform.location.y
    dz = wz - vehicle_transform.location.z

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

    lx -= self.camera_offset_x
    lz -= self.camera_height

    return (lx, ly, -lz)

  def _velocity_to_calibrated(self, vx, vy, vz, vehicle_transform):
    """Convert world-frame velocity vector to calibrated frame.

    Only applies rotation (no translation offset for velocities).
    """
    yaw = radians(vehicle_transform.rotation.yaw)
    pitch = radians(vehicle_transform.rotation.pitch)
    roll = radians(vehicle_transform.rotation.roll)

    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)
    cr, sr = cos(roll), sin(roll)

    # R^T @ [vx, vy, vz]
    lx = (cy * cp) * vx + (sy * cp) * vy + (-sp) * vz
    ly = (cy * sp * sr - sy * cr) * vx + (sy * sp * sr + cy * cr) * vy + (cp * sr) * vz
    lz = (cy * sp * cr + sy * sr) * vx + (sy * sp * cr - cy * sr) * vy + (cp * cr) * vz

    return (lx, ly, -lz)
