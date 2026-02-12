"""Camera calibration: known pose mode + online calibration (ported from calibrationd)."""

import numpy as np
from openpilot.common.transformations.orientation import rot_from_euler, euler_from_rot

# Constants from calibrationd
MIN_SPEED_FILTER = 15 * 0.44704  # 15 MPH -> m/s
MAX_VEL_ANGLE_STD = np.radians(0.25)
MAX_YAW_RATE_FILTER = np.radians(2)  # per second
MAX_HEIGHT_STD = np.exp(-3.5)

SMOOTH_CYCLES = 10
BLOCK_SIZE = 100
INPUTS_NEEDED = 5
INPUTS_WANTED = 50
MAX_ALLOWED_YAW_SPREAD = np.radians(2)
MAX_ALLOWED_PITCH_SPREAD = np.radians(4)
RPY_INIT = np.array([0.0, 0.0, 0.0])
HEIGHT_INIT = np.array([1.22])

PITCH_LIMITS = np.array([-0.09074112085129739, 0.17])
YAW_LIMITS = np.array([-0.06912048084718224, 0.06912048084718235])


def is_calibration_valid(rpy):
  return (PITCH_LIMITS[0] < rpy[1] < PITCH_LIMITS[1]) and (YAW_LIMITS[0] < rpy[2] < YAW_LIMITS[1])


def sanity_clip(rpy):
  if np.isnan(rpy).any():
    rpy = RPY_INIT.copy()
  return np.array([rpy[0],
                   np.clip(rpy[1], PITCH_LIMITS[0] - .005, PITCH_LIMITS[1] + .005),
                   np.clip(rpy[2], YAW_LIMITS[0] - .005, YAW_LIMITS[1] + .005)])


def moving_avg_with_linear_decay(prev_mean, new_val, idx, block_size):
  return (idx * prev_mean + (block_size - idx) * new_val) / block_size


class KnownPoseCalibrator:
  """Calibrator with known camera pose (for debugging/validation)."""

  def __init__(self, pitch_deg=0.0, yaw_deg=0.0, height=1.22):
    self.rpy = np.array([0.0, np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)])
    self._height = np.array([height])
    self.cal_status = 'calibrated'
    self.valid_blocks = INPUTS_WANTED

  @property
  def rpyCalib(self):
    return self.rpy

  @property
  def height(self):
    return float(self._height[0])

  @property
  def calibrated(self):
    return True

  def update(self, vision_output, vehicle_speed):
    pass  # No-op for known pose


class OnlineCalibrator:
  """Online camera calibration, ported from calibrationd.py Calibrator."""

  def __init__(self, pitch_deg_init=0.0, yaw_deg_init=0.0, height_init=1.22):
    rpy_init = np.array([0.0, np.deg2rad(pitch_deg_init), np.deg2rad(yaw_deg_init)])
    self._reset(rpy_init, 0, HEIGHT_INIT if height_init is None else np.array([height_init]))

  def _reset(self, rpy_init, valid_blocks=0, height_init=HEIGHT_INIT, smooth_from=None):
    if not np.isfinite(rpy_init).all():
      self.rpy = RPY_INIT.copy()
    else:
      self.rpy = rpy_init.copy()

    if not np.isfinite(height_init).all() or len(height_init) != 1:
      self._height = HEIGHT_INIT.copy()
    else:
      self._height = height_init.copy()

    self.wide_from_device_euler = np.zeros(3)
    self.valid_blocks = max(0, valid_blocks)

    self.rpys = np.tile(self.rpy, (INPUTS_WANTED, 1))
    self.wide_from_device_eulers = np.tile(self.wide_from_device_euler, (INPUTS_WANTED, 1))
    self.heights = np.tile(self._height, (INPUTS_WANTED, 1))

    self.idx = 0
    self.block_idx = 0
    self.v_ego = 0.0
    self.calib_spread = np.zeros(3)

    if smooth_from is None:
      self.old_rpy = RPY_INIT.copy()
      self.old_rpy_weight = 0.0
    else:
      self.old_rpy = smooth_from.copy()
      self.old_rpy_weight = 1.0

    self.cal_status = 'uncalibrated'

  @property
  def rpyCalib(self):
    return self.get_smooth_rpy()

  @property
  def height(self):
    return float(self._height[0])

  @property
  def calibrated(self):
    return self.cal_status == 'calibrated'

  def get_smooth_rpy(self):
    if self.old_rpy_weight > 0:
      return self.old_rpy_weight * self.old_rpy + (1.0 - self.old_rpy_weight) * self.rpy
    return self.rpy

  def _get_valid_idxs(self):
    before_current = list(range(self.block_idx))
    after_current = list(range(min(self.valid_blocks, self.block_idx + 1), self.valid_blocks))
    return before_current + after_current

  def _update_status(self):
    valid_idxs = self._get_valid_idxs()
    if valid_idxs:
      self.wide_from_device_euler = np.mean(self.wide_from_device_eulers[valid_idxs], axis=0)
      self._height = np.mean(self.heights[valid_idxs], axis=0)
      rpys = self.rpys[valid_idxs]
      self.rpy = np.mean(rpys, axis=0)
      max_rpy = np.max(rpys, axis=0)
      min_rpy = np.min(rpys, axis=0)
      self.calib_spread = np.abs(max_rpy - min_rpy)
    else:
      self.calib_spread = np.zeros(3)

    if self.valid_blocks < INPUTS_NEEDED:
      if self.cal_status != 'recalibrating':
        self.cal_status = 'uncalibrated'
    elif is_calibration_valid(self.rpy):
      self.cal_status = 'calibrated'
    else:
      self.cal_status = 'invalid'

    spread_too_high = (self.calib_spread[1] > MAX_ALLOWED_PITCH_SPREAD or
                       self.calib_spread[2] > MAX_ALLOWED_YAW_SPREAD)
    if spread_too_high and self.cal_status == 'calibrated':
      self._reset(self.rpys[self.block_idx - 1], valid_blocks=1, smooth_from=self.rpy)
      self.cal_status = 'recalibrating'

  def _handle_cam_odom(self, trans, rot, wide_from_device_euler, trans_std,
                       road_transform_trans, road_transform_trans_std):
    """Core calibration update from camera odometry, matching calibrationd logic."""
    self.old_rpy_weight = max(0.0, self.old_rpy_weight - 1 / SMOOTH_CYCLES)

    # In simulation, visual odometry underestimates speed due to low realtime ratio,
    # so we relax the trans[0] threshold (matching calibrationd SIMULATION logic)
    min_trans_speed = MIN_SPEED_FILTER * 0.2
    straight_and_fast = ((self.v_ego > MIN_SPEED_FILTER) and (trans[0] > min_trans_speed) and
                         (abs(rot[2]) < MAX_YAW_RATE_FILTER))
    rpy_certain = np.arctan2(trans_std[1], trans[0]) < MAX_VEL_ANGLE_STD
    height_certain = road_transform_trans_std[2] < MAX_HEIGHT_STD if len(road_transform_trans_std) == 3 else True
    certain_if_calib = (rpy_certain and height_certain) or (self.valid_blocks < INPUTS_NEEDED)

    if not (straight_and_fast and certain_if_calib):
      return None

    observed_rpy = np.array([0,
                             -np.arctan2(trans[2], trans[0]),
                             np.arctan2(trans[1], trans[0])])
    new_rpy = euler_from_rot(rot_from_euler(self.get_smooth_rpy()).dot(rot_from_euler(observed_rpy)))
    new_rpy = sanity_clip(new_rpy)

    new_wide = np.array(wide_from_device_euler) if len(wide_from_device_euler) == 3 else np.zeros(3)
    new_height = np.array([road_transform_trans[2]]) if len(road_transform_trans) == 3 else HEIGHT_INIT.copy()

    self.rpys[self.block_idx] = moving_avg_with_linear_decay(
      self.rpys[self.block_idx], new_rpy, self.idx, float(BLOCK_SIZE))
    self.wide_from_device_eulers[self.block_idx] = moving_avg_with_linear_decay(
      self.wide_from_device_eulers[self.block_idx], new_wide, self.idx, float(BLOCK_SIZE))
    self.heights[self.block_idx] = moving_avg_with_linear_decay(
      self.heights[self.block_idx], new_height, self.idx, float(BLOCK_SIZE))

    self.idx = (self.idx + 1) % BLOCK_SIZE
    if self.idx == 0:
      self.block_idx += 1
      self.valid_blocks = max(self.block_idx, self.valid_blocks)
      self.block_idx = self.block_idx % INPUTS_WANTED

    self._update_status()
    return new_rpy

  def update(self, vision_output, vehicle_speed):
    """Update calibration from vision model output and vehicle speed."""
    self.v_ego = vehicle_speed

    trans = vision_output['pose'][0, :3].tolist()
    rot = vision_output['pose'][0, 3:].tolist()
    trans_std = vision_output['pose_stds'][0, :3].tolist()
    wide_euler = vision_output['wide_from_device_euler'][0].tolist()
    road_trans = vision_output['road_transform'][0, :3].tolist()
    road_trans_std = vision_output['road_transform_stds'][0, :3].tolist()

    new_rpy = self._handle_cam_odom(trans, rot, wide_euler, trans_std, road_trans, road_trans_std)
    return new_rpy
