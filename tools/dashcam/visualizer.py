"""Visualization: draw perception results on camera images, display and record video.

Adapted for cereal modelV2 message format with continuous probability rendering
matching the openpilot UI approach (no hard probability thresholds).
"""

import math
import cv2
import numpy as np

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler

from openpilot.tools.dashcam.carla_world import W, H

# Height compensation constants
_HEIGHT_Z_IDX_START = 5   # ~4.7m forward, skip noisy near-range
_HEIGHT_Z_IDX_END = 21    # ~82.7m forward, avoid far-range road slope
_HEIGHT_MIN_PROB = 0.3    # minimum lane line probability for height estimation
_HEIGHT_MIN_Z = 0.5       # minimum reasonable camera height (meters)

# Bird's eye view (BEV) panel constants
_BEV_W, _BEV_H = 300, 450   # panel pixel size
_BEV_X_MAX = 80.0            # forward range (meters)
_BEV_Y_HALF = 10.0           # lateral half-range (meters), ±10m
_BEV_MARGIN = 15             # pixels from edge of display

# Match openpilot UI: 2160x1080, camera zoomed to fill then center-cropped
UI_W, UI_H = 2160, 1080
ZOOM = max(UI_W / W, UI_H / H)  # ~1.12, fill width then crop height

# Distance clipping constants (aligned with model_renderer.py)
CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 5.0
MAX_DRAW_DISTANCE = 100.0

# Calibration constants for info panel display
INPUTS_NEEDED = 5

# Status colors (BGR) for calibration display
_STATUS_COLORS = {
  'uncalibrated': (0, 200, 255),    # yellow
  'calibrated': (0, 220, 0),        # green
  'invalid': (0, 0, 220),           # red
  'recalibrating': (0, 165, 255),   # orange
}


def project_points_to_image(xs, ys, zs, intrinsics_3x3, rpyCalib):
  """Project calibration-frame 3D points to image pixel coordinates.

  Calibration frame: x=forward, y=right, z=down.
  Model outputs lane_lines/road_edges with z ~ camera_height (road surface).
  Projection: intrinsic @ view_frame_from_device_frame @ device_from_calib.
  Returns Nx2 array of (u, v) pixel coords. Invalid points have NaN.
  """
  device_from_calib = rot_from_euler(rpyCalib)
  transform = intrinsics_3x3 @ view_frame_from_device_frame @ device_from_calib  # 3x3
  pts = np.stack([xs, ys, zs], axis=0)  # 3xN
  proj = transform @ pts  # 3xN
  # Filter out points behind camera (proj[2] is depth in view frame)
  behind = proj[2] <= 0
  proj[2, behind] = np.nan
  uv = proj[:2] / proj[2:3]
  return uv.T  # Nx2


def _build_transform(K, rpyCalib):
  """Build 3x3 projection matrix from intrinsics and calibration RPY."""
  device_from_calib = rot_from_euler(rpyCalib)
  return K @ view_frame_from_device_frame @ device_from_calib


def _get_path_length_idx(pos_x, distance):
  """Get the index of the last point with x <= distance."""
  indices = np.where(pos_x <= distance)[0]
  return indices[-1] if indices.size > 0 else 0


def _map_line_to_polygon(points_3d, y_off, z_off, max_idx, max_distance, transform):
  """Convert 3D line to 2D closed polygon for rendering.

  Aligned with model_renderer.py:_map_line_to_polygon.
  Returns Mx2 int32 array of polygon vertices (left forward + right reversed).
  """
  if points_3d.shape[0] == 0:
    return np.empty((0, 2), dtype=np.int32)

  points = points_3d[:max_idx + 1]

  # Interpolate around max_idx for smooth path end
  if 0 < max_idx < points_3d.shape[0] - 1:
    p0 = points_3d[max_idx]
    p1 = points_3d[max_idx + 1]
    interp_y = np.interp(max_distance, [p0[0], p1[0]], [p0[1], p1[1]])
    interp_z = np.interp(max_distance, [p0[0], p1[0]], [p0[2], p1[2]])
    interp_point = np.array([max_distance, interp_y, interp_z], dtype=points.dtype)
    points = np.concatenate((points, interp_point[None, :]), axis=0)

  # Filter non-negative x (points in front of camera)
  points = points[points[:, 0] >= 0]
  if points.shape[0] == 0:
    return np.empty((0, 2), dtype=np.int32)

  N = points.shape[0]
  # Generate left and right 3D points with ±y_off
  offsets = np.array([[0, -y_off, z_off], [0, y_off, z_off]], dtype=np.float32)
  points_lr = points[None, :, :] + offsets[:, None, :]  # 2xNx3
  points_lr = points_lr.reshape(2 * N, 3)

  # Project to 2D
  proj = transform @ points_lr.T  # 3x(2*N)
  proj = proj.reshape(3, 2, N)
  left_proj = proj[:, 0, :]
  right_proj = proj[:, 1, :]

  # Filter valid depth
  valid = (np.abs(left_proj[2]) >= 1e-6) & (np.abs(right_proj[2]) >= 1e-6)
  if not np.any(valid):
    return np.empty((0, 2), dtype=np.int32)

  # Compute screen coordinates
  left_screen = left_proj[:2, valid] / left_proj[2, valid][None, :]
  right_screen = right_proj[:2, valid] / right_proj[2, valid][None, :]

  # Clip to image bounds with margin
  x_min, x_max = -CLIP_MARGIN, W + CLIP_MARGIN
  y_min, y_max = -CLIP_MARGIN, H + CLIP_MARGIN

  both_in = (
    (left_screen[0] >= x_min) & (left_screen[0] <= x_max) &
    (left_screen[1] >= y_min) & (left_screen[1] <= y_max) &
    (right_screen[0] >= x_min) & (right_screen[0] <= x_max) &
    (right_screen[1] >= y_min) & (right_screen[1] <= y_max)
  )

  if not np.any(both_in):
    return np.empty((0, 2), dtype=np.int32)

  left_screen = left_screen[:, both_in]
  right_screen = right_screen[:, both_in]

  # Build closed polygon: left forward, right reversed
  return np.vstack((left_screen.T, right_screen[:, ::-1].T)).astype(np.int32)


def _draw_polygon_alpha(img, polygon, color_bgr, alpha):
  """Draw a filled polygon with alpha blending using ROI optimization."""
  x, y, w, h = cv2.boundingRect(polygon)
  # Clip ROI to image bounds
  x0 = max(x, 0)
  y0 = max(y, 0)
  x1 = min(x + w, img.shape[1])
  y1 = min(y + h, img.shape[0])
  if x1 <= x0 or y1 <= y0:
    return

  roi = img[y0:y1, x0:x1]
  overlay = roi.copy()
  shifted = polygon - np.array([x0, y0])
  cv2.fillPoly(overlay, [shifted], color_bgr)
  img[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)


def compensate_lane_lines_for_height(lane_lines_xyz, lane_line_probs, actual_height, left_scale=0.25, right_scale=0.1):
  """Compensate lane line z coordinates for camera height difference.

  Only z is corrected (z' = z + delta_height); x and y are unchanged.
  This breaks projection invariance: v' = fy*(z+dh)/x ≠ fy*z/x,
  so compensated lines appear at different vertical positions in the 2D image.

  Args:
    lane_lines_xyz: list of Nx3 float32 arrays [(x, y, z) per point], one per lane line.
    lane_line_probs: list of floats, existence probability per lane line.
    actual_height: actual camera mounting height in meters (ground truth from CLI).

  Returns:
    tuple: (compensated_lines, model_height, delta_height)
      compensated_lines: list of Nx3 float32 arrays with corrected z.
      model_height: estimated model perceived camera height (meters).
      delta_height: correction applied (meters), = actual_height - model_height.
  """
  # Estimate model's perceived camera height from lane line z values.
  z_samples = []
  for pts, prob in zip(lane_lines_xyz, lane_line_probs, strict=True):
    if prob > _HEIGHT_MIN_PROB and pts.shape[0] > _HEIGHT_Z_IDX_END:
      z_samples.append(pts[_HEIGHT_Z_IDX_START:_HEIGHT_Z_IDX_END, 2])

  if len(z_samples) == 0:
    return lane_lines_xyz, actual_height, 0.0

  model_height = float(np.median(np.concatenate(z_samples)))
  if model_height < _HEIGHT_MIN_Z:
    return lane_lines_xyz, model_height, 0.0

  delta_height = actual_height - model_height
  compensated = []
  for i, pts in enumerate(lane_lines_xyz):
    new_pts = pts.copy()
    if i < 2:  # left lanes (0=far-left, 1=near-left)
      dh = delta_height * left_scale
    else:       # right lanes (2=near-right, 3=far-right)
      dh = delta_height * right_scale
    new_pts[:, 2] = pts[:, 2] + dh
    #safe_z = np.maximum(pts[:, 2], _HEIGHT_MIN_Z)
    #scale = (safe_z + dh) / safe_z
    #new_pts[:, 1] = pts[:, 1] * scale
    compensated.append(new_pts)

  return compensated, model_height, delta_height


class Visualizer:
  """Draw perception results on camera images, optionally save video."""

  def __init__(self, save_video_path='', no_display=False, source_fps=20.0, actual_height=0.0, show_bev=False):
    self.actual_height = actual_height
    self.show_bev = show_bev or actual_height > 0
    self._model_height = 0.0
    self._delta_height = 0.0
    # Stored for BEV rendering (set each frame by _draw_lane_lines)
    self._lane_lines_xyz = []
    self._lane_probs = []
    self._comp_lines = []
    # GT and evaluation state
    self._gt_lines = None
    self._gt_probs = None
    self._eval_metrics = None
    self.no_display = no_display
    self.writer = None
    if save_video_path:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      self.writer = cv2.VideoWriter(save_video_path, fourcc, source_fps, (UI_W, UI_H))
      print(f"Video recording to {save_video_path} at {source_fps} FPS, {UI_W}x{UI_H}")

    if not no_display:
      cv2.namedWindow('dashcam', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('dashcam', UI_W, UI_H)

  def draw(self, frame_rgb, model_msg, fcam_intrinsics_3x3, rpyCalib,
           camera_height, vehicle_speed, cal_status, valid_blocks, cal_perc, fps,
           gt_lines=None, gt_probs=None, eval_metrics=None):
    """Draw all perception results on frame and display/record.

    Args:
      frame_rgb: HxWx3 RGB image from Carla camera.
      model_msg: cereal modelV2 message object, or None if no model output yet.
      fcam_intrinsics_3x3: 3x3 camera intrinsic matrix.
      rpyCalib: [roll, pitch, yaw] calibration in radians.
      camera_height: camera height in meters.
      vehicle_speed: ego vehicle speed in m/s.
      cal_status: calibration status string (e.g. 'calibrated').
      valid_blocks: number of valid calibration blocks.
      cal_perc: calibration percentage (0-100).
      fps: current FPS.
      gt_lines: list of 4 arrays (33x3), GT lane lines in calibrated frame, or None.
      gt_probs: list of 4 floats, GT lane existence probabilities, or None.
      eval_metrics: dict of evaluation metrics for this frame, or None.

    Returns True if should continue, False if user pressed 'q'.
    """
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if model_msg is not None:
      self._draw_lane_lines(img, model_msg, fcam_intrinsics_3x3, rpyCalib)
      self._draw_road_edges(img, model_msg, fcam_intrinsics_3x3, rpyCalib)
      self._draw_lead(img, model_msg, fcam_intrinsics_3x3, rpyCalib, camera_height)

    # Reset GT state each frame (caller provides None when near junction or non-eval frame)
    self._gt_lines = gt_lines
    self._gt_probs = gt_probs
    self._eval_metrics = eval_metrics

    if gt_lines is not None and gt_probs is not None:
      self._draw_gt_lane_lines(img, gt_lines, gt_probs, fcam_intrinsics_3x3, rpyCalib)

    # Blue crosshair at image center
    cx, cy = W // 2, H // 2
    cross_size = 20
    cv2.line(img, (cx - cross_size, cy), (cx + cross_size, cy), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - cross_size), (cx, cy + cross_size), (255, 0, 0), 1, cv2.LINE_AA)

    # Zoom 1.1x then center-crop to UI size (matching openpilot UI)
    zoomed_w, zoomed_h = int(W * ZOOM), int(H * ZOOM)
    zoomed = cv2.resize(img, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)
    x0 = (zoomed_w - UI_W) // 2
    y0 = (zoomed_h - UI_H) // 2
    display = zoomed[y0:y0 + UI_H, x0:x0 + UI_W]

    # Draw HUD on final display (after crop, so it's always visible)
    if self.show_bev:
      self._draw_bev_panel(display)
    self._draw_info_panel(display, model_msg, vehicle_speed, rpyCalib,
                          cal_status, valid_blocks, cal_perc, camera_height, fps)
    if self._eval_metrics is not None:
      self._draw_eval_panel(display)

    if self.writer is not None:
      self.writer.write(display)

    if not self.no_display:
      cv2.imshow('dashcam', display)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q') or key == 27:  # q or ESC
        return False

    return True

  def _draw_lane_lines(self, img, model, K, rpyCalib):
    """Draw 4 lane lines as filled polygons with alpha blending.

    Original lines in green; when height compensation is active (z-only),
    compensated lines in cyan are drawn on top. Since only z changes,
    the 2D vertical position shifts while horizontal stays the same.
    """
    transform = _build_transform(K, rpyCalib)
    lane_probs = list(model.laneLineProbs)

    # Extract lane line data as numpy arrays
    lane_lines_xyz = []
    for ll in model.laneLines:
      lane_lines_xyz.append(np.array([ll.x, ll.y, ll.z], dtype=np.float32).T)

    # Draw parameters (shared by original and compensated)
    path_xs = lane_lines_xyz[0][:, 0] if len(lane_lines_xyz) > 0 else np.array([])
    max_distance = np.clip(path_xs[-1] if len(path_xs) > 0 else 0, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = _get_path_length_idx(path_xs, max_distance)

    # Draw original lane lines (green)
    for i, pts in enumerate(lane_lines_xyz):
      prob = lane_probs[i]
      if prob < 0.01:
        continue
      y_off = 0.025 * prob
      polygon = _map_line_to_polygon(pts, y_off, 0.0, max_idx, max_distance, transform)
      if len(polygon) < 3:
        continue
      alpha = float(np.clip(prob, 0.0, 0.7))
      _draw_polygon_alpha(img, polygon, (64, 255, 0), alpha)  # green

    # Compute height compensation and draw compensated lines (cyan) on perspective view
    # Lane line order: 0=far-left, 1=near-left, 2=near-right, 3=far-right
    self._lane_lines_xyz = lane_lines_xyz
    self._lane_probs = lane_probs
    self._comp_lines = []
    if self.actual_height > 0:
      comp_lines, self._model_height, self._delta_height = compensate_lane_lines_for_height(
        lane_lines_xyz, lane_probs, self.actual_height)
      if abs(self._delta_height) > 0.05:
        self._comp_lines = comp_lines
        for i, pts in enumerate(comp_lines):
          prob = lane_probs[i]
          if prob < 0.01:
            continue
          y_off = 0.025 * prob
          polygon = _map_line_to_polygon(pts, y_off, 0.0, max_idx, max_distance, transform)
          if len(polygon) < 3:
            continue
          alpha = float(np.clip(prob, 0.0, 0.7))
          _draw_polygon_alpha(img, polygon, (255, 255, 0), alpha)  # cyan

  def _draw_road_edges(self, img, model, K, rpyCalib):
    """Draw 2 road edges as filled polygons with alpha blending.

    Aligned with model_renderer.py: polygon fill, alpha = clip(1 - std, 0, 1),
    fixed width = 0.025m, red color.
    """
    transform = _build_transform(K, rpyCalib)
    path_xs = np.array(model.laneLines[0].x) if len(model.laneLines) > 0 else np.array([])
    max_distance = np.clip(path_xs[-1] if len(path_xs) > 0 else 0, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = _get_path_length_idx(path_xs, max_distance)

    road_stds = list(model.roadEdgeStds)
    for i, re in enumerate(model.roadEdges):
      std = road_stds[i]
      alpha = float(np.clip(1.0 - std, 0.0, 1.0))
      if alpha < 0.01:
        continue

      points_3d = np.array([re.x, re.y, re.z], dtype=np.float32).T  # Nx3
      y_off = 0.025  # fixed width in meters
      polygon = _map_line_to_polygon(points_3d, y_off, 0.0, max_idx, max_distance, transform)
      if len(polygon) < 3:
        continue

      color_bgr = (0, 0, 255)  # BGR for red (aligned with rl.Color(255, 0, 0))
      _draw_polygon_alpha(img, polygon, color_bgr, alpha)

  def _draw_lead(self, img, model, K, rpyCalib, height):
    """Draw lead vehicle chevron indicator, aligned with model_renderer.py.

    Renders a yellow glow triangle (outer) and a red chevron triangle (inner)
    with fill alpha based on distance and relative speed.
    """
    if len(model.leadsV3) == 0:
      return

    lead = model.leadsV3[0]
    if lead.prob < 0.3:
      return

    x_dist = float(lead.x[0])
    y_offset = float(lead.y[0])

    if x_dist < 1.0 or x_dist > 200.0:
      return

    # Project lead position to image
    pts = project_points_to_image(
      np.array([x_dist]), np.array([y_offset]), np.array([height]),
      K, rpyCalib)

    if np.isnan(pts[0]).any():
      return

    x, y = float(pts[0, 0]), float(pts[0, 1])

    # Compute chevron size (aligned with model_renderer.py:_update_lead_vehicle)
    sz = np.clip((25 * 30) / (x_dist / 3 + 30), 15.0, 30.0) * 2.35
    x = np.clip(x, 0.0, W - sz / 2)
    y = min(y, H - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    # Glow triangle (outer, yellow) — aligned with model_renderer.py
    glow = np.array([
      [x + sz * 1.35 + g_xo, y + sz + g_yo],
      [x, y - g_yo],
      [x - sz * 1.35 - g_xo, y + sz + g_yo],
    ], dtype=np.int32)

    # Chevron triangle (inner, red)
    chevron = np.array([
      [x + sz * 1.25, y + sz],
      [x, y],
      [x - sz * 1.25, y + sz],
    ], dtype=np.int32)

    # Fill alpha based on distance and relative speed
    speed_buff, lead_buff = 10.0, 40.0
    v_rel = float(lead.v[0]) if len(lead.v) > 0 else 0.0
    fill_alpha = 0.0
    if x_dist < lead_buff:
      fill_alpha = 1.0 - (x_dist / lead_buff)
      if v_rel < 0:
        fill_alpha += -v_rel / speed_buff
      fill_alpha = min(fill_alpha, 1.0)

    # Draw glow (yellow, fully opaque) — BGR for rl.Color(218, 202, 37)
    cv2.fillPoly(img, [glow], (37, 202, 218), cv2.LINE_AA)

    # Draw chevron (red, alpha blended) — BGR for rl.Color(201, 34, 49)
    if fill_alpha > 0.01:
      _draw_polygon_alpha(img, chevron, (49, 34, 201), fill_alpha)

    # Label
    label = f"{x_dist:.0f}m v:{v_rel:+.1f}"
    label_x = int(x + sz * 1.35 + g_xo + 5)
    label_y = int(y + sz / 2)
    cv2.putText(img, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (37, 202, 218), 2, cv2.LINE_AA)

  def _draw_bev_panel(self, img):
    """Draw bird's eye view panel showing lane lines from above.

    When height compensation is active, shows both original (green) and
    compensated (cyan) lane lines. This is the correct way to visualize
    compensation effects — perspective projection is invariant to the
    compensation transform, so the difference is only visible in BEV.
    """
    if len(self._lane_lines_xyz) == 0:
      return

    bev_w, bev_h = _BEV_W, _BEV_H
    # Position: bottom-right corner of display
    x0 = img.shape[1] - bev_w - _BEV_MARGIN
    y0 = img.shape[0] - bev_h - _BEV_MARGIN

    # Semi-transparent black background
    roi = img[y0:y0 + bev_h, x0:x0 + bev_w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (bev_w, bev_h), (0, 0, 0), -1)
    img[y0:y0 + bev_h, x0:x0 + bev_w] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)

    # Coordinate mapping: 3D (x=fwd, y=right) -> BEV pixel
    scale_x = bev_h / _BEV_X_MAX          # px per meter forward
    scale_y = bev_w / (2 * _BEV_Y_HALF)   # px per meter lateral
    cx_bev = bev_w // 2                    # lateral center

    def to_bev(x_fwd, y_lat):
      """Map 3D forward/lateral to BEV pixel coords (relative to panel)."""
      px = cx_bev + y_lat * scale_y
      py = bev_h - x_fwd * scale_x  # forward = up
      return int(np.clip(px, 0, bev_w - 1)), int(np.clip(py, 0, bev_h - 1))

    # Draw grid lines
    grid_color = (60, 60, 60)
    for dist in [20, 40, 60]:
      _, gy = to_bev(dist, 0)
      cv2.line(img, (x0, y0 + gy), (x0 + bev_w, y0 + gy), grid_color, 1)
      cv2.putText(img, f"{dist}m", (x0 + 3, y0 + gy - 3),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
    # Center line (ego forward)
    cv2.line(img, (x0 + cx_bev, y0), (x0 + cx_bev, y0 + bev_h), grid_color, 1)

    # Draw ego vehicle marker
    ego_bx, ego_by = to_bev(0, 0)
    cv2.circle(img, (x0 + ego_bx, y0 + ego_by - 3), 5, (255, 255, 255), -1)

    # Helper: draw lane line polyline on BEV
    def draw_bev_line(pts_3d, prob, color, thickness):
      if prob < 0.01:
        return
      xs, ys = pts_3d[:, 0], pts_3d[:, 1]
      # Filter to valid forward range
      mask = (xs > 0) & (xs < _BEV_X_MAX) & (np.abs(ys) < _BEV_Y_HALF)
      if np.sum(mask) < 2:
        return
      bev_pts = []
      for xi, yi in zip(xs[mask], ys[mask], strict=True):
        bx, by = to_bev(xi, yi)
        bev_pts.append([x0 + bx, y0 + by])
      bev_pts = np.array(bev_pts, dtype=np.int32)
      cv2.polylines(img, [bev_pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Draw original lane lines (green)
    for i, pts in enumerate(self._lane_lines_xyz):
      draw_bev_line(pts, self._lane_probs[i], (64, 255, 0), 2)

    # Draw compensated lane lines (cyan), if active
    if len(self._comp_lines) > 0:
      for i, pts in enumerate(self._comp_lines):
        draw_bev_line(pts, self._lane_probs[i], (255, 255, 0), 2)

    # Draw GT lane lines (magenta) on BEV
    if self._gt_lines is not None and self._gt_probs is not None:
      for i, pts in enumerate(self._gt_lines):
        draw_bev_line(pts, self._gt_probs[i], (255, 0, 255), 2)

    # Title and legend
    cv2.putText(img, "BEV (top-down)", (x0 + 5, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    legend_y = y0 + 32
    has_comp = len(self._comp_lines) > 0
    has_gt = self._gt_lines is not None
    if has_comp or has_gt:
      cv2.putText(img, "--- model", (x0 + 5, legend_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (64, 255, 0), 1, cv2.LINE_AA)
      col_x = 100
      if has_comp:
        cv2.putText(img, "--- comp", (x0 + col_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        col_x += 90
      if has_gt:
        cv2.putText(img, "--- GT", (x0 + col_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
      if has_comp:
        cv2.putText(img, f"dh={self._delta_height:+.2f}m", (x0 + 5, legend_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

  def _draw_gt_lane_lines(self, img, gt_lines, gt_probs, K, rpyCalib):
    """Draw GT lane lines as magenta polylines on perspective view.

    Uses direct point projection + polylines instead of _map_line_to_polygon
    to avoid polygon closure artifacts from near-camera singularities
    (X_IDXS starts at x=0 where projection depth ≈ 0).
    """
    for i, pts in enumerate(gt_lines):
      if gt_probs[i] < 0.5:
        continue
      # Keep only points with valid data in well-conditioned projection range.
      # GT points beyond sampling coverage have NaN y/z — exclude them.
      mask = (pts[:, 0] >= MIN_DRAW_DISTANCE) & (pts[:, 0] <= 100.0) & \
             ~np.isnan(pts[:, 1]) & ~np.isnan(pts[:, 2])
      valid_pts = pts[mask]
      if valid_pts.shape[0] < 2:
        continue

      uv = project_points_to_image(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], K, rpyCalib)

      # Filter to valid projections within image bounds
      good = ~np.isnan(uv).any(axis=1) & \
             (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < H)

      # Split into contiguous segments at gaps to avoid spurious lines
      # connecting non-adjacent points (e.g. lane exits then re-enters image on curves)
      indices = np.where(good)[0]
      if len(indices) < 2:
        continue
      breaks = np.where(np.diff(indices) > 1)[0] + 1
      for seg_idx in np.split(indices, breaks):
        if len(seg_idx) < 2:
          continue
        seg_uv = uv[seg_idx].astype(np.int32)
        cv2.polylines(img, [seg_uv], isClosed=False, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)

  def _draw_eval_panel(self, img):
    """Draw evaluation metrics panel in the top-right corner."""
    metrics = self._eval_metrics
    if metrics is None:
      return

    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_w, panel_h = 320, 160
    x0 = img.shape[1] - panel_w - _BEV_MARGIN
    y0 = _BEV_MARGIN

    # Semi-transparent background
    roi = img[y0:y0 + panel_h, x0:x0 + panel_w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    img[y0:y0 + panel_h, x0:x0 + panel_w] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)

    def _mae_color(val):
      """Color-code MAE: green < 0.3m, yellow < 0.5m, red >= 0.5m."""
      if val < 0.3:
        return (0, 220, 0)
      elif val < 0.5:
        return (0, 220, 220)
      return (0, 0, 220)

    y = y0 + 20
    dy = 22
    scale = 0.45

    cv2.putText(img, "Lane Eval", (x0 + 5, y), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    y += dy

    # Near-left / near-right MAE
    for name, label in [('near_left_all_mae', 'L-MAE'), ('near_right_all_mae', 'R-MAE')]:
      val = metrics.get(name)
      if val is not None:
        color = _mae_color(val)
        cv2.putText(img, f"{label}: {val:.3f}m", (x0 + 5, y), font, scale, color, 1, cv2.LINE_AA)
        y += dy

    # Overall MAE
    val = metrics.get('overall_mae')
    if val is not None:
      color = _mae_color(val)
      cv2.putText(img, f"Overall: {val:.3f}m", (x0 + 5, y), font, scale, color, 1, cv2.LINE_AA)
      y += dy

    # Width MAE
    val = metrics.get('width_mae')
    if val is not None:
      color = _mae_color(val)
      cv2.putText(img, f"Width: {val:.3f}m", (x0 + 5, y), font, scale, color, 1, cv2.LINE_AA)
      y += dy

    # Recall
    val = metrics.get('recall')
    if val is not None:
      color = (0, 220, 0) if val > 0.8 else (0, 220, 220) if val > 0.5 else (0, 0, 220)
      cv2.putText(img, f"Recall: {val:.0%}", (x0 + 5, y), font, scale, color, 1, cv2.LINE_AA)

  def _draw_info_panel(self, img, model, speed, rpyCalib, cal_status, valid_blocks, cal_perc, height, fps):
    """Draw semi-transparent info panel with calibration progress bar."""
    # Background
    panel_h = 260 if abs(self._delta_height) > 0.05 else 230
    panel_w = 550
    overlay = img[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    img[0:panel_h, 0:panel_w] = cv2.addWeighted(overlay, 0.6, img[0:panel_h, 0:panel_w], 0.4, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    gray = (160, 160, 160)
    y = 30
    dy = 28
    scale = 0.65

    # Line 1: FPS + Speed
    cv2.putText(img, f"FPS: {fps:.1f}", (10, y), font, scale, white, 1, cv2.LINE_AA)
    cv2.putText(img, f"Speed: {speed:.1f} m/s ({speed*3.6:.1f} km/h)", (180, y), font, scale, white, 1, cv2.LINE_AA)
    y += dy

    # Line 2: Calibration status (color-coded)
    status_str = str(cal_status).upper()
    status_color = _STATUS_COLORS.get(str(cal_status), white)
    cv2.putText(img, "Calib: ", (10, y), font, scale, gray, 1, cv2.LINE_AA)
    cv2.putText(img, status_str, (105, y), font, scale, status_color, 2, cv2.LINE_AA)
    y += dy

    # Line 3: Progress bar
    bar_x, bar_w, bar_h = 10, 200, 18
    # Background
    cv2.rectangle(img, (bar_x, y - 2), (bar_x + bar_w, y - 2 + bar_h), (60, 60, 60), -1)
    # Fill
    fill_w = int(bar_w * min(cal_perc, 100) / 100)
    if fill_w > 0:
      cv2.rectangle(img, (bar_x, y - 2), (bar_x + fill_w, y - 2 + bar_h), status_color, -1)
    # Border
    cv2.rectangle(img, (bar_x, y - 2), (bar_x + bar_w, y - 2 + bar_h), (120, 120, 120), 1)
    # Percentage + blocks text
    perc_text = f"{cal_perc}%  ({valid_blocks}/{INPUTS_NEEDED} blocks)"
    cv2.putText(img, perc_text, (bar_x + bar_w + 10, y + 12), font, 0.55, white, 1, cv2.LINE_AA)
    y += bar_h + 12

    # Line 4: Calibration results (pitch, yaw, height)
    pitch_d = math.degrees(rpyCalib[1])
    yaw_d = math.degrees(rpyCalib[2])
    cv2.putText(img, f"pitch={pitch_d:+.2f}deg  yaw={yaw_d:+.2f}deg  height={height:.2f}m",
                (10, y), font, scale, white, 1, cv2.LINE_AA)
    y += dy

    # Line 4b: Height compensation (when active)
    if abs(self._delta_height) > 0.05:
      cv2.putText(img, f"HeightComp: model={self._model_height:.2f}m actual={self.actual_height:.2f}m "
                        + f"dh={self._delta_height:+.2f}m",
                  (10, y), font, scale, (255, 255, 0), 1, cv2.LINE_AA)  # cyan
      y += dy

    # Line 5: Lead info (optional)
    if model is not None and len(model.leadsV3) > 0:
      lead = model.leadsV3[0]
      if lead.prob > 0.3:
        x_dist = float(lead.x[0])
        v_rel = float(lead.v[0]) if len(lead.v) > 0 else 0.0
        cv2.putText(img, f"Lead: {x_dist:.1f}m  v_rel={v_rel:+.1f} m/s  prob={lead.prob:.2f}",
                    (10, y), font, scale, white, 1, cv2.LINE_AA)

  def close(self):
    """Release resources (idempotent)."""
    if self.writer is not None:
      self.writer.release()
      self.writer = None
      print("Video saved")
    if not self.no_display:
      cv2.destroyAllWindows()
