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

# Match openpilot UI: 2160x1080, camera zoomed to fill then center-cropped
UI_W, UI_H = 2160, 1080
ZOOM = max(UI_W / W, UI_H / H)  # ~1.12, fill width then crop height

# Distance clipping constants (aligned with model_renderer.py)
CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0

# Calibration constants for info panel display
INPUTS_NEEDED = 5


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


class Visualizer:
  """Draw perception results on camera images, optionally save video."""

  def __init__(self, save_video_path='', no_display=False, source_fps=20.0):
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
           camera_height, vehicle_speed, cal_status, valid_blocks, fps):
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
      fps: current FPS.

    Returns True if should continue, False if user pressed 'q'.
    """
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if model_msg is not None:
      self._draw_lane_lines(img, model_msg, fcam_intrinsics_3x3, rpyCalib)
      self._draw_road_edges(img, model_msg, fcam_intrinsics_3x3, rpyCalib)
      self._draw_lead(img, model_msg, fcam_intrinsics_3x3, rpyCalib, camera_height)

    self._draw_info_panel(img, model_msg, vehicle_speed, rpyCalib,
                          cal_status, valid_blocks, camera_height, fps)

    # Zoom 1.1x then center-crop to UI size (matching openpilot UI)
    zoomed_w, zoomed_h = int(W * ZOOM), int(H * ZOOM)
    zoomed = cv2.resize(img, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)
    x0 = (zoomed_w - UI_W) // 2
    y0 = (zoomed_h - UI_H) // 2
    display = zoomed[y0:y0 + UI_H, x0:x0 + UI_W]

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

    Aligned with model_renderer.py: polygon fill, alpha = clip(prob, 0, 0.7),
    width = 0.025 * prob (meters in 3D), green color.
    """
    transform = _build_transform(K, rpyCalib)
    path_xs = np.array(model.laneLines[0].x) if len(model.laneLines) > 0 else np.array([])
    max_distance = np.clip(path_xs[-1] if len(path_xs) > 0 else 0, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = _get_path_length_idx(path_xs, max_distance)

    lane_probs = list(model.laneLineProbs)
    for i, ll in enumerate(model.laneLines):
      prob = lane_probs[i]
      if prob < 0.01:
        continue

      points_3d = np.array([ll.x, ll.y, ll.z], dtype=np.float32).T  # Nx3
      y_off = 0.025 * prob  # 3D width in meters
      polygon = _map_line_to_polygon(points_3d, y_off, 0.0, max_idx, max_distance, transform)
      if len(polygon) < 3:
        continue

      alpha = float(np.clip(prob, 0.0, 0.7))
      color_bgr = (64, 255, 0)  # BGR for green (aligned with rl.Color(0, 255, 64))
      _draw_polygon_alpha(img, polygon, color_bgr, alpha)

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
    """Draw lead vehicle detection from modelV2.leadsV3."""
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
    # In calibration frame z=down, so road level is z=camera_height
    pts = project_points_to_image(
      np.array([x_dist]), np.array([y_offset]), np.array([height]),
      K, rpyCalib)

    if np.isnan(pts[0]).any():
      return

    u, v = int(pts[0, 0]), int(pts[0, 1])
    if 0 <= u < W and 0 <= v < H:
      # Draw marker
      radius = max(10, int(600 / max(x_dist, 1)))
      color = (255, 255, 0)  # cyan in BGR
      cv2.circle(img, (u, v), radius, color, 2, cv2.LINE_AA)

      # Label
      v_rel = float(lead.v[0]) if len(lead.v) > 0 else 0.0
      label = f"{x_dist:.0f}m v:{v_rel:+.1f}"
      cv2.putText(img, label, (u + radius + 5, v + 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

  def _draw_info_panel(self, img, model, speed, rpyCalib, cal_status, valid_blocks, height, fps):
    """Draw semi-transparent info panel in top-left corner."""
    # Background
    panel_h = 200
    panel_w = 550
    overlay = img[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    img[0:panel_h, 0:panel_w] = cv2.addWeighted(overlay, 0.6, img[0:panel_h, 0:panel_w], 0.4, 0)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    y0 = 30
    dy = 28
    scale = 0.65

    status_str = str(cal_status).upper()
    lines = [
      f"FPS: {fps:.1f}",
      f"Speed: {speed:.1f} m/s ({speed*3.6:.1f} km/h)",
      f"Calib: pitch={math.degrees(rpyCalib[1]):.2f} yaw={math.degrees(rpyCalib[2]):.2f} [{status_str}]",
      f"Calib blocks: {valid_blocks}/{INPUTS_NEEDED}",
      f"Height: {height:.2f}m",
    ]

    if model is not None:
      # Lead info from leadsV3
      if len(model.leadsV3) > 0:
        lead = model.leadsV3[0]
        if lead.prob > 0.3:
          x_dist = float(lead.x[0])
          v_rel = float(lead.v[0]) if len(lead.v) > 0 else 0.0
          lines.append(f"Lead: {x_dist:.1f}m  v_rel={v_rel:+.1f} m/s  prob={lead.prob:.2f}")

    for i, line in enumerate(lines):
      cv2.putText(img, line, (10, y0 + i * dy), font, scale, color, 1, cv2.LINE_AA)

  def close(self):
    """Release resources (idempotent)."""
    if self.writer is not None:
      self.writer.release()
      self.writer = None
      print("Video saved")
    if not self.no_display:
      cv2.destroyAllWindows()
