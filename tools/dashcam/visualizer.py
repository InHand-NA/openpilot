"""Visualization: draw perception results on camera images, display and record video."""

import math
import cv2
import numpy as np

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.calibrator import INPUTS_NEEDED

from openpilot.tools.dashcam.carla_world import W, H

# Match openpilot UI: 2160x1080, camera zoomed to fill then center-cropped
UI_W, UI_H = 2160, 1080
ZOOM = max(UI_W / W, UI_H / H)  # ~1.12, fill width then crop height


def project_points_to_image(xs, ys, zs, intrinsics_3x3, rpyCalib):
  """Project calibration-frame 3D points to image pixel coordinates.

  Calibration frame: x=forward, y=right, z=down.
  Model outputs lane_lines/road_edges with z ≈ camera_height (road surface).
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


def _filter_points(points):
  """Filter Nx2 pixel points: remove NaN and out-of-bounds."""
  valid = ~np.isnan(points).any(axis=1)
  pts = points[valid].astype(np.int32)
  margin = 200
  in_bounds = ((pts[:, 0] > -margin) & (pts[:, 0] < W + margin) &
               (pts[:, 1] > -margin) & (pts[:, 1] < H + margin))
  return pts[in_bounds]


def draw_path(img, points, color, thickness=2):
  """Draw a solid polyline on image from Nx2 pixel coordinates."""
  pts = _filter_points(points)
  if len(pts) >= 2:
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_dashed_path(img, points, color, thickness=2, dash_px=20, gap_px=15):
  """Draw a dashed polyline on image from Nx2 pixel coordinates."""
  pts = _filter_points(points)
  if len(pts) < 2:
    return
  # Walk along the polyline, alternating dash/gap
  drawing = True
  remaining = dash_px
  seg_start = 0
  for i in range(1, len(pts)):
    dx = float(pts[i, 0] - pts[i - 1, 0])
    dy = float(pts[i, 1] - pts[i - 1, 1])
    seg_len = math.hypot(dx, dy)
    if seg_len < 1:
      continue
    consumed = 0.0
    while consumed < seg_len:
      step = min(remaining, seg_len - consumed)
      t0 = consumed / seg_len
      t1 = (consumed + step) / seg_len
      p0 = (int(pts[i - 1, 0] + dx * t0), int(pts[i - 1, 1] + dy * t0))
      p1 = (int(pts[i - 1, 0] + dx * t1), int(pts[i - 1, 1] + dy * t1))
      if drawing:
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
      consumed += step
      remaining -= step
      if remaining <= 0:
        drawing = not drawing
        remaining = gap_px if not drawing else dash_px


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

    self.x_idxs = np.array(ModelConstants.X_IDXS)

  def draw(self, frame_rgb, vision_output, fcam_intrinsics_3x3, rpyCalib,
           camera_height, vehicle_speed, calibrator_status, valid_blocks, fps):
    """Draw all perception results on frame and display/record.

    Returns True if should continue, False if user pressed 'q'.
    """
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if vision_output is not None:
      self._draw_lane_lines(img, vision_output, fcam_intrinsics_3x3, rpyCalib, camera_height)
      self._draw_road_edges(img, vision_output, fcam_intrinsics_3x3, rpyCalib, camera_height)
      self._draw_lead(img, vision_output, fcam_intrinsics_3x3, rpyCalib, camera_height)

    self._draw_info_panel(img, vision_output, vehicle_speed, rpyCalib,
                          calibrator_status, valid_blocks, camera_height, fps)

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

  def _draw_lane_lines(self, img, out, K, rpyCalib, height):
    """Draw 4 lane lines: inner (1,2) green solid, outer (0,3) green dashed."""
    lane_lines = out['lane_lines'][0]  # (4, 33, 2): y, z at each x
    raw_probs = out['lane_lines_prob'][0]  # (8,) - take every other for 4 lane probs
    lane_probs = raw_probs[1::2]  # indices 1,3,5,7 = 4 lane line probabilities
    color = (0, 220, 0)  # green in BGR

    for i in range(4):
      prob = float(lane_probs[i])
      if prob < 0.3:
        continue
      ys = lane_lines[i, :, 0]  # lateral offset
      zs = lane_lines[i, :, 1]  # height offset
      pixels = project_points_to_image(self.x_idxs, ys, zs, K, rpyCalib)

      if i == 1 or i == 2:  # inner lanes (current lane boundaries) - solid
        draw_path(img, pixels, color, thickness=3)
      else:  # outer lanes (0, 3) - dashed
        draw_dashed_path(img, pixels, color, thickness=3)

  def _draw_road_edges(self, img, out, K, rpyCalib, height):
    """Draw 2 road edges in red."""
    road_edges = out['road_edges'][0]  # (2, 33, 2)
    color = (0, 0, 220)  # red in BGR

    for i in range(2):
      ys = road_edges[i, :, 0]
      zs = road_edges[i, :, 1]
      pixels = project_points_to_image(self.x_idxs, ys, zs, K, rpyCalib)
      draw_path(img, pixels, color, thickness=2)

  def _draw_lead(self, img, out, K, rpyCalib, height):
    """Draw lead vehicle detection."""
    lead = out['lead']  # (1, 6, 4) or (1, 3, 6, 4) depending on MHP
    lead_prob = out['lead_prob'][0]  # (3,) probabilities

    if lead.ndim == 4:
      # MHP format: (1, n_selections, traj_len, 4)
      lead_data = lead[0, 0]  # best hypothesis, first time step
    else:
      # (1, traj_len, 4)
      lead_data = lead[0]

    prob = float(lead_prob[0])
    if prob < 0.3:
      return

    # lead_data[0] = (x_dist, y_offset, v_rel, a_rel) at t=0
    x_dist = float(lead_data[0, 0])
    y_offset = float(lead_data[0, 1])
    v_rel = float(lead_data[0, 2])

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
      alpha = np.clip(prob, 0.3, 1.0)
      color = (255, 255, 0)  # cyan in BGR
      cv2.circle(img, (u, v), radius, color, 2, cv2.LINE_AA)

      # Label
      label = f"{x_dist:.0f}m v:{v_rel:+.1f}"
      cv2.putText(img, label, (u + radius + 5, v + 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

  def _draw_info_panel(self, img, out, speed, rpyCalib, cal_status, valid_blocks, height, fps):
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

    lines = [
      f"FPS: {fps:.1f}",
      f"Speed: {speed:.1f} m/s ({speed*3.6:.1f} km/h)",
      f"Calib: pitch={math.degrees(rpyCalib[1]):.2f} yaw={math.degrees(rpyCalib[2]):.2f} [{cal_status.upper()}]",
      f"Calib blocks: {valid_blocks}/{INPUTS_NEEDED}",
      f"Height: {height:.2f}m",
    ]

    if out is not None:
      # Camera odometry
      trans = out['pose'][0, :3]
      lines.append(f"Cam Odom: vx={trans[0]:.2f} vy={trans[1]:.3f} vz={trans[2]:.3f} m/s")

      # Lead info
      lead_prob = out['lead_prob'][0]
      if lead_prob[0] > 0.3:
        lead = out['lead']
        if lead.ndim == 4:
          ld = lead[0, 0, 0]
        else:
          ld = lead[0, 0]
        lines.append(f"Lead: {ld[0]:.1f}m  v_rel={ld[2]:+.1f} m/s  prob={lead_prob[0]:.2f}")

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
