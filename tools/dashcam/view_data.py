"""Interactive training data viewer — browse .npz frames with GT overlays.

Usage:
  python tools/dashcam/view_data.py data/training/town04_h2.0/
  python tools/dashcam/view_data.py data/training/town04_h2.0/ --start 50

Keys:
  Left/Right    previous/next frame
  PageUp/Down   ±10 frames
  Home/End      first/last frame
  l             toggle lane lines
  e             toggle road edges
  v             toggle lead vehicles
  i             toggle info panel
  b             toggle BEV panel
  s             screenshot (PNG)
  q / ESC       quit
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.dashcam.visualizer import (
  W, H,
  _BEV_W, _BEV_H, _BEV_Y_HALF, _BEV_MARGIN,
  project_points_to_image, _build_transform,
  _map_line_to_polygon, _draw_polygon_alpha, _get_path_length_idx,
)

# Draw range constants (full X_IDXS coverage: 0-192m)
MIN_DRAW_DISTANCE = 5.0
MAX_DRAW_DISTANCE = 192.0
_BEV_X_MAX = 192.0

# Camera intrinsics for road-only fcam (pc/simulator matches Carla capture)
K = DEVICE_CAMERAS[("pc", "unknown")].fcam.intrinsics

# OpenCV key codes (Linux GTK backend)
KEY_RIGHT = 65363
KEY_LEFT = 65361
KEY_PGUP = 65365
KEY_PGDN = 65366
KEY_HOME = 65360
KEY_END = 65367


def _load_frame(path):
  """Load a single .npz frame and return its dict."""
  return dict(np.load(path, allow_pickle=True))


def _prob_color(prob):
  """BGR color by probability: green >= 0.8, yellow >= 0.3, red < 0.3."""
  if prob >= 0.8:
    return (0, 220, 0)
  if prob >= 0.3:
    return (0, 220, 220)
  return (0, 0, 220)


def _check_warnings(data):
  """Return list of warning strings for data quality issues."""
  warnings = []
  for key in ('lane_lines', 'lead', 'pose', 'road_edges'):
    if key in data and np.any(np.isnan(data[key])):
      warnings.append(f"NaN in {key}")
  probs = data.get('lane_lines_prob')
  if probs is not None and np.all(probs == 0):
    warnings.append("all lane_lines_prob = 0")
  lead_probs = data.get('lead_prob')
  if lead_probs is not None and np.all(lead_probs == 0):
    warnings.append("all lead_prob = 0")
  return warnings


def _draw_lane_lines(img, data, rpyCalib):
  """Draw GT lane lines on perspective image."""
  lane_lines = data['lane_lines']       # (4, 33, 3)
  lane_probs = data['lane_lines_prob']   # (4,)
  transform = _build_transform(K, rpyCalib)

  path_xs = lane_lines[0, :, 0]
  max_distance = np.clip(path_xs[-1] if len(path_xs) > 0 else 0, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(path_xs, max_distance)

  for i in range(4):
    prob = float(lane_probs[i])
    pts = lane_lines[i]  # (33, 3)
    color = _prob_color(prob)

    # --- filled polygon (low alpha for area sense) ---
    if prob > 0.01:
      y_off = 0.025 * prob
      polygon = _map_line_to_polygon(pts, y_off, 0.0, max_idx, max_distance, transform)
      if len(polygon) >= 3:
        _draw_polygon_alpha(img, polygon, color, 0.25)

    # --- polyline (like _draw_gt_lane_lines) ---
    if prob < 0.3:
      continue
    mask = ((pts[:, 0] >= MIN_DRAW_DISTANCE) & (pts[:, 0] <= MAX_DRAW_DISTANCE) &
            ~np.isnan(pts[:, 1]) & ~np.isnan(pts[:, 2]))
    valid_pts = pts[mask]
    if valid_pts.shape[0] < 2:
      continue

    uv = project_points_to_image(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], K, rpyCalib)
    good = (~np.isnan(uv).any(axis=1) &
            (uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H))
    indices = np.where(good)[0]
    if len(indices) < 2:
      continue
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    for seg_idx in np.split(indices, breaks):
      if len(seg_idx) < 2:
        continue
      seg_uv = uv[seg_idx].astype(np.int32)
      cv2.polylines(img, [seg_uv], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    # probability label near first visible point
    first_uv = uv[indices[0]]
    lx, ly = int(first_uv[0]) + 5, int(first_uv[1]) - 8
    cv2.putText(img, f"p={prob:.2f}", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# Road edge colors (BGR): left=magenta, right=magenta
_ROAD_EDGE_COLOR = (220, 0, 220)


def _draw_road_edges(img, data, rpyCalib):
  """Draw GT road edges on perspective image."""
  road_edges = data.get('road_edges')
  road_edge_probs = data.get('road_edges_prob')
  if road_edges is None or road_edge_probs is None:
    return

  transform = _build_transform(K, rpyCalib)
  path_xs = road_edges[0, :, 0]
  max_distance = np.clip(path_xs[-1] if len(path_xs) > 0 else 0, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(path_xs, max_distance)

  for i in range(2):
    prob = float(road_edge_probs[i])
    pts = road_edges[i]  # (33, 3)
    color = _ROAD_EDGE_COLOR

    # --- filled polygon (low alpha) ---
    if prob > 0.01:
      y_off = 0.025 * prob
      polygon = _map_line_to_polygon(pts, y_off, 0.0, max_idx, max_distance, transform)
      if len(polygon) >= 3:
        _draw_polygon_alpha(img, polygon, color, 0.2)

    # --- dashed polyline ---
    if prob < 0.3:
      continue
    mask = ((pts[:, 0] >= MIN_DRAW_DISTANCE) & (pts[:, 0] <= MAX_DRAW_DISTANCE) &
            ~np.isnan(pts[:, 1]) & ~np.isnan(pts[:, 2]))
    valid_pts = pts[mask]
    if valid_pts.shape[0] < 2:
      continue

    uv = project_points_to_image(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], K, rpyCalib)
    good = (~np.isnan(uv).any(axis=1) &
            (uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H))
    indices = np.where(good)[0]
    if len(indices) < 2:
      continue
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    for seg_idx in np.split(indices, breaks):
      if len(seg_idx) < 2:
        continue
      seg_uv = uv[seg_idx].astype(np.int32)
      cv2.polylines(img, [seg_uv], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    # probability label
    first_uv = uv[indices[0]]
    label = "RE" if i == 1 else "LE"
    lx, ly = int(first_uv[0]) + 5, int(first_uv[1]) - 8
    cv2.putText(img, f"{label} p={prob:.2f}", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# Lead vehicle colors per selection (BGR): yellow, orange, cyan
_LEAD_COLORS = [(0, 220, 220), (0, 140, 255), (255, 220, 0)]


def _draw_leads(img, data, rpyCalib, camera_height):
  """Draw lead vehicles (t=0) on perspective image."""
  leads = data['lead']           # (3, 6, 4) — x, y, v_rel, a
  lead_probs = data['lead_prob']  # (3,)

  for sel in range(3):
    prob = float(lead_probs[sel])
    if prob < 0.3:
      continue
    x_dist = float(leads[sel, 0, 0])
    y_off = float(leads[sel, 0, 1])
    v_rel = float(leads[sel, 0, 2])
    if x_dist < 1.0 or x_dist > 200.0:
      continue

    uv = project_points_to_image(
      np.array([x_dist]), np.array([y_off]), np.array([camera_height]),
      K, rpyCalib)
    if np.isnan(uv[0]).any():
      continue
    cx, cy = int(uv[0, 0]), int(uv[0, 1])
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
      continue

    color = _LEAD_COLORS[sel]
    radius = int(np.clip(800.0 / max(x_dist, 5.0), 8, 40))
    cv2.circle(img, (cx, cy), radius, color, 2, cv2.LINE_AA)
    label = f"#{sel} {x_dist:.1f}m v:{v_rel:+.1f}"
    cv2.putText(img, label, (cx + radius + 4, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_info_panel(img, data, frame_idx, total_frames):
  """Draw metadata info panel at top-left."""
  font = cv2.FONT_HERSHEY_SIMPLEX
  white = (255, 255, 255)

  lines = []
  lines.append(f"Frame: {frame_idx}/{total_frames - 1}")
  town = str(data.get('town', '?'))
  height = float(data.get('camera_height', 0))
  lines.append(f"Town: {town}  Height: {height:.2f}m")
  pitch = float(data.get('camera_pitch', 0))
  yaw = float(data.get('camera_yaw', 0))
  lines.append(f"Pitch: {pitch:.3f}rad ({np.degrees(pitch):.2f}deg)  Yaw: {yaw:.3f}rad ({np.degrees(yaw):.2f}deg)")
  v_ego = float(data.get('v_ego', 0))
  lines.append(f"Speed: {v_ego:.1f} m/s ({v_ego * 3.6:.1f} km/h)")
  probs = data.get('lane_lines_prob', np.zeros(4))
  lines.append(f"Lane probs: [{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}, {probs[3]:.2f}]")
  road_edge_probs = data.get('road_edges_prob', np.zeros(2))
  lines.append(f"Road edge probs: [{road_edge_probs[0]:.2f}, {road_edge_probs[1]:.2f}]")
  lead_probs = data.get('lead_prob', np.zeros(3))
  lines.append(f"Lead probs: [{lead_probs[0]:.2f}, {lead_probs[1]:.2f}, {lead_probs[2]:.2f}]")
  pose = data.get('pose')
  if pose is not None:
    lines.append(f"Pose: v=[{pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}] w=[{pose[3]:.4f},{pose[4]:.4f},{pose[5]:.4f}]")
  rt = data.get('road_transform')
  if rt is not None:
    lines.append(f"RoadTf: [{rt[0]:.2f},{rt[1]:.2f},{rt[2]:.2f},{rt[3]:.2f},{rt[4]:.2f},{rt[5]:.2f}]")

  dy = 26
  panel_h = dy * len(lines) + 12
  panel_w = 680
  # Semi-transparent background
  ph = min(panel_h, img.shape[0])
  pw = min(panel_w, img.shape[1])
  roi = img[0:ph, 0:pw]
  overlay = roi.copy()
  cv2.rectangle(overlay, (0, 0), (pw, ph), (0, 0, 0), -1)
  img[0:ph, 0:pw] = cv2.addWeighted(overlay, 0.65, roi, 0.35, 0)

  y = 22
  for line in lines:
    cv2.putText(img, line, (10, y), font, 0.55, white, 1, cv2.LINE_AA)
    y += dy


def _draw_bev_panel(img, data, camera_height):
  """Draw bird's eye view panel at bottom-right."""
  bev_w, bev_h = _BEV_W, _BEV_H
  x0 = img.shape[1] - bev_w - _BEV_MARGIN
  y0 = img.shape[0] - bev_h - _BEV_MARGIN
  if x0 < 0 or y0 < 0:
    return

  # Semi-transparent background
  roi = img[y0:y0 + bev_h, x0:x0 + bev_w]
  overlay = roi.copy()
  cv2.rectangle(overlay, (0, 0), (bev_w, bev_h), (0, 0, 0), -1)
  img[y0:y0 + bev_h, x0:x0 + bev_w] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)

  scale_x = bev_h / _BEV_X_MAX
  scale_y = bev_w / (2 * _BEV_Y_HALF)
  cx_bev = bev_w // 2

  def to_bev(x_fwd, y_lat):
    px = cx_bev + y_lat * scale_y
    py = bev_h - x_fwd * scale_x
    return int(np.clip(px, 0, bev_w - 1)), int(np.clip(py, 0, bev_h - 1))

  # Grid
  grid_color = (60, 60, 60)
  for dist in [50, 100, 150]:
    _, gy = to_bev(dist, 0)
    cv2.line(img, (x0, y0 + gy), (x0 + bev_w, y0 + gy), grid_color, 1)
    cv2.putText(img, f"{dist}m", (x0 + 3, y0 + gy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
  cv2.line(img, (x0 + cx_bev, y0), (x0 + cx_bev, y0 + bev_h), grid_color, 1)

  # Ego marker
  ego_bx, ego_by = to_bev(0, 0)
  cv2.circle(img, (x0 + ego_bx, y0 + ego_by - 3), 5, (255, 255, 255), -1)

  # Lane lines
  lane_lines = data['lane_lines']
  lane_probs = data['lane_lines_prob']
  for i in range(4):
    prob = float(lane_probs[i])
    if prob < 0.01:
      continue
    pts = lane_lines[i]
    xs, ys = pts[:, 0], pts[:, 1]
    mask = (xs > 0) & (xs < _BEV_X_MAX) & (np.abs(ys) < _BEV_Y_HALF) & ~np.isnan(ys)
    if np.sum(mask) < 2:
      continue
    bev_pts = []
    for xi, yi in zip(xs[mask], ys[mask], strict=True):
      bx, by = to_bev(xi, yi)
      bev_pts.append([x0 + bx, y0 + by])
    color = _prob_color(prob)
    cv2.polylines(img, [np.array(bev_pts, dtype=np.int32)],
                  isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

  # Road edges
  road_edges = data.get('road_edges')
  road_edge_probs = data.get('road_edges_prob')
  if road_edges is not None and road_edge_probs is not None:
    for i in range(2):
      prob = float(road_edge_probs[i])
      if prob < 0.01:
        continue
      pts = road_edges[i]
      xs, ys = pts[:, 0], pts[:, 1]
      mask = (xs > 0) & (xs < _BEV_X_MAX) & (np.abs(ys) < _BEV_Y_HALF) & ~np.isnan(ys)
      if np.sum(mask) < 2:
        continue
      bev_pts = []
      for xi, yi in zip(xs[mask], ys[mask], strict=True):
        bx, by = to_bev(xi, yi)
        bev_pts.append([x0 + bx, y0 + by])
      cv2.polylines(img, [np.array(bev_pts, dtype=np.int32)],
                    isClosed=False, color=_ROAD_EDGE_COLOR, thickness=2, lineType=cv2.LINE_AA)

  # Lead vehicles
  leads = data.get('lead')
  lead_probs = data.get('lead_prob')
  if leads is not None and lead_probs is not None:
    for sel in range(3):
      if float(lead_probs[sel]) < 0.3:
        continue
      lx = float(leads[sel, 0, 0])
      ly = float(leads[sel, 0, 1])
      if lx < 1.0 or lx > _BEV_X_MAX:
        continue
      bx, by = to_bev(lx, ly)
      cv2.circle(img, (x0 + bx, y0 + by), 5, _LEAD_COLORS[sel], -1)

  # Title
  cv2.putText(img, "BEV", (x0 + 5, y0 + 15),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_warnings(img, warnings):
  """Draw red warning banner at top of image."""
  if not warnings:
    return
  banner_h = 30 * len(warnings)
  cv2.rectangle(img, (0, 0), (img.shape[1], banner_h), (0, 0, 180), -1)
  y = 22
  for w in warnings:
    cv2.putText(img, f"WARNING: {w}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += 30


def main():
  parser = argparse.ArgumentParser(description="Interactive training data viewer")
  parser.add_argument("directory", help="Directory containing .npz training frames")
  parser.add_argument("--start", type=int, default=0, help="Starting frame index")
  args = parser.parse_args()

  npz_files = sorted(glob.glob(os.path.join(args.directory, '*.npz')))
  if not npz_files:
    print(f"No .npz files found in {args.directory}")
    sys.exit(1)
  print(f"Found {len(npz_files)} frames in {args.directory}")

  total = len(npz_files)
  idx = max(0, min(args.start, total - 1))

  # Toggle state
  show_lanes = True
  show_road_edges = True
  show_leads = True
  show_info = True
  show_bev = True

  cv2.namedWindow('view_data', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('view_data', W, H)

  while True:
    data = _load_frame(npz_files[idx])
    frame_rgb = data['frame_rgb']   # (H, W, 3) RGB
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    camera_pitch = float(data.get('camera_pitch', 0))
    camera_yaw = float(data.get('camera_yaw', 0))
    camera_height = float(data.get('camera_height', 1.2))
    rpyCalib = np.array([0.0, -camera_pitch, -camera_yaw])

    # Quality warnings
    warnings = _check_warnings(data)
    if warnings:
      _draw_warnings(img, warnings)

    # Overlays
    if show_lanes:
      _draw_lane_lines(img, data, rpyCalib)
    if show_road_edges:
      _draw_road_edges(img, data, rpyCalib)
    if show_leads:
      _draw_leads(img, data, rpyCalib, camera_height)
    if show_info:
      _draw_info_panel(img, data, idx, total)
    if show_bev:
      _draw_bev_panel(img, data, camera_height)

    # Title bar
    fname = os.path.basename(npz_files[idx])
    cv2.setWindowTitle('view_data', f"[{idx}/{total - 1}] {fname}")
    cv2.imshow('view_data', img)

    key = cv2.waitKeyEx(0)
    if key == ord('q') or key == 27:  # q or ESC
      break
    elif key == KEY_RIGHT:
      idx = min(idx + 1, total - 1)
    elif key == KEY_LEFT:
      idx = max(idx - 1, 0)
    elif key == KEY_PGDN:
      idx = min(idx + 10, total - 1)
    elif key == KEY_PGUP:
      idx = max(idx - 10, 0)
    elif key == KEY_HOME:
      idx = 0
    elif key == KEY_END:
      idx = total - 1
    elif key == ord('l'):
      show_lanes = not show_lanes
    elif key == ord('e'):
      show_road_edges = not show_road_edges
    elif key == ord('v'):
      show_leads = not show_leads
    elif key == ord('i'):
      show_info = not show_info
    elif key == ord('b'):
      show_bev = not show_bev
    elif key == ord('s'):
      screenshot_path = f"screenshot_{idx:06d}.png"
      cv2.imwrite(screenshot_path, img)
      print(f"Screenshot saved: {screenshot_path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
