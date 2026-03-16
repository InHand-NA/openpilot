#!/usr/bin/env python3
"""Inspect annotated multi-height data (Step B output).

Displays the warp-corrected model input image with annotation overlays
(lane lines, road edges, lead vehicles) matching the run.py / Visualizer
drawing style: filled alpha-blended polygons for lane lines/road edges,
chevron triangles for lead vehicles.

Projection pipeline for annotations:
  3D calib-frame point [x,y,z]
    → K @ view_frame_from_device @ rot_from_euler(rpyCalib)   (camera pixels)
    → inv(M_warp)                                              (512×256 warped pixels)
    → scale × 2                                               (1024×512 display pixels)

Keys:
  Left / Right   - prev / next frame
  PgUp / PgDn    - ±10 frames
  Home / End     - first / last frame
  Tab            - switch narrow / wide camera
  h              - cycle heights  H1 → H2 → … → H6 → H1
  l              - toggle lane lines
  e              - toggle road edges
  v              - toggle lead vehicles
  b              - toggle BEV panel
  i              - toggle info panel
  r              - toggle raw / warped image
  f              - toggle filtered-only mode
  s              - screenshot
  q / ESC        - quit
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.modeld.constants import ModelConstants


KEY_LEFT  = 65361
KEY_RIGHT = 65363
KEY_PGUP  = 65365
KEY_PGDN  = 65366
KEY_HOME  = 65360
KEY_END   = 65367
KEY_TAB   = 9

# Model input size → display size (×2)
MODEL_W, MODEL_H = 512, 256
DISPLAY_W, DISPLAY_H = MODEL_W * 2, MODEL_H * 2   # 1024 × 512
BEV_W = 300

X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)   # (33,) forward distances
MIN_DRAW_DISTANCE = 2.0
MAX_DRAW_DISTANCE = 100.0

# ---------------------------------------------------------------------------
# JSON annotation loader
# ---------------------------------------------------------------------------

# Fields stored as nested lists in JSON that must be converted back to float32 arrays.
_ARRAY_FIELDS: dict[str, type] = {
  'lane_lines':             np.float32,
  'lane_lines_prob':        np.float32,
  'road_edges':             np.float32,
  'road_edges_prob':        np.float32,
  'lead':                   np.float32,
  'lead_prob':              np.float32,
  'pose':                   np.float32,
  'road_transform':         np.float32,
  'wide_from_device_euler': np.float32,
  'world_pose':             np.float32,
}


def _load_json_annotation(path: Path) -> dict:
  """Load a JSON annotation file and convert list fields back to numpy arrays."""
  with open(path) as f:
    record = json.load(f)
  for key, dtype in _ARRAY_FIELDS.items():
    if key in record:
      record[key] = np.array(record[key], dtype=dtype)
  return record

# Polygon clip margin beyond display bounds (pixels)
CLIP_MARGIN = 200

# BEV panel constants (matching visualizer.py style)
_BEV_X_MAX  = 80.0
_BEV_Y_HALF = 10.0
_BEV_MARGIN = 15


# ---------------------------------------------------------------------------
# Core projection helpers
# ---------------------------------------------------------------------------

def build_display_transform(M_warp: np.ndarray, K: np.ndarray,
                             rpyCalib: np.ndarray) -> np.ndarray:
  """Build 3×3 matrix projecting calibration-frame 3D → display pixels.

  Pipeline:
    T_cam   = K @ view_frame_from_device @ rot_from_euler(rpyCalib)
    T_warp  = inv(M_warp) @ T_cam          (→ 512×256 warped pixels)
    T_disp  = S @ T_warp                   (→ DISPLAY_W×DISPLAY_H pixels)
  where S = diag(DISPLAY_W/512, DISPLAY_H/256, 1).
  """
  device_from_calib = rot_from_euler(rpyCalib)
  T_cam  = K @ view_frame_from_device_frame @ device_from_calib
  T_warp = np.linalg.inv(M_warp) @ T_cam
  S = np.diag([DISPLAY_W / MODEL_W, DISPLAY_H / MODEL_H, 1.0])
  return S @ T_warp


def project_pts(xs, ys, zs, T_disp: np.ndarray) -> np.ndarray:
  """Project calibration-frame 3D points to display pixels. Returns (N,2) float."""
  pts  = np.stack([xs, ys, zs], axis=0)   # (3, N)
  proj = T_disp @ pts                     # (3, N)
  behind = proj[2] <= 0
  proj[2, behind] = np.nan
  uv = (proj[:2] / proj[2:3]).T           # (N, 2)
  return uv


def _in_display(uv: np.ndarray) -> np.ndarray:
  """Boolean mask: points inside display bounds."""
  return (~np.isnan(uv).any(axis=1)
          & (uv[:, 0] >= 0) & (uv[:, 0] < DISPLAY_W)
          & (uv[:, 1] >= 0) & (uv[:, 1] < DISPLAY_H))


def _get_path_length_idx(xs: np.ndarray, distance: float) -> int:
  """Return index of last point with x <= distance."""
  indices = np.where(xs <= distance)[0]
  return int(indices[-1]) if indices.size > 0 else 0


# ---------------------------------------------------------------------------
# Polygon helpers (matching visualizer.py style)
# ---------------------------------------------------------------------------

def _map_line_to_polygon_disp(pts_xyz: np.ndarray, y_off: float,
                                max_idx: int, max_distance: float,
                                T_disp: np.ndarray) -> np.ndarray:
  """Convert 3D line to 2D closed polygon for display-space rendering.

  Adapted from visualizer._map_line_to_polygon for warped display space.
  Returns Mx2 int32 polygon (left forward + right reversed), or empty array.
  """
  if pts_xyz.shape[0] == 0:
    return np.empty((0, 2), dtype=np.int32)

  points = pts_xyz[:max_idx + 1]

  # Interpolate at max_distance for smooth path end
  if 0 < max_idx < pts_xyz.shape[0] - 1:
    p0, p1 = pts_xyz[max_idx], pts_xyz[max_idx + 1]
    interp_y = np.interp(max_distance, [p0[0], p1[0]], [p0[1], p1[1]])
    interp_z = np.interp(max_distance, [p0[0], p1[0]], [p0[2], p1[2]])
    points = np.concatenate((points, np.array([[max_distance, interp_y, interp_z]])), axis=0)

  points = points[points[:, 0] >= 0]
  if points.shape[0] == 0:
    return np.empty((0, 2), dtype=np.int32)

  N = points.shape[0]
  offsets = np.array([[0, -y_off, 0], [0, y_off, 0]], dtype=np.float32)
  points_lr = (points[None, :, :] + offsets[:, None, :]).reshape(2 * N, 3)

  # Project all points at once
  proj = (T_disp @ points_lr.T).reshape(3, 2, N)
  left_proj  = proj[:, 0, :]
  right_proj = proj[:, 1, :]

  valid = (np.abs(left_proj[2]) >= 1e-6) & (np.abs(right_proj[2]) >= 1e-6)
  if not np.any(valid):
    return np.empty((0, 2), dtype=np.int32)

  left_screen  = left_proj[:2, valid]  / left_proj[2, valid][None, :]
  right_screen = right_proj[:2, valid] / right_proj[2, valid][None, :]

  x_min, x_max_ = -CLIP_MARGIN, DISPLAY_W + CLIP_MARGIN
  y_min, y_max_ = -CLIP_MARGIN, DISPLAY_H + CLIP_MARGIN

  both_in = (
    (left_screen[0] >= x_min) & (left_screen[0] <= x_max_) &
    (left_screen[1] >= y_min) & (left_screen[1] <= y_max_) &
    (right_screen[0] >= x_min) & (right_screen[0] <= x_max_) &
    (right_screen[1] >= y_min) & (right_screen[1] <= y_max_)
  )
  if not np.any(both_in):
    return np.empty((0, 2), dtype=np.int32)

  left_screen  = left_screen[:, both_in]
  right_screen = right_screen[:, both_in]

  # Build closed polygon: left forward + right reversed
  return np.vstack((left_screen.T, right_screen[:, ::-1].T)).astype(np.int32)


def _draw_polygon_alpha(img: np.ndarray, polygon: np.ndarray,
                         color_bgr: tuple, alpha: float) -> None:
  """Draw a filled polygon with alpha blending (ROI-optimized)."""
  x, y, w, h = cv2.boundingRect(polygon)
  x0, y0 = max(x, 0), max(y, 0)
  x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
  if x1 <= x0 or y1 <= y0:
    return
  roi     = img[y0:y1, x0:x1]
  overlay = roi.copy()
  cv2.fillPoly(overlay, [polygon - np.array([x0, y0])], color_bgr)
  img[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)


# ---------------------------------------------------------------------------
# Annotation drawing (visualizer.py polygon style)
# ---------------------------------------------------------------------------

def draw_lane_lines(img: np.ndarray, data: dict, T_disp: np.ndarray) -> None:
  """Draw 4 lane lines as filled alpha-blended polygons (green), matching run.py style."""
  ll      = data.get('lane_lines')       # (4, 33, 3) [x, y, z]
  ll_prob = data.get('lane_lines_prob')  # (4,)
  if ll is None or ll_prob is None:
    return

  # Determine draw range from first lane line x values
  path_xs = ll[0][:, 0]
  max_dist = float(np.clip(path_xs[-1] if len(path_xs) > 0 else 0,
                            MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
  max_idx = _get_path_length_idx(path_xs, max_dist)

  for i in range(4):
    prob = float(ll_prob[i])
    if prob < 0.01:
      continue
    pts = ll[i]  # (33, 3)
    y_off   = 0.025 * prob
    polygon = _map_line_to_polygon_disp(pts, y_off, max_idx, max_dist, T_disp)
    if len(polygon) < 3:
      continue
    alpha = float(np.clip(prob, 0.0, 0.7))
    _draw_polygon_alpha(img, polygon, (64, 255, 0), alpha)  # green, same as visualizer


def draw_road_edges(img: np.ndarray, data: dict, T_disp: np.ndarray) -> None:
  """Draw 2 road edges as filled alpha-blended polygons (red), matching run.py style."""
  re      = data.get('road_edges')       # (2, 33, 3)
  re_prob = data.get('road_edges_prob')  # (2,)  probability ≈ 1 − std
  if re is None or re_prob is None:
    return

  ll = data.get('lane_lines')
  path_xs = ll[0][:, 0] if ll is not None else np.array([MAX_DRAW_DISTANCE])
  max_dist = float(np.clip(path_xs[-1] if len(path_xs) > 0 else 0,
                            MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
  max_idx = _get_path_length_idx(path_xs, max_dist)

  for i in range(2):
    prob = float(re_prob[i])
    alpha = float(np.clip(prob, 0.0, 1.0))
    if alpha < 0.01:
      continue
    pts     = re[i]   # (33, 3)
    y_off   = 0.025   # fixed width in meters (matching visualizer)
    polygon = _map_line_to_polygon_disp(pts, y_off, max_idx, max_dist, T_disp)
    if len(polygon) < 3:
      continue
    _draw_polygon_alpha(img, polygon, (0, 0, 255), alpha)  # red, same as visualizer


def draw_leads(img: np.ndarray, data: dict, T_disp: np.ndarray, camera_height: float) -> None:
  """Draw lead vehicles as glow+chevron triangles, matching run.py style."""
  leads     = data.get('lead')       # (3, 6, 4)
  lead_prob = data.get('lead_prob')  # (3,)
  if leads is None or lead_prob is None:
    return

  # Only draw the most-probable lead (index 0) like visualizer._draw_lead
  for sel in range(3):
    prob = float(lead_prob[sel])
    if prob < 0.3:
      continue
    x_dist = float(leads[sel, 0, 0])
    y_off  = float(leads[sel, 0, 1])
    v_rel  = float(leads[sel, 0, 2])

    if x_dist < 1.0 or x_dist > 200.0:
      continue

    uv = project_pts(np.array([x_dist]), np.array([y_off]),
                     np.array([camera_height]), T_disp)
    if np.isnan(uv[0]).any():
      continue

    x, y = float(uv[0, 0]), float(uv[0, 1])

    # Chevron size (matching visualizer._draw_lead)
    sz = np.clip((25 * 30) / (x_dist / 3 + 30), 15.0, 30.0) * 2.35
    x  = float(np.clip(x, 0.0, DISPLAY_W - sz / 2))
    y  = min(y, DISPLAY_H - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    # Glow triangle (outer, yellow BGR=(37, 202, 218))
    glow = np.array([
      [x + sz * 1.35 + g_xo, y + sz + g_yo],
      [x, y - g_yo],
      [x - sz * 1.35 - g_xo, y + sz + g_yo],
    ], dtype=np.int32)

    # Chevron triangle (inner, red BGR=(49, 34, 201))
    chevron = np.array([
      [x + sz * 1.25, y + sz],
      [x, y],
      [x - sz * 1.25, y + sz],
    ], dtype=np.int32)

    # Fill alpha based on distance and relative speed
    speed_buff, lead_buff = 10.0, 40.0
    fill_alpha = 0.0
    if x_dist < lead_buff:
      fill_alpha = 1.0 - (x_dist / lead_buff)
      if v_rel < 0:
        fill_alpha += -v_rel / speed_buff
      fill_alpha = min(fill_alpha, 1.0)

    cv2.fillPoly(img, [glow], (37, 202, 218), cv2.LINE_AA)

    if fill_alpha > 0.01:
      _draw_polygon_alpha(img, chevron, (49, 34, 201), fill_alpha)

    label = f"#{sel} p:{prob:.2f} {x_dist:.0f}m v:{v_rel:+.1f}"
    cv2.putText(img, label,
                (int(x + sz * 1.35 + g_xo + 5), int(y + sz / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (37, 202, 218), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# BEV panel (matching visualizer.py _draw_bev_panel style)
# ---------------------------------------------------------------------------

def draw_bev(canvas: np.ndarray, data: dict) -> None:
  """Draw bird's-eye-view panel, matching visualizer._draw_bev_panel style."""
  bev_h, bev_w = canvas.shape[:2]
  scale_x = bev_h / _BEV_X_MAX
  scale_y = bev_w / (2 * _BEV_Y_HALF)
  cx_bev  = bev_w // 2

  # Semi-transparent black background
  roi     = canvas[:bev_h, :bev_w]
  overlay = roi.copy()
  cv2.rectangle(overlay, (0, 0), (bev_w, bev_h), (0, 0, 0), -1)
  canvas[:bev_h, :bev_w] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)

  def to_bev(x_fwd: float, y_lat: float) -> tuple[int, int]:
    px = cx_bev + y_lat * scale_y
    py = bev_h  - x_fwd * scale_x
    return int(np.clip(px, 0, bev_w - 1)), int(np.clip(py, 0, bev_h - 1))

  # Grid lines at 20, 40, 60 m
  grid_color = (60, 60, 60)
  for dist in [20, 40, 60]:
    _, gy = to_bev(dist, 0)
    cv2.line(canvas, (0, gy), (bev_w, gy), grid_color, 1)
    cv2.putText(canvas, f"{dist}m", (3, gy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
  cv2.line(canvas, (cx_bev, 0), (cx_bev, bev_h), grid_color, 1)

  # Ego vehicle marker
  ego_bx, ego_by = to_bev(0, 0)
  cv2.circle(canvas, (ego_bx, ego_by - 3), 5, (255, 255, 255), -1)

  def draw_bev_line(pts_xyz: np.ndarray, prob: float, color: tuple, thickness: int) -> None:
    if prob < 0.01:
      return
    xs, ys = pts_xyz[:, 0], pts_xyz[:, 1]
    mask = (xs > 0) & (xs < _BEV_X_MAX) & (np.abs(ys) < _BEV_Y_HALF)
    if np.sum(mask) < 2:
      return
    bev_pts = np.array([list(to_bev(xi, yi)) for xi, yi in zip(xs[mask], ys[mask])],
                       dtype=np.int32)
    cv2.polylines(canvas, [bev_pts], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)

  ll      = data.get('lane_lines',      np.zeros((4, 33, 3), np.float32))
  ll_prob = data.get('lane_lines_prob', np.zeros(4, np.float32))
  for i in range(4):
    draw_bev_line(ll[i], float(ll_prob[i]), (64, 255, 0), 2)

  re      = data.get('road_edges',      np.zeros((2, 33, 3), np.float32))
  re_prob = data.get('road_edges_prob', np.zeros(2, np.float32))
  for i in range(2):
    draw_bev_line(re[i], float(re_prob[i]), (0, 0, 255), 2)

  leads     = data.get('lead',      np.zeros((3, 6, 4), np.float32))
  lead_prob = data.get('lead_prob', np.zeros(3, np.float32))
  for i in range(3):
    if lead_prob[i] < 0.3:
      continue
    lx, ly = float(leads[i, 0, 0]), float(leads[i, 0, 1])
    if 1.0 < lx < _BEV_X_MAX:
      cv2.circle(canvas, to_bev(lx, ly), 5, (37, 202, 218), -1)

  # Title
  cv2.putText(canvas, "BEV (top-down)", (3, 15),
              cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Probability bar
# ---------------------------------------------------------------------------

def draw_prob_bar(img: np.ndarray, probs: np.ndarray, y0: int, min_prob: float) -> None:
  labels = ['L-out', 'L-in', 'R-in', 'R-out']
  bar_w  = img.shape[1] // len(probs)
  for i, p in enumerate(probs):
    x0    = i * bar_w
    color = (0, 200, 0) if p >= 0.8 else ((0, 200, 200) if p >= min_prob else (0, 0, 200))
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + 28), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + int(bar_w * float(p)), y0 + 28), color, -1)
    cv2.putText(img, f"{labels[i]} {float(p):.2f}", (x0 + 4, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)


def draw_lead_prob_bar(img: np.ndarray, probs: np.ndarray, y0: int) -> None:
  """Draw a probability bar for lead vehicles (up to 3 leads)."""
  labels = ['Lead-0', 'Lead-1', 'Lead-2']
  n = min(len(probs), 3)
  bar_w = img.shape[1] // 3
  for i in range(n):
    p = float(probs[i])
    x0 = i * bar_w
    # Lead prob threshold is 0.3 for drawing chevrons
    color = (218, 202, 37) if p >= 0.5 else ((0, 200, 200) if p >= 0.3 else (80, 80, 80))
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + 28), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + int(bar_w * p), y0 + 28), color, -1)
    cv2.putText(img, f"{labels[i]} {p:.2f}", (x0 + 4, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def render_frame(
  data: dict,
  frame_idx: str,
  total: int,
  height_tag: str,
  height_m: float,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  warp_wide: np.ndarray,
  K_road: np.ndarray,
  K_wide: np.ndarray,
  cam_idx: int,          # 0 = narrow/road, 1 = wide
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_bev: bool,
  show_info: bool,
  show_raw: bool,
  min_ll_prob: float,
) -> np.ndarray:

  # --- Select camera ---
  if cam_idx == 0:
    rgb_key, warp, K = 'road_rgb', warp_road, K_road
    cam_label = 'NARROW'
  else:
    rgb_key, warp, K = 'wide_rgb', warp_wide, K_wide
    cam_label = 'WIDE'

  rgb = data.get(rgb_key, data.get('road_rgb'))

  # --- Build display image ---
  if show_raw:
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (DISPLAY_W, DISPLAY_H))
  else:
    warped = cv2.warpPerspective(rgb, warp, (MODEL_W, MODEL_H),
                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    img = cv2.resize(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR), (DISPLAY_W, DISPLAY_H))

  # --- Build projection matrix ---
  T_disp = build_display_transform(warp, K, rpyCalib)

  # --- Annotation overlays (visualizer.py polygon style) ---
  camera_height = float(data.get('camera_height', height_m))
  if not show_raw:
    if show_lanes:
      draw_lane_lines(img, data, T_disp)
    if show_edges:
      draw_road_edges(img, data, T_disp)
    if show_leads:
      draw_leads(img, data, T_disp, camera_height)

  # --- Quality pass/fail flag ---
  ll_prob = data.get('lane_lines_prob', np.zeros(4))
  passes  = bool(data.get('ll_quality_pass',
                           ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob))

  # --- Camera + height label (top-right, matching visualizer color scheme) ---
  label = f"{cam_label}  {height_tag} {height_m:.2f}m"
  (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
  lx = img.shape[1] - tw - 14
  cv2.rectangle(img, (lx - 4, 0), (img.shape[1], 30), (0, 0, 0), -1)
  cv2.putText(img, label, (lx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

  # --- Quality filter banner (top-left) ---
  if not passes:
    cv2.rectangle(img, (0, 0), (160, 30), (0, 0, 180), -1)
    cv2.putText(img, "[FILTERED]", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

  # --- Info panel (matching visualizer._draw_info_panel style) ---
  if show_info:
    v_ego     = float(data.get('v_ego', 0.0))
    lead_prob = data.get('lead_prob', np.zeros(3))
    pitch_d   = math.degrees(rpyCalib[1])
    yaw_d     = math.degrees(rpyCalib[2])

    lines = [
      (f"Frame: {frame_idx}", (255, 255, 255)),
      (f"Speed: {v_ego:.1f} m/s  ({v_ego * 3.6:.1f} km/h)", (255, 255, 255)),
      (f"pitch={pitch_d:+.2f}deg  yaw={yaw_d:+.2f}deg  h={camera_height:.2f}m",
       (255, 255, 255)),
      (f"LL prob: {ll_prob[0]:.2f}  {ll_prob[1]:.2f}  {ll_prob[2]:.2f}  {ll_prob[3]:.2f}",
       (255, 255, 255)),
      (f"Lead:  {lead_prob[0]:.2f}  {lead_prob[1]:.2f}  {lead_prob[2]:.2f}",
       (255, 255, 255)),
      (f"{'PASS' if passes else 'FAIL'}  thresh={min_ll_prob:.2f}"
       + f"  {'[raw]' if show_raw else '[warped]'}",
       (0, 220, 0) if passes else (0, 0, 220)),
    ]
    dy, pw = 26, 480
    ph = dy * len(lines) + 12
    roi     = img[0:ph, 0:pw]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (pw, ph), (0, 0, 0), -1)
    img[0:ph, 0:pw] = cv2.addWeighted(overlay, 0.60, roi, 0.40, 0)
    y = 22
    for text, color in lines:
      cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
      y += dy

  # --- Probability bars at bottom ---
  prob_bar = np.zeros((30, DISPLAY_W, 3), dtype=np.uint8)
  draw_prob_bar(prob_bar, ll_prob, 1, min_ll_prob)
  lead_prob = data.get('lead_prob', np.zeros(3))
  lead_bar = np.zeros((30, DISPLAY_W, 3), dtype=np.uint8)
  draw_lead_prob_bar(lead_bar, lead_prob, 1)
  main = np.vstack([img, prob_bar, lead_bar])

  # --- BEV panel (right column, matching visualizer style) ---
  if show_bev:
    bev = np.zeros((main.shape[0], BEV_W, 3), dtype=np.uint8)
    draw_bev(bev, data)
    main = np.hstack([main, bev])

  return main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description='Inspect annotated multi-height data')
  parser.add_argument('annotated_dir',
                      help='Annotated session directory (contains H1/, H2/, …)')
  parser.add_argument('--height', default='H1', help='Initial height (default: H1)')
  parser.add_argument('--start', type=int, default=0, help='Starting frame index')
  parser.add_argument('--filter-low', action='store_true',
                      help='Show only low-confidence frames')
  parser.add_argument('--min-ll-prob', type=float, default=0.5,
                      help='Quality filter threshold (default: 0.5)')
  args = parser.parse_args()

  annotated_dir = Path(args.annotated_dir)
  if not annotated_dir.exists():
    print(f"ERROR: not found: {annotated_dir}", file=sys.stderr); sys.exit(1)

  clip_info_path = annotated_dir / 'clip_info.json'
  if not clip_info_path.exists():
    print("ERROR: clip_info.json not found", file=sys.stderr); sys.exit(1)
  with open(clip_info_path) as f:
    clip_info = json.load(f)

  pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
  yaw_rad   = math.radians(clip_info['camera']['yaw_deg'])
  rpyCalib  = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
  warp_wide = compute_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True)
  K_road    = dc.fcam.intrinsics
  K_wide    = dc.ecam.intrinsics

  heights_info: dict[str, float] = clip_info.get('heights', {})

  # Annotated NPZs no longer embed road_rgb/wide_rgb (removed to reduce file size).
  # source_session_dir in clip_info.json points to the original session for RGB lookup.
  source_session_dir: Path | None = None
  src = clip_info.get('source_session_dir')
  if src:
    # Resolve relative to annotated_dir (e.g. '..' → parent session dir)
    p = (annotated_dir / src).resolve()
    source_session_dir = p if p.exists() else None
    if source_session_dir is None:
      print(f"WARN: source_session_dir not found: {p}", file=sys.stderr)

  all_tags = sorted([d.name for d in annotated_dir.iterdir()
                     if d.is_dir() and d.name.startswith('H') and d.name[1:].isdigit()])
  if not all_tags:
    print("ERROR: no height directories found", file=sys.stderr); sys.exit(1)

  current_height_idx = all_tags.index(args.height) if args.height in all_tags else 0
  height_tag = all_tags[current_height_idx]

  def get_frame_files(tag: str) -> list[Path]:
    return sorted((annotated_dir / tag).glob('*.json'))

  frame_files = get_frame_files(height_tag)
  total = len(frame_files)
  if total == 0:
    print(f"ERROR: no frames in {annotated_dir / height_tag}", file=sys.stderr); sys.exit(1)

  idx         = max(0, min(args.start, total - 1))
  cam_idx     = 0      # 0=narrow, 1=wide
  show_lanes  = True
  show_edges  = True
  show_leads  = True
  show_bev    = True
  show_info   = True
  show_raw    = False
  filter_low  = args.filter_low
  min_ll_prob = args.min_ll_prob

  print(f"Session : {annotated_dir.name}")
  print(f"Heights : {all_tags}  Current: {height_tag}  Frames: {total}")
  print("Keys    : ←→ frames  PgUp/PgDn ±10  Tab=narrow/wide  h=height  "
        "l/e/v=overlays  b=BEV  i=info  r=raw/warp  f=filter  s=screenshot  q=quit")

  win = 'inspect_annotated'
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win, DISPLAY_W + BEV_W, DISPLAY_H + 60)

  while True:
    data     = _load_json_annotation(frame_files[idx])
    height_m = heights_info.get(height_tag, float(data.get('camera_height', 1.22)))

    # JSON never embeds images — always load RGB from source_session_dir
    stem = frame_files[idx].stem  # original tick-based frame id, e.g. '000004'
    if source_session_dir is not None:
      height_dir = source_session_dir / height_tag
      road_bgr = cv2.imread(str(height_dir / f'road_{stem}.png'))
      wide_bgr = cv2.imread(str(height_dir / f'wide_{stem}.png'))
      if road_bgr is not None:
        data['road_rgb'] = cv2.cvtColor(road_bgr, cv2.COLOR_BGR2RGB)
      if wide_bgr is not None:
        data['wide_rgb'] = cv2.cvtColor(wide_bgr, cv2.COLOR_BGR2RGB)

    frame_num = stem  # already computed above
    img = render_frame(
      data=data, frame_idx=frame_num, total=total,
      height_tag=height_tag, height_m=height_m,
      rpyCalib=rpyCalib,
      warp_road=warp_road, warp_wide=warp_wide,
      K_road=K_road, K_wide=K_wide,
      cam_idx=cam_idx,
      show_lanes=show_lanes, show_edges=show_edges, show_leads=show_leads,
      show_bev=show_bev, show_info=show_info, show_raw=show_raw,
      min_ll_prob=min_ll_prob,
    )

    cam_name = 'NARROW' if cam_idx == 0 else 'WIDE'
    cv2.setWindowTitle(win,
      f"[{frame_num}] {height_tag} {height_m:.2f}m  [{cam_name}] | {annotated_dir.name}")
    cv2.imshow(win, img)

    key = cv2.waitKeyEx(0)
    if key in (ord('q'), 27):
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
    elif key == KEY_TAB:
      cam_idx = 1 - cam_idx
      print(f"Camera: {'WIDE' if cam_idx else 'NARROW'}")
    elif key == ord('h'):
      current_height_idx = (current_height_idx + 1) % len(all_tags)
      height_tag  = all_tags[current_height_idx]
      frame_files = get_frame_files(height_tag)
      total       = len(frame_files)
      idx         = min(idx, total - 1)
      print(f"Height: {height_tag}")
    elif key == ord('l'):
      show_lanes = not show_lanes
    elif key == ord('e'):
      show_edges = not show_edges
    elif key == ord('v'):
      show_leads = not show_leads
    elif key == ord('b'):
      show_bev = not show_bev
    elif key == ord('i'):
      show_info = not show_info
    elif key == ord('r'):
      show_raw = not show_raw
    elif key == ord('f'):
      filter_low = not filter_low
      print(f"Filter-low: {'ON' if filter_low else 'OFF'}")
    elif key == ord('s'):
      path = f"screenshot_inspect_{height_tag}_{cam_name}_{idx:06d}.png"
      cv2.imwrite(path, img)
      print(f"Screenshot: {path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
