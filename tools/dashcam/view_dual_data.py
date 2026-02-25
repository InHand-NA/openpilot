#!/usr/bin/env python3
"""双目训练数据可视化工具：浏览 road_rgb + wide_rgb 双目帧，叠加 modeld 标注。

支持两种布局：
  - 并排模式（默认）：road + wide 左右并排，各半宽
  - 切换模式（按 Tab）：全幅显示单个相机，Tab 切换

用法：
  python tools/dashcam/view_dual_data.py data/dual_camera_exp_002/
  python tools/dashcam/view_dual_data.py data/dual_camera_exp_002/ --start 100
  python tools/dashcam/view_dual_data.py data/dual_camera_exp/ data/dual_camera_exp_002/  # 多目录

操作：
  Left/Right    上/下一帧
  PageUp/Down   ±10 帧
  Home/End      首/末帧
  l             切换车道线
  e             切换路沿
  v             切换前车
  i             切换信息面板
  b             切换 BEV 面板
  Tab           切换并排/单相机模式
  1/2           单相机模式下选择 road/wide
  s             截图 (PNG)
  q / ESC       退出
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.dashcam.visualizer import (
  W,
  H,
  _BEV_H,
  _BEV_MARGIN,
  _BEV_W,
  _BEV_Y_HALF,
  _build_transform,
  _draw_polygon_alpha,
  _get_path_length_idx,
  _map_line_to_polygon,
  project_points_to_image,
)

# Draw range
MIN_DRAW_DISTANCE = 5.0
MAX_DRAW_DISTANCE = 192.0
_BEV_X_MAX = 192.0

# Camera intrinsics
dc = DEVICE_CAMERAS[("pc", "unknown")]
K_ROAD = dc.fcam.intrinsics
K_WIDE = dc.ecam.intrinsics

# OpenCV key codes (Linux GTK)
KEY_RIGHT = 65363
KEY_LEFT = 65361
KEY_PGUP = 65365
KEY_PGDN = 65366
KEY_HOME = 65360
KEY_END = 65367
KEY_TAB = 9

# Lead vehicle colors (BGR)
_LEAD_COLORS = [(0, 220, 220), (0, 140, 255), (255, 220, 0)]
_ROAD_EDGE_COLOR = (220, 0, 220)


def _prob_color(prob):
  if prob >= 0.8:
    return (0, 220, 0)
  if prob >= 0.3:
    return (0, 220, 220)
  return (0, 0, 220)


def _draw_lane_lines(img, data, rpyCalib, K):
  """绘制 4 条车道线。"""
  lane_lines = data['lane_lines']  # (4, 33, 3)
  lane_probs = data['lane_lines_prob']  # (4,)
  transform = _build_transform(K, rpyCalib)

  path_xs = lane_lines[0, :, 0]
  max_distance = np.clip(path_xs[-1] if len(path_xs) > 0 else 0, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(path_xs, max_distance)

  for i in range(4):
    prob = float(lane_probs[i])
    pts = lane_lines[i]
    color = _prob_color(prob)

    if prob > 0.01:
      polygon = _map_line_to_polygon(pts, 0.025 * prob, 0.0, max_idx, max_distance, transform)
      if len(polygon) >= 3:
        _draw_polygon_alpha(img, polygon, color, 0.25)

    if prob < 0.3:
      continue
    mask = (pts[:, 0] >= MIN_DRAW_DISTANCE) & (pts[:, 0] <= MAX_DRAW_DISTANCE) & ~np.isnan(pts[:, 1]) & ~np.isnan(pts[:, 2])
    valid_pts = pts[mask]
    if valid_pts.shape[0] < 2:
      continue
    uv = project_points_to_image(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], K, rpyCalib)
    good = ~np.isnan(uv).any(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    indices = np.where(good)[0]
    if len(indices) < 2:
      continue
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    for seg_idx in np.split(indices, breaks):
      if len(seg_idx) < 2:
        continue
      cv2.polylines(img, [uv[seg_idx].astype(np.int32)], False, color, 2, cv2.LINE_AA)
    first_uv = uv[indices[0]]
    cv2.putText(img, f"p={prob:.2f}", (int(first_uv[0]) + 5, int(first_uv[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_road_edges(img, data, rpyCalib, K):
  """绘制 2 条路沿。"""
  road_edges = data.get('road_edges')
  road_probs = data.get('road_edges_prob')
  if road_edges is None or road_probs is None:
    return

  transform = _build_transform(K, rpyCalib)
  path_xs = road_edges[0, :, 0]
  max_distance = np.clip(path_xs[-1] if len(path_xs) > 0 else 0, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(path_xs, max_distance)

  for i in range(2):
    prob = float(road_probs[i])
    pts = road_edges[i]

    if prob > 0.01:
      polygon = _map_line_to_polygon(pts, 0.025 * prob, 0.0, max_idx, max_distance, transform)
      if len(polygon) >= 3:
        _draw_polygon_alpha(img, polygon, _ROAD_EDGE_COLOR, 0.2)

    if prob < 0.3:
      continue
    mask = (pts[:, 0] >= MIN_DRAW_DISTANCE) & (pts[:, 0] <= MAX_DRAW_DISTANCE) & ~np.isnan(pts[:, 1]) & ~np.isnan(pts[:, 2])
    valid_pts = pts[mask]
    if valid_pts.shape[0] < 2:
      continue
    uv = project_points_to_image(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], K, rpyCalib)
    good = ~np.isnan(uv).any(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    indices = np.where(good)[0]
    if len(indices) < 2:
      continue
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    for seg_idx in np.split(indices, breaks):
      if len(seg_idx) < 2:
        continue
      cv2.polylines(img, [uv[seg_idx].astype(np.int32)], False, _ROAD_EDGE_COLOR, 2, cv2.LINE_AA)


def _draw_leads(img, data, rpyCalib, K, camera_height):
  """绘制前车。"""
  leads = data['lead']  # (3, 6, 4)
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

    uv = project_points_to_image(np.array([x_dist]), np.array([y_off]), np.array([camera_height]), K, rpyCalib)
    if np.isnan(uv[0]).any():
      continue
    cx, cy = int(uv[0, 0]), int(uv[0, 1])
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
      continue

    color = _LEAD_COLORS[sel]
    radius = int(np.clip(800.0 / max(x_dist, 5.0), 8, 40))
    cv2.circle(img, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"#{sel} {x_dist:.1f}m v:{v_rel:+.1f}", (cx + radius + 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_info_panel(img, data, frame_idx, total_frames, cam_label=""):
  """绘制信息面板（左上角）。"""
  font = cv2.FONT_HERSHEY_SIMPLEX
  white = (255, 255, 255)

  lines = []
  lines.append(f"Frame: {frame_idx}/{total_frames - 1}  {cam_label}")

  label_source = str(data.get('label_source', '?'))
  town = str(data.get('town', '?'))
  height = float(data.get('camera_height', 0))
  lines.append(f"Town: {town}  Height: {height:.2f}m  Labels: {label_source}")

  v_ego = float(data.get('v_ego', 0))
  rpyCalib = np.degrees(data['rpyCalib'].astype(np.float64))
  lines.append(f"Speed: {v_ego:.1f} m/s ({v_ego * 3.6:.1f} km/h)  rpyCalib: [{rpyCalib[0]:.2f}, {rpyCalib[1]:.2f}, {rpyCalib[2]:.2f}] deg")

  probs = data.get('lane_lines_prob', np.zeros(4))
  lines.append(f"Lane probs: [{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}, {probs[3]:.2f}]")

  road_probs = data.get('road_edges_prob', np.zeros(2))
  lead_probs = data.get('lead_prob', np.zeros(3))
  lines.append(f"Road edge probs: [{road_probs[0]:.2f}, {road_probs[1]:.2f}]  Lead probs: [{lead_probs[0]:.2f}, {lead_probs[1]:.2f}, {lead_probs[2]:.2f}]")

  pose = data.get('pose')
  if pose is not None:
    lines.append(f"Pose: v=[{pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}] w=[{pose[3]:.4f},{pose[4]:.4f},{pose[5]:.4f}]")

  rt = data.get('road_transform')
  if rt is not None:
    lines.append(f"RoadTf: [{rt[0]:.2f},{rt[1]:.2f},{rt[2]:.2f},{rt[3]:.2f},{rt[4]:.2f},{rt[5]:.2f}]")

  dy = 26
  panel_h = dy * len(lines) + 12
  panel_w = 700
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


def _draw_bev_panel(img, data):
  """绘制 BEV 俯视面板（右下角）。"""
  bev_w, bev_h = _BEV_W, _BEV_H
  x0 = img.shape[1] - bev_w - _BEV_MARGIN
  y0 = img.shape[0] - bev_h - _BEV_MARGIN
  if x0 < 0 or y0 < 0:
    return

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
    cv2.putText(img, f"{dist}m", (x0 + 3, y0 + gy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
  cv2.line(img, (x0 + cx_bev, y0), (x0 + cx_bev, y0 + bev_h), grid_color, 1)

  # Ego
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
      bev_pts.append([x0 + to_bev(xi, yi)[0], y0 + to_bev(xi, yi)[1]])
    cv2.polylines(img, [np.array(bev_pts, dtype=np.int32)], False, _prob_color(prob), 2, cv2.LINE_AA)

  # Road edges
  road_edges = data.get('road_edges')
  road_probs = data.get('road_edges_prob')
  if road_edges is not None and road_probs is not None:
    for i in range(2):
      prob = float(road_probs[i])
      if prob < 0.01:
        continue
      pts = road_edges[i]
      xs, ys = pts[:, 0], pts[:, 1]
      mask = (xs > 0) & (xs < _BEV_X_MAX) & (np.abs(ys) < _BEV_Y_HALF) & ~np.isnan(ys)
      if np.sum(mask) < 2:
        continue
      bev_pts = []
      for xi, yi in zip(xs[mask], ys[mask], strict=True):
        bev_pts.append([x0 + to_bev(xi, yi)[0], y0 + to_bev(xi, yi)[1]])
      cv2.polylines(img, [np.array(bev_pts, dtype=np.int32)], False, _ROAD_EDGE_COLOR, 2, cv2.LINE_AA)

  # Lead vehicles
  leads = data.get('lead')
  lead_probs = data.get('lead_prob')
  if leads is not None and lead_probs is not None:
    for sel in range(3):
      if float(lead_probs[sel]) < 0.3:
        continue
      lx, ly = float(leads[sel, 0, 0]), float(leads[sel, 0, 1])
      if lx < 1.0 or lx > _BEV_X_MAX:
        continue
      bx, by = to_bev(lx, ly)
      cv2.circle(img, (x0 + bx, y0 + by), 5, _LEAD_COLORS[sel], -1)

  cv2.putText(img, "BEV", (x0 + 5, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_camera_label(img, label, position='top-right'):
  """在图像角落绘制相机标签。"""
  font = cv2.FONT_HERSHEY_SIMPLEX
  (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
  if position == 'top-right':
    x = img.shape[1] - tw - 12
    y = th + 10
  else:
    x = 10
    y = th + 10
  cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
  cv2.putText(img, label, (x, y), font, 0.7, (0, 200, 255), 2, cv2.LINE_AA)


def _check_warnings(data):
  """检查数据质量问题。"""
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


def _draw_warnings(img, warnings):
  if not warnings:
    return
  banner_h = 30 * len(warnings)
  cv2.rectangle(img, (0, 0), (img.shape[1], banner_h), (0, 0, 180), -1)
  y = 22
  for w in warnings:
    cv2.putText(img, f"WARNING: {w}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += 30


def render_side_by_side(data, frame_idx, total_frames, show_lanes, show_edges, show_leads, show_info, show_bev):
  """渲染并排双目视图。"""
  road_bgr = cv2.cvtColor(data['road_rgb'], cv2.COLOR_RGB2BGR)
  wide_bgr = cv2.cvtColor(data['wide_rgb'], cv2.COLOR_RGB2BGR)

  camera_height = float(data.get('camera_height', 1.2))
  rpyCalib = np.array(data['rpyCalib'], dtype=np.float64)

  # 在原始分辨率上绘制标注
  if show_lanes:
    _draw_lane_lines(road_bgr, data, rpyCalib, K_ROAD)
    _draw_lane_lines(wide_bgr, data, rpyCalib, K_WIDE)
  if show_edges:
    _draw_road_edges(road_bgr, data, rpyCalib, K_ROAD)
    _draw_road_edges(wide_bgr, data, rpyCalib, K_WIDE)
  if show_leads:
    _draw_leads(road_bgr, data, rpyCalib, K_ROAD, camera_height)
    _draw_leads(wide_bgr, data, rpyCalib, K_WIDE, camera_height)

  _draw_camera_label(road_bgr, "ROAD (narrow)", 'top-right')
  _draw_camera_label(wide_bgr, "WIDE", 'top-right')

  warnings = _check_warnings(data)
  if warnings:
    _draw_warnings(road_bgr, warnings)

  # 缩小到半宽并排
  half_w = W // 2
  half_h = H // 2
  road_small = cv2.resize(road_bgr, (half_w, half_h))
  wide_small = cv2.resize(wide_bgr, (half_w, half_h))
  combo = np.hstack([road_small, wide_small])

  if show_info:
    _draw_info_panel(combo, data, frame_idx, total_frames, cam_label="[DUAL]")
  if show_bev:
    _draw_bev_panel(combo, data)

  return combo


def render_single(data, frame_idx, total_frames, cam_idx, show_lanes, show_edges, show_leads, show_info, show_bev):
  """渲染单相机全幅视图。"""
  if cam_idx == 0:
    img = cv2.cvtColor(data['road_rgb'], cv2.COLOR_RGB2BGR)
    K = K_ROAD
    cam_label = "[ROAD]"
  else:
    img = cv2.cvtColor(data['wide_rgb'], cv2.COLOR_RGB2BGR)
    K = K_WIDE
    cam_label = "[WIDE]"

  camera_height = float(data.get('camera_height', 1.2))
  rpyCalib = np.array(data['rpyCalib'], dtype=np.float64)

  warnings = _check_warnings(data)
  if warnings:
    _draw_warnings(img, warnings)

  if show_lanes:
    _draw_lane_lines(img, data, rpyCalib, K)
  if show_edges:
    _draw_road_edges(img, data, rpyCalib, K)
  if show_leads:
    _draw_leads(img, data, rpyCalib, K, camera_height)
  if show_info:
    _draw_info_panel(img, data, frame_idx, total_frames, cam_label=cam_label)
  if show_bev:
    _draw_bev_panel(img, data)

  return img


def main():
  parser = argparse.ArgumentParser(description="双目训练数据可视化工具")
  parser.add_argument("directories", nargs='+', help="包含 .npz 文件的目录（支持多个）")
  parser.add_argument("--start", type=int, default=0, help="起始帧索引")
  args = parser.parse_args()

  # 收集所有 NPZ 文件
  npz_files = []
  for d in args.directories:
    found = sorted(glob.glob(os.path.join(d, '*.npz')))
    npz_files.extend(found)

  if not npz_files:
    print(f"未找到 .npz 文件")
    sys.exit(1)
  print(f"共 {len(npz_files)} 帧 (来自 {len(args.directories)} 个目录)")

  # 验证第一帧是否为双目数据
  test_data = dict(np.load(npz_files[0], allow_pickle=True))
  if 'road_rgb' not in test_data or 'wide_rgb' not in test_data:
    print("错误：NPZ 文件中缺少 road_rgb / wide_rgb 字段（非双目数据）")
    print(f"  可用字段: {sorted(test_data.keys())}")
    sys.exit(1)

  total = len(npz_files)
  idx = max(0, min(args.start, total - 1))

  # 显示状态
  show_lanes = True
  show_edges = True
  show_leads = True
  show_info = True
  show_bev = True
  side_by_side = True  # True=并排, False=单相机
  cam_idx = 0  # 0=road, 1=wide

  win_name = 'view_dual_data'
  cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win_name, W, H // 2)

  print("操作: ←→翻页  PgUp/PgDn±10  Home/End首末  Tab并排/单相机  1/2选相机  l/e/v/i/b切换图层  s截图  q退出")

  while True:
    data = dict(np.load(npz_files[idx], allow_pickle=True))

    if side_by_side:
      img = render_side_by_side(data, idx, total, show_lanes, show_edges, show_leads, show_info, show_bev)
    else:
      img = render_single(data, idx, total, cam_idx, show_lanes, show_edges, show_leads, show_info, show_bev)

    fname = os.path.basename(npz_files[idx])
    mode_str = "DUAL" if side_by_side else ("ROAD" if cam_idx == 0 else "WIDE")
    cv2.setWindowTitle(win_name, f"[{idx}/{total - 1}] {fname} [{mode_str}]")
    cv2.imshow(win_name, img)

    key = cv2.waitKeyEx(0)
    if key == ord('q') or key == 27:
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
      side_by_side = not side_by_side
      if side_by_side:
        cv2.resizeWindow(win_name, W, H // 2)
      else:
        cv2.resizeWindow(win_name, W, H)
    elif key == ord('1'):
      cam_idx = 0
      side_by_side = False
      cv2.resizeWindow(win_name, W, H)
    elif key == ord('2'):
      cam_idx = 1
      side_by_side = False
      cv2.resizeWindow(win_name, W, H)
    elif key == ord('l'):
      show_lanes = not show_lanes
    elif key == ord('e'):
      show_edges = not show_edges
    elif key == ord('v'):
      show_leads = not show_leads
    elif key == ord('i'):
      show_info = not show_info
    elif key == ord('b'):
      show_bev = not show_bev
    elif key == ord('s'):
      screenshot_path = f"screenshot_dual_{idx:06d}.png"
      cv2.imwrite(screenshot_path, img)
      print(f"截图已保存: {screenshot_path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
