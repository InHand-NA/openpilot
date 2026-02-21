#!/usr/bin/env python3
"""NPZ 训练数据可视化工具：在 warped 模型输入图像上叠加所有 GT 标注。

加载 npz → warp 原始图像到 512×256 模型输入空间 → 绘制车道线/路沿/前车/信息 → 显示或保存。

用法：
  python tools/dashcam/view_npz.py data/training/town04_xxx/000100.npz     # 单帧
  python tools/dashcam/view_npz.py data/training/town04_xxx/                # 目录，←→ 键翻页
  python tools/dashcam/view_npz.py data/training/town04_xxx/ --save-dir out/ # 批量保存 PNG
"""

import argparse
import os
import sys

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, medmodel_intrinsics
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.visualizer import (
  _build_transform,
  _draw_polygon_alpha,
  _get_path_length_idx,
  _map_line_to_polygon,
  MAX_DRAW_DISTANCE,
  MIN_DRAW_DISTANCE,
  project_points_to_image,
)
from openpilot.tools.dashcam.warp_example import warp_image

MW, MH = MEDMODEL_INPUT_SIZE  # 512, 256
INFO_BAR_H = 54
X_IDXS = np.array(ModelConstants.X_IDXS)


def render_frame(data, filename=""):
  """渲染单帧：warp 图像 + 叠加所有标注。返回 BGR 图像 (INFO_BAR_H+MH) x MW x 3。"""
  rgb = data['frame_rgb']
  rpyCalib = data['rpyCalib'].astype(np.float64)
  camera_height = float(data['camera_height'])

  # Warp to model input space (512x256)
  camera_intrinsics = DEVICE_CAMERAS[("pc", "unknown")].fcam.intrinsics
  warped = warp_image(rgb, rpyCalib, camera_intrinsics)
  img = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)

  # In warped space, projection uses medmodel_intrinsics with rpyCalib=[0,0,0]
  K = medmodel_intrinsics
  rpy_zero = np.zeros(3)
  transform = _build_transform(K, rpy_zero)

  _draw_lanes(img, data, transform)
  _draw_road_edges(img, data, transform)
  _draw_lead(img, data, K, rpy_zero, camera_height)

  return np.vstack([_make_info_bar(data, filename), img])


def _draw_lanes(img, data, transform):
  """绘制 4 条车道线（绿色半透明多边形）。"""
  lane_lines = data['lane_lines']       # (4, 33, 3)
  lane_probs = data['lane_lines_prob']  # (4,)

  max_distance = np.clip(X_IDXS[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(X_IDXS, max_distance)

  for i in range(lane_lines.shape[0]):
    prob = float(lane_probs[i])
    if prob < 0.01:
      continue
    polygon = _map_line_to_polygon(lane_lines[i], 0.025 * prob, 0.0, max_idx, max_distance, transform)
    if len(polygon) < 3:
      continue
    _draw_polygon_alpha(img, polygon, (0, 255, 0), float(np.clip(prob, 0.0, 0.7)))


def _draw_road_edges(img, data, transform):
  """绘制 2 条路沿（橙色半透明多边形）。"""
  road_edges = data['road_edges']       # (2, 33, 3)
  road_probs = data['road_edges_prob']  # (2,)

  max_distance = np.clip(X_IDXS[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
  max_idx = _get_path_length_idx(X_IDXS, max_distance)

  for i in range(road_edges.shape[0]):
    prob = float(road_probs[i])
    if prob < 0.01:
      continue
    polygon = _map_line_to_polygon(road_edges[i], 0.025, 0.0, max_idx, max_distance, transform)
    if len(polygon) < 3:
      continue
    _draw_polygon_alpha(img, polygon, (0, 165, 255), float(np.clip(prob, 0.0, 0.7)))


def _draw_lead(img, data, K, rpyCalib, camera_height):
  """绘制前车标记（三角形 + 距离/速度标签）。"""
  lead_prob = data['lead_prob']  # (3,)
  if lead_prob[0] < 0.3:
    return

  lead = data['lead']  # (3, 6, 4) -> [x_dist, y_offset, v_abs, accel]
  x_dist, y_offset, v_abs, accel = (float(v) for v in lead[0, 0])

  if x_dist < 1.0 or x_dist > 200.0:
    return

  pts = project_points_to_image(
    np.array([x_dist]), np.array([y_offset]), np.array([camera_height]), K, rpyCalib)
  if np.isnan(pts[0]).any():
    return

  x, y = float(pts[0, 0]), float(pts[0, 1])
  sz = np.clip((25 * 30) / (x_dist / 3 + 30), 15.0, 30.0)
  x = np.clip(x, sz, MW - sz)
  y = min(y, MH - sz * 0.6)

  tri = np.array([
    [x, y - sz * 0.4],
    [x - sz * 0.8, y + sz * 0.4],
    [x + sz * 0.8, y + sz * 0.4],
  ], dtype=np.int32)
  cv2.fillPoly(img, [tri], (0, 200, 255))
  cv2.polylines(img, [tri], True, (0, 0, 255), 1, cv2.LINE_AA)

  label = f"{x_dist:.1f}m v={v_abs:.1f} a={accel:.1f}"
  cv2.putText(img, label, (int(x + sz), int(y)),
              cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)


def _make_info_bar(data, filename):
  """创建图像上方的信息栏（3 行）。"""
  bar = np.zeros((INFO_BAR_H, MW, 3), dtype=np.uint8)
  font = cv2.FONT_HERSHEY_SIMPLEX
  white, gray, orange = (255, 255, 255), (180, 180, 180), (0, 200, 255)
  s = 0.33

  pose = data['pose']  # (6,) translational + rotational velocities
  v_ego = float(data['v_ego'])
  rpyCalib = np.degrees(data['rpyCalib'].astype(np.float64))
  camera_height = float(data['camera_height'])

  # Line 1: pose velocities
  cv2.putText(bar, f"v={pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f} w={pose[3]:.2f},{pose[4]:.2f},{pose[5]:.2f}",
              (4, 14), font, s, gray, 1, cv2.LINE_AA)
  if filename:
    cv2.putText(bar, filename, (MW - 7 * len(filename), 14), font, s, gray, 1, cv2.LINE_AA)

  # Line 2: ego speed, height, rpyCalib
  cv2.putText(bar, f"v_ego={v_ego:.1f} m/s | h={camera_height:.2f}m | rpyCalib=[{rpyCalib[0]:.1f}, {rpyCalib[1]:.1f}, {rpyCalib[2]:.1f}] deg",
              (4, 30), font, s, white, 1, cv2.LINE_AA)

  # Line 3: lead + lane/edge probs
  lead_str = ""
  if data['lead_prob'][0] > 0.3:
    ld = data['lead']
    lead_str = f"lead: x={ld[0,0,0]:.1f}m y={ld[0,0,1]:.1f}m v={ld[0,0,2]:.1f} a={ld[0,0,3]:.1f} | "
  lp = data['lane_lines_prob']
  rp = data['road_edges_prob']
  cv2.putText(bar, f"{lead_str}lanes=[{lp[0]:.2f},{lp[1]:.2f},{lp[2]:.2f},{lp[3]:.2f}] edges=[{rp[0]:.2f},{rp[1]:.2f}]",
              (4, 46), font, 0.30, orange if lead_str else gray, 1, cv2.LINE_AA)

  return bar


def collect_npz_files(path):
  """收集 npz 文件列表（单文件或目录下所有 .npz 按名排序）。"""
  if os.path.isfile(path):
    return [path]
  files = sorted(f for f in os.listdir(path) if f.endswith('.npz'))
  if not files:
    print(f"目录中无 .npz 文件: {path}")
    sys.exit(1)
  return [os.path.join(path, f) for f in files]


def main():
  parser = argparse.ArgumentParser(description='NPZ 训练数据可视化工具')
  parser.add_argument('path', help='.npz 文件或包含 .npz 的目录')
  parser.add_argument('--save-dir', help='批量保存 PNG 到指定目录（不显示窗口）')
  args = parser.parse_args()

  if not os.path.exists(args.path):
    print(f"路径不存在: {args.path}")
    sys.exit(1)

  files = collect_npz_files(args.path)
  print(f"共 {len(files)} 帧")

  # Batch save mode
  if args.save_dir:
    os.makedirs(args.save_dir, exist_ok=True)
    for i, f in enumerate(files):
      data = dict(np.load(f, allow_pickle=True))
      img = render_frame(data, os.path.basename(f))
      out_path = os.path.join(args.save_dir, os.path.basename(f).replace('.npz', '.png'))
      cv2.imwrite(out_path, img)
      print(f"\r[{i+1}/{len(files)}] {out_path}", end="", flush=True)
    print("\n完成")
    return

  # Interactive mode
  cv2.namedWindow('view_npz', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('view_npz', MW * 2, (MH + INFO_BAR_H) * 2)

  idx = 0
  cached_img = None
  cached_idx = -1

  while True:
    if idx != cached_idx:
      data = dict(np.load(files[idx], allow_pickle=True))
      cached_img = render_frame(data, os.path.basename(files[idx]))
      cached_idx = idx
      print(f"\r[{idx+1}/{len(files)}] {os.path.basename(files[idx])}", end="", flush=True)

    cv2.imshow('view_npz', cached_img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q') or key == 27:  # q / ESC
      break
    elif key == 83 or key == ord('d'):  # → or d
      idx = min(idx + 1, len(files) - 1)
    elif key == 81 or key == ord('a'):  # ← or a
      idx = max(idx - 1, 0)
    elif key == ord('g'):  # jump to first
      idx = 0
    elif key == ord('G'):  # jump to last
      idx = len(files) - 1

  print()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
