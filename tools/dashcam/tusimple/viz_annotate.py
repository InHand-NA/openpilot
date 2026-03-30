#!/usr/bin/env python3
"""TuSimple Phase 2 可视化：11 相机 4×3 网格 + 3D 标注。

布局 (4 列 × 3 行, 每格 480×360):
  ┌─────────┬─────────┬─────────┬─────────┐
  │ H0 road │ H0 wide │ H1 mono │ H2 mono │  row 0
  ├─────────┼─────────┼─────────┼─────────┤
  │ H3 mono │ H4 mono │ H5 mono │ H6 mono │  row 1
  ├─────────┼─────────┼─────────┼─────────┤
  │ H7 mono │ H8 mono │ H9 mono │ [info]  │  row 2
  └─────────┴─────────┴─────────┴─────────┘

H0 road/wide 使用 canonical 标注 (K_ROAD / K_WIDE)。
H1~H9 mono 使用各自高度变换标注 (K_MONO)。

绘制风格与 tools/dashcam/viz/inspect_annotated.py 保持一致：
  lane_lines  → 填充半透明绿色多边形 (64,255,0)  alpha=clip(prob,0,0.7)
  road_edges  → 填充半透明红色多边形 (0,0,255)   alpha=clip(prob,0,1.0)

操作:
  Left/Right    上/下一帧
  PageUp/Down   ±10 帧
  Home/End      首/末帧
  l             toggle lane_lines
  e             toggle road_edges
  s             截图 (PNG)
  q / ESC       退出

用法:
  python tools/dashcam/tusimple/viz_annotate.py \\
      data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \\
      --max-frames 20
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.tools.dashcam.tusimple.config import HEIGHT_DEFS, K_MONO
from openpilot.tools.dashcam.tusimple.viz_collect import (
  resize_keep_ar,
  load_session_frames,
  load_metadata,
  KEY_RIGHT, KEY_LEFT, KEY_PGUP, KEY_PGDN, KEY_HOME, KEY_END,
)

# H0 camera intrinsics (from openpilot DEVICE_CAMERAS)
_dc = DEVICE_CAMERAS[('pc', 'unknown')]
K_ROAD = _dc.fcam.intrinsics   # narrow, focal≈2648, 1928×1208
K_WIDE = _dc.ecam.intrinsics   # wide,   focal≈567,  1928×1208

# Grid: 4 columns × 3 rows  (12 slots: H0 road/wide + H1~H9 mono + info)
GRID_COLS = 4
GRID_ROWS = 3
CELL_W = 480
CELL_H = 360
INFO_BAR_H = 0  # info drawn in the last cell instead
MONO_TAGS = sorted(HEIGHT_DEFS.keys())

# Polygon clip margin (pixels beyond image bounds)
CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 2.0
MAX_DRAW_DISTANCE = 100.0


# ---------------------------------------------------------------------------
# Projection and polygon helpers (matching inspect_annotated.py style)
# ---------------------------------------------------------------------------

def _build_cam_transform(K: np.ndarray, rpyCalib: np.ndarray) -> np.ndarray:
  """Build 3x3 matrix: calibration-frame 3D → camera pixel coords."""
  device_from_calib = rot_from_euler(rpyCalib)
  return K @ view_frame_from_device_frame @ device_from_calib


def _get_path_length_idx(xs: np.ndarray, distance: float) -> int:
  indices = np.where(xs <= distance)[0]
  return int(indices[-1]) if indices.size > 0 else 0


def _map_line_to_polygon(pts_xyz: np.ndarray, y_off: float,
                         max_idx: int, max_distance: float,
                         T_cam: np.ndarray,
                         img_w: int, img_h: int) -> np.ndarray:
  """Convert 3D line to 2D closed polygon in camera pixel space.

  Matching inspect_annotated.py / visualizer.py polygon building logic.
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

  proj = (T_cam @ points_lr.T).reshape(3, 2, N)
  left_proj = proj[:, 0, :]
  right_proj = proj[:, 1, :]

  valid = (np.abs(left_proj[2]) >= 1e-6) & (np.abs(right_proj[2]) >= 1e-6)
  if not np.any(valid):
    return np.empty((0, 2), dtype=np.int32)

  left_screen = left_proj[:2, valid] / left_proj[2, valid][None, :]
  right_screen = right_proj[:2, valid] / right_proj[2, valid][None, :]

  x_min, x_max = -CLIP_MARGIN, img_w + CLIP_MARGIN
  y_min, y_max = -CLIP_MARGIN, img_h + CLIP_MARGIN

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

  return np.vstack((left_screen.T, right_screen[:, ::-1].T)).astype(np.int32)


def _draw_polygon_alpha(img: np.ndarray, polygon: np.ndarray,
                        color_bgr: tuple, alpha: float) -> None:
  """Draw a filled polygon with alpha blending (ROI-optimized)."""
  x, y, w, h = cv2.boundingRect(polygon)
  x0, y0 = max(x, 0), max(y, 0)
  x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
  if x1 <= x0 or y1 <= y0:
    return
  roi = img[y0:y1, x0:x1]
  overlay = roi.copy()
  cv2.fillPoly(overlay, [polygon - np.array([x0, y0])], color_bgr)
  img[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)


# ---------------------------------------------------------------------------
# Annotation drawing (matching inspect_annotated.py / visualizer.py style)
# ---------------------------------------------------------------------------

def _draw_lane_lines(img: np.ndarray, anno: dict, T_cam: np.ndarray) -> None:
  """Draw 4 lane lines as filled alpha-blended green polygons."""
  ll = np.array(anno['lane_lines'], dtype=np.float32)        # (4, 33, 3)
  ll_prob = np.array(anno['lane_lines_prob'], dtype=np.float32)  # (4,)

  path_xs = ll[0][:, 0]
  max_dist = float(np.clip(path_xs[-1] if len(path_xs) > 0 else 0,
                            MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
  max_idx = _get_path_length_idx(path_xs, max_dist)
  img_h, img_w = img.shape[:2]

  for i in range(4):
    prob = float(ll_prob[i])
    if prob < 0.01:
      continue
    y_off = 0.025 * prob
    polygon = _map_line_to_polygon(ll[i], y_off, max_idx, max_dist, T_cam, img_w, img_h)
    if len(polygon) < 3:
      continue
    alpha = float(np.clip(prob, 0.0, 0.7))
    _draw_polygon_alpha(img, polygon, (64, 255, 0), alpha)  # green


def _draw_road_edges(img: np.ndarray, anno: dict, T_cam: np.ndarray) -> None:
  """Draw 2 road edges as filled alpha-blended red polygons."""
  re = np.array(anno['road_edges'], dtype=np.float32)          # (2, 33, 3)
  re_prob = np.array(anno['road_edges_prob'], dtype=np.float32)  # (2,)

  # Use lane_lines x range for draw distance
  ll = np.array(anno.get('lane_lines', [[[0]] * 33] * 4), dtype=np.float32)
  path_xs = ll[0][:, 0]
  max_dist = float(np.clip(path_xs[-1] if len(path_xs) > 0 else 0,
                            MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
  max_idx = _get_path_length_idx(path_xs, max_dist)
  img_h, img_w = img.shape[:2]

  for i in range(2):
    prob = float(re_prob[i])
    alpha = float(np.clip(prob, 0.0, 1.0))
    if alpha < 0.01:
      continue
    y_off = 0.025  # fixed width in meters
    polygon = _map_line_to_polygon(re[i], y_off, max_idx, max_dist, T_cam, img_w, img_h)
    if len(polygon) < 3:
      continue
    _draw_polygon_alpha(img, polygon, (0, 0, 255), alpha)  # red


# ---------------------------------------------------------------------------
# Loading and rendering
# ---------------------------------------------------------------------------

def load_annotation(label_dir: Path, frame_id: str) -> dict | None:
  """Load a single annotation JSON."""
  json_path = label_dir / f'{frame_id}.json'
  if not json_path.exists():
    return None
  with open(json_path) as f:
    return json.load(f)


def _load_mono_image(session_dir: Path, tag: str, frame_id: str,
                     clip_info: dict) -> np.ndarray | None:
  """Load Hk mono image (PNG or JPEG)."""
  mono_fmt = clip_info.get('mono_camera', {}).get('format', 'png')
  mono_ext = '.jpg' if mono_fmt == 'jpeg' else '.png'
  mono_path = session_dir / tag / f'{frame_id}{mono_ext}'
  if not mono_path.exists():
    # Fallback to the other extension
    alt_ext = '.png' if mono_ext == '.jpg' else '.jpg'
    mono_path = session_dir / tag / f'{frame_id}{alt_ext}'
  if not mono_path.exists():
    return None
  return cv2.imread(str(mono_path))


def _draw_overlays(img: np.ndarray, anno: dict | None, T_cam: np.ndarray,
                   show_lanes: bool, show_edges: bool) -> None:
  """Draw lane lines and road edges on img if annotation exists."""
  if anno is None:
    return
  if show_lanes and 'lane_lines' in anno:
    _draw_lane_lines(img, anno, T_cam)
  if show_edges and 'road_edges' in anno:
    _draw_road_edges(img, anno, T_cam)


def _make_cell(img: np.ndarray | None, label: str) -> np.ndarray:
  """Resize image to CELL_W×CELL_H, add label. Black if img is None."""
  if img is not None:
    cell = resize_keep_ar(img, CELL_W, CELL_H)
  else:
    cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
  cv2.putText(cell, label, (3, 14),
              cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
  return cell


def _make_info_cell(frame_id: str, clip_info: dict, metadata: dict,
                    anno_canonical: dict | None,
                    show_lanes: bool, show_edges: bool) -> np.ndarray:
  """Render the 9th cell as an info panel."""
  cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
  font = cv2.FONT_HERSHEY_SIMPLEX
  white = (220, 220, 220)
  gray = (140, 140, 140)
  y, dy = 30, 28

  cv2.putText(cell, f'frame  {frame_id}', (10, y), font, 0.6, white, 1, cv2.LINE_AA)
  y += dy

  frame_tick = int(frame_id)
  meta_rec = metadata.get(frame_tick, {})
  v_ego = meta_rec.get('v_ego')
  if v_ego is not None:
    cv2.putText(cell, f'speed  {v_ego:.1f} m/s  ({v_ego * 3.6:.0f} km/h)',
                (10, y), font, 0.55, white, 1, cv2.LINE_AA)
  y += dy

  cam = clip_info.get('camera', {})
  cv2.putText(cell, f"pitch  {cam.get('pitch_deg', '?')} deg   yaw  {cam.get('yaw_deg', '?')} deg",
              (10, y), font, 0.55, white, 1, cv2.LINE_AA)
  y += dy + 5

  # Lane line probs
  if anno_canonical is not None and 'lane_lines_prob' in anno_canonical:
    p = anno_canonical['lane_lines_prob']
    cv2.putText(cell, 'lane probs', (10, y), font, 0.5, gray, 1, cv2.LINE_AA)
    y += 24
    labels = ['L-out', 'L-in', 'R-in', 'R-out']
    for i in range(4):
      prob = float(p[i])
      color = (0, 200, 0) if prob >= 0.5 else ((0, 200, 200) if prob >= 0.2 else (0, 0, 200))
      bar_w = int(200 * prob)
      cv2.rectangle(cell, (110, y - 14), (110 + 200, y + 2), (40, 40, 40), -1)
      cv2.rectangle(cell, (110, y - 14), (110 + bar_w, y + 2), color, -1)
      cv2.putText(cell, f'{labels[i]} {prob:.2f}', (10, y), font, 0.45, white, 1, cv2.LINE_AA)
      y += 22
  y += 8

  # Toggle state
  lanes_color = (0, 220, 0) if show_lanes else (80, 80, 80)
  edges_color = (0, 220, 0) if show_edges else (80, 80, 80)
  cv2.putText(cell, '[L]anes', (10, y), font, 0.5, lanes_color, 1, cv2.LINE_AA)
  cv2.putText(cell, '[E]dges', (120, y), font, 0.5, edges_color, 1, cv2.LINE_AA)
  y += dy

  # Keys help
  cv2.putText(cell, 'keys: Left/Right  PgUp/PgDn  s=screenshot', (10, y),
              font, 0.4, gray, 1, cv2.LINE_AA)

  return cell


def render_grid(session_dir: Path, frame_id: str, label_dir: Path,
                clip_info: dict, rpyCalib: np.ndarray, metadata: dict,
                show_lanes: bool, show_edges: bool) -> np.ndarray:
  """Render 3×3 grid (1920×1080): 8 cameras + 1 info panel."""
  T_road = _build_cam_transform(K_ROAD, rpyCalib)
  T_wide = _build_cam_transform(K_WIDE, rpyCalib)
  T_mono = _build_cam_transform(K_MONO, rpyCalib)
  anno_canonical = load_annotation(label_dir / 'H1', frame_id)
  heights_info = clip_info.get('heights', {})

  cells: list[np.ndarray] = []

  # Cell 0: H0 road (canonical)
  h0_road_path = session_dir / 'H0' / f'road_{frame_id}.png'
  road_img = cv2.imread(str(h0_road_path)) if h0_road_path.exists() else None
  if road_img is not None:
    _draw_overlays(road_img, anno_canonical, T_road, show_lanes, show_edges)
  cells.append(_make_cell(road_img, 'H0 road (canonical)'))

  # Cell 1: H0 wide (canonical)
  h0_wide_path = session_dir / 'H0' / f'wide_{frame_id}.png'
  wide_img = cv2.imread(str(h0_wide_path)) if h0_wide_path.exists() else None
  if wide_img is not None:
    _draw_overlays(wide_img, anno_canonical, T_wide, show_lanes, show_edges)
  cells.append(_make_cell(wide_img, 'H0 wide (canonical)'))

  # Cells 2~10: H1~H9 mono
  for tag in MONO_TAGS:
    h_val = heights_info.get(tag, '?')
    mono_img = _load_mono_image(session_dir, tag, frame_id, clip_info)
    anno_hk = load_annotation(label_dir / tag, frame_id)
    if mono_img is not None:
      _draw_overlays(mono_img, anno_hk, T_mono, show_lanes, show_edges)
    cells.append(_make_cell(mono_img, f'{tag} ({h_val}m)'))

  # Last cell: info panel
  cells.append(_make_info_cell(frame_id, clip_info, metadata,
                               anno_canonical, show_lanes, show_edges))

  # Assemble grid
  rows = []
  for r in range(GRID_ROWS):
    s = r * GRID_COLS
    rows.append(np.hstack(cells[s:s + GRID_COLS]))
  return np.vstack(rows)


def main():
  parser = argparse.ArgumentParser(description="TuSimple Phase 2 可视化：8 相机 2×4 网格 + 3D 标注")
  parser.add_argument("session_dir", help="session 目录")
  parser.add_argument("--label-dir", default=None,
                      help="标注目录 (default: <session_dir>/3d_labels/)")
  parser.add_argument("--max-frames", type=int, default=0,
                      help="最大浏览帧数 (0=全部)")
  parser.add_argument("--output-dir", default=None,
                      help="批量输出目录 (非交互模式)")
  args = parser.parse_args()

  session_dir = Path(args.session_dir)
  if not session_dir.exists():
    print(f"目录不存在: {session_dir}")
    sys.exit(1)

  if args.label_dir:
    label_dir = Path(args.label_dir)
  else:
    label_dir = session_dir / '3d_labels'
  if not label_dir.exists():
    print(f"标注目录不存在: {label_dir}")
    print(f"请先运行 annotate_3d.py 生成标注")
    sys.exit(1)

  clip_info_path = session_dir / 'clip_info.json'
  if clip_info_path.exists():
    with open(clip_info_path) as f:
      clip_info = json.load(f)
  else:
    print(f"[WARN] clip_info.json 不存在: {clip_info_path}")
    clip_info = {}

  cam = clip_info.get('camera', {})
  pitch_rad = math.radians(cam.get('pitch_deg', 4.0))
  yaw_rad = math.radians(cam.get('yaw_deg', 0.0))
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  frame_ids = load_session_frames(session_dir)
  if not frame_ids:
    print(f"未找到 H0/road_*.png 帧文件")
    sys.exit(1)

  if args.max_frames > 0:
    frame_ids = frame_ids[:args.max_frames]
  total = len(frame_ids)

  metadata = load_metadata(session_dir)
  n_mono = len(MONO_TAGS)
  print(f"共 {total} 帧  {2 + n_mono} cameras (H0 road/wide + H1~H{n_mono} mono)  {CELL_W*GRID_COLS}x{CELL_H*GRID_ROWS}")

  # Batch output mode
  if args.output_dir:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fid in enumerate(frame_ids):
      img = render_grid(session_dir, fid, label_dir, clip_info,
                        rpyCalib, metadata, True, True)
      cv2.imwrite(str(out_dir / f'viz_{fid}.png'), img)
      if (i + 1) % 50 == 0:
        print(f"  {i+1}/{total} saved")
    print(f"Done: {total} grid images saved to {out_dir}")
    return

  # Interactive mode (no scroll needed — 1920×1080 fits exactly)
  idx = 0
  show_lanes = True
  show_edges = True
  win_name = 'viz_annotate_tusimple'
  cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
  cv2.resizeWindow(win_name, CELL_W * GRID_COLS, CELL_H * GRID_ROWS)

  print("操作: ←→翻页  PgUp/PgDn±10  Home/End首末  l车道线  e路沿  s截图  q退出")

  while True:
    img = render_grid(session_dir, frame_ids[idx], label_dir, clip_info,
                      rpyCalib, metadata, show_lanes, show_edges)

    cv2.setWindowTitle(win_name, f"[{idx}/{total - 1}] frame={frame_ids[idx]}")
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
    elif key == ord('l'):
      show_lanes = not show_lanes
      print(f"车道线: {'ON' if show_lanes else 'OFF'}")
    elif key == ord('e'):
      show_edges = not show_edges
      print(f"路沿: {'ON' if show_edges else 'OFF'}")
    elif key == ord('s'):
      screenshot_path = f"screenshot_annotate_{frame_ids[idx]}.png"
      cv2.imwrite(screenshot_path, img)
      print(f"截图已保存: {screenshot_path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
