#!/usr/bin/env python3
"""Compare annotations across multiple heights side by side (Step B output).

Core validation tool: shows warp+annotation for all heights simultaneously.
Verifies that:
  1. All heights have consistent pitch→warp (same horizon position)
  2. z_height values increase with height (transform_annotation correctness)
  3. Near-field blind zone grows with height (H5/H6 blank near X=0)

Usage:
  python tools/dashcam/viz/compare_heights.py data/multi_height/quick_..._annotated/
  python tools/dashcam/viz/compare_heights.py <annotated_dir> --heights H1 H4 H6
  python tools/dashcam/viz/compare_heights.py <annotated_dir> --no-annotations

Keys:
  Left/Right   - prev/next frame
  PgUp/PgDn    - ±10 frames
  Home/End     - first/last frame
  a            - toggle annotation overlay
  z            - toggle z_height values on lane lines
  l / e / v    - toggle lane lines / edges / leads
  Tab          - toggle 2×N grid / single-height full view
  1-6          - select height in single view (H1=1, H2=2, ...)
  s            - screenshot (saves full grid)
  q / ESC      - quit
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.dashcam.view_dual_data import (
  _draw_lane_lines,
  _draw_leads,
  _draw_road_edges,
  _prob_color,
)
from openpilot.tools.dashcam.visualizer import (
  _build_transform,
  _get_path_length_idx,
  _map_line_to_polygon,
  _draw_polygon_alpha,
  project_points_to_image,
)


KEY_LEFT  = 65361
KEY_RIGHT = 65363
KEY_PGUP  = 65365
KEY_PGDN  = 65366
KEY_HOME  = 65360
KEY_END   = 65367
KEY_TAB   = 9

X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)
PANEL_W, PANEL_H = 512, 256   # per-cell size in grid mode
FULL_W, FULL_H = 1024, 512    # single-height full view


def get_warp_matrices(clip_info: dict):
  pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
  yaw_rad = math.radians(clip_info['camera']['yaw_deg'])
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)
  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
  return warp_road, rpyCalib


def warp_rgb(rgb: np.ndarray, warp: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
  warped = cv2.warpPerspective(rgb, warp, (512, 256), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
  bgr = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
  return cv2.resize(bgr, (out_w, out_h))


def draw_z_values(img: np.ndarray, data: dict, rpyCalib: np.ndarray, K: np.ndarray):
  """Overlay z_height values at X=20,50,100m for each lane line."""
  ll = data.get('lane_lines')
  if ll is None:
    return
  ll_prob = data.get('lane_lines_prob', np.zeros(4))
  check_xs = [20.0, 50.0, 100.0]

  for i in range(4):
    if ll_prob[i] < 0.3:
      continue
    for x_target in check_xs:
      idx = np.searchsorted(X_IDXS, x_target)
      if idx >= len(ll[i]):
        continue
      pt = ll[i, idx]
      if np.isnan(pt[1]) or np.isnan(pt[2]):
        continue
      uv = project_points_to_image(
        np.array([pt[0]]), np.array([pt[1]]), np.array([pt[2]]), K, rpyCalib)
      if np.isnan(uv[0]).any():
        continue
      u, v = int(uv[0, 0]), int(uv[0, 1])
      if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.putText(img, f"z={pt[2]:.2f}", (u, v), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 220, 0), 1, cv2.LINE_AA)


def render_panel(
  data: dict,
  tag: str,
  height_m: float,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  show_ann: bool,
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_z: bool,
  min_ll_prob: float,
  out_w: int,
  out_h: int,
) -> np.ndarray:
  """Render one height panel."""
  img = warp_rgb(data['road_rgb'], warp_road, out_w, out_h)

  # Scale intrinsics to match output size
  dc = DEVICE_CAMERAS[('pc', 'unknown')]
  K = dc.fcam.intrinsics.copy()
  K[0] *= out_w / 512.0
  K[1] *= out_h / 256.0

  if show_ann:
    if show_lanes and 'lane_lines' in data:
      _draw_lane_lines(img, data, rpyCalib, K)
    if show_edges and 'road_edges' in data:
      _draw_road_edges(img, data, rpyCalib, K)
    if show_leads and 'lead' in data:
      camera_height = float(data.get('camera_height', height_m))
      _draw_leads(img, data, rpyCalib, K, camera_height)
    if show_z:
      draw_z_values(img, data, rpyCalib, K)

  # Height tag label
  cv2.rectangle(img, (0, 0), (150, 26), (0, 0, 0), -1)
  cv2.putText(img, f"{tag} {height_m:.2f}m", (4, 18),
              cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

  # Lane prob summary below label
  ll_prob = data.get('lane_lines_prob', np.zeros(4))
  passes = bool(ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob)

  # Border: green=PASS, red=FAIL
  border_color = (0, 200, 0) if passes else (0, 0, 200)
  cv2.rectangle(img, (0, 0), (out_w - 1, out_h - 1), border_color, 2)

  # Prob text at bottom
  prob_text = f"L={ll_prob[1]:.2f} R={ll_prob[2]:.2f} {'PASS' if passes else 'FAIL'}"
  cv2.rectangle(img, (0, out_h - 22), (260, out_h), (0, 0, 0), -1)
  cv2.putText(img, prob_text, (4, out_h - 6),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

  return img


def render_grid(
  data_per_height: dict[str, dict],
  tags: list[str],
  heights_info: dict[str, float],
  idx: int,
  total: int,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  clip_info: dict,
  show_ann: bool,
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_z: bool,
  min_ll_prob: float,
) -> np.ndarray:
  n = len(tags)
  cols = min(3, n)
  rows = (n + cols - 1) // cols
  grid_w = cols * PANEL_W
  grid_h = rows * PANEL_H
  grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

  for i, tag in enumerate(tags):
    if tag not in data_per_height or data_per_height[tag] is None:
      continue
    r, c = divmod(i, cols)
    x0, y0 = c * PANEL_W, r * PANEL_H
    height_m = heights_info.get(tag, 0.0)
    panel = render_panel(
      data=data_per_height[tag],
      tag=tag,
      height_m=height_m,
      rpyCalib=rpyCalib,
      warp_road=warp_road,
      show_ann=show_ann,
      show_lanes=show_lanes,
      show_edges=show_edges,
      show_leads=show_leads,
      show_z=show_z,
      min_ll_prob=min_ll_prob,
      out_w=PANEL_W,
      out_h=PANEL_H,
    )
    grid[y0:y0 + PANEL_H, x0:x0 + PANEL_W] = panel

  # Status bar
  bar_h = 28
  bar = np.zeros((bar_h, grid_w, 3), dtype=np.uint8)
  cam = clip_info.get('camera', {})
  pitch = cam.get('pitch_deg', 0.0)
  yaw = cam.get('yaw_deg', 0.0)
  weather = clip_info.get('weather', '?')
  map_name = clip_info.get('map', '?')
  status = (f"Frame {idx}/{total - 1}   {map_name} {weather}   "
            f"pitch={pitch:.1f}° yaw={yaw:.1f}°   "
            f"a=ann  z=z_vals  Tab=single  s=screenshot  q=quit")
  cv2.putText(bar, status, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
  return np.vstack([grid, bar])


def render_single(
  data: dict,
  tag: str,
  height_m: float,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  show_ann: bool,
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_z: bool,
  min_ll_prob: float,
  idx: int,
  total: int,
) -> np.ndarray:
  panel = render_panel(
    data=data,
    tag=tag,
    height_m=height_m,
    rpyCalib=rpyCalib,
    warp_road=warp_road,
    show_ann=show_ann,
    show_lanes=show_lanes,
    show_edges=show_edges,
    show_leads=show_leads,
    show_z=show_z,
    min_ll_prob=min_ll_prob,
    out_w=FULL_W,
    out_h=FULL_H,
  )
  cv2.putText(panel, f"Frame {idx}/{total - 1}   1-6=height  Tab=grid  a/z/l/e/v  s=screenshot  q=quit",
              (6, FULL_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
  return panel


def main():
  parser = argparse.ArgumentParser(description='Compare multi-height annotations side by side')
  parser.add_argument('annotated_dir', help='Annotated session directory')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='Subset of heights (default: all found)')
  parser.add_argument('--start', type=int, default=0, help='Starting frame index')
  parser.add_argument('--no-annotations', action='store_true', help='Show warp images only')
  parser.add_argument('--min-ll-prob', type=float, default=0.5, help='Quality filter threshold')
  args = parser.parse_args()

  annotated_dir = Path(args.annotated_dir)
  if not annotated_dir.exists():
    print(f"ERROR: not found: {annotated_dir}", file=sys.stderr)
    sys.exit(1)

  clip_info_path = annotated_dir / 'clip_info.json'
  if not clip_info_path.exists():
    print(f"ERROR: clip_info.json not found", file=sys.stderr)
    sys.exit(1)
  with open(clip_info_path) as f:
    clip_info = json.load(f)

  warp_road, rpyCalib = get_warp_matrices(clip_info)
  heights_info: dict[str, float] = clip_info.get('heights', {})

  all_tags = sorted([d.name for d in annotated_dir.iterdir()
                     if d.is_dir() and d.name.startswith('H') and d.name[1:].isdigit()])
  if args.heights:
    tags = [t for t in args.heights if t in all_tags]
  else:
    tags = all_tags

  if not tags:
    print(f"ERROR: no height dirs found in {annotated_dir}", file=sys.stderr)
    sys.exit(1)

  # Frame file lists per height
  frame_files_per_height: dict[str, list[Path]] = {}
  for tag in tags:
    files = sorted((annotated_dir / tag).glob('*.npz'))
    if files:
      frame_files_per_height[tag] = files
  tags = [t for t in tags if t in frame_files_per_height]

  total = min(len(frame_files_per_height[t]) for t in tags)
  idx = max(0, min(args.start, total - 1))

  show_ann = not args.no_annotations
  show_lanes = True
  show_edges = True
  show_leads = True
  show_z = False
  grid_mode = True
  single_height_idx = 0

  print(f"Session: {annotated_dir.name}")
  print(f"Heights: {tags}  Frames: {total}")
  print("Keys: ←→ frames  a=ann  z=z_vals  l/e/v=layers  Tab=single  1-6=height  s=screenshot  q=quit")

  win = 'compare_heights'
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cols = min(3, len(tags))
  rows = (len(tags) + cols - 1) // cols
  cv2.resizeWindow(win, cols * PANEL_W, rows * PANEL_H + 28)

  while True:
    # Load all heights for current frame
    data_per_height: dict[str, dict | None] = {}
    for tag in tags:
      files = frame_files_per_height[tag]
      if idx < len(files):
        try:
          data_per_height[tag] = dict(np.load(files[idx], allow_pickle=True))
        except Exception:
          data_per_height[tag] = None

    if grid_mode:
      img = render_grid(
        data_per_height=data_per_height,
        tags=tags,
        heights_info=heights_info,
        idx=idx,
        total=total,
        rpyCalib=rpyCalib,
        warp_road=warp_road,
        clip_info=clip_info,
        show_ann=show_ann,
        show_lanes=show_lanes,
        show_edges=show_edges,
        show_leads=show_leads,
        show_z=show_z,
        min_ll_prob=args.min_ll_prob,
      )
    else:
      tag = tags[single_height_idx % len(tags)]
      height_m = heights_info.get(tag, 0.0)
      data = data_per_height.get(tag) or {}
      img = render_single(
        data=data,
        tag=tag,
        height_m=height_m,
        rpyCalib=rpyCalib,
        warp_road=warp_road,
        show_ann=show_ann,
        show_lanes=show_lanes,
        show_edges=show_edges,
        show_leads=show_leads,
        show_z=show_z,
        min_ll_prob=args.min_ll_prob,
        idx=idx,
        total=total,
      )

    cv2.setWindowTitle(win, f"[{idx}/{total - 1}] compare_heights | {annotated_dir.name}")
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
      grid_mode = not grid_mode
      if grid_mode:
        cv2.resizeWindow(win, cols * PANEL_W, rows * PANEL_H + 28)
      else:
        cv2.resizeWindow(win, FULL_W, FULL_H)
    elif ord('1') <= key <= ord('6'):
      n = key - ord('1')
      if n < len(tags):
        single_height_idx = n
        grid_mode = False
        cv2.resizeWindow(win, FULL_W, FULL_H)
    elif key == ord('a'):
      show_ann = not show_ann
    elif key == ord('z'):
      show_z = not show_z
    elif key == ord('l'):
      show_lanes = not show_lanes
    elif key == ord('e'):
      show_edges = not show_edges
    elif key == ord('v'):
      show_leads = not show_leads
    elif key == ord('s'):
      path = f"screenshot_compare_{idx:06d}.png"
      cv2.imwrite(path, img)
      print(f"Screenshot saved: {path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
