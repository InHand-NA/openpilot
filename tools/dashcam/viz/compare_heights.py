#!/usr/bin/env python3
"""Compare annotations across multiple heights side by side.

Displays warp-corrected images with annotation overlays for all heights
simultaneously. Projection and drawing functions are imported from
inspect_annotated.py to avoid duplication. Grid panels are rendered at
full resolution (1024×512) then downscaled to 512×256, so all drawing
code runs in the same coordinate space.

Usage:
  python tools/dashcam/viz/compare_heights.py <annotated_dir>
  python tools/dashcam/viz/compare_heights.py <annotated_dir> --heights H1 H4 H6
  python tools/dashcam/viz/compare_heights.py <annotated_dir> --no-annotations

Keys:
  Left / Right  - prev / next frame
  PgUp / PgDn   - ±10 frames
  Home / End    - first / last frame
  Tab           - toggle narrow / wide camera (all panels)
  g             - toggle grid / single-height view
  1-6           - select height in single view
  a             - toggle annotation overlay
  z             - toggle z_height values on lane lines
  l / e / v     - toggle lane lines / road edges / lead vehicles
  b             - toggle BEV panel (single view only)
  r             - toggle raw / warped image
  f             - toggle filtered-only mode (skip low-quality frames)
  s             - screenshot
  q / ESC       - quit
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

# Reuse all projection and drawing helpers from inspect_annotated.py
from openpilot.tools.dashcam.viz.inspect_annotated import (
  build_display_transform,
  project_pts,
  _get_path_length_idx,
  _map_line_to_polygon_disp,
  _draw_polygon_alpha,
  draw_lane_lines,
  draw_road_edges,
  draw_leads,
  draw_bev,
  draw_prob_bar,
  KEY_LEFT, KEY_RIGHT, KEY_PGUP, KEY_PGDN, KEY_HOME, KEY_END, KEY_TAB,
  MODEL_W, MODEL_H,
  DISPLAY_W, DISPLAY_H,    # 1024 × 512 — full render size
  BEV_W,
  X_IDXS,
  MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE,
)


# Grid cell: render at DISPLAY size then downscale → avoids re-parameterising draw functions
PANEL_W, PANEL_H = 512, 256


# ---------------------------------------------------------------------------
# z_height label overlay  (not in inspect_annotated.py)
# ---------------------------------------------------------------------------

def draw_z_values(img: np.ndarray, data: dict, T_disp: np.ndarray) -> None:
  """Overlay z_height values at X = 20, 50, 100 m for each visible lane line."""
  ll      = data.get('lane_lines')
  ll_prob = data.get('lane_lines_prob', np.zeros(4))
  if ll is None:
    return
  for i in range(4):
    if ll_prob[i] < 0.3:
      continue
    for x_target in [20.0, 50.0, 100.0]:
      idx = int(np.searchsorted(X_IDXS, x_target))
      if idx >= len(ll[i]):
        continue
      pt = ll[i, idx]
      if np.isnan(pt[1]) or np.isnan(pt[2]):
        continue
      uv = project_pts(np.array([pt[0]]), np.array([pt[1]]), np.array([pt[2]]), T_disp)
      if np.isnan(uv[0]).any():
        continue
      u, v = int(uv[0, 0]), int(uv[0, 1])
      if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.putText(img, f"z={pt[2]:.2f}", (u, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# RGB loading
# ---------------------------------------------------------------------------

def _load_rgb(source_session_dir: Path | None,
              tag: str, frame_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
  """Load road_rgb / wide_rgb from the original session directory."""
  if source_session_dir is None:
    return None, None
  src_path = source_session_dir / tag / frame_name
  if not src_path.exists():
    return None, None
  try:
    src = np.load(str(src_path), allow_pickle=True)
    return src.get('road_rgb'), src.get('wide_rgb')
  except Exception:
    return None, None


# ---------------------------------------------------------------------------
# Per-height panel renderer  (renders at DISPLAY size, optionally downscaled)
# ---------------------------------------------------------------------------

def render_panel(
  data: dict,
  tag: str,
  height_m: float,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  warp_wide: np.ndarray,
  K_road: np.ndarray,
  K_wide: np.ndarray,
  cam_idx: int,
  show_ann: bool,
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_z: bool,
  show_raw: bool,
  min_ll_prob: float,
  out_w: int = DISPLAY_W,
  out_h: int = DISPLAY_H,
) -> np.ndarray:
  """Render one height panel.

  Always draws at DISPLAY_W × DISPLAY_H (matching inspect_annotated.py coordinate
  space), then resizes to out_w × out_h at the end.  This lets all imported
  drawing functions run without any coordinate-space adjustments.
  """
  if cam_idx == 0:
    rgb_key, warp, K = 'road_rgb', warp_road, K_road
    cam_label = 'NARROW'
  else:
    rgb_key, warp, K = 'wide_rgb', warp_wide, K_wide
    cam_label = 'WIDE'

  rgb = data.get(rgb_key)
  if rgb is None:
    rgb = data.get('road_rgb')

  # Build DISPLAY_W × DISPLAY_H base image
  if rgb is None:
    img = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(img, "NO RGB", (DISPLAY_W // 2 - 60, DISPLAY_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA)
  elif show_raw:
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (DISPLAY_W, DISPLAY_H))
  else:
    warped = cv2.warpPerspective(rgb, warp, (MODEL_W, MODEL_H),
                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    img = cv2.resize(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR), (DISPLAY_W, DISPLAY_H))

  # Projection matrix (same as inspect_annotated.py: 3D → DISPLAY pixels)
  T_disp = build_display_transform(warp, K, rpyCalib)

  # Annotation overlays
  camera_height = float(data.get('camera_height', height_m))
  if not show_raw and show_ann:
    if show_lanes:
      draw_lane_lines(img, data, T_disp)
    if show_edges:
      draw_road_edges(img, data, T_disp)
    if show_leads:
      draw_leads(img, data, T_disp, camera_height)
    if show_z:
      draw_z_values(img, data, T_disp)

  # Height + camera label (top-right)
  ll_prob = data.get('lane_lines_prob', np.zeros(4))
  passes  = bool(data.get('ll_quality_pass',
                           bool(ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob)))
  label = f"{cam_label}  {tag} {height_m:.2f}m"
  (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
  lx = img.shape[1] - tw - 14
  cv2.rectangle(img, (lx - 4, 0), (img.shape[1], 30), (0, 0, 0), -1)
  cv2.putText(img, label, (lx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

  # PASS / FAIL banner (top-left)
  if not passes:
    cv2.rectangle(img, (0, 0), (160, 30), (0, 0, 180), -1)
    cv2.putText(img, "[FILTERED]", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

  # Border colour: green = PASS, red = FAIL
  cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                (0, 200, 0) if passes else (0, 0, 200), 3)

  # Prob text at bottom-left
  prob_text = f"L={ll_prob[1]:.2f} R={ll_prob[2]:.2f} {'PASS' if passes else 'FAIL'}"
  cv2.rectangle(img, (0, DISPLAY_H - 22), (260, DISPLAY_H), (0, 0, 0), -1)
  cv2.putText(img, prob_text, (4, DISPLAY_H - 6),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

  # Downscale if requested (grid mode)
  if out_w != DISPLAY_W or out_h != DISPLAY_H:
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
  return img


# ---------------------------------------------------------------------------
# Grid renderer
# ---------------------------------------------------------------------------

def render_grid(
  data_per_height: dict[str, dict | None],
  tags: list[str],
  heights_info: dict[str, float],
  idx: int,
  total: int,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  warp_wide: np.ndarray,
  K_road: np.ndarray,
  K_wide: np.ndarray,
  clip_info: dict,
  cam_idx: int,
  show_ann: bool,
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_z: bool,
  show_raw: bool,
  min_ll_prob: float,
) -> np.ndarray:
  n    = len(tags)
  cols = min(3, n)
  rows = (n + cols - 1) // cols
  grid = np.zeros((rows * PANEL_H, cols * PANEL_W, 3), dtype=np.uint8)

  for i, tag in enumerate(tags):
    data = data_per_height.get(tag)
    if data is None:
      continue
    r, c = divmod(i, cols)
    x0, y0  = c * PANEL_W, r * PANEL_H
    height_m = heights_info.get(tag, float(data.get('camera_height', 1.22)))
    panel = render_panel(
      data=data, tag=tag, height_m=height_m,
      rpyCalib=rpyCalib,
      warp_road=warp_road, warp_wide=warp_wide, K_road=K_road, K_wide=K_wide,
      cam_idx=cam_idx,
      show_ann=show_ann, show_lanes=show_lanes, show_edges=show_edges,
      show_leads=show_leads, show_z=show_z, show_raw=show_raw,
      min_ll_prob=min_ll_prob,
      out_w=PANEL_W, out_h=PANEL_H,
    )
    grid[y0:y0 + PANEL_H, x0:x0 + PANEL_W] = panel

  cam = clip_info.get('camera', {})
  pitch, yaw = cam.get('pitch_deg', 0.0), cam.get('yaw_deg', 0.0)
  cam_name = 'NARROW' if cam_idx == 0 else 'WIDE'
  status = (f"Frame {idx}/{total - 1}   "
            f"{clip_info.get('map', '?')} {clip_info.get('weather', '?')}   "
            f"pitch={pitch:.1f}° yaw={yaw:.1f}°   [{cam_name}]   "
            f"Tab=cam  g=single  a/z/l/e/v/r  f=filter  s=shot  q=quit")
  bar = np.zeros((28, cols * PANEL_W, 3), dtype=np.uint8)
  cv2.putText(bar, status, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
  return np.vstack([grid, bar])


# ---------------------------------------------------------------------------
# Single-height full view  (with prob bar + BEV, matching inspect_annotated.py)
# ---------------------------------------------------------------------------

def render_single(
  data: dict,
  tag: str,
  height_m: float,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  warp_wide: np.ndarray,
  K_road: np.ndarray,
  K_wide: np.ndarray,
  cam_idx: int,
  show_ann: bool,
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_z: bool,
  show_bev: bool,
  show_raw: bool,
  min_ll_prob: float,
  idx: int,
  total: int,
) -> np.ndarray:
  panel = render_panel(
    data=data, tag=tag, height_m=height_m,
    rpyCalib=rpyCalib,
    warp_road=warp_road, warp_wide=warp_wide, K_road=K_road, K_wide=K_wide,
    cam_idx=cam_idx,
    show_ann=show_ann, show_lanes=show_lanes, show_edges=show_edges,
    show_leads=show_leads, show_z=show_z, show_raw=show_raw,
    min_ll_prob=min_ll_prob,
    out_w=DISPLAY_W, out_h=DISPLAY_H,
  )

  ll_prob  = data.get('lane_lines_prob', np.zeros(4))
  prob_bar = np.zeros((30, DISPLAY_W, 3), dtype=np.uint8)
  draw_prob_bar(prob_bar, ll_prob, 1, min_ll_prob)
  hint = (f"Frame {idx}/{total - 1}   1-6=height  g=grid  Tab=cam  "
          f"a/z/l/e/v/b/r  f=filter  s=screenshot  q=quit")
  cv2.putText(prob_bar, hint, (6, 24),
              cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

  main = np.vstack([panel, prob_bar])

  if show_bev:
    bev = np.zeros((main.shape[0], BEV_W, 3), dtype=np.uint8)
    draw_bev(bev, data)
    main = np.hstack([main, bev])

  return main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description='Compare multi-height annotations side by side')
  parser.add_argument('annotated_dir', help='Annotated session directory (contains H1/, H2/, …)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='Subset of heights to display (default: all found)')
  parser.add_argument('--start', type=int, default=0, help='Starting frame index')
  parser.add_argument('--no-annotations', action='store_true', help='Show warped images only')
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

  dc        = DEVICE_CAMERAS[('pc', 'unknown')]
  warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
  warp_wide = compute_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True)
  K_road, K_wide = dc.fcam.intrinsics, dc.ecam.intrinsics

  heights_info: dict[str, float] = clip_info.get('heights', {})

  # source_session_dir: where the original RGB frames live
  source_session_dir: Path | None = None
  src = clip_info.get('source_session_dir')
  if src:
    p = Path(src)
    source_session_dir = p if p.exists() else None
    if source_session_dir is None:
      print(f"WARN: source_session_dir not found: {src}", file=sys.stderr)

  all_tags = sorted([d.name for d in annotated_dir.iterdir()
                     if d.is_dir() and d.name.startswith('H') and d.name[1:].isdigit()])
  tags = ([t for t in args.heights if t in all_tags] if args.heights else all_tags)
  if not tags:
    print(f"ERROR: no height directories found in {annotated_dir}", file=sys.stderr); sys.exit(1)

  frame_files_per_height: dict[str, list[Path]] = {}
  for tag in tags:
    files = sorted((annotated_dir / tag).glob('*.npz'))
    if files:
      frame_files_per_height[tag] = files
  tags = [t for t in tags if t in frame_files_per_height]
  if not tags:
    print("ERROR: no NPZ frames found", file=sys.stderr); sys.exit(1)

  total = min(len(frame_files_per_height[t]) for t in tags)
  idx   = max(0, min(args.start, total - 1))

  # View state
  cam_idx        = 0
  show_ann       = not args.no_annotations
  show_lanes     = True
  show_edges     = True
  show_leads     = True
  show_z         = False
  show_bev       = True
  show_raw       = False
  filter_low     = False
  grid_mode      = True
  single_tag_idx = 0
  min_ll_prob    = args.min_ll_prob

  print(f"Session : {annotated_dir.name}")
  print(f"Heights : {tags}  Frames: {total}")
  print("Keys    : ←→ PgUp/PgDn Home/End  Tab=cam  g=grid/single  "
        "1-6=height  a/z/l/e/v/b/r=overlays  f=filter  s=screenshot  q=quit")

  win  = 'compare_heights'
  cols = min(3, len(tags))
  rows = (len(tags) + cols - 1) // cols
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win, cols * PANEL_W, rows * PANEL_H + 28)

  while True:
    frame_name = frame_files_per_height[tags[0]][idx].name

    data_per_height: dict[str, dict | None] = {}
    for tag in tags:
      files = frame_files_per_height[tag]
      if idx >= len(files):
        data_per_height[tag] = None
        continue
      try:
        d = dict(np.load(files[idx], allow_pickle=True))
      except Exception:
        data_per_height[tag] = None
        continue
      # Load RGB from source_session_dir if not embedded in annotated NPZ
      if 'road_rgb' not in d or 'wide_rgb' not in d:
        road_rgb, wide_rgb = _load_rgb(source_session_dir, tag, frame_name)
        if road_rgb is not None:
          d['road_rgb'] = road_rgb
        if wide_rgb is not None:
          d['wide_rgb'] = wide_rgb
      data_per_height[tag] = d

    # Filter-low: advance past frames where all heights are PASS
    if filter_low:
      all_pass = all(
        bool((d or {}).get('ll_quality_pass', True)) for d in data_per_height.values()
      )
      if all_pass:
        idx = min(idx + 1, total - 1)
        continue

    if grid_mode:
      img = render_grid(
        data_per_height=data_per_height, tags=tags,
        heights_info=heights_info, idx=idx, total=total,
        rpyCalib=rpyCalib,
        warp_road=warp_road, warp_wide=warp_wide, K_road=K_road, K_wide=K_wide,
        clip_info=clip_info, cam_idx=cam_idx,
        show_ann=show_ann, show_lanes=show_lanes, show_edges=show_edges,
        show_leads=show_leads, show_z=show_z, show_raw=show_raw,
        min_ll_prob=min_ll_prob,
      )
    else:
      tag      = tags[single_tag_idx % len(tags)]
      height_m = heights_info.get(tag, 1.22)
      data     = data_per_height.get(tag) or {}
      img = render_single(
        data=data, tag=tag, height_m=height_m,
        rpyCalib=rpyCalib,
        warp_road=warp_road, warp_wide=warp_wide, K_road=K_road, K_wide=K_wide,
        cam_idx=cam_idx,
        show_ann=show_ann, show_lanes=show_lanes, show_edges=show_edges,
        show_leads=show_leads, show_z=show_z, show_bev=show_bev,
        show_raw=show_raw, min_ll_prob=min_ll_prob,
        idx=idx, total=total,
      )

    cam_name = 'NARROW' if cam_idx == 0 else 'WIDE'
    mode_str = 'GRID' if grid_mode else f'SINGLE {tags[single_tag_idx % len(tags)]}'
    cv2.setWindowTitle(win, f"[{idx}/{total-1}] {mode_str} [{cam_name}] | {annotated_dir.name}")
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
    elif key == ord('g'):
      grid_mode = not grid_mode
      if grid_mode:
        cv2.resizeWindow(win, cols * PANEL_W, rows * PANEL_H + 28)
      else:
        w = DISPLAY_W + (BEV_W if show_bev else 0)
        cv2.resizeWindow(win, w, DISPLAY_H + 30)
    elif ord('1') <= key <= ord('6'):
      n = key - ord('1')
      if n < len(tags):
        single_tag_idx = n
        grid_mode = False
        w = DISPLAY_W + (BEV_W if show_bev else 0)
        cv2.resizeWindow(win, w, DISPLAY_H + 30)
        print(f"Height: {tags[single_tag_idx]}")
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
    elif key == ord('b'):
      show_bev = not show_bev
      if not grid_mode:
        w = DISPLAY_W + (BEV_W if show_bev else 0)
        cv2.resizeWindow(win, w, DISPLAY_H + 30)
    elif key == ord('r'):
      show_raw = not show_raw
    elif key == ord('f'):
      filter_low = not filter_low
      print(f"Filter-low: {'ON' if filter_low else 'OFF'}")
    elif key == ord('s'):
      path = f"screenshot_compare_{idx:06d}.png"
      cv2.imwrite(path, img)
      print(f"Screenshot saved: {path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
