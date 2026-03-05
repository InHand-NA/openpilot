#!/usr/bin/env python3
"""Browse raw multi-height session data (Step A output).

Displays a grid of road_rgb (or wide_rgb) images for all height slots simultaneously.
Useful for verifying that all cameras captured data correctly after collection.

Usage:
  python tools/dashcam/viz/browse_raw_session.py data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/
  python tools/dashcam/viz/browse_raw_session.py <session_dir> --heights H1 H4 H6
  python tools/dashcam/viz/browse_raw_session.py <session_dir> --wide-road  # show wide camera

Keys:
  Left/Right   - prev/next frame
  PgUp/PgDn    - ±20 frames (1 second of sim time at 20 FPS)
  Home/End     - first/last frame
  w            - toggle road/wide camera
  i            - toggle info overlay
  s            - save screenshot
  q / ESC      - quit
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# Key codes (Linux GTK)
KEY_LEFT  = 65361
KEY_RIGHT = 65363
KEY_PGUP  = 65365
KEY_PGDN  = 65366
KEY_HOME  = 65360
KEY_END   = 65367


def load_frame(path: Path) -> dict | None:
  try:
    return dict(np.load(path, allow_pickle=True))
  except Exception as e:
    print(f"  WARN: failed to load {path}: {e}")
    return None


def render_grid(
  frame_files_per_height: dict[str, list[Path]],
  heights_info: dict[str, float],
  idx: int,
  clip_info: dict,
  show_wide: bool,
  show_info: bool,
  panel_w: int = 640,
  panel_h: int = 400,
) -> np.ndarray:
  """Render a grid of images, one per height."""
  tags = list(frame_files_per_height.keys())
  n = len(tags)
  cols = min(3, n)
  rows = (n + cols - 1) // cols

  grid_w = cols * panel_w
  grid_h = rows * panel_h
  grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

  for i, tag in enumerate(tags):
    files = frame_files_per_height[tag]
    row, col = divmod(i, cols)
    x0, y0 = col * panel_w, row * panel_h

    if idx >= len(files):
      continue
    data = load_frame(files[idx])
    if data is None:
      continue

    rgb_key = 'wide_rgb' if show_wide else 'road_rgb'
    if rgb_key not in data:
      rgb_key = 'road_rgb'
    rgb = data[rgb_key]
    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (panel_w, panel_h))

    # Height label (top-left)
    h_m = heights_info.get(tag, 0.0)
    label = f"{tag}  {h_m:.2f}m"
    cv2.rectangle(bgr, (0, 0), (180, 28), (0, 0, 0), -1)
    cv2.putText(bgr, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)

    # Frame number (top-right)
    total_h = len(files)
    fn_text = f"{idx:06d}/{total_h - 1:06d}"
    (tw, _), _ = cv2.getTextSize(fn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(bgr, fn_text, (panel_w - tw - 6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # v_ego (bottom-left)
    v_ego = float(data.get('v_ego', 0))
    v_text = f"{v_ego:.1f} m/s  ({v_ego * 3.6:.0f} km/h)"
    cv2.putText(bgr, v_text, (6, panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

    grid[y0:y0 + panel_h, x0:x0 + panel_w] = bgr

  # Status bar at bottom
  if show_info:
    bar_h = 30
    bar = np.zeros((bar_h, grid_w, 3), dtype=np.uint8)
    map_name = clip_info.get('map', '?')
    weather = clip_info.get('weather', '?')
    cam = clip_info.get('camera', {})
    pitch = cam.get('pitch_deg', 0.0)
    yaw = cam.get('yaw_deg', 0.0)
    cam_label = 'wide' if show_wide else 'road'
    info = (f"Frame {idx}   map={map_name}  weather={weather}  "
            f"pitch={pitch:.1f}°  yaw={yaw:.1f}°   [{cam_label.upper()} cam]   "
            f"w=toggle wide  s=screenshot  q=quit")
    cv2.putText(bar, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    grid = np.vstack([grid, bar])

  return grid


def main():
  parser = argparse.ArgumentParser(description='Browse raw multi-height session data')
  parser.add_argument('session_dir', help='Session directory (contains H1/, H2/, ... and clip_info.json)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='Subset of heights to display (default: all found)')
  parser.add_argument('--start', type=int, default=0, help='Starting frame index')
  parser.add_argument('--wide-road', action='store_true', help='Start with wide camera view')
  args = parser.parse_args()

  session_dir = Path(args.session_dir)
  if not session_dir.exists():
    print(f"ERROR: not found: {session_dir}", file=sys.stderr)
    sys.exit(1)

  # Load clip_info.json
  clip_info_path = session_dir / 'clip_info.json'
  clip_info = {}
  if clip_info_path.exists():
    with open(clip_info_path) as f:
      clip_info = json.load(f)
  heights_info = clip_info.get('heights', {})

  # Discover height directories
  all_tags = sorted([d.name for d in session_dir.iterdir()
                     if d.is_dir() and d.name.startswith('H') and d.name[1:].isdigit()])
  if args.heights:
    tags = [t for t in args.heights if t in all_tags]
    if not tags:
      print(f"ERROR: none of specified heights {args.heights} found in {session_dir}", file=sys.stderr)
      sys.exit(1)
  else:
    tags = all_tags

  if not tags:
    print(f"ERROR: no height directories (H1, H2, ...) found in {session_dir}", file=sys.stderr)
    sys.exit(1)

  # Collect frame file lists per height
  frame_files_per_height: dict[str, list[Path]] = {}
  for tag in tags:
    files = sorted((session_dir / tag).glob('*.npz'))
    if files:
      frame_files_per_height[tag] = files
    else:
      print(f"  WARN: no frames in {session_dir / tag}")
  tags = [t for t in tags if t in frame_files_per_height]

  if not tags:
    print("ERROR: no frames found", file=sys.stderr)
    sys.exit(1)

  total = min(len(frame_files_per_height[t]) for t in tags)
  print(f"Session: {session_dir.name}")
  print(f"Heights: {tags}")
  print(f"Frames: {total}")
  print(f"Keys: ←→ frames  PgUp/PgDn ±20  Home/End  w=wide  i=info  s=screenshot  q=quit")

  idx = max(0, min(args.start, total - 1))
  show_wide = args.wide_road
  show_info = True

  win = 'browse_raw_session'
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  n_cols = min(3, len(tags))
  n_rows = (len(tags) + n_cols - 1) // n_cols
  cv2.resizeWindow(win, n_cols * 640, n_rows * 400 + 30)

  while True:
    img = render_grid(frame_files_per_height, heights_info, idx, clip_info,
                      show_wide, show_info)
    cv2.setWindowTitle(win, f"[{idx}/{total - 1}] {session_dir.name}")
    cv2.imshow(win, img)

    key = cv2.waitKeyEx(0)
    if key in (ord('q'), 27):
      break
    elif key == KEY_RIGHT:
      idx = min(idx + 1, total - 1)
    elif key == KEY_LEFT:
      idx = max(idx - 1, 0)
    elif key == KEY_PGDN:
      idx = min(idx + 20, total - 1)
    elif key == KEY_PGUP:
      idx = max(idx - 20, 0)
    elif key == KEY_HOME:
      idx = 0
    elif key == KEY_END:
      idx = total - 1
    elif key == ord('w'):
      show_wide = not show_wide
    elif key == ord('i'):
      show_info = not show_info
    elif key == ord('s'):
      path = f"screenshot_raw_{idx:06d}.png"
      cv2.imwrite(path, img)
      print(f"Screenshot saved: {path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
