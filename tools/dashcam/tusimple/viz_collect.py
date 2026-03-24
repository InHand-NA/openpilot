#!/usr/bin/env python3
"""TuSimple Phase 1 可视化：多相机网格视图。

显示 8 相机采集结果（H0 road/wide + H1~H6 mono），支持交互式帧浏览。

网格布局:
  ┌──────────────┬──────────────┐
  │  H0 road     │  H0 wide     │  640×400 each
  ├────┬────┬────┼────┬────┬────┤
  │ H1 │ H2 │ H3 │ H4 │ H5 │ H6 │  ~213×180 each
  └────┴────┴────┴────┴────┴────┘
  [info bar: frame_id, session, pitch/yaw, speed]

用法:
  python tools/dashcam/tusimple/viz_collect.py data/tusimple/Town04_ClearNoon_p5.0_y0.0/
  python tools/dashcam/tusimple/viz_collect.py data/tusimple/Town04_ClearNoon_p5.0_y0.0/ --max-frames 20

操作:
  Left/Right    上/下一帧
  PageUp/Down   ±10 帧
  Home/End      首/末帧
  s             截图 (PNG)
  q / ESC       退出
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# OpenCV key codes (Linux GTK)
KEY_RIGHT = 65363
KEY_LEFT = 65361
KEY_PGUP = 65365
KEY_PGDN = 65366
KEY_HOME = 65360
KEY_END = 65367

# Grid layout constants
INFO_BAR_H = 30


def resize_keep_ar(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
  """Resize image to fit within target_w × target_h, preserving aspect ratio.

  Scales uniformly to fit the target box, then centers on a black canvas.
  """
  src_h, src_w = img.shape[:2]
  scale = min(target_w / src_w, target_h / src_h)
  new_w = int(src_w * scale)
  new_h = int(src_h * scale)
  resized = cv2.resize(img, (new_w, new_h))
  canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
  y_off = (target_h - new_h) // 2
  x_off = (target_w - new_w) // 2
  canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
  return canvas


def load_session_frames(session_dir: Path) -> list[str]:
  """Extract sorted frame ID list from H0/road_*.png filenames."""
  h0_dir = session_dir / 'H0'
  if not h0_dir.exists():
    return []
  pattern = re.compile(r'road_(\d+)\.png')
  frame_ids = []
  for f in sorted(h0_dir.iterdir()):
    m = pattern.match(f.name)
    if m:
      frame_ids.append(m.group(1))
  return frame_ids


def load_metadata(session_dir: Path) -> dict[int, dict]:
  """Load H0/metadata.jsonl into {frame_tick: {v_ego, world_pose, ...}} dict."""
  meta_path = session_dir / 'H0' / 'metadata.jsonl'
  meta = {}
  if meta_path.exists():
    with open(meta_path) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        rec = json.loads(line)
        meta[rec['frame']] = rec
  return meta


def load_frame(session_dir: Path, frame_id: str, heights: list[str], clip_info: dict,
               metadata: dict[int, dict] | None = None) -> dict:
  """Load all camera images and metadata for a given frame.

  Returns:
    {
      'h0_road': np.ndarray (BGR),
      'h0_wide': np.ndarray (BGR),
      'mono': {'H1': np.ndarray (BGR), ...},
      'frame_id': str,
      'v_ego': float | None,
    }
  """
  result = {'frame_id': frame_id, 'mono': {}, 'v_ego': None}

  # Look up v_ego from metadata
  if metadata is not None:
    frame_tick = int(frame_id)
    rec = metadata.get(frame_tick)
    if rec is not None:
      result['v_ego'] = rec.get('v_ego')

  # H0 images
  h0_road_path = session_dir / 'H0' / f'road_{frame_id}.png'
  h0_wide_path = session_dir / 'H0' / f'wide_{frame_id}.png'
  result['h0_road'] = cv2.imread(str(h0_road_path)) if h0_road_path.exists() else None
  result['h0_wide'] = cv2.imread(str(h0_wide_path)) if h0_wide_path.exists() else None

  # Mono images (detect format from clip_info or try both)
  mono_fmt = clip_info.get('mono_camera', {}).get('format', 'png')
  mono_ext = '.jpg' if mono_fmt == 'jpeg' else '.png'
  for h in heights:
    mono_path = session_dir / h / f'{frame_id}{mono_ext}'
    if not mono_path.exists() and mono_ext == '.jpg':
      mono_path = session_dir / h / f'{frame_id}.png'  # fallback
    if not mono_path.exists() and mono_ext == '.png':
      mono_path = session_dir / h / f'{frame_id}.jpg'  # fallback
    result['mono'][h] = cv2.imread(str(mono_path)) if mono_path.exists() else None

  return result


def render_grid(frame_data: dict, frame_id: str, clip_info: dict, heights: list[str]) -> np.ndarray:
  """Render 8-camera grid image (aspect-ratio preserving).

  Layout:
    Top row: H0 road + H0 wide (640×400 each) → 1280×400
    Bottom row: H1~H6 mono (~213×120 each, letterboxed) → 1280×120
    Info bar: 1280×30
  """
  h0_panel_w, h0_panel_h = 640, 400
  grid_w = h0_panel_w * 2  # 1280
  n_mono = len(heights)
  mono_w = grid_w // max(n_mono, 1)
  # Mono panel height: fit 16:9 source into mono_w, preserving AR
  mono_h = int(mono_w * 9 / 16) if n_mono > 0 else 120

  # Top row: H0 road + wide
  if frame_data.get('h0_road') is not None:
    road_panel = resize_keep_ar(frame_data['h0_road'], h0_panel_w, h0_panel_h)
  else:
    road_panel = np.zeros((h0_panel_h, h0_panel_w, 3), dtype=np.uint8)
  cv2.putText(road_panel, 'H0 road (narrow)', (5, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  if frame_data.get('h0_wide') is not None:
    wide_panel = resize_keep_ar(frame_data['h0_wide'], h0_panel_w, h0_panel_h)
  else:
    wide_panel = np.zeros((h0_panel_h, h0_panel_w, 3), dtype=np.uint8)
  cv2.putText(wide_panel, 'H0 wide', (5, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  top_row = np.hstack([road_panel, wide_panel])

  # Bottom row: H1~H6 mono
  mono_panels = []
  height_defs = clip_info.get('heights', {})
  for h in heights:
    mono_img = frame_data['mono'].get(h)
    if mono_img is not None:
      panel = resize_keep_ar(mono_img, mono_w, mono_h)
    else:
      panel = np.zeros((mono_h, mono_w, 3), dtype=np.uint8)
    h_val = height_defs.get(h, '?')
    cv2.putText(panel, f'{h} {h_val}m', (3, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    mono_panels.append(panel)

  if mono_panels:
    bottom_row = np.hstack(mono_panels)
    # Pad to match top row width
    if bottom_row.shape[1] < grid_w:
      pad = np.zeros((mono_h, grid_w - bottom_row.shape[1], 3), dtype=np.uint8)
      bottom_row = np.hstack([bottom_row, pad])
    elif bottom_row.shape[1] > grid_w:
      bottom_row = bottom_row[:, :grid_w]
  else:
    bottom_row = np.zeros((mono_h, grid_w, 3), dtype=np.uint8)

  # Info bar
  info_bar = np.zeros((INFO_BAR_H, grid_w, 3), dtype=np.uint8)
  session_id = clip_info.get('session_id', '?')
  cam = clip_info.get('camera', {})
  pitch = cam.get('pitch_deg', '?')
  yaw = cam.get('yaw_deg', '?')
  v_ego = frame_data.get('v_ego')
  speed_str = f"v={v_ego:.1f}m/s ({v_ego * 3.6:.0f}km/h)" if v_ego is not None else "v=?"
  info_text = f"frame={frame_id}  {speed_str}  session={session_id}  pitch={pitch}  yaw={yaw}"
  cv2.putText(info_bar, info_text, (5, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

  grid = np.vstack([top_row, bottom_row, info_bar])
  return grid


def main():
  parser = argparse.ArgumentParser(description="TuSimple Phase 1 可视化：多相机网格视图")
  parser.add_argument("session_dir", help="session 目录 (包含 H0/, H1/, ... 和 clip_info.json)")
  parser.add_argument("--max-frames", type=int, default=0,
                      help="最大浏览帧数 (0=全部)")
  parser.add_argument("--output-dir", default=None,
                      help="批量输出目录 (非交互模式，保存所有帧网格图)")
  parser.add_argument("--heights", nargs='+', default=None,
                      help="要显示的 mono 高度 (默认: clip_info 中所有高度)")
  args = parser.parse_args()

  session_dir = Path(args.session_dir)
  if not session_dir.exists():
    print(f"目录不存在: {session_dir}")
    sys.exit(1)

  # Load clip_info
  clip_info_path = session_dir / 'clip_info.json'
  if clip_info_path.exists():
    with open(clip_info_path) as f:
      clip_info = json.load(f)
  else:
    print(f"[WARN] clip_info.json 不存在: {clip_info_path}")
    clip_info = {}

  # Determine heights
  if args.heights:
    heights = args.heights
  else:
    heights = sorted(clip_info.get('heights', {}).keys())
  if not heights:
    heights = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']

  # Load frame IDs
  frame_ids = load_session_frames(session_dir)
  if not frame_ids:
    print(f"未找到 H0/road_*.png 帧文件")
    sys.exit(1)

  if args.max_frames > 0:
    frame_ids = frame_ids[:args.max_frames]
  total = len(frame_ids)

  # Load metadata (frame tick → v_ego etc.)
  metadata = load_metadata(session_dir)
  print(f"共 {total} 帧  heights={heights}  metadata={len(metadata)} 条")

  # Batch output mode
  if args.output_dir:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fid in enumerate(frame_ids):
      data = load_frame(session_dir, fid, heights, clip_info, metadata)
      grid = render_grid(data, fid, clip_info, heights)
      out_path = out_dir / f'grid_{fid}.png'
      cv2.imwrite(str(out_path), grid)
      if (i + 1) % 50 == 0:
        print(f"  {i+1}/{total} saved")
    print(f"Done: {total} grid images saved to {out_dir}")
    return

  # Interactive mode
  idx = 0
  n_mono = len(heights)
  mono_row_h = int((1280 // max(n_mono, 1)) * 9 / 16) if n_mono > 0 else 120
  win_h = 400 + mono_row_h + INFO_BAR_H
  win_name = 'viz_collect_tusimple'
  cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win_name, 1280, win_h)

  print("操作: ←→翻页  PgUp/PgDn±10  Home/End首末  s截图  q退出")

  while True:
    data = load_frame(session_dir, frame_ids[idx], heights, clip_info, metadata)
    grid = render_grid(data, frame_ids[idx], clip_info, heights)

    cv2.setWindowTitle(win_name, f"[{idx}/{total - 1}] frame={frame_ids[idx]}")
    cv2.imshow(win_name, grid)

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
    elif key == ord('s'):
      screenshot_path = f"screenshot_tusimple_{frame_ids[idx]}.png"
      cv2.imwrite(screenshot_path, grid)
      print(f"截图已保存: {screenshot_path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
