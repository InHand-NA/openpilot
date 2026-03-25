#!/usr/bin/env python3
"""TuSimple Phase 1 可视化：多相机网格视图。

显示采集结果（H0 road/wide + H1~H6 mono），支持交互式帧浏览。
当存在 _prev.png 时自动切换为田字格布局，显示 main + prev 对比。

无 prev 时布局:
  ┌──────────────┬──────────────┐
  │  H0 road     │  H0 wide     │  640×400
  ├────┬────┬────┼────┬────┬────┤
  │ H1 │ H2 │ H3 │ H4 │ H5 │ H6 │  ~213×120
  └────┴────┴────┴────┴────┴────┘
  [info bar]

有 prev 时田字格布局:
  ┌──────────────┬──────────────┐
  │  H0 road     │  H0 wide     │  main (640×300)
  ├──────────────┼──────────────┤
  │  road prev   │  wide prev   │  prev (640×300)
  ├────┬────┬────┼────┬────┬────┤
  │ H1 │ H2 │ H3 │ H4 │ H5 │ H6 │  ~213×120
  └────┴────┴────┴────┴────┴────┘
  [info bar]

操作:
  Left/Right    上/下一帧
  PageUp/Down   ±10 帧
  Home/End      首/末帧
  s             截图 (PNG)
  q / ESC       退出

用法:
  python tools/dashcam/tusimple/viz_collect.py data/tusimple/Town04_ClearNoon_p5.0_y0.0/
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

INFO_BAR_H = 30


def resize_keep_ar(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
  """Resize image to fit within target_w × target_h, preserving aspect ratio."""
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
  """Extract sorted main frame ID list from H0/road_*.png (exclude _prev)."""
  h0_dir = session_dir / 'H0'
  if not h0_dir.exists():
    return []
  pattern = re.compile(r'^road_(\d+)\.png$')
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


def _has_prev(session_dir: Path) -> bool:
  """Check if any _prev.png exists in H0/."""
  h0_dir = session_dir / 'H0'
  return h0_dir.exists() and any(h0_dir.glob('road_*_prev.png'))


def load_frame(session_dir: Path, frame_id: str, heights: list[str], clip_info: dict,
               metadata: dict[int, dict] | None = None, has_prev: bool = False) -> dict:
  """Load all camera images and metadata for a given frame."""
  result = {'frame_id': frame_id, 'mono': {}, 'v_ego': None,
            'h0_road': None, 'h0_wide': None,
            'h0_road_prev': None, 'h0_wide_prev': None}

  if metadata is not None:
    frame_tick = int(frame_id)
    rec = metadata.get(frame_tick)
    if rec is not None:
      result['v_ego'] = rec.get('v_ego')

  h0_dir = session_dir / 'H0'
  # Main H0
  road_path = h0_dir / f'road_{frame_id}.png'
  wide_path = h0_dir / f'wide_{frame_id}.png'
  result['h0_road'] = cv2.imread(str(road_path)) if road_path.exists() else None
  result['h0_wide'] = cv2.imread(str(wide_path)) if wide_path.exists() else None

  # Prev H0 (named by main frame ID)
  if has_prev:
    road_prev = h0_dir / f'road_{frame_id}_prev.png'
    wide_prev = h0_dir / f'wide_{frame_id}_prev.png'
    result['h0_road_prev'] = cv2.imread(str(road_prev)) if road_prev.exists() else None
    result['h0_wide_prev'] = cv2.imread(str(wide_prev)) if wide_prev.exists() else None

  # Mono images
  mono_fmt = clip_info.get('mono_camera', {}).get('format', 'png')
  mono_ext = '.jpg' if mono_fmt == 'jpeg' else '.png'
  for h in heights:
    mono_path = session_dir / h / f'{frame_id}{mono_ext}'
    if not mono_path.exists() and mono_ext == '.jpg':
      mono_path = session_dir / h / f'{frame_id}.png'
    if not mono_path.exists() and mono_ext == '.png':
      mono_path = session_dir / h / f'{frame_id}.jpg'
    result['mono'][h] = cv2.imread(str(mono_path)) if mono_path.exists() else None

  return result


def _make_h0_panel(img: np.ndarray | None, label: str, w: int, h: int) -> np.ndarray:
  if img is not None:
    panel = resize_keep_ar(img, w, h)
  else:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
  cv2.putText(panel, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
  return panel


def render_grid(frame_data: dict, frame_id: str, clip_info: dict,
                heights: list[str], has_prev: bool) -> np.ndarray:
  """Render camera grid image."""
  grid_w = 1280

  if has_prev:
    # 田字格: 2 rows of H0 (main + prev), each 640×300
    h0_w, h0_h = 640, 300
    road_main = _make_h0_panel(frame_data['h0_road'], 'H0 road (main)', h0_w, h0_h)
    wide_main = _make_h0_panel(frame_data['h0_wide'], 'H0 wide (main)', h0_w, h0_h)
    road_prev = _make_h0_panel(frame_data['h0_road_prev'], 'H0 road (prev)', h0_w, h0_h)
    wide_prev = _make_h0_panel(frame_data['h0_wide_prev'], 'H0 wide (prev)', h0_w, h0_h)
    h0_top = np.hstack([road_main, wide_main])
    h0_bot = np.hstack([road_prev, wide_prev])
    h0_section = np.vstack([h0_top, h0_bot])
  else:
    # Original: single row of H0, 640×400
    h0_w, h0_h = 640, 400
    road_panel = _make_h0_panel(frame_data['h0_road'], 'H0 road (narrow)', h0_w, h0_h)
    wide_panel = _make_h0_panel(frame_data['h0_wide'], 'H0 wide', h0_w, h0_h)
    h0_section = np.hstack([road_panel, wide_panel])

  # Mono row: H1~H6
  n_mono = len(heights)
  mono_w = grid_w // max(n_mono, 1)
  mono_h = int(mono_w * 9 / 16) if n_mono > 0 else 120

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
  mode_str = 'paired' if has_prev else 'dense'
  info_text = f"frame={frame_id}  {speed_str}  {mode_str}  pitch={pitch}  yaw={yaw}"
  cv2.putText(info_bar, info_text, (5, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

  return np.vstack([h0_section, bottom_row, info_bar])


def main():
  parser = argparse.ArgumentParser(description="TuSimple Phase 1 可视化：多相机网格视图")
  parser.add_argument("session_dir", help="session 目录 (包含 H0/, H1/, ... 和 clip_info.json)")
  parser.add_argument("--max-frames", type=int, default=0,
                      help="最大浏览帧数 (0=全部)")
  parser.add_argument("--output-dir", default=None,
                      help="批量输出目录 (非交互模式)")
  parser.add_argument("--heights", nargs='+', default=None,
                      help="要显示的 mono 高度 (默认: clip_info 中所有高度)")
  args = parser.parse_args()

  session_dir = Path(args.session_dir)
  if not session_dir.exists():
    print(f"目录不存在: {session_dir}")
    sys.exit(1)

  clip_info_path = session_dir / 'clip_info.json'
  if clip_info_path.exists():
    with open(clip_info_path) as f:
      clip_info = json.load(f)
  else:
    print(f"[WARN] clip_info.json 不存在: {clip_info_path}")
    clip_info = {}

  if args.heights:
    heights = args.heights
  else:
    heights = sorted(clip_info.get('heights', {}).keys())
  if not heights:
    heights = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']

  frame_ids = load_session_frames(session_dir)
  if not frame_ids:
    print(f"未找到 H0/road_*.png 主帧文件")
    sys.exit(1)

  if args.max_frames > 0:
    frame_ids = frame_ids[:args.max_frames]
  total = len(frame_ids)

  metadata = load_metadata(session_dir)
  has_prev = _has_prev(session_dir)
  mode_str = 'paired (main+prev)' if has_prev else 'dense'
  print(f"共 {total} 帧  heights={heights}  mode={mode_str}  metadata={len(metadata)} 条")

  # Batch output
  if args.output_dir:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fid in enumerate(frame_ids):
      data = load_frame(session_dir, fid, heights, clip_info, metadata, has_prev)
      grid = render_grid(data, fid, clip_info, heights, has_prev)
      cv2.imwrite(str(out_dir / f'grid_{fid}.png'), grid)
      if (i + 1) % 50 == 0:
        print(f"  {i+1}/{total} saved")
    print(f"Done: {total} grid images saved to {out_dir}")
    return

  # Interactive
  idx = 0
  win_name = 'viz_collect_tusimple'
  cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

  print("操作: ←→翻页  PgUp/PgDn±10  Home/End首末  s截图  q退出")

  while True:
    data = load_frame(session_dir, frame_ids[idx], heights, clip_info, metadata, has_prev)
    grid = render_grid(data, frame_ids[idx], clip_info, heights, has_prev)

    cv2.setWindowTitle(win_name, f"[{idx}/{total - 1}] frame={frame_ids[idx]}")
    cv2.resizeWindow(win_name, grid.shape[1], grid.shape[0])
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
