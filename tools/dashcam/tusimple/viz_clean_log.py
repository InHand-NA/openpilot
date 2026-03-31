#!/usr/bin/env python3
"""TuSimple Phase 3 可视化：浏览 clean_log 中的帧，查看 pass/fail 原因。

布局 (1280×460):
  ┌──────────────────┬──────────────────┐
  │ H0 road (narrow) │ H0 wide          │  640×400 each
  │ + lane/edge 标注  │ + lane/edge 标注  │
  └──────────────────┴──────────────────┘
  [info bar: frame_id, pass/fail, reason, v_ego, ll_prob, pitch/yaw]

pass 帧 info bar 为绿色，fail 帧为红色。

操作:
  Left/Right    上/下一帧
  PageUp/Down   ±10 帧
  Home/End      首/末帧
  f             切换 pass-only / fail-only / all 模式
  l             toggle lane_lines
  e             toggle road_edges
  s             截图 (PNG)
  q / ESC       退出

用法:
  python tools/dashcam/tusimple/viz_clean_log.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/

  # 只看 fail 帧
  python tools/dashcam/tusimple/viz_clean_log.py ... --filter fail
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.dashcam.tusimple.viz_annotate import (
  _build_cam_transform,
  _draw_lane_lines,
  _draw_road_edges,
  load_annotation,
)
from openpilot.tools.dashcam.tusimple.viz_collect import (
  resize_keep_ar,
  load_metadata,
  KEY_RIGHT, KEY_LEFT, KEY_PGUP, KEY_PGDN, KEY_HOME, KEY_END,
)

_dc = DEVICE_CAMERAS[('pc', 'unknown')]
K_ROAD = _dc.fcam.intrinsics
K_WIDE = _dc.ecam.intrinsics

PANEL_W = 640
PANEL_H = 400
INFO_BAR_H = 60


def _find_ref_height(label_dir: Path) -> str | None:
  """找到 label_dir 下第一个存在的 H* 子目录名。"""
  for d in sorted(label_dir.iterdir()):
    if d.is_dir() and d.name.startswith('H'):
      return d.name
  return None


def load_clean_log(path: Path) -> list[tuple[str, str]]:
  """Load clean_log.txt → [(frame_id, 'pass'|'fail'), ...]."""
  entries = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = [p.strip() for p in line.split(',')]
      if len(parts) == 2:
        entries.append((parts[0], parts[1]))
  return entries


def render_frame(session_dir: Path, frame_id: str, status: str, reason: str,
                 label_dir: Path, ref_height: str, rpyCalib: np.ndarray, metadata: dict,
                 show_lanes: bool, show_edges: bool) -> np.ndarray:
  """Render H0 road + H0 wide with annotations and pass/fail info bar."""
  anno = load_annotation(label_dir / ref_height, frame_id)
  T_road = _build_cam_transform(K_ROAD, rpyCalib)
  T_wide = _build_cam_transform(K_WIDE, rpyCalib)

  # Left: H0 road
  road_path = session_dir / 'H0' / f'road_{frame_id}.png'
  if road_path.exists():
    road_img = cv2.imread(str(road_path))
    if anno is not None and show_lanes and 'lane_lines' in anno:
      _draw_lane_lines(road_img, anno, T_road)
    if anno is not None and show_edges and 'road_edges' in anno:
      _draw_road_edges(road_img, anno, T_road)
    left = resize_keep_ar(road_img, PANEL_W, PANEL_H)
  else:
    left = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
  cv2.putText(left, 'H0 road', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  # Right: H0 wide
  wide_path = session_dir / 'H0' / f'wide_{frame_id}.png'
  if wide_path.exists():
    wide_img = cv2.imread(str(wide_path))
    if anno is not None and show_lanes and 'lane_lines' in anno:
      _draw_lane_lines(wide_img, anno, T_wide)
    if anno is not None and show_edges and 'road_edges' in anno:
      _draw_road_edges(wide_img, anno, T_wide)
    right = resize_keep_ar(wide_img, PANEL_W, PANEL_H)
  else:
    right = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
  cv2.putText(right, 'H0 wide', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  combined = np.hstack([left, right])

  # Info bar
  is_pass = (status == 'pass')
  bar_color = (0, 60, 0) if is_pass else (0, 0, 60)
  info_bar = np.full((INFO_BAR_H, PANEL_W * 2, 3), bar_color, dtype=np.uint8)

  # Line 1: status + frame + reason
  status_str = 'PASS' if is_pass else 'FAIL'
  status_color = (0, 255, 0) if is_pass else (0, 0, 255)
  cv2.putText(info_bar, f'[{status_str}]', (5, 22),
              cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
  cv2.putText(info_bar, f'frame={frame_id}', (110, 22),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
  if reason:
    cv2.putText(info_bar, f'reason: {reason}', (310, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255) if not is_pass else (180, 255, 180),
                1, cv2.LINE_AA)

  # Line 2: v_ego + ll_prob + pitch/yaw
  frame_tick = int(frame_id)
  meta_rec = metadata.get(frame_tick, {})
  v_ego = meta_rec.get('v_ego')
  speed_str = f"v={v_ego:.1f}m/s" if v_ego is not None else "v=?"

  ll_str = ''
  if anno is not None and 'lane_lines_prob' in anno:
    p = anno['lane_lines_prob']
    ll_str = f"  ll=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, {p[3]:.2f}]"

  info2 = f"{speed_str}{ll_str}"
  cv2.putText(info_bar, info2, (5, 48),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

  # Toggle indicators
  lanes_c = (0, 220, 0) if show_lanes else (80, 80, 80)
  edges_c = (0, 220, 0) if show_edges else (80, 80, 80)
  cv2.putText(info_bar, '[L]anes', (PANEL_W * 2 - 200, 48),
              cv2.FONT_HERSHEY_SIMPLEX, 0.4, lanes_c, 1)
  cv2.putText(info_bar, '[E]dges', (PANEL_W * 2 - 120, 48),
              cv2.FONT_HERSHEY_SIMPLEX, 0.4, edges_c, 1)

  return np.vstack([combined, info_bar])


def main():
  parser = argparse.ArgumentParser(description='TuSimple Phase 3 可视化: clean_log pass/fail 浏览')
  parser.add_argument('session_dir', help='Session 目录')
  parser.add_argument('--filter', choices=['all', 'pass', 'fail'], default='all',
                      help='显示模式 (default: all)')
  parser.add_argument('--max-frames', type=int, default=0,
                      help='最大浏览帧数 (0=全部)')
  parser.add_argument('--output-dir', default=None,
                      help='批量输出目录 (非交互模式)')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"目录不存在: {session_dir}")
    sys.exit(1)

  splits_dir = session_dir / 'splits'
  clean_log_path = splits_dir / 'clean_log.txt'
  if not clean_log_path.exists():
    print(f"clean_log.txt 不存在: {clean_log_path}")
    print(f"请先运行 clean_and_sample.py")
    sys.exit(1)

  label_dir = session_dir / '3d_labels'
  if not label_dir.exists():
    print(f"3d_labels 不存在: {label_dir}")
    sys.exit(1)

  # Load clip_info for rpyCalib
  clip_info_path = session_dir / 'clip_info.json'
  if clip_info_path.exists():
    with open(clip_info_path) as f:
      clip_info = json.load(f)
  else:
    clip_info = {}
  cam = clip_info.get('camera', {})
  pitch_rad = math.radians(cam.get('pitch_deg', 4.0))
  yaw_rad = math.radians(cam.get('yaw_deg', 0.0))
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  # Load clean_log
  all_entries = load_clean_log(clean_log_path)

  # Build fail reasons from annotation data
  # Re-run the quality checks to get reasons (lightweight — just reads JSON)
  reasons: dict[str, str] = {}
  stats_path = splits_dir / 'stats.json'
  if stats_path.exists():
    with open(stats_path) as f:
      stats = json.load(f)
    min_ll_prob = stats.get('params', {}).get('min_ll_prob', 0.5)
    min_speed = stats.get('params', {}).get('min_speed', 1.0)
  else:
    min_ll_prob, min_speed = 0.5, 1.0

  ref_height = _find_ref_height(label_dir)
  if ref_height is None:
    print(f"3d_labels 中无 H* 子目录: {label_dir}")
    sys.exit(1)
  print(f"标注参考高度: {ref_height}")

  for fid, status in all_entries:
    if status == 'fail':
      anno = load_annotation(label_dir / ref_height, fid)
      if anno is None:
        reasons[fid] = 'no_annotation'
        continue
      ll_prob = anno.get('lane_lines_prob', [0] * 4)
      v_ego = float(anno.get('v_ego', 0))
      if not (float(ll_prob[1]) > min_ll_prob and float(ll_prob[2]) > min_ll_prob):
        reasons[fid] = f'll_prob L={float(ll_prob[1]):.3f} R={float(ll_prob[2]):.3f} (thresh={min_ll_prob})'
      elif v_ego <= min_speed:
        reasons[fid] = f'v_ego={v_ego:.2f} (thresh={min_speed})'
      else:
        reasons[fid] = 'unknown'

  # Apply filter
  filter_mode = args.filter

  def apply_filter(mode: str) -> list[tuple[str, str]]:
    if mode == 'pass':
      return [(f, s) for f, s in all_entries if s == 'pass']
    elif mode == 'fail':
      return [(f, s) for f, s in all_entries if s == 'fail']
    return all_entries

  entries = apply_filter(filter_mode)
  if args.max_frames > 0:
    entries = entries[:args.max_frames]

  n_pass = sum(1 for _, s in all_entries if s == 'pass')
  n_fail = sum(1 for _, s in all_entries if s == 'fail')
  total = len(all_entries)
  print(f"Session: {session_dir.name}")
  print(f"clean_log: {total} 帧  pass={n_pass} ({n_pass/max(total,1):.1%})  fail={n_fail}")
  print(f"当前模式: {filter_mode}  显示 {len(entries)} 帧")

  metadata = load_metadata(session_dir)

  # Batch output
  if args.output_dir:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (fid, status) in enumerate(entries):
      img = render_frame(session_dir, fid, status, reasons.get(fid, ''),
                         label_dir, ref_height, rpyCalib, metadata, True, True)
      cv2.imwrite(str(out_dir / f'clean_{status}_{fid}.png'), img)
      if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(entries)} saved")
    print(f"Done: {len(entries)} images saved to {out_dir}")
    return

  if not entries:
    print("无帧可显示")
    sys.exit(0)

  # Interactive mode
  idx = 0
  show_lanes = True
  show_edges = True
  win_name = 'viz_clean_log'
  cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
  cv2.resizeWindow(win_name, PANEL_W * 2, PANEL_H + INFO_BAR_H)

  print("操作: ←→翻页  PgUp/PgDn±10  Home/End首末  f切换模式  l车道线  e路沿  s截图  q退出")

  while True:
    fid, status = entries[idx]
    img = render_frame(session_dir, fid, status, reasons.get(fid, ''),
                       label_dir, ref_height, rpyCalib, metadata, show_lanes, show_edges)

    mode_str = filter_mode.upper()
    cv2.setWindowTitle(win_name,
      f"[{idx}/{len(entries)-1}] frame={fid} {status.upper()} | mode={mode_str}")
    cv2.imshow(win_name, img)

    key = cv2.waitKeyEx(0)
    if key == ord('q') or key == 27:
      break
    elif key == KEY_RIGHT:
      idx = min(idx + 1, len(entries) - 1)
    elif key == KEY_LEFT:
      idx = max(idx - 1, 0)
    elif key == KEY_PGDN:
      idx = min(idx + 10, len(entries) - 1)
    elif key == KEY_PGUP:
      idx = max(idx - 10, 0)
    elif key == KEY_HOME:
      idx = 0
    elif key == KEY_END:
      idx = len(entries) - 1
    elif key == ord('f'):
      # Cycle: all → pass → fail → all
      cycle = {'all': 'pass', 'pass': 'fail', 'fail': 'all'}
      filter_mode = cycle[filter_mode]
      entries = apply_filter(filter_mode)
      if args.max_frames > 0:
        entries = entries[:args.max_frames]
      idx = min(idx, max(0, len(entries) - 1))
      print(f"模式: {filter_mode}  ({len(entries)} 帧)")
      if not entries:
        print("当前模式无帧可显示，切换到 all")
        filter_mode = 'all'
        entries = apply_filter(filter_mode)
        idx = 0
    elif key == ord('l'):
      show_lanes = not show_lanes
      print(f"车道线: {'ON' if show_lanes else 'OFF'}")
    elif key == ord('e'):
      show_edges = not show_edges
      print(f"路沿: {'ON' if show_edges else 'OFF'}")
    elif key == ord('s'):
      screenshot_path = f"screenshot_clean_{status}_{fid}.png"
      cv2.imwrite(screenshot_path, img)
      print(f"截图已保存: {screenshot_path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
