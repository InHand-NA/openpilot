#!/usr/bin/env python3
"""TuSimple Phase 4 可视化：在裁剪后的 1280×720 图像上检查 TuSimple 2D 车道线标注。

直接加载 tusimple/ 目录下的 JSON Lines 标签文件和对应 JPEG 图像进行渲染。

绘制规则:
  slot 0 (L-out): 蓝色  (255, 0, 0)
  slot 1 (L-inn): 绿色  (0, 220, 0)
  slot 2 (R-inn): 红色  (0, 0, 220)
  slot 3 (R-out): 黄色  (0, 220, 220)
  有效采样点画实心圆，有效点间画连线。

操作:
  Left/Right    上/下一帧
  PageUp/Down   ±10 帧
  Home/End      首/末帧
  s             截图 (PNG)
  q / ESC       退出

用法:
  # 查看 train.json
  python tools/dashcam/tusimple/viz_tusimple.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/ \\
      --split train --max-frames 20

  # 查看指定高度的 labels.json
  python tools/dashcam/tusimple/viz_tusimple.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/ \\
      --height H1

  # 批量输出
  python tools/dashcam/tusimple/viz_tusimple.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/ \\
      --split train --output-dir /tmp/viz_tusimple
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from openpilot.tools.dashcam.tusimple.config import TUSIMPLE_W, TUSIMPLE_H

KEY_RIGHT = 65363
KEY_LEFT = 65361
KEY_PGUP = 65365
KEY_PGDN = 65366
KEY_HOME = 65360
KEY_END = 65367

# Lane slot colors (BGR)
LANE_COLORS = [
  (255, 0, 0),      # slot 0 L-out: blue
  (0, 220, 0),      # slot 1 L-inn: green
  (0, 0, 220),      # slot 2 R-inn: red
  (0, 220, 220),    # slot 3 R-out: yellow
]
LANE_LABELS = ['L-out', 'L-inn', 'R-inn', 'R-out']

INFO_BAR_H = 48


def load_tusimple_labels(path: Path) -> list[dict]:
  """Load TuSimple JSON Lines file → list of records."""
  records = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return records


def draw_tusimple_lanes(img: np.ndarray, lanes: list[list[int]],
                        h_samples: list[int]) -> None:
  """Draw TuSimple lanes on a 1280×720 image."""
  for i, lane in enumerate(lanes):
    color = LANE_COLORS[i % len(LANE_COLORS)]
    # Collect valid points
    pts = []
    for j, x in enumerate(lane):
      if x != -2:
        pts.append((x, h_samples[j]))
    # Draw connecting lines
    for k in range(len(pts) - 1):
      cv2.line(img, pts[k], pts[k + 1], color, 2, cv2.LINE_AA)
    # Draw sample points
    for pt in pts:
      cv2.circle(img, pt, 4, color, -1, cv2.LINE_AA)


def render_frame(tusimple_dir: Path, record: dict, idx: int, total: int) -> np.ndarray:
  """Render one TuSimple frame: cropped image + lane annotations + info bar."""
  raw_file = record['raw_file']
  lanes = record['lanes']
  h_samples = record['h_samples']

  # Load cropped JPEG
  img_path = tusimple_dir / raw_file
  if img_path.exists():
    img = cv2.imread(str(img_path))
  else:
    img = np.zeros((TUSIMPLE_H, TUSIMPLE_W, 3), dtype=np.uint8)
    cv2.putText(img, f'Image not found: {raw_file}', (10, TUSIMPLE_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)

  # Draw lanes
  draw_tusimple_lanes(img, lanes, h_samples)

  # Info bar
  info_bar = np.zeros((INFO_BAR_H, TUSIMPLE_W, 3), dtype=np.uint8)

  # Count visible points per lane
  lane_info_parts = []
  for i, lane in enumerate(lanes):
    n_vis = sum(1 for x in lane if x != -2)
    label = LANE_LABELS[i] if i < len(LANE_LABELS) else f'L{i}'
    lane_info_parts.append(f'{label}={n_vis}')
  lane_str = '  '.join(lane_info_parts)

  cv2.putText(info_bar, f'[{idx}/{total - 1}]  {raw_file}', (5, 18),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
  cv2.putText(info_bar, lane_str, (5, 40),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

  # Legend
  lx = TUSIMPLE_W - 380
  for i, (label, color) in enumerate(zip(LANE_LABELS, LANE_COLORS)):
    x = lx + i * 90
    cv2.circle(info_bar, (x, 36), 5, color, -1)
    cv2.putText(info_bar, label, (x + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

  return np.vstack([img, info_bar])


def main():
  parser = argparse.ArgumentParser(description='TuSimple Phase 4 可视化: 2D 车道线标注检查')
  parser.add_argument('session_dir', help='Session 目录 (含 tusimple/)')
  parser.add_argument('--split', choices=['train', 'val', 'test'], default=None,
                      help='加载 train.json / val.json / test.json')
  parser.add_argument('--height', default=None,
                      help='加载指定高度的 labels.json (如 H1)')
  parser.add_argument('--label-file', default=None,
                      help='直接指定 JSON Lines 标签文件路径')
  parser.add_argument('--max-frames', type=int, default=0,
                      help='最大浏览帧数 (0=全部)')
  parser.add_argument('--output-dir', default=None,
                      help='批量输出目录 (非交互模式)')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  tusimple_dir = session_dir / 'tusimple'
  if not tusimple_dir.exists():
    print(f"tusimple/ 不存在: {tusimple_dir}")
    print(f"请先运行 project_tusimple.py")
    sys.exit(1)

  # Determine label file
  if args.label_file:
    label_path = Path(args.label_file)
  elif args.split:
    label_path = tusimple_dir / f'{args.split}.json'
  elif args.height:
    label_path = tusimple_dir / args.height / 'labels.json'
  else:
    # Default: try train.json
    label_path = tusimple_dir / 'train.json'
    if not label_path.exists():
      # Fallback to first height's labels.json
      for d in sorted(tusimple_dir.iterdir()):
        if d.is_dir() and (d / 'labels.json').exists():
          label_path = d / 'labels.json'
          break

  if not label_path.exists():
    print(f"标签文件不存在: {label_path}")
    sys.exit(1)

  records = load_tusimple_labels(label_path)
  if not records:
    print(f"标签文件为空: {label_path}")
    sys.exit(1)

  if args.max_frames > 0:
    records = records[:args.max_frames]
  total = len(records)

  print(f"Session: {session_dir.name}")
  print(f"标签: {label_path.name}  ({total} 帧)")

  # Batch output
  if args.output_dir:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(records):
      img = render_frame(tusimple_dir, rec, i, total)
      raw_file = rec['raw_file']
      safe_name = raw_file.replace('/', '_').replace('.jpg', '.png')
      cv2.imwrite(str(out_dir / safe_name), img)
      if (i + 1) % 100 == 0:
        print(f"  {i+1}/{total} saved")
    print(f"Done: {total} images saved to {out_dir}")
    return

  # Interactive
  idx = 0
  win_name = 'viz_tusimple'
  cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
  cv2.resizeWindow(win_name, TUSIMPLE_W, TUSIMPLE_H + INFO_BAR_H)

  print("操作: ←→翻页  PgUp/PgDn±10  Home/End首末  s截图  q退出")

  while True:
    img = render_frame(tusimple_dir, records[idx], idx, total)
    cv2.setWindowTitle(win_name, f"[{idx}/{total-1}] {records[idx]['raw_file']}")
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
    elif key == ord('s'):
      raw_file = records[idx]['raw_file']
      safe_name = f"screenshot_tusimple_{raw_file.replace('/', '_').replace('.jpg', '.png')}"
      cv2.imwrite(safe_name, img)
      print(f"截图: {safe_name}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
