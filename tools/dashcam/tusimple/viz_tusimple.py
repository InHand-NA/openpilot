#!/usr/bin/env python3
"""TuSimple Phase 4 可视化：在裁剪后的 1280×720 图像上检查 TuSimple 2D 车道线标注。

支持两种模式:
  1. 浏览 tusimple_merged/ 下的合并标签 (raw_file 为 data_root 相对路径)
  2. 浏览 per-session tusimple/ 下的标签 (raw_file 为 session 内相对路径)

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
  # 浏览合并标签 (推荐)
  python tools/dashcam/tusimple/viz_tusimple.py data/tusimple-sample --split train

  # 浏览 per-session 标签
  python tools/dashcam/tusimple/viz_tusimple.py data/tusimple-sample/Town04_.../tusimple --split train

  # 直接指定标签文件
  python tools/dashcam/tusimple/viz_tusimple.py data/tusimple-sample --label-file data/tusimple-sample/tusimple_merged/train_1.json
"""

import argparse
import json
import re
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

LANE_COLORS = [
  (255, 0, 0),      # slot 0 L-out: blue
  (0, 220, 0),      # slot 1 L-inn: green
  (0, 0, 220),      # slot 2 R-inn: red
  (0, 220, 220),    # slot 3 R-out: yellow
]
LANE_LABELS = ['L-out', 'L-inn', 'R-inn', 'R-out']
INFO_BAR_H = 48
SPLITS = ['train', 'val', 'test']


def _discover_merged_label(merged_dir: Path, split: str) -> Path | None:
  """Find merged label file: single or first chunk."""
  single = merged_dir / f'{split}.json'
  if single.exists():
    return single
  pattern = re.compile(rf'^{re.escape(split)}_(\d+)\.json$')
  chunks = sorted(
    (p for p in merged_dir.iterdir() if pattern.match(p.name)),
    key=lambda p: int(pattern.match(p.name).group(1)),
  )
  return chunks[0] if chunks else None


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
    pts = []
    for j, x in enumerate(lane):
      if x != -2:
        pts.append((x, h_samples[j]))
    for k in range(len(pts) - 1):
      cv2.line(img, pts[k], pts[k + 1], color, 2, cv2.LINE_AA)
    for pt in pts:
      cv2.circle(img, pt, 4, color, -1, cv2.LINE_AA)


def render_frame(img_root: Path, record: dict, idx: int, total: int) -> np.ndarray:
  """Render one TuSimple frame: cropped image + lane annotations + info bar."""
  raw_file = record['raw_file']
  lanes = record['lanes']
  h_samples = record['h_samples']

  img_path = img_root / raw_file
  if img_path.exists():
    img = cv2.imread(str(img_path))
  else:
    img = np.zeros((TUSIMPLE_H, TUSIMPLE_W, 3), dtype=np.uint8)
    cv2.putText(img, f'Image not found: {raw_file}', (10, TUSIMPLE_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

  draw_tusimple_lanes(img, lanes, h_samples)

  # Info bar
  info_bar = np.zeros((INFO_BAR_H, TUSIMPLE_W, 3), dtype=np.uint8)

  lane_info_parts = []
  for i, lane in enumerate(lanes):
    n_vis = sum(1 for x in lane if x != -2)
    label = LANE_LABELS[i] if i < len(LANE_LABELS) else f'L{i}'
    lane_info_parts.append(f'{label}={n_vis}')
  lane_str = '  '.join(lane_info_parts)

  cv2.putText(info_bar, f'[{idx}/{total - 1}]  {raw_file}', (5, 18),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
  cv2.putText(info_bar, lane_str, (5, 40),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

  lx = TUSIMPLE_W - 380
  for i, (label, color) in enumerate(zip(LANE_LABELS, LANE_COLORS)):
    x = lx + i * 90
    cv2.circle(info_bar, (x, 36), 5, color, -1)
    cv2.putText(info_bar, label, (x + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

  return np.vstack([img, info_bar])


def main():
  parser = argparse.ArgumentParser(description='TuSimple Phase 4 可视化: 2D 车道线标注检查')
  parser.add_argument('data_dir', help='数据根目录 (含 tusimple_merged/) 或 per-session tusimple/ 目录')
  parser.add_argument('--split', choices=SPLITS, default=None,
                      help='加载指定 split 的标签')
  parser.add_argument('--label-file', default=None,
                      help='直接指定 JSON Lines 标签文件路径')
  parser.add_argument('--max-frames', type=int, default=0,
                      help='最大浏览帧数 (0=全部)')
  parser.add_argument('--output-dir', default=None,
                      help='批量输出目录 (非交互模式)')
  args = parser.parse_args()

  data_dir = Path(args.data_dir).resolve()
  if not data_dir.exists():
    print(f"目录不存在: {data_dir}")
    sys.exit(1)

  # Determine label file and image root
  label_path: Path | None = None
  img_root: Path = data_dir  # default: resolve raw_file relative to data_dir

  if args.label_file:
    label_path = Path(args.label_file).resolve()
  else:
    merged_dir = data_dir / 'tusimple_merged'
    per_session_dir = data_dir / 'tusimple' if (data_dir / 'tusimple').exists() else None

    if merged_dir.exists():
      # Mode 1: data_root with tusimple_merged/
      split = args.split or 'train'
      label_path = _discover_merged_label(merged_dir, split)
      img_root = data_dir  # raw_file: "session/tusimple/H1/images/xxx.jpg"
    elif per_session_dir:
      # Mode 2: session_dir with tusimple/ (legacy per-session)
      split = args.split or 'train'
      label_path = per_session_dir / f'{split}.json'
      if not label_path.exists():
        for d in sorted(per_session_dir.iterdir()):
          if d.is_dir() and (d / 'labels.json').exists():
            label_path = d / 'labels.json'
            break
      img_root = per_session_dir  # raw_file: "H1/images/xxx.jpg"
    elif args.split:
      # data_dir itself might be the tusimple_merged dir
      label_path = _discover_merged_label(data_dir, args.split)

  if label_path is None or not label_path.exists():
    print(f"未找到标签文件。用法:")
    print(f"  viz_tusimple.py <data_root> --split train   (读 tusimple_merged/)")
    print(f"  viz_tusimple.py <session_dir>               (读 tusimple/)")
    print(f"  viz_tusimple.py <any> --label-file <path>")
    sys.exit(1)

  records = load_tusimple_labels(label_path)
  if not records:
    print(f"标签文件为空: {label_path}")
    sys.exit(1)

  if args.max_frames > 0:
    records = records[:args.max_frames]
  total = len(records)

  print(f"标签: {label_path}")
  print(f"图像根: {img_root}")
  print(f"帧数: {total}")

  # Batch output
  if args.output_dir:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(records):
      img = render_frame(img_root, rec, i, total)
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
    img = render_frame(img_root, records[idx], idx, total)
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
