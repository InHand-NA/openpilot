#!/usr/bin/env python3
"""TuSimple Phase 4: 3D→2D 投影，生成 TuSimple 格式车道线标签。

读取 Phase 3 的 sampled_log 帧列表 + Phase 2 的 3D 标注 + Phase 1 的 Mono 图像，
执行 ROI 裁剪 + 3D→2D 投影，输出 TuSimple JSON Lines + JPEG 图像。

输入:
  - session_dir/H1~H9/          (Mono 图像, 1920×1080)
  - session_dir/3d_labels/H1~H9/ (3D 标注)
  - session_dir/splits/          (sampled_log.txt, train.txt, val.txt, test.txt)

输出:
  session_dir/tusimple/
  ├── H1/images/*.jpg + labels.json
  ├── H2/ ... H9/
  ├── train.json   (合并所有高度)
  ├── val.json
  └── test.json

用法:
  python tools/dashcam/tusimple/project_tusimple.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/

  python tools/dashcam/tusimple/project_tusimple.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/ \\
      --crop-hfov 70 --lane-prob-threshold 0.2
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from openpilot.tools.dashcam.tusimple.config import (
  TUSIMPLE_W, TUSIMPLE_H, TUSIMPLE_H_SAMPLES,
  compute_crop_params,
)
from openpilot.tools.dashcam.tusimple.projection import lanes_3d_to_tusimple


def _load_split_list(splits_dir: Path, name: str) -> list[str]:
  """Load a split file (train.txt etc.) → list of 'Hk/frame_id'."""
  path = splits_dir / name
  if not path.exists():
    return []
  with open(path) as f:
    return [line.strip() for line in f if line.strip()]


def _load_sampled_log(splits_dir: Path) -> list[str]:
  """Load sampled_log.txt → list of frame_id strings."""
  path = splits_dir / 'sampled_log.txt'
  if not path.exists():
    return []
  with open(path) as f:
    return [line.strip() for line in f if line.strip()]


def _load_annotation(label_dir: Path, tag: str, frame_id: str) -> dict | None:
  path = label_dir / tag / f'{frame_id}.json'
  if not path.exists():
    return None
  with open(path) as f:
    return json.load(f)


def _load_mono_image(session_dir: Path, tag: str, frame_id: str,
                     clip_info: dict) -> np.ndarray | None:
  mono_fmt = clip_info.get('mono_camera', {}).get('format', 'png')
  mono_ext = '.jpg' if mono_fmt == 'jpeg' else '.png'
  mono_path = session_dir / tag / f'{frame_id}{mono_ext}'
  if not mono_path.exists():
    alt_ext = '.png' if mono_ext == '.jpg' else '.jpg'
    mono_path = session_dir / tag / f'{frame_id}{alt_ext}'
  if not mono_path.exists():
    return None
  return cv2.imread(str(mono_path))


def _crop_and_resize(img: np.ndarray, crop_rect: tuple) -> np.ndarray:
  """ROI 裁剪 + 缩放到 1280×720。"""
  x, y, w, h = crop_rect
  roi = img[y:y + h, x:x + w]
  return cv2.resize(roi, (TUSIMPLE_W, TUSIMPLE_H))


def project_session(
  session_dir: Path,
  output_dir: Path,
  crop_hfov: float,
  lane_prob_threshold: float,
  min_visible_pts: int,
  jpeg_quality: int,
  force: bool = False,
) -> dict:
  """Execute Phase 4 for one session. Returns stats dict."""
  clip_info_path = session_dir / 'clip_info.json'
  if not clip_info_path.exists():
    raise FileNotFoundError(f"clip_info.json not found: {session_dir}")
  with open(clip_info_path) as f:
    clip_info = json.load(f)

  pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
  yaw_rad = math.radians(clip_info['camera']['yaw_deg'])
  rpyCalib = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

  crop_params = compute_crop_params(crop_hfov=crop_hfov)
  K_crop = crop_params['K_crop']
  crop_rect = crop_params['crop_rect']

  label_dir = session_dir / '3d_labels'
  splits_dir = session_dir / 'splits'

  # Load split lists
  splits = {}
  for name in ['train', 'val', 'test']:
    splits[name] = _load_split_list(splits_dir, f'{name}.txt')

  # Build frame→split mapping
  frame_to_split: dict[str, str] = {}
  for split_name, entries in splits.items():
    for entry in entries:
      frame_to_split[entry] = split_name  # 'H1/000004' → 'train'

  # Collect all unique height/frame pairs
  all_entries: list[str] = []
  for entries in splits.values():
    all_entries.extend(entries)
  all_entries = sorted(set(all_entries))

  if not all_entries:
    # Fallback: use sampled_log directly
    sampled_ids = _load_sampled_log(splits_dir)
    heights = sorted(clip_info.get('heights', {}).keys())
    for fid in sampled_ids:
      for h in heights:
        entry = f'{h}/{fid}'
        all_entries.append(entry)
        frame_to_split[entry] = 'train'

  print(f"  Entries: {len(all_entries)}  crop_rect={crop_rect}  K_crop focal={crop_params['effective_focal']:.1f}")

  output_dir.mkdir(parents=True, exist_ok=True)

  # Per-split JSON lines accumulators
  split_lines: dict[str, list[str]] = {'train': [], 'val': [], 'test': []}
  # Per-height labels.json accumulators
  height_lines: dict[str, list[str]] = {}

  stats = {'total': 0, 'projected': 0, 'skipped_no_anno': 0,
           'skipped_no_image': 0, 'skipped_all_invalid': 0, 'skipped_exists': 0}

  t_start = time.monotonic()

  for entry in all_entries:
    tag, frame_id = entry.split('/', 1)
    stats['total'] += 1

    # Output paths
    height_dir = output_dir / tag / 'images'
    img_out_path = height_dir / f'{frame_id}.jpg'
    raw_file = f'{tag}/images/{frame_id}.jpg'

    # Idempotency: skip if image and labels already exist (unless force)
    if not force and img_out_path.exists():
      label_json_path = output_dir / tag / 'labels.json'
      if label_json_path.exists():
        stats['skipped_exists'] += 1
        continue

    # Load 3D annotation
    anno = _load_annotation(label_dir, tag, frame_id)
    if anno is None:
      stats['skipped_no_anno'] += 1
      continue

    # Load mono image
    mono_img = _load_mono_image(session_dir, tag, frame_id, clip_info)
    if mono_img is None:
      stats['skipped_no_image'] += 1
      continue

    # 3D → 2D projection
    lane_lines_3d = np.array(anno['lane_lines'], dtype=np.float32)
    lane_lines_prob = np.array(anno['lane_lines_prob'], dtype=np.float32)
    road_edges_3d = np.array(anno['road_edges'], dtype=np.float32)

    lanes, lane_names = lanes_3d_to_tusimple(
      lane_lines_3d=lane_lines_3d,
      lane_lines_prob=lane_lines_prob,
      road_edges_3d=road_edges_3d,
      K_tusimple=K_crop,
      rpyCalib=rpyCalib,
      h_samples=TUSIMPLE_H_SAMPLES,
      lane_prob_threshold=lane_prob_threshold,
      min_visible_pts=min_visible_pts,
    )

    # Filter out empty lanes (all -2), keep names in sync
    valid = [(lane, name) for lane, name in zip(lanes, lane_names)
             if any(x != -2 for x in lane)]
    if not valid:
      stats['skipped_all_invalid'] += 1
      continue
    lanes, lane_names = zip(*valid)
    lanes = list(lanes)
    lane_names = list(lane_names)

    # Save cropped JPEG
    height_dir.mkdir(parents=True, exist_ok=True)
    tusimple_img = _crop_and_resize(mono_img, crop_rect)
    cv2.imwrite(str(img_out_path), tusimple_img,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    # Build TuSimple JSON line
    record = {
      'lanes': lanes,
      'lane_names': lane_names,
      'h_samples': TUSIMPLE_H_SAMPLES,
      'raw_file': raw_file,
    }
    json_line = json.dumps(record, separators=(',', ':'))

    # Accumulate for per-height labels.json
    if tag not in height_lines:
      height_lines[tag] = []
    height_lines[tag].append(json_line)

    # Accumulate for split files
    split_name = frame_to_split.get(entry, 'train')
    split_lines[split_name].append(json_line)

    stats['projected'] += 1

    if stats['projected'] % 500 == 0:
      elapsed = time.monotonic() - t_start
      fps = stats['projected'] / elapsed if elapsed > 0 else 0
      print(f"    {stats['projected']}/{len(all_entries)} | {fps:.1f} fps")

  # Write per-height labels.json
  for tag, lines in height_lines.items():
    labels_path = output_dir / tag / 'labels.json'
    with open(labels_path, 'w') as f:
      for line in lines:
        f.write(line + '\n')

  # Write merged split files (train.json, val.json, test.json)
  for split_name, lines in split_lines.items():
    if lines:
      with open(output_dir / f'{split_name}.json', 'w') as f:
        for line in lines:
          f.write(line + '\n')

  elapsed = time.monotonic() - t_start
  print(f"  Done in {elapsed:.1f}s: projected={stats['projected']} "
        f"skip_exists={stats['skipped_exists']} skip_no_anno={stats['skipped_no_anno']} "
        f"skip_no_img={stats['skipped_no_image']} skip_invalid={stats['skipped_all_invalid']}")

  return stats


def main():
  parser = argparse.ArgumentParser(description='TuSimple Phase 4: 3D→2D 投影')
  parser.add_argument('session_dir', help='Session 目录')
  parser.add_argument('--output', default=None,
                      help='输出目录 (default: <session_dir>/tusimple/)')
  parser.add_argument('--crop-hfov', type=float, default=70.0,
                      help='ROI 裁剪 HFOV (default: 70)')
  parser.add_argument('--lane-prob-threshold', type=float, default=0.2,
                      help='lane_line 使用/补位置信度阈值 (default: 0.2)')
  parser.add_argument('--min-visible-pts', type=int, default=2,
                      help='最少可见采样点 (default: 2)')
  parser.add_argument('--jpeg-quality', type=int, default=95,
                      help='JPEG 输出质量 (default: 95)')
  parser.add_argument('--force', action='store_true',
                      help='强制重新生成已存在的文件')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"ERROR: 目录不存在: {session_dir}", file=sys.stderr)
    sys.exit(1)

  output_dir = Path(args.output).resolve() if args.output else session_dir / 'tusimple'

  print(f"Session: {session_dir.name}")
  print(f"Output:  {output_dir}")
  print(f"crop_hfov={args.crop_hfov}  lane_prob_thresh={args.lane_prob_threshold}  "
        f"min_visible={args.min_visible_pts}  jpeg_q={args.jpeg_quality}")

  try:
    project_session(
      session_dir=session_dir,
      output_dir=output_dir,
      crop_hfov=args.crop_hfov,
      lane_prob_threshold=args.lane_prob_threshold,
      min_visible_pts=args.min_visible_pts,
      jpeg_quality=args.jpeg_quality,
      force=args.force,
    )
    print(f"\nTuSimple output: {output_dir}")
  except KeyboardInterrupt:
    print("\n[Interrupted]")
    sys.exit(0)


if __name__ == '__main__':
  main()
