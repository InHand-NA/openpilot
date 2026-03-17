#!/usr/bin/env python3
"""T3.3 — Mini 数据集提取

从 T3.1/T3.2 的划分结果中摘取一份小子集，用于调试训练脚本。
每个子集按 session 分组后各取前 N 帧，保证覆盖多个 session。

用法：
  python tools/dashcam/train/make_mini_dataset.py \
      --output-dir data/multi_height-0312/ \
      --mini-dir data/multi_height-0312-mini/ \
      --max-frames 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


SPLITS = ['train', 'val', 'test']
HOLDOUT = 'h1_holdout'


def load_entries(path: Path) -> list[str]:
  if not path.exists():
    return []
  with open(path) as f:
    return [line.strip() for line in f if line.strip()]


def subsample_entries(entries: list[str], max_frames: int) -> list[str]:
  """按 session 均匀采样，总计不超过 max_frames 帧"""
  if len(entries) <= max_frames:
    return entries

  # Group by session
  by_session: dict[str, list[str]] = defaultdict(list)
  for e in entries:
    session = e.rsplit('/', 1)[0]
    by_session[session].append(e)

  # Round-robin from each session
  n_sessions = len(by_session)
  per_session = max(1, max_frames // n_sessions)
  result = []
  for session in sorted(by_session):
    frames = by_session[session]
    result.extend(frames[:per_session])

  # Trim to exact max
  return sorted(result[:max_frames])


def write_list(path: Path, entries: list[str]) -> None:
  with open(path, 'w') as f:
    for e in entries:
      f.write(e + '\n')


def main():
  parser = argparse.ArgumentParser(description='T3.3 Mini 数据集提取')
  parser.add_argument('--output-dir', type=Path, required=True,
                      help='T3.1/T3.2 输出目录（含 train.txt, val.txt 等）')
  parser.add_argument('--mini-dir', type=Path, required=True,
                      help='Mini 数据集输出目录')
  parser.add_argument('--max-frames', type=int, default=100,
                      help='train 子集最大帧数 (default: 100), val/test/holdout 按比例缩放')
  args = parser.parse_args()

  output_dir = args.output_dir.resolve()
  mini_dir = args.mini_dir.resolve()
  mini_dir.mkdir(parents=True, exist_ok=True)

  max_train = args.max_frames
  # val/test/holdout 按 train 的 1/8 比例
  max_other = max(10, max_train // 8)

  info = {'max_frames_train': max_train, 'max_frames_other': max_other}

  for split in SPLITS:
    src = output_dir / f'{split}.txt'
    entries = load_entries(src)
    if not entries:
      print(f'WARN: {src} not found or empty')
      continue
    limit = max_train if split == 'train' else max_other
    mini = subsample_entries(entries, limit)
    dst = mini_dir / f'{split}.txt'
    write_list(dst, mini)
    info[split] = {'original': len(entries), 'mini': len(mini)}
    print(f'{split}: {len(entries)} → {len(mini)}')

  # H1 holdout
  src = output_dir / f'{HOLDOUT}.txt'
  entries = load_entries(src)
  if entries:
    mini = subsample_entries(entries, max_other)
    write_list(mini_dir / f'{HOLDOUT}.txt', mini)
    info[HOLDOUT] = {'original': len(entries), 'mini': len(mini)}
    print(f'{HOLDOUT}: {len(entries)} → {len(mini)}')

  # Copy clean_log (for reference)
  clean_src = output_dir / 'clean_log.txt'
  if clean_src.exists():
    # Mini clean_log = union of all mini splits
    all_mini = set()
    for split in SPLITS:
      p = mini_dir / f'{split}.txt'
      if p.exists():
        all_mini.update(load_entries(p))
    write_list(mini_dir / 'clean_log.txt', sorted(all_mini))
    info['clean_log_mini'] = len(all_mini)

  # Write info
  info_path = mini_dir / 'mini_info.json'
  with open(info_path, 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

  print(f'\nmini_info.json: {info_path}')
  print(f'Mini 数据集已写入: {mini_dir}')


if __name__ == '__main__':
  main()
