#!/usr/bin/env python3
"""T2 验证工具 — 自动检查数据清洗结果

3 项检查全部 PASS 才通过：
  1. 抽帧正确性: 每个帧号 % step == 0
  2. 置信度过滤: 每帧至少一条内侧线 prob > threshold
  3. PRE 帧存在性: 帧 N-save_every 的图像存在

用法：
  python tools/dashcam/train/verify_clean.py \
      --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
      --output-dir data/multi_height-0312/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


L_INNER_IDX = 1
R_INNER_IDX = 2


def load_clean_log(output_dir: Path) -> list[str]:
  """读取 clean_log.txt，返回 session/frame_id 列表"""
  log_path = output_dir / 'clean_log.txt'
  if not log_path.exists():
    raise FileNotFoundError(f'clean_log.txt not found: {log_path}')
  with open(log_path) as f:
    return [line.strip() for line in f if line.strip()]


def parse_entry(entry: str) -> tuple[str, int]:
  """解析 'session_name/frame_id' → (session_name, frame_id_int)"""
  parts = entry.rsplit('/', 1)
  return parts[0], int(parts[1])


def check_subsample(entries: list[str], step: int) -> tuple[bool, str]:
  """检查 1: 抽帧正确性"""
  bad = []
  for entry in entries:
    _, fid = parse_entry(entry)
    if fid % step != 0:
      bad.append(entry)
      if len(bad) >= 5:
        break
  if bad:
    return False, f'{len(bad)} 帧号不是 {step} 的倍数，例: {bad[:3]}'
  return True, f'{len(entries)}/{len(entries)} 帧号均为 {step} 的倍数'


def check_confidence(entries: list[str], dataset_dir: Path, min_ll_prob: float) -> tuple[bool, str]:
  """检查 2: 置信度过滤（从原始标注读取）"""
  bad = []
  for entry in entries:
    session, fid = parse_entry(entry)
    ann_path = dataset_dir / session / 'annotations' / 'H1' / f'{fid:06d}.json'
    with open(ann_path) as f:
      ann = json.load(f)
    ll_prob = ann['lane_lines_prob']
    l_inner = ll_prob[L_INNER_IDX]
    r_inner = ll_prob[R_INNER_IDX]
    if l_inner <= min_ll_prob and r_inner <= min_ll_prob:
      bad.append(f'{entry} (L={l_inner:.4f}, R={r_inner:.4f})')
      if len(bad) >= 5:
        break
  if bad:
    return False, f'{len(bad)} 帧不满足置信度条件，例: {bad[:3]}'
  return True, f'{len(entries)}/{len(entries)} 帧至少一条内侧线 prob>{min_ll_prob}'


def check_pre_frame(entries: list[str], dataset_dir: Path, save_every: int) -> tuple[bool, str]:
  """检查 3: PRE 帧存在性（从原始图像目录检查）"""
  bad = []
  for entry in entries:
    session, fid = parse_entry(entry)
    pre_id = fid - save_every
    h1_dir = dataset_dir / session / 'H1'
    road = h1_dir / f'road_{pre_id:06d}.png'
    wide = h1_dir / f'wide_{pre_id:06d}.png'
    if not road.exists() or not wide.exists():
      bad.append(f'{entry} (pre={pre_id:06d})')
      if len(bad) >= 5:
        break
  if bad:
    return False, f'{len(bad)} 帧缺少 PRE 帧，例: {bad[:3]}'
  return True, f'{len(entries)}/{len(entries)} 帧均有对应 PRE 帧'


def main():
  parser = argparse.ArgumentParser(description='验证数据清洗结果')
  parser.add_argument('--dataset-dir', type=Path, required=True,
                      help='原始数据集根目录（只读，用于读取原始标注和图像）')
  parser.add_argument('--output-dir', type=Path, default=None,
                      help='clean_log.txt 所在目录 (default: dataset_dir)')
  parser.add_argument('--subsample', type=int, default=5, help='抽帧间隔 (default: 5)')
  parser.add_argument('--min-ll-prob', type=float, default=0.05, help='最低置信度 (default: 0.05)')
  args = parser.parse_args()

  dataset_dir = args.dataset_dir.resolve()
  output_dir = (args.output_dir or dataset_dir).resolve()

  # 读取 save_every
  sessions = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('Town'))
  if not sessions:
    print('ERROR: no Town* sessions found', file=sys.stderr)
    sys.exit(1)

  clip_path = dataset_dir / sessions[0] / 'clip_info.json'
  with open(clip_path) as f:
    clip_info = json.load(f)
  save_every = clip_info.get('save_every', 4)
  step = save_every * args.subsample

  # 读取 clean_log
  entries = load_clean_log(output_dir)
  print(f'clean_log.txt: {len(entries)} 帧')
  print(f'dataset_dir: {dataset_dir}')
  print(f'output_dir:  {output_dir}')
  print(f'save_every={save_every}, subsample={args.subsample}, step={step}')
  print()

  checks = [
    ('抽帧正确性', lambda: check_subsample(entries, step)),
    ('置信度过滤', lambda: check_confidence(entries, dataset_dir, args.min_ll_prob)),
    ('PRE 帧存在性', lambda: check_pre_frame(entries, dataset_dir, save_every)),
  ]

  results = []
  for name, check_fn in checks:
    passed, msg = check_fn()
    status = 'PASS' if passed else 'FAIL'
    print(f'[{status}] {name}: {msg}')
    results.append(passed)

  print()
  pass_count = sum(results)
  total = len(results)
  print(f'===== {pass_count}/{total} PASS =====')

  if pass_count < total:
    sys.exit(1)


if __name__ == '__main__':
  main()
