#!/usr/bin/env python3
"""T2 — 数据清洗脚本

按 training_evaluation_methodology.md §2.1 规则清洗标注数据：
1. 时序抽帧：每 subsample 个保存帧取 1 帧（等效 1 FPS）
2. 最低置信度过滤：L-inner prob ≤ threshold AND R-inner prob ≤ threshold 时丢弃
3. PRE 帧存在性：确保前一帧（帧号差 save_every）存在
4. H1~H6 同步：仅对 H1 做分析，所有高度共享清洗结果

输出（写到 --output-dir，默认为 dataset_dir）：
  - <output_dir>/clean_log.txt：被选中帧索引（每行 session_name/frame_number）

用法：
  python tools/dashcam/train/clean_data.py /nfs/openpilot-datasets/multi_height-0312/ --output-dir data/multi_height-0312/
  python tools/dashcam/train/clean_data.py /nfs/openpilot-datasets/multi_height-0312/ --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# lane_lines_prob 顺序: [L-outer(0), L-inner(1), R-inner(2), R-outer(3)]
L_INNER_IDX = 1
R_INNER_IDX = 2


def discover_sessions(dataset_dir: Path) -> list[str]:
  """发现所有 Town* session 目录（按名称排序）"""
  sessions = sorted(
    d.name for d in dataset_dir.iterdir()
    if d.is_dir() and d.name.startswith('Town')
  )
  return sessions


def load_clip_info(dataset_dir: Path, session: str) -> dict:
  """加载 session 的 clip_info.json"""
  clip_path = dataset_dir / session / 'clip_info.json'
  with open(clip_path) as f:
    return json.load(f)


def get_annotation_frame_ids(ann_dir: Path) -> list[int]:
  """获取标注目录下所有帧号（排序）"""
  frame_ids = []
  for p in ann_dir.iterdir():
    if p.suffix == '.json':
      try:
        frame_ids.append(int(p.stem))
      except ValueError:
        continue
  return sorted(frame_ids)


def load_annotation(ann_dir: Path, frame_id: int) -> dict:
  """加载单个标注 JSON"""
  path = ann_dir / f'{frame_id:06d}.json'
  with open(path) as f:
    return json.load(f)


def check_pre_frame_exists(session_dir: Path, frame_id: int, save_every: int) -> bool:
  """检查 PRE 帧的图像是否存在（在 H1 目录中检查）"""
  pre_id = frame_id - save_every
  if pre_id < 0:
    return False
  h1_dir = session_dir / 'H1'
  road_path = h1_dir / f'road_{pre_id:06d}.png'
  wide_path = h1_dir / f'wide_{pre_id:06d}.png'
  return road_path.exists() and wide_path.exists()


def clean_session(
  dataset_dir: Path,
  session: str,
  subsample: int,
  min_ll_prob: float,
  save_every: int,
) -> tuple[list[str], dict]:
  """清洗单个 session，返回 (选中帧列表, 统计信息)"""
  session_dir = dataset_dir / session
  ann_h1_dir = session_dir / 'annotations' / 'H1'

  if not ann_h1_dir.exists():
    return [], {'error': f'annotations/H1 not found for {session}'}

  # 获取所有帧号
  all_frame_ids = get_annotation_frame_ids(ann_h1_dir)
  if not all_frame_ids:
    return [], {'error': f'no annotations found for {session}'}

  # 统计
  stats = {
    'total_frames': len(all_frame_ids),
    'after_subsample': 0,
    'filtered_no_ll': 0,
    'filtered_no_pre': 0,
    'selected': 0,
  }

  # 抽帧步长 = save_every * subsample
  step = save_every * subsample
  selected_entries = []

  for frame_id in all_frame_ids:
    # 规则 1: 时序抽帧 — 帧号必须是 step 的倍数
    if frame_id % step != 0:
      continue
    stats['after_subsample'] += 1

    # 规则 3: PRE 帧存在性
    if not check_pre_frame_exists(session_dir, frame_id, save_every):
      stats['filtered_no_pre'] += 1
      continue

    # 规则 2: 最低置信度过滤（仅分析 H1）
    ann = load_annotation(ann_h1_dir, frame_id)
    ll_prob = ann['lane_lines_prob']
    l_inner = ll_prob[L_INNER_IDX]
    r_inner = ll_prob[R_INNER_IDX]

    # 仅当两条内侧线都 ≤ threshold 时丢弃
    if l_inner <= min_ll_prob and r_inner <= min_ll_prob:
      stats['filtered_no_ll'] += 1
      continue

    # 通过所有规则
    entry = f'{session}/{frame_id:06d}'
    selected_entries.append(entry)
    stats['selected'] += 1

  return selected_entries, stats


def main():
  parser = argparse.ArgumentParser(description='数据清洗：时序抽帧 + 置信度过滤')
  parser.add_argument('dataset_dir', type=Path, help='数据集根目录')
  parser.add_argument('--subsample', type=int, default=5,
                      help='每 N 个保存帧取 1 帧 (default: 5)')
  parser.add_argument('--min-ll-prob', type=float, default=0.05,
                      help='内侧车道线最低置信度阈值 (default: 0.05)')
  parser.add_argument('--output-dir', type=Path, default=None,
                      help='输出目录 (default: dataset_dir)')
  parser.add_argument('--dry-run', action='store_true',
                      help='仅统计，不写入文件')
  args = parser.parse_args()

  dataset_dir = args.dataset_dir.resolve()
  if not dataset_dir.exists():
    print(f'ERROR: dataset_dir not found: {dataset_dir}', file=sys.stderr)
    sys.exit(1)

  output_dir = (args.output_dir or dataset_dir).resolve()
  if not args.dry_run:
    output_dir.mkdir(parents=True, exist_ok=True)

  sessions = discover_sessions(dataset_dir)
  if not sessions:
    print(f'ERROR: no Town* sessions found in {dataset_dir}', file=sys.stderr)
    sys.exit(1)

  # 读取 save_every（从第一个 session 的 clip_info.json）
  clip_info = load_clip_info(dataset_dir, sessions[0])
  save_every = clip_info.get('save_every', 4)
  step = save_every * args.subsample

  print(f'数据集: {dataset_dir}')
  print(f'输出: {output_dir}')
  print(f'Sessions: {len(sessions)}')
  print(f'save_every={save_every}, subsample={args.subsample}, step={step}')
  print(f'min_ll_prob={args.min_ll_prob}')
  print(f'dry_run={args.dry_run}')
  print()

  all_entries = []
  total_stats = {
    'total_frames': 0,
    'after_subsample': 0,
    'filtered_no_ll': 0,
    'filtered_no_pre': 0,
    'selected': 0,
    'sessions_processed': 0,
    'sessions_with_errors': 0,
  }

  for i, session in enumerate(sessions):
    entries, stats = clean_session(
      dataset_dir, session, args.subsample, args.min_ll_prob, save_every,
    )

    if 'error' in stats:
      print(f'  [{i+1}/{len(sessions)}] {session}: ERROR - {stats["error"]}')
      total_stats['sessions_with_errors'] += 1
      continue

    all_entries.extend(entries)
    total_stats['sessions_processed'] += 1
    for key in ['total_frames', 'after_subsample', 'filtered_no_ll', 'filtered_no_pre', 'selected']:
      total_stats[key] += stats[key]

    if (i + 1) % 50 == 0 or i == len(sessions) - 1:
      print(f'  [{i+1}/{len(sessions)}] 已处理，累计选中 {len(all_entries)} 帧')

  # 排序（按 session 名 + 帧号）
  all_entries.sort()

  print()
  print('=' * 60)
  print('清洗统计:')
  print(f'  Sessions: {total_stats["sessions_processed"]} processed, '
        f'{total_stats["sessions_with_errors"]} errors')
  print(f'  总帧数: {total_stats["total_frames"]}')
  print(f'  抽帧后: {total_stats["after_subsample"]}')
  print(f'  过滤(无PRE帧): {total_stats["filtered_no_pre"]}')
  print(f'  过滤(无车道线): {total_stats["filtered_no_ll"]}')
  print(f'  最终选中: {total_stats["selected"]}')
  if total_stats['after_subsample'] > 0:
    keep_rate = total_stats['selected'] / total_stats['after_subsample'] * 100
    print(f'  保留率(抽帧后): {keep_rate:.1f}%')

  # 写入 clean_log.txt
  if not args.dry_run:
    log_path = output_dir / 'clean_log.txt'
    with open(log_path, 'w') as f:
      for entry in all_entries:
        f.write(entry + '\n')
    print(f'\nclean_log.txt 已写入: {log_path} ({len(all_entries)} 行)')
  else:
    print(f'\n[dry-run] 将写入 {len(all_entries)} 行到 clean_log.txt')


if __name__ == '__main__':
  main()
