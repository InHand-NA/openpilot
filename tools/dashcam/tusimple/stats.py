#!/usr/bin/env python3
"""TuSimple 数据集统计：收集所有 session 的 TuSimple 标签，生成文件列表和统计报告。

功能:
  1. 扫描 data_root 下所有 session 的 tusimple/{train,val,test}.json
  2. 合并生成 train_list.txt / val_list.txt / test_list.txt (每行一个 JSON 文件路径)
  3. 统计每个 split 的总帧数、总车道线数、平均车道线数、可见点分布

输出写到 data_root/tusimple_stats/ 目录。

用法:
  python tools/dashcam/tusimple/stats.py data/tusimple-sample

  # 指定输出目录
  python tools/dashcam/tusimple/stats.py data/tusimple-sample --output data/tusimple-sample/tusimple_stats
"""

import argparse
import json
import sys
from pathlib import Path

SPLITS = ['train', 'val', 'test']


def discover_split_files(data_root: Path) -> dict[str, list[Path]]:
  """扫描 data_root 下所有 session 的 tusimple/{split}.json。"""
  result: dict[str, list[Path]] = {s: [] for s in SPLITS}
  for d in sorted(data_root.iterdir()):
    if not d.is_dir():
      continue
    tusimple_dir = d / 'tusimple'
    if not tusimple_dir.exists():
      continue
    for split in SPLITS:
      path = tusimple_dir / f'{split}.json'
      if path.exists():
        result[split].append(path)
  return result


def count_labels(path: Path) -> dict:
  """统计一个 JSON Lines 文件的帧数、车道线数，以及按高度分组的帧数/车道线数。

  Returns:
    {
      'frames': int, 'lanes': int, 'visible_pts': int,
      'per_height': {tag: {'frames': int, 'lanes': int, 'visible_pts': int}, ...}
    }
  """
  n_frames = 0
  n_lanes = 0
  n_visible_pts = 0
  per_height: dict[str, dict[str, int]] = {}
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      n_frames += 1

      # Extract height tag from raw_file, e.g. "H1/images/000004.jpg" → "H1"
      raw_file = rec.get('raw_file', '')
      tag = raw_file.split('/')[0] if '/' in raw_file else 'unknown'
      if tag not in per_height:
        per_height[tag] = {'frames': 0, 'lanes': 0, 'visible_pts': 0}
      per_height[tag]['frames'] += 1

      for lane in rec.get('lanes', []):
        vis = sum(1 for x in lane if x != -2)
        if vis > 0:
          n_lanes += 1
          n_visible_pts += vis
          per_height[tag]['lanes'] += 1
          per_height[tag]['visible_pts'] += vis

  return {'frames': n_frames, 'lanes': n_lanes, 'visible_pts': n_visible_pts,
          'per_height': per_height}


def main():
  parser = argparse.ArgumentParser(description='TuSimple 数据集统计')
  parser.add_argument('data_root', help='数据根目录 (含多个 session 子目录)')
  parser.add_argument('--output', default=None,
                      help='输出目录 (default: <data_root>/tusimple_stats/)')
  args = parser.parse_args()

  data_root = Path(args.data_root).resolve()
  if not data_root.exists():
    print(f"ERROR: 目录不存在: {data_root}", file=sys.stderr)
    sys.exit(1)

  output_dir = Path(args.output).resolve() if args.output else data_root / 'tusimple_stats'
  output_dir.mkdir(parents=True, exist_ok=True)

  split_files = discover_split_files(data_root)

  # 1) 生成文件列表
  for split in SPLITS:
    files = split_files[split]
    list_path = output_dir / f'{split}_list.txt'
    with open(list_path, 'w') as f:
      for p in files:
        f.write(str(p) + '\n')
    print(f"{split}_list.txt: {len(files)} 个文件")

  # 2) 统计 — 总览
  print(f"\n{'split':<8} {'files':>6} {'frames':>8} {'lanes':>8} {'avg_lanes':>10} {'vis_pts':>10} {'avg_pts/lane':>13}")
  print('-' * 75)

  summary = {}
  for split in SPLITS:
    files = split_files[split]
    total = {'frames': 0, 'lanes': 0, 'visible_pts': 0}
    merged_heights: dict[str, dict[str, int]] = {}
    for p in files:
      c = count_labels(p)
      total['frames'] += c['frames']
      total['lanes'] += c['lanes']
      total['visible_pts'] += c['visible_pts']
      for tag, h_stats in c['per_height'].items():
        if tag not in merged_heights:
          merged_heights[tag] = {'frames': 0, 'lanes': 0, 'visible_pts': 0}
        for k in ['frames', 'lanes', 'visible_pts']:
          merged_heights[tag][k] += h_stats[k]

    avg_lanes = total['lanes'] / max(total['frames'], 1)
    avg_pts = total['visible_pts'] / max(total['lanes'], 1)

    print(f"{split:<8} {len(files):>6} {total['frames']:>8} {total['lanes']:>8} "
          f"{avg_lanes:>10.2f} {total['visible_pts']:>10} {avg_pts:>13.1f}")

    summary[split] = {
      'files': len(files),
      'frames': total['frames'],
      'lanes': total['lanes'],
      'avg_lanes_per_frame': round(avg_lanes, 2),
      'visible_pts': total['visible_pts'],
      'avg_pts_per_lane': round(avg_pts, 1),
      'per_height': {tag: dict(h) for tag, h in sorted(merged_heights.items())},
    }

  # 3) 统计 — 按高度
  all_tags = sorted({tag for s in summary.values() for tag in s.get('per_height', {})})
  if all_tags:
    print(f"\n按高度统计:")
    print(f"  {'split':<8} ", end='')
    for tag in all_tags:
      print(f"  {tag:>10}", end='')
    print()
    print('  ' + '-' * (10 + 12 * len(all_tags)))
    for split in SPLITS:
      ph = summary[split].get('per_height', {})
      print(f"  {split:<8} ", end='')
      for tag in all_tags:
        n = ph.get(tag, {}).get('frames', 0)
        print(f"  {n:>10}", end='')
      print()

  # 写 summary.json
  summary_path = output_dir / 'summary.json'
  with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

  print(f"\n输出: {output_dir}")
  print(f"  train_list.txt / val_list.txt / test_list.txt")
  print(f"  summary.json")


if __name__ == '__main__':
  main()
