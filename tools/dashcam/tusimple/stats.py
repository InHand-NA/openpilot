#!/usr/bin/env python3
"""TuSimple 数据集统计。

读取 data_root/tusimple_merged/ 下的全局合并文件 (含分片 train_1.json 等)，
统计每个 split 的总帧数、车道线数、按高度分布。

用法:
  python tools/dashcam/tusimple/stats.py data/tusimple-sample
"""

import argparse
import json
import re
import sys
from pathlib import Path

SPLITS = ['train', 'val', 'test']


def discover_split_files(merged_dir: Path) -> dict[str, list[Path]]:
  """发现 tusimple_merged/ 下的 split 文件。

  支持单文件 (train.json) 和分片 (train_1.json, train_2.json, ...)。
  """
  result: dict[str, list[Path]] = {s: [] for s in SPLITS}
  if not merged_dir.exists():
    return result

  for split in SPLITS:
    # 单文件
    single = merged_dir / f'{split}.json'
    if single.exists():
      result[split].append(single)
    # 分片: {split}_1.json, {split}_2.json, ...
    pattern = re.compile(rf'^{re.escape(split)}_(\d+)\.json$')
    chunks = sorted(
      (p for p in merged_dir.iterdir() if pattern.match(p.name)),
      key=lambda p: int(pattern.match(p.name).group(1)),
    )
    result[split].extend(chunks)

  return result


def count_labels(path: Path) -> dict:
  """统计一个 JSON Lines 文件的帧数、车道线数，按高度和 session 分组。"""
  n_frames = 0
  n_lanes = 0
  n_visible_pts = 0
  per_height: dict[str, dict[str, int]] = {}
  per_session: dict[str, int] = {}
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      n_frames += 1

      raw_file = rec.get('raw_file', '')
      parts = raw_file.split('/')
      # raw_file: "session/tusimple/H1/images/000004.jpg" → session=parts[0], height=parts[2]
      # or legacy: "H1/images/000004.jpg" → height=parts[0]
      if len(parts) >= 4 and parts[1] == 'tusimple':
        session = parts[0]
        tag = parts[2]
      else:
        session = 'unknown'
        tag = parts[0] if parts else 'unknown'

      per_session[session] = per_session.get(session, 0) + 1

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
          'per_height': per_height, 'per_session': per_session}


def main():
  parser = argparse.ArgumentParser(description='TuSimple 数据集统计')
  parser.add_argument('data_root', help='数据根目录 (含 tusimple_merged/)')
  args = parser.parse_args()

  data_root = Path(args.data_root).resolve()
  merged_dir = data_root / 'tusimple_merged'
  if not merged_dir.exists():
    print(f"ERROR: tusimple_merged/ 不存在: {merged_dir}", file=sys.stderr)
    print(f"请先运行 batch_project_tusimple.py")
    sys.exit(1)

  split_files = discover_split_files(merged_dir)
  if not any(split_files[s] for s in SPLITS):
    print(f"ERROR: tusimple_merged/ 中无 split 文件", file=sys.stderr)
    sys.exit(1)

  # 文件列表
  print(f"数据源: {merged_dir}")
  for split in SPLITS:
    files = split_files[split]
    names = ', '.join(p.name for p in files) if files else '(无)'
    print(f"  {split}: {names}")

  # 总览
  print(f"\n{'split':<8} {'files':>6} {'frames':>8} {'lanes':>8} {'avg_lanes':>10} {'vis_pts':>10} {'avg_pts/lane':>13} {'sessions':>9}")
  print('-' * 85)

  summary = {}
  for split in SPLITS:
    files = split_files[split]
    total = {'frames': 0, 'lanes': 0, 'visible_pts': 0}
    merged_heights: dict[str, dict[str, int]] = {}
    all_sessions: set[str] = set()
    for p in files:
      c = count_labels(p)
      total['frames'] += c['frames']
      total['lanes'] += c['lanes']
      total['visible_pts'] += c['visible_pts']
      all_sessions.update(c['per_session'].keys())
      for tag, h_stats in c['per_height'].items():
        if tag not in merged_heights:
          merged_heights[tag] = {'frames': 0, 'lanes': 0, 'visible_pts': 0}
        for k in ['frames', 'lanes', 'visible_pts']:
          merged_heights[tag][k] += h_stats[k]

    avg_lanes = total['lanes'] / max(total['frames'], 1)
    avg_pts = total['visible_pts'] / max(total['lanes'], 1)

    print(f"{split:<8} {len(files):>6} {total['frames']:>8} {total['lanes']:>8} "
          f"{avg_lanes:>10.2f} {total['visible_pts']:>10} {avg_pts:>13.1f} {len(all_sessions):>9}")

    summary[split] = {
      'files': len(files),
      'frames': total['frames'],
      'lanes': total['lanes'],
      'avg_lanes_per_frame': round(avg_lanes, 2),
      'visible_pts': total['visible_pts'],
      'avg_pts_per_lane': round(avg_pts, 1),
      'sessions': len(all_sessions),
      'per_height': {tag: dict(h) for tag, h in sorted(merged_heights.items())},
    }

  # 按高度
  all_tags = sorted({tag for s in summary.values() for tag in s.get('per_height', {})})
  if all_tags:
    print(f"\n按高度统计 (帧数):")
    print(f"  {'split':<8} ", end='')
    for tag in all_tags:
      print(f"  {tag:>8}", end='')
    print()
    print('  ' + '-' * (10 + 10 * len(all_tags)))
    for split in SPLITS:
      ph = summary[split].get('per_height', {})
      print(f"  {split:<8} ", end='')
      for tag in all_tags:
        n = ph.get(tag, {}).get('frames', 0)
        print(f"  {n:>8}", end='')
      print()

  # 写 summary.json
  summary_path = merged_dir / 'summary.json'
  with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

  print(f"\nsummary.json: {summary_path}")


if __name__ == '__main__':
  main()
