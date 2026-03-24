#!/usr/bin/env python3
"""批量清洗、抽样、跨 session 划分 tusimple 数据集。

流程:
  Step A: 逐 session 执行 clean_and_sample (质量过滤 + 时间抽样)
  Step B: 跨 session 做 session 级别 train/val/test 划分
          整个 session 分配到同一个 split，避免同场景泄漏
  Step C: 写回每个 session 的 splits/train.txt, val.txt, test.txt

用法：
  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample

  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample \\
      --heights H1 H3 H6 --min-ll-prob 0.5 --sample-every 5 \\
      --split-ratio 0.8 0.1 0.1 --seed 42

  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample --dry-run
  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample --force
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from openpilot.tools.dashcam.tusimple.config import HEIGHT_DEFS


def discover_sessions(data_root: Path) -> list[Path]:
  """发现 data_root 下所有含 3d_labels/H1/ 的 session 目录。"""
  sessions = []
  for d in sorted(data_root.iterdir()):
    if d.is_dir() and (d / '3d_labels' / 'H1').exists():
      sessions.append(d)
  return sessions


def _load_sampled_log(session_dir: Path) -> list[str]:
  path = session_dir / 'splits' / 'sampled_log.txt'
  if not path.exists():
    return []
  with open(path) as f:
    return [l.strip() for l in f if l.strip()]


def _get_heights(session_dir: Path, override: list[str] | None) -> list[str]:
  if override:
    heights = override
  else:
    ci_path = session_dir / 'clip_info.json'
    if ci_path.exists():
      with open(ci_path) as f:
        ci = json.load(f)
      heights = sorted(ci.get('heights', HEIGHT_DEFS).keys())
    else:
      heights = sorted(HEIGHT_DEFS.keys())
  label_dir = session_dir / '3d_labels'
  return [h for h in heights if (label_dir / h).exists()]


def main():
  parser = argparse.ArgumentParser(description='批量清洗 + 跨 session 划分 (Phase 3)')
  parser.add_argument('data_root', help='数据根目录 (含多个 session 子目录)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='要处理的高度 (default: clip_info 中所有高度)')
  parser.add_argument('--min-ll-prob', type=float, default=0.5,
                      help='内侧车道线最低概率 (default: 0.5)')
  parser.add_argument('--min-speed', type=float, default=1.0,
                      help='最低自车速度 m/s (default: 1.0)')
  parser.add_argument('--sample-every', type=int, default=5,
                      help='每 N 帧取 1 帧 (default: 5)')
  parser.add_argument('--split-ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                      help='train/val/test 比例 (default: 0.8 0.1 0.1)')
  parser.add_argument('--seed', type=int, default=42,
                      help='随机种子 (default: 42)')
  parser.add_argument('--dry-run', action='store_true',
                      help='仅预览，不实际处理')
  parser.add_argument('--force', action='store_true',
                      help='强制重新处理已有 splits 的 session')
  args = parser.parse_args()

  data_root = Path(args.data_root).resolve()
  if not data_root.exists():
    print(f"ERROR: 目录不存在: {data_root}", file=sys.stderr)
    sys.exit(1)

  sessions = discover_sessions(data_root)
  if not sessions:
    print(f"未找到 session (需含 3d_labels/H1/): {data_root}")
    sys.exit(0)

  r_sum = sum(args.split_ratio)
  ratios = [r / r_sum for r in args.split_ratio]

  print(f"数据根目录: {data_root}")
  print(f"发现 {len(sessions)} 个 session")
  if args.heights:
    print(f"高度: {args.heights}")
  print(f"min_ll_prob={args.min_ll_prob}  min_speed={args.min_speed}  sample_every={args.sample_every}")
  print(f"split={ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}  seed={args.seed}")
  print()

  if args.dry_run:
    print("[DRY-RUN] 仅预览")

  # ── Step A: per-session clean + sample ──────────────────────────────────
  from openpilot.tools.dashcam.tusimple.clean_and_sample import clean_and_sample, expand_to_heights, write_lines

  t_total = time.monotonic()
  completed = 0
  failed: list[tuple[str, str]] = []
  total_sampled = 0

  # Need to track which sessions need Step A vs already done
  for i, session_dir in enumerate(sessions):
    already_done = (session_dir / 'splits' / 'sampled_log.txt').exists()
    if already_done and not args.force:
      n = len(_load_sampled_log(session_dir))
      total_sampled += n
      completed += 1
      continue

    heights = _get_heights(session_dir, args.heights)
    if not heights:
      continue

    print(f"[{i + 1}/{len(sessions)}] {session_dir.name}")
    try:
      stats = clean_and_sample(
        session_dir=session_dir,
        heights=heights,
        min_ll_prob=args.min_ll_prob,
        min_speed=args.min_speed,
        sample_every=args.sample_every,
        dry_run=args.dry_run,
      )
      total_sampled += stats['sampled']
      completed += 1
    except Exception as e:
      failed.append((session_dir.name, str(e)))
      print(f"  -> 失败: {e}")

  print(f"\nStep A 完成: {completed}/{len(sessions)} sessions, {total_sampled} sampled frames")

  if args.dry_run:
    return

  # ── Step B: session-level split ─────────────────────────────────────────
  print(f"\nStep B: 跨 session 划分 (session 级别)...")

  # Collect sessions with sampled frames
  session_info: list[tuple[Path, list[str], list[str]]] = []  # (dir, sampled_ids, heights)
  for session_dir in sessions:
    sampled = _load_sampled_log(session_dir)
    if not sampled:
      continue
    heights = _get_heights(session_dir, args.heights)
    if not heights:
      continue
    session_info.append((session_dir, sampled, heights))

  if not session_info:
    print("  无可用 session")
    return

  # Shuffle sessions and split
  indices = list(range(len(session_info)))
  random.seed(args.seed)
  random.shuffle(indices)

  n = len(indices)
  n_train = round(n * ratios[0])
  n_val = round(n * ratios[1])

  train_indices = sorted(indices[:n_train])
  val_indices = sorted(indices[n_train:n_train + n_val])
  test_indices = sorted(indices[n_train + n_val:])

  split_assignment: dict[str, str] = {}  # session_name → split
  split_counts = {'train': 0, 'val': 0, 'test': 0}

  for idx_group, split_name in [(train_indices, 'train'), (val_indices, 'val'), (test_indices, 'test')]:
    for idx in idx_group:
      session_dir, sampled_ids, heights = session_info[idx]
      split_assignment[session_dir.name] = split_name

      entries = expand_to_heights(sampled_ids, heights)
      split_counts[split_name] += len(entries)

      # Write per-session split files
      splits_dir = session_dir / 'splits'
      splits_dir.mkdir(parents=True, exist_ok=True)
      for sn in ['train', 'val', 'test']:
        write_lines(splits_dir / f'{sn}.txt', entries if sn == split_name else [])

  print(f"  {len(session_info)} sessions → "
        f"train={len(train_indices)} sessions ({split_counts['train']} entries)  "
        f"val={len(val_indices)} sessions ({split_counts['val']} entries)  "
        f"test={len(test_indices)} sessions ({split_counts['test']} entries)")

  # Verify ratio
  total_entries = sum(split_counts.values())
  if total_entries > 0:
    actual = [split_counts[s] / total_entries for s in ['train', 'val', 'test']]
    print(f"  实际比例: {actual[0]:.3f}/{actual[1]:.3f}/{actual[2]:.3f}")

  # Write global split info
  global_stats = {
    'split_level': 'session',
    'seed': args.seed,
    'ratios': ratios,
    'sessions_total': len(session_info),
    'sessions_train': len(train_indices),
    'sessions_val': len(val_indices),
    'sessions_test': len(test_indices),
    'entries_train': split_counts['train'],
    'entries_val': split_counts['val'],
    'entries_test': split_counts['test'],
    'session_assignment': split_assignment,
  }
  global_stats_path = data_root / 'split_info.json'
  with open(global_stats_path, 'w') as f:
    json.dump(global_stats, f, indent=2, ensure_ascii=False)

  elapsed = time.monotonic() - t_total
  print(f"\n完成 ({elapsed:.1f}s)")
  print(f"  split_info.json: {global_stats_path}")
  if failed:
    print(f"\n失败列表:")
    for name, err in failed:
      print(f"  {name}: {err}")


if __name__ == '__main__':
  main()
