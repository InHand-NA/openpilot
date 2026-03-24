#!/usr/bin/env python3
"""批量清洗与抽样 tusimple 数据目录下的所有 session。

扫描给定根目录下含 3d_labels/ 的 session 子目录，
逐一调用 clean_and_sample.clean_and_sample() 执行 Phase 3。

支持断点续跑：已有 splits/stats.json 的 session 自动跳过。

用法：
  # 清洗 data/tusimple-sample 下所有 session
  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample

  # 指定高度和参数
  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample \\
      --heights H1 H3 H6 --min-ll-prob 0.5 --sample-every 5

  # 预览
  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample --dry-run

  # 强制重跑
  python tools/dashcam/tusimple/batch_clean_and_sample.py data/tusimple-sample --force
"""

import argparse
import json
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


def main():
  parser = argparse.ArgumentParser(description='批量清洗与抽样 tusimple session (Phase 3)')
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

  # 分类: skip / todo
  todo: list[Path] = []
  skipped: list[Path] = []

  for s in sessions:
    if not args.force and (s / 'splits' / 'stats.json').exists():
      skipped.append(s)
      continue
    todo.append(s)

  # Normalize ratios
  r_sum = sum(args.split_ratio)
  ratios = [r / r_sum for r in args.split_ratio]

  print(f"数据根目录: {data_root}")
  print(f"发现 {len(sessions)} 个 session: {len(todo)} 待处理, {len(skipped)} 跳过")
  if args.heights:
    print(f"高度: {args.heights}")
  print(f"min_ll_prob={args.min_ll_prob}  min_speed={args.min_speed}  sample_every={args.sample_every}")
  print(f"split={ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}  seed={args.seed}")
  print()

  if not todo:
    print("所有 session 已处理完成，无需操作。")
    sys.exit(0)

  if args.dry_run:
    print("[DRY-RUN] 待处理 session:")
    for i, s in enumerate(todo):
      n = len(list((s / '3d_labels' / 'H1').glob('*.json')))
      print(f"  [{i + 1}/{len(todo)}] {s.name}  ({n} H1 frames)")
    sys.exit(0)

  from openpilot.tools.dashcam.tusimple.clean_and_sample import clean_and_sample

  t_total_start = time.monotonic()
  completed = 0
  failed: list[tuple[str, str]] = []
  total_stats = {'train': 0, 'val': 0, 'test': 0, 'quality_pass': 0, 'sampled': 0}

  try:
    for i, session_dir in enumerate(todo):
      print(f"{'=' * 70}")
      print(f"[{i + 1}/{len(todo)}] {session_dir.name}")
      print(f"{'=' * 70}")

      # Determine heights for this session
      if args.heights:
        heights = args.heights
      else:
        ci_path = session_dir / 'clip_info.json'
        if ci_path.exists():
          with open(ci_path) as f:
            ci = json.load(f)
          heights = sorted(ci.get('heights', HEIGHT_DEFS).keys())
        else:
          heights = sorted(HEIGHT_DEFS.keys())
      label_dir = session_dir / '3d_labels'
      heights = [h for h in heights if (label_dir / h).exists()]

      if not heights:
        print(f"  -> 跳过 (无可用高度标注)")
        continue

      t_sess = time.monotonic()
      try:
        stats = clean_and_sample(
          session_dir=session_dir,
          heights=heights,
          min_ll_prob=args.min_ll_prob,
          min_speed=args.min_speed,
          sample_every=args.sample_every,
          split_ratio=ratios,
          seed=args.seed,
        )
        elapsed = time.monotonic() - t_sess
        completed += 1
        for k in ['train', 'val', 'test', 'quality_pass', 'sampled']:
          total_stats[k] += stats[k]
        print(f"  -> 完成 ({elapsed:.1f}s)")
      except Exception as e:
        elapsed = time.monotonic() - t_sess
        failed.append((session_dir.name, str(e)))
        print(f"  -> 失败 ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()

  except KeyboardInterrupt:
    print(f"\n[中断] 已完成 {completed}/{len(todo)} 个 session")

  # 汇总
  total_elapsed = time.monotonic() - t_total_start
  print(f"\n{'=' * 70}")
  print(f"批量清洗完成")
  print(f"  总耗时: {total_elapsed:.1f}s")
  print(f"  成功: {completed}/{len(todo)}  跳过: {len(skipped)}  失败: {len(failed)}")
  print(f"  累计: pass={total_stats['quality_pass']}  sampled={total_stats['sampled']}  "
        f"train={total_stats['train']}  val={total_stats['val']}  test={total_stats['test']}")
  if failed:
    print(f"\n失败列表:")
    for name, err in failed:
      print(f"  {name}: {err}")


if __name__ == '__main__':
  main()
