#!/usr/bin/env python3
"""批量 TuSimple 投影：遍历数据目录下所有 session 执行 Phase 4。

扫描给定根目录下含 splits/ 的 session 子目录，
逐一调用 project_tusimple.project_session() 生成 TuSimple 格式标签。

支持断点续跑：已有 tusimple/train.json 的 session 自动跳过。

用法：
  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample

  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample \\
      --crop-hfov 70 --lane-prob-threshold 0.2

  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample --dry-run

  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample --force
"""

import argparse
import sys
import time
from pathlib import Path


def discover_sessions(data_root: Path) -> list[Path]:
  """发现 data_root 下所有含 splits/ 目录的 session。"""
  sessions = []
  for d in sorted(data_root.iterdir()):
    if d.is_dir() and (d / 'splits').exists():
      sessions.append(d)
  return sessions


def main():
  parser = argparse.ArgumentParser(description='批量 TuSimple 投影 (Phase 4)')
  parser.add_argument('data_root', help='数据根目录 (含多个 session 子目录)')
  parser.add_argument('--crop-hfov', type=float, default=70.0,
                      help='ROI 裁剪 HFOV (default: 70)')
  parser.add_argument('--lane-prob-threshold', type=float, default=0.2,
                      help='lane_line 使用/补位置信度阈值 (default: 0.2)')
  parser.add_argument('--min-visible-pts', type=int, default=2,
                      help='最少可见采样点 (default: 2)')
  parser.add_argument('--jpeg-quality', type=int, default=95,
                      help='JPEG 输出质量 (default: 95)')
  parser.add_argument('--dry-run', action='store_true',
                      help='仅预览，不实际处理')
  parser.add_argument('--force', action='store_true',
                      help='强制重新处理已有 tusimple/ 的 session')
  args = parser.parse_args()

  data_root = Path(args.data_root).resolve()
  if not data_root.exists():
    print(f"ERROR: 目录不存在: {data_root}", file=sys.stderr)
    sys.exit(1)

  sessions = discover_sessions(data_root)
  if not sessions:
    print(f"未找到 session (需含 splits/): {data_root}")
    sys.exit(0)

  todo: list[Path] = []
  skipped: list[Path] = []

  for s in sessions:
    if not args.force and (s / 'tusimple' / 'train.json').exists():
      skipped.append(s)
      continue
    todo.append(s)

  print(f"数据根目录: {data_root}")
  print(f"发现 {len(sessions)} 个 session: {len(todo)} 待处理, {len(skipped)} 跳过")
  print(f"crop_hfov={args.crop_hfov}  lane_prob_thresh={args.lane_prob_threshold}  "
        f"min_visible={args.min_visible_pts}  jpeg_q={args.jpeg_quality}")
  print()

  if not todo:
    print("所有 session 已处理完成，无需操作。")
    sys.exit(0)

  if args.dry_run:
    print("[DRY-RUN] 待处理 session:")
    for i, s in enumerate(todo):
      print(f"  [{i + 1}/{len(todo)}] {s.name}")
    sys.exit(0)

  from openpilot.tools.dashcam.tusimple.project_tusimple import project_session

  t_total_start = time.monotonic()
  completed = 0
  failed: list[tuple[str, str]] = []
  total_projected = 0

  try:
    for i, session_dir in enumerate(todo):
      output_dir = session_dir / 'tusimple'
      print(f"{'=' * 70}")
      print(f"[{i + 1}/{len(todo)}] {session_dir.name}")
      print(f"{'=' * 70}")

      t_sess = time.monotonic()
      try:
        stats = project_session(
          session_dir=session_dir,
          output_dir=output_dir,
          crop_hfov=args.crop_hfov,
          lane_prob_threshold=args.lane_prob_threshold,
          min_visible_pts=args.min_visible_pts,
          jpeg_quality=args.jpeg_quality,
        )
        elapsed = time.monotonic() - t_sess
        completed += 1
        total_projected += stats.get('projected', 0)
        print(f"  -> 完成 ({elapsed:.1f}s)")
      except Exception as e:
        elapsed = time.monotonic() - t_sess
        failed.append((session_dir.name, str(e)))
        print(f"  -> 失败 ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()

  except KeyboardInterrupt:
    print(f"\n[中断] 已完成 {completed}/{len(todo)} 个 session")

  total_elapsed = time.monotonic() - t_total_start
  print(f"\n{'=' * 70}")
  print(f"批量投影完成")
  print(f"  总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
  print(f"  成功: {completed}/{len(todo)}  跳过: {len(skipped)}  失败: {len(failed)}")
  print(f"  累计投影: {total_projected} 帧")
  if failed:
    print(f"\n失败列表:")
    for name, err in failed:
      print(f"  {name}: {err}")


if __name__ == '__main__':
  main()
