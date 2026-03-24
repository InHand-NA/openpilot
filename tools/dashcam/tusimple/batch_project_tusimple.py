#!/usr/bin/env python3
"""批量 TuSimple 投影 + 合并输出。

Step A: 遍历各 session 执行 Phase 4 投影 (per-session tusimple/)
Step B: 合并所有 session 的 tusimple/{split}.json 为全局文件
        超过 1000 帧的 split 自动分片: train_1.json, train_2.json, ...

输出: data_root/tusimple_merged/{split}.json 或 {split}_N.json

用法：
  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample

  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample \\
      --crop-hfov 70 --lane-prob-threshold 0.2

  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample --dry-run
  python tools/dashcam/tusimple/batch_project_tusimple.py data/tusimple-sample --force
"""

import argparse
import json
import sys
import time
from pathlib import Path

SPLITS = ['train', 'val', 'test']
CHUNK_SIZE = 1000


def discover_sessions(data_root: Path) -> list[Path]:
  """发现 data_root 下所有含 splits/ 目录的 session。"""
  sessions = []
  for d in sorted(data_root.iterdir()):
    if d.is_dir() and (d / 'splits').exists():
      sessions.append(d)
  return sessions


def merge_split_files(data_root: Path, sessions: list[Path], output_dir: Path) -> dict:
  """合并所有 session 的 tusimple/{split}.json → 全局文件，>1000 帧分片。

  raw_file 重写为以 data_root 为起点的相对路径:
    per-session: "H1/images/000004.jpg"
    → merged:    "Town04_.../tusimple/H1/images/000004.jpg"

  Returns: {split: {'total': N, 'files': [path, ...]}}
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  merge_stats: dict[str, dict] = {}

  for split in SPLITS:
    # Collect all lines, rewriting raw_file to data_root-relative path
    all_lines: list[str] = []
    for s in sessions:
      path = s / 'tusimple' / f'{split}.json'
      if not path.exists():
        continue
      # session_prefix: e.g. "Town04_ClearNoon_p4.0_y0.0/tusimple"
      session_prefix = f'{s.name}/tusimple'
      with open(path) as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          rec = json.loads(line)
          rec['raw_file'] = f"{session_prefix}/{rec['raw_file']}"
          all_lines.append(json.dumps(rec, separators=(',', ':')))

    if not all_lines:
      merge_stats[split] = {'total': 0, 'files': []}
      continue

    # Write: single file or chunked
    files_written: list[Path] = []
    if len(all_lines) <= CHUNK_SIZE:
      out_path = output_dir / f'{split}.json'
      with open(out_path, 'w') as f:
        for line in all_lines:
          f.write(line + '\n')
      files_written.append(out_path)
    else:
      for chunk_idx in range(0, len(all_lines), CHUNK_SIZE):
        chunk = all_lines[chunk_idx:chunk_idx + CHUNK_SIZE]
        file_num = chunk_idx // CHUNK_SIZE + 1
        out_path = output_dir / f'{split}_{file_num}.json'
        with open(out_path, 'w') as f:
          for line in chunk:
            f.write(line + '\n')
        files_written.append(out_path)

    merge_stats[split] = {'total': len(all_lines), 'files': [str(p) for p in files_written]}
    file_desc = ', '.join(p.name for p in files_written)
    print(f"  {split}: {len(all_lines)} entries → {file_desc}")

  return merge_stats


def main():
  parser = argparse.ArgumentParser(description='批量 TuSimple 投影 + 合并 (Phase 4)')
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
  print(f"发现 {len(sessions)} 个 session: {len(todo)} 待投影, {len(skipped)} 跳过")
  print(f"crop_hfov={args.crop_hfov}  lane_prob_thresh={args.lane_prob_threshold}  "
        f"min_visible={args.min_visible_pts}  jpeg_q={args.jpeg_quality}")
  print()

  if args.dry_run:
    print("[DRY-RUN] 待投影 session:")
    for i, s in enumerate(todo):
      print(f"  [{i + 1}/{len(todo)}] {s.name}")
    sys.exit(0)

  # ── Step A: per-session projection ──────────────────────────────────────
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
          force=args.force,
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

  # ── Step B: merge all sessions into global files ────────────────────────
  print(f"\n{'=' * 70}")
  print(f"Step B: 合并全局 TuSimple 文件 (>1000 帧自动分片)")
  print(f"{'=' * 70}")

  # Use ALL sessions (including skipped) for merge
  all_sessions = sessions
  merged_dir = data_root / 'tusimple_merged'
  merge_stats = merge_split_files(data_root, all_sessions, merged_dir)

  total_elapsed = time.monotonic() - t_total_start
  print(f"\n{'=' * 70}")
  print(f"批量投影完成")
  print(f"  总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
  print(f"  投影: {completed}/{len(todo)} 成功  {len(skipped)} 跳过  {len(failed)} 失败")
  print(f"  累计投影: {total_projected} 帧")
  print(f"  合并输出: {merged_dir}")
  for split in SPLITS:
    ms = merge_stats.get(split, {})
    print(f"    {split}: {ms.get('total', 0)} entries, {len(ms.get('files', []))} files")
  if failed:
    print(f"\n失败列表:")
    for name, err in failed:
      print(f"  {name}: {err}")


if __name__ == '__main__':
  main()
