#!/usr/bin/env python3
"""删除 data_root 下所有 session 的 H0/ PNG 图像，保留 metadata.jsonl。

H0 图像仅用于 Phase 2 modeld 推理，标注完成后不再需要。
删除后可回收约 60% 磁盘空间 (H0 road+wide 占总存储大头)。

用法:
  # 预览 (不删除)
  python tools/dashcam/tusimple/purge_h0_images.py data/tusimple-sample-new --dry-run

  # 执行删除
  python tools/dashcam/tusimple/purge_h0_images.py data/tusimple-sample-new
"""

import argparse
import sys
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description='删除 H0/ PNG 图像，回收磁盘空间')
  parser.add_argument('data_root', help='数据根目录')
  parser.add_argument('--dry-run', action='store_true', help='仅统计，不删除')
  args = parser.parse_args()

  data_root = Path(args.data_root).resolve()
  if not data_root.exists():
    print(f"ERROR: 目录不存在: {data_root}", file=sys.stderr)
    sys.exit(1)

  # 先统计
  total_files = 0
  total_bytes = 0
  sessions = 0

  all_pngs: list[Path] = []
  for d in sorted(data_root.iterdir()):
    h0_dir = d / 'H0'
    if not h0_dir.is_dir():
      continue
    pngs = list(h0_dir.glob('*.png'))
    if not pngs:
      continue
    sessions += 1
    for p in pngs:
      total_bytes += p.stat().st_size
      total_files += 1
      all_pngs.append(p)

  gb = total_bytes / (1024 ** 3)

  if total_files == 0:
    print("无 H0 PNG 文件可删除。")
    return

  print(f"将删除: {total_files} 个 PNG 文件 ({gb:.2f} GB) from {sessions} sessions")

  if args.dry_run:
    print("(使用不带 --dry-run 执行删除)")
    return

  # 用户确认
  print()
  print("WARNING: 删除 H0 PNG 图像后将无法重新运行 Phase 2 3D 标注程序 (annotate_3d.py)。")
  print("         请确认所有 session 的 3D 标注已完成 (3d_labels/ 目录存在且完整)。")
  print()
  answer = input("确认删除? [y/N] ").strip().lower()
  if answer != 'y':
    print("取消。")
    return

  for p in all_pngs:
    p.unlink()
  print(f"已删除: {total_files} 个 PNG 文件 ({gb:.2f} GB) from {sessions} sessions")


if __name__ == '__main__':
  main()
