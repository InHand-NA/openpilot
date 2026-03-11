#!/usr/bin/env python3
"""批量标注 multi_height 数据目录下的所有 session。

扫描给定根目录下的所有 session 子目录（含 clip_info.json），
逐一调用 annotate_multi_height.annotate_session() 进行标注。
模型仅在首次调用时加载（后续从 pkl 缓存读取），GPU 预处理器跨 session 复用。

支持断点续跑：已完成标注的 session（annotations/ 中帧数 ≥ 源帧数）自动跳过。

用法：
  # 标注 data/multi_height_0311 下所有 session
  python tools/dashcam/annotate_batch.py data/multi_height_0311

  # 指定 ONNX 模型和高度
  python tools/dashcam/annotate_batch.py data/multi_height_0311 \\
      --onnx selfdrive/modeld/models/driving_vision.onnx --heights H1 H6

  # 预览，不实际标注
  python tools/dashcam/annotate_batch.py data/multi_height_0311 --dry-run

  # 强制重新标注已有 annotations 的 session
  python tools/dashcam/annotate_batch.py data/multi_height_0311 --force
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def discover_sessions(data_root: Path) -> list[Path]:
  """发现 data_root 下所有含 clip_info.json 的 session 目录（仅一级子目录）。"""
  sessions = []
  for d in sorted(data_root.iterdir()):
    if d.is_dir() and (d / 'clip_info.json').exists():
      sessions.append(d)
  return sessions


def count_annotated_frames(annotations_dir: Path, heights: list[str]) -> int:
  """统计 annotations 目录中第一个 height 子目录的 JSON 帧数。"""
  for tag in heights:
    tag_dir = annotations_dir / tag
    if tag_dir.is_dir():
      return len(list(tag_dir.glob('*.json')))
  return 0


def count_source_frames(session_dir: Path) -> int:
  """统计 session 中 H1 的源帧数。"""
  h1_dir = session_dir / 'H1'
  if not h1_dir.exists():
    return 0
  return len(list(h1_dir.glob('road_*.png')))


def main():
  parser = argparse.ArgumentParser(description='批量标注 multi_height 数据')
  parser.add_argument('data_root', help='数据根目录 (含多个 session 子目录)')
  parser.add_argument('--onnx', default='selfdrive/modeld/models/driving_vision.onnx',
                      help='ONNX 模型路径 (default: selfdrive/modeld/models/driving_vision.onnx)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='仅标注指定高度 (default: 全部)')
  parser.add_argument('--min-ll-prob', type=float, default=0.1,
                      help='车道线置信度过滤阈值 (default: 0.1)')
  parser.add_argument('--no-gpu-preprocess', action='store_true',
                      help='禁用 GPU OpenCL 预处理，回退到 CPU')
  parser.add_argument('--dry-run', action='store_true',
                      help='仅预览，不实际标注')
  parser.add_argument('--force', action='store_true',
                      help='强制重新标注已有 annotations 的 session')
  args = parser.parse_args()

  data_root = Path(args.data_root).resolve()
  if not data_root.exists():
    print(f"ERROR: 目录不存在: {data_root}", file=sys.stderr)
    sys.exit(1)

  sessions = discover_sessions(data_root)
  if not sessions:
    print(f"未找到 session (需含 clip_info.json): {data_root}")
    sys.exit(0)

  # 分类: skip / todo
  todo: list[Path] = []
  skipped: list[Path] = []

  for s in sessions:
    ann_dir = s / 'annotations'
    src_frames = count_source_frames(s)
    if src_frames == 0:
      skipped.append(s)
      continue

    if not args.force and ann_dir.exists():
      # 读取 heights 列表用于检查
      with open(s / 'clip_info.json') as f:
        ci = json.load(f)
      heights_list = args.heights or list(ci.get('heights', {}).keys())
      ann_frames = count_annotated_frames(ann_dir, heights_list)
      if ann_frames >= src_frames:
        skipped.append(s)
        continue

    todo.append(s)

  print(f"数据根目录: {data_root}")
  print(f"发现 {len(sessions)} 个 session: {len(todo)} 待标注, {len(skipped)} 跳过")
  if args.heights:
    print(f"标注高度: {args.heights}")
  print(f"ONNX: {args.onnx}")
  print(f"GPU 预处理: {'OFF' if args.no_gpu_preprocess else 'ON'}")
  print()

  if not todo:
    print("所有 session 已标注完成，无需操作。")
    sys.exit(0)

  if args.dry_run:
    print("[DRY-RUN] 待标注 session:")
    for i, s in enumerate(todo):
      src = count_source_frames(s)
      print(f"  [{i + 1}/{len(todo)}] {s.name}  ({src} frames)")
    sys.exit(0)

  # 延迟导入 — 避免 dry-run 时触发 tinygrad/CUDA 初始化
  os.environ.setdefault('PYOPENCL_CTX', '')
  if 'DEV' not in os.environ:
    os.environ['DEV'] = 'CUDA'

  from openpilot.tools.dashcam.annotate_multi_height import annotate_session

  t_total_start = time.monotonic()
  completed = 0
  failed: list[tuple[str, str]] = []

  try:
    for i, session_dir in enumerate(todo):
      output_dir = session_dir / 'annotations'
      print(f"\n{'=' * 70}")
      print(f"[{i + 1}/{len(todo)}] {session_dir.name}")
      print(f"{'=' * 70}")

      t_sess = time.monotonic()
      try:
        stats = annotate_session(
          session_dir=session_dir,
          output_dir=output_dir,
          onnx_path=args.onnx,
          heights_to_annotate=args.heights,
          min_ll_prob=args.min_ll_prob,
          use_gpu_preprocess=not args.no_gpu_preprocess,
        )
        elapsed = time.monotonic() - t_sess
        completed += 1
        print(f"  -> 完成 ({elapsed:.1f}s)")
      except Exception as e:
        elapsed = time.monotonic() - t_sess
        failed.append((session_dir.name, str(e)))
        print(f"  -> 失败 ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        continue

  except KeyboardInterrupt:
    print(f"\n[中断] 已完成 {completed}/{len(todo)} 个 session")

  # 汇总
  total_elapsed = time.monotonic() - t_total_start
  print(f"\n{'=' * 70}")
  print(f"批量标注完成")
  print(f"  总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
  print(f"  成功: {completed}/{len(todo)}  跳过: {len(skipped)}  失败: {len(failed)}")
  if failed:
    print(f"\n失败列表:")
    for name, err in failed:
      print(f"  {name}: {err}")


if __name__ == '__main__':
  main()
