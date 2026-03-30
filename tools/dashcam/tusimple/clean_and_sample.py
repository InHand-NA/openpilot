#!/usr/bin/env python3
"""TuSimple Phase 3: per-session 数据清洗。

对 Phase 2 输出的 3D 标注进行质量过滤。
采集工具已固定 1 FPS，无需再做时间抽样。
训练/验证/测试划分由 batch_clean_and_sample.py 在 session 级别完成。

输入: session_dir/3d_labels/H*/*.json  (Phase 2 输出)
输出: session_dir/splits/
  ├── clean_log.txt      "<frame_id>, pass" 或 "<frame_id>, fail"
  ├── pass_frames.txt    通过质量过滤的 frame_id 列表
  └── stats.json

用法:
  python tools/dashcam/tusimple/clean_and_sample.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/ \\
      --min-ll-prob 0.5 --min-speed 1.0
"""

import argparse
import json
import sys
from pathlib import Path

from openpilot.tools.dashcam.tusimple.config import HEIGHT_DEFS

L_INNER_IDX = 1
R_INNER_IDX = 2


# ---------------------------------------------------------------------------
# 质量过滤
# ---------------------------------------------------------------------------

def quality_filter(
  label_dir: Path,
  min_ll_prob: float,
  min_speed: float,
) -> tuple[list[str], list[str], dict]:
  """对任意高度的 canonical 标注做质量过滤。

  自动选择 label_dir 下第一个存在的 H* 子目录作为基准
  （所有高度共享同一份 3D 标注，质量指标相同）。

  Returns:
    pass_ids:  通过的 frame_id 列表 (sorted)
    fail_ids:  未通过的 frame_id 列表 (sorted)
    reasons:   {frame_id: reason_str} 失败原因
  """
  ref_dir = None
  for d in sorted(label_dir.iterdir()):
    if d.is_dir() and d.name.startswith('H'):
      ref_dir = d
      break
  if ref_dir is None:
    raise FileNotFoundError(f"标注目录中无 H* 子目录: {label_dir}")

  frame_files = sorted(ref_dir.glob('*.json'))
  if not frame_files:
    raise ValueError(f"{ref_dir.name} 标注为空: {ref_dir}")

  all_frame_ids = sorted(p.stem for p in frame_files)
  first_frame_id = all_frame_ids[0] if all_frame_ids else None

  pass_ids: list[str] = []
  fail_ids: list[str] = []
  reasons: dict[str, str] = {}

  for fpath in frame_files:
    frame_id = fpath.stem

    with open(fpath) as f:
      anno = json.load(f)

    if frame_id == first_frame_id:
      fail_ids.append(frame_id)
      reasons[frame_id] = 'first_frame'
      continue

    ll_prob = anno.get('lane_lines_prob', [0.0] * 4)
    l_inner = float(ll_prob[L_INNER_IDX])
    r_inner = float(ll_prob[R_INNER_IDX])
    if not (l_inner > min_ll_prob and r_inner > min_ll_prob):
      fail_ids.append(frame_id)
      reasons[frame_id] = f'll_prob L={l_inner:.3f} R={r_inner:.3f}'
      continue

    v_ego = float(anno.get('v_ego', 0.0))
    if v_ego <= min_speed:
      fail_ids.append(frame_id)
      reasons[frame_id] = f'v_ego={v_ego:.2f}'
      continue

    pass_ids.append(frame_id)

  pass_ids.sort()
  fail_ids.sort()
  return pass_ids, fail_ids, reasons


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_lines(path: Path, lines: list[str]) -> None:
  with open(path, 'w') as f:
    for line in lines:
      f.write(line + '\n')


def write_clean_log(path: Path, pass_ids: list[str], fail_ids: list[str]) -> None:
  """Write clean_log.txt: <frame_id>, pass/fail (sorted by frame_id)."""
  all_entries = [(fid, 'pass') for fid in pass_ids] + [(fid, 'fail') for fid in fail_ids]
  all_entries.sort(key=lambda x: x[0])
  with open(path, 'w') as f:
    for fid, status in all_entries:
      f.write(f'{fid}, {status}\n')


def expand_to_heights(frame_ids: list[str], heights: list[str]) -> list[str]:
  """将帧 ID 列表按高度展开为 height_tag/frame_id 格式。"""
  entries = []
  for fid in frame_ids:
    for h in heights:
      entries.append(f'{h}/{fid}')
  return sorted(entries)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean_session(
  session_dir: Path,
  heights: list[str],
  min_ll_prob: float,
  min_speed: float,
  dry_run: bool = False,
) -> dict:
  """Execute per-session quality filter. Returns stats dict."""
  label_dir = session_dir / '3d_labels'
  splits_dir = session_dir / 'splits'

  pass_ids, fail_ids, reasons = quality_filter(label_dir, min_ll_prob, min_speed)
  total = len(pass_ids) + len(fail_ids)
  pass_rate = len(pass_ids) / max(total, 1)

  print(f"  质量过滤: {total} 帧 → {len(pass_ids)} pass ({pass_rate:.1%}), {len(fail_ids)} fail")

  stats = {
    'total_frames': total,
    'quality_pass': len(pass_ids),
    'quality_pass_rate': pass_rate,
    'heights': heights,
    'params': {
      'min_ll_prob': min_ll_prob,
      'min_speed': min_speed,
    },
  }

  if dry_run:
    print("  [DRY-RUN] 不写入文件")
    return stats

  splits_dir.mkdir(parents=True, exist_ok=True)
  write_clean_log(splits_dir / 'clean_log.txt', pass_ids, fail_ids)
  write_lines(splits_dir / 'pass_frames.txt', pass_ids)
  with open(splits_dir / 'stats.json', 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

  print(f"  输出: {splits_dir}  ({len(pass_ids)} pass frames)")
  return stats


def main():
  parser = argparse.ArgumentParser(description='TuSimple Phase 3: per-session 数据清洗')
  parser.add_argument('session_dir', help='Session 目录 (含 3d_labels/)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help='要处理的高度 (default: clip_info 中所有高度)')
  parser.add_argument('--min-ll-prob', type=float, default=0.5,
                      help='内侧车道线最低概率 (default: 0.5)')
  parser.add_argument('--min-speed', type=float, default=1.0,
                      help='最低自车速度 m/s (default: 1.0)')
  parser.add_argument('--dry-run', action='store_true',
                      help='仅预览统计，不写入文件')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"ERROR: 目录不存在: {session_dir}", file=sys.stderr)
    sys.exit(1)

  label_dir = session_dir / '3d_labels'
  if not label_dir.exists():
    print(f"ERROR: 3d_labels 目录不存在: {label_dir}", file=sys.stderr)
    sys.exit(1)

  if args.heights:
    heights = args.heights
  else:
    clip_info_path = session_dir / 'clip_info.json'
    if clip_info_path.exists():
      with open(clip_info_path) as f:
        clip_info = json.load(f)
      heights = sorted(clip_info.get('heights', HEIGHT_DEFS).keys())
    else:
      heights = sorted(HEIGHT_DEFS.keys())
  heights = [h for h in heights if (label_dir / h).exists()]
  if not heights:
    print(f"ERROR: 无可用高度标注目录", file=sys.stderr)
    sys.exit(1)

  print(f"Session: {session_dir.name}")
  print(f"Heights: {heights}")
  print(f"min_ll_prob={args.min_ll_prob}  min_speed={args.min_speed}")
  print()

  try:
    clean_session(
      session_dir=session_dir,
      heights=heights,
      min_ll_prob=args.min_ll_prob,
      min_speed=args.min_speed,
      dry_run=args.dry_run,
    )
  except (FileNotFoundError, ValueError) as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  main()
