#!/usr/bin/env python3
"""TuSimple Phase 3: 数据清洗与抽样。

对 Phase 2 输出的 3D 标注进行质量过滤、时间抽样和训练/验证/测试集划分。

三步流程:
  Step 1  质量过滤 — 基于 H1 canonical 标注 (所有高度共享结果)
  Step 2  时间抽样 — 每 N 帧取 1 帧，降低时序冗余
  Step 3  训练/验证/测试划分 — 按高度展开后全局随机划分

输入: session_dir/3d_labels/H1/*.json  (Phase 2 输出)
输出: session_dir/splits/
  ├── clean_log.txt      "<frame_id>, pass" 或 "<frame_id>, fail"
  ├── sampled_log.txt    "<frame_id>"
  ├── train.txt          "<height_tag>/<frame_id>"
  ├── val.txt
  ├── test.txt
  └── stats.json

用法:
  python tools/dashcam/tusimple/clean_and_sample.py \\
      data/tusimple-sample/Town04_ClearNoon_p4.0_y0.0/ \\
      --heights H1 H3 H6 \\
      --min-ll-prob 0.7 --min-speed 1.0 \\
      --sample-every 5 --split-ratio 0.8 0.1 0.1 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

from openpilot.tools.dashcam.tusimple.config import HEIGHT_DEFS

L_INNER_IDX = 1
R_INNER_IDX = 2


# ---------------------------------------------------------------------------
# Step 1: 质量过滤
# ---------------------------------------------------------------------------

def quality_filter(
  label_dir: Path,
  min_ll_prob: float,
  min_speed: float,
) -> tuple[list[str], list[str], dict]:
  """对 H1 canonical 标注做质量过滤。

  Returns:
    pass_ids:  通过的 frame_id 列表 (sorted)
    fail_ids:  未通过的 frame_id 列表 (sorted)
    reasons:   {frame_id: reason_str} 失败原因
  """
  h1_dir = label_dir / 'H1'
  if not h1_dir.exists():
    raise FileNotFoundError(f"H1 标注目录不存在: {h1_dir}")

  frame_files = sorted(h1_dir.glob('*.json'))
  if not frame_files:
    raise ValueError(f"H1 标注为空: {h1_dir}")

  # 找到第 0 帧 (session 中最小帧号)
  all_frame_ids = sorted(p.stem for p in frame_files)
  first_frame_id = all_frame_ids[0] if all_frame_ids else None

  pass_ids: list[str] = []
  fail_ids: list[str] = []
  reasons: dict[str, str] = {}

  for fpath in frame_files:
    frame_id = fpath.stem

    with open(fpath) as f:
      anno = json.load(f)

    # 规则 1: 跳过第 0 帧
    if frame_id == first_frame_id:
      fail_ids.append(frame_id)
      reasons[frame_id] = 'first_frame'
      continue

    # 规则 2: 内侧车道线概率
    ll_prob = anno.get('lane_lines_prob', [0.0] * 4)
    l_inner = float(ll_prob[L_INNER_IDX])
    r_inner = float(ll_prob[R_INNER_IDX])
    if not (l_inner > min_ll_prob and r_inner > min_ll_prob):
      fail_ids.append(frame_id)
      reasons[frame_id] = f'll_prob L={l_inner:.3f} R={r_inner:.3f}'
      continue

    # 规则 3: 最低车速
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
# Step 2: 时间抽样
# ---------------------------------------------------------------------------

def time_sample(pass_ids: list[str], sample_every: int) -> list[str]:
  """每 sample_every 帧取 1 帧。"""
  return [fid for i, fid in enumerate(pass_ids) if i % sample_every == 0]


# ---------------------------------------------------------------------------
# Step 3: 训练/验证/测试划分
# ---------------------------------------------------------------------------

def expand_to_heights(sampled_ids: list[str], heights: list[str]) -> list[str]:
  """将帧 ID 列表按高度展开为 height_tag/frame_id 格式。"""
  entries = []
  for fid in sampled_ids:
    for h in heights:
      entries.append(f'{h}/{fid}')
  return entries


def split_entries(
  entries: list[str],
  ratios: list[float],
  seed: int,
) -> tuple[list[str], list[str], list[str]]:
  """全局随机划分 train/val/test。"""
  shuffled = entries.copy()
  random.seed(seed)
  random.shuffle(shuffled)

  n = len(shuffled)
  n_train = round(n * ratios[0])
  n_val = round(n * ratios[1])
  train = sorted(shuffled[:n_train])
  val = sorted(shuffled[n_train:n_train + n_val])
  test = sorted(shuffled[n_train + n_val:])
  return train, val, test


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clean_and_sample(
  session_dir: Path,
  heights: list[str],
  min_ll_prob: float,
  min_speed: float,
  sample_every: int,
  split_ratio: list[float],
  seed: int,
  dry_run: bool = False,
) -> dict:
  """Execute Phase 3 pipeline for one session. Returns stats dict."""
  label_dir = session_dir / '3d_labels'
  splits_dir = session_dir / 'splits'

  # Step 1: 质量过滤
  pass_ids, fail_ids, reasons = quality_filter(label_dir, min_ll_prob, min_speed)
  total = len(pass_ids) + len(fail_ids)
  pass_rate = len(pass_ids) / max(total, 1)

  print(f"  Step 1 质量过滤: {total} 帧 → {len(pass_ids)} pass ({pass_rate:.1%}), {len(fail_ids)} fail")

  # Step 2: 时间抽样
  sampled_ids = time_sample(pass_ids, sample_every)
  print(f"  Step 2 时间抽样: {len(pass_ids)} → {len(sampled_ids)} (every {sample_every})")

  # Step 3: 按高度展开 + 划分
  entries = expand_to_heights(sampled_ids, heights)
  train, val, test = split_entries(entries, split_ratio, seed)
  print(f"  Step 3 划分: {len(entries)} entries → train={len(train)} val={len(val)} test={len(test)}")

  # Per-height stats
  per_height = {}
  for h in heights:
    h_pass = len(pass_ids)  # 所有高度共享
    h_sampled = len(sampled_ids)
    h_train = sum(1 for e in train if e.startswith(f'{h}/'))
    h_val = sum(1 for e in val if e.startswith(f'{h}/'))
    h_test = sum(1 for e in test if e.startswith(f'{h}/'))
    per_height[h] = {
      'pass': h_pass, 'sampled': h_sampled,
      'train': h_train, 'val': h_val, 'test': h_test,
    }

  stats = {
    'total_frames': total * len(heights),
    'quality_pass': len(pass_ids) * len(heights),
    'quality_pass_rate': pass_rate,
    'sampled': len(sampled_ids) * len(heights),
    'per_height': per_height,
    'train': len(train),
    'val': len(val),
    'test': len(test),
    'params': {
      'min_ll_prob': min_ll_prob,
      'min_speed': min_speed,
      'sample_every': sample_every,
      'split_ratio': split_ratio,
      'seed': seed,
      'heights': heights,
    },
  }

  if dry_run:
    print("  [DRY-RUN] 不写入文件")
    return stats

  # Write outputs
  splits_dir.mkdir(parents=True, exist_ok=True)
  write_clean_log(splits_dir / 'clean_log.txt', pass_ids, fail_ids)
  write_lines(splits_dir / 'sampled_log.txt', sampled_ids)
  write_lines(splits_dir / 'train.txt', train)
  write_lines(splits_dir / 'val.txt', val)
  write_lines(splits_dir / 'test.txt', test)
  with open(splits_dir / 'stats.json', 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

  print(f"  输出: {splits_dir}")
  return stats


def main():
  parser = argparse.ArgumentParser(description='TuSimple Phase 3: 数据清洗与抽样')
  parser.add_argument('session_dir', help='Session 目录 (含 3d_labels/)')
  parser.add_argument('--heights', nargs='+', default=None,
                      help=f'要处理的高度 (default: clip_info 中所有高度)')
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
                      help='仅预览统计，不写入文件')
  args = parser.parse_args()

  session_dir = Path(args.session_dir).resolve()
  if not session_dir.exists():
    print(f"ERROR: 目录不存在: {session_dir}", file=sys.stderr)
    sys.exit(1)

  label_dir = session_dir / '3d_labels'
  if not label_dir.exists():
    print(f"ERROR: 3d_labels 目录不存在: {label_dir}", file=sys.stderr)
    print(f"请先运行 annotate_3d.py 生成标注")
    sys.exit(1)

  # Determine heights
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

  # Filter to heights that have label dirs
  heights = [h for h in heights if (label_dir / h).exists()]
  if not heights:
    print(f"ERROR: 无可用高度标注目录", file=sys.stderr)
    sys.exit(1)

  # Normalize ratios
  r_sum = sum(args.split_ratio)
  ratios = [r / r_sum for r in args.split_ratio]

  print(f"Session: {session_dir.name}")
  print(f"Heights: {heights}")
  print(f"min_ll_prob={args.min_ll_prob}  min_speed={args.min_speed}")
  print(f"sample_every={args.sample_every}  split={ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}  seed={args.seed}")
  print()

  try:
    stats = clean_and_sample(
      session_dir=session_dir,
      heights=heights,
      min_ll_prob=args.min_ll_prob,
      min_speed=args.min_speed,
      sample_every=args.sample_every,
      split_ratio=ratios,
      seed=args.seed,
      dry_run=args.dry_run,
    )
  except (FileNotFoundError, ValueError) as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  main()
