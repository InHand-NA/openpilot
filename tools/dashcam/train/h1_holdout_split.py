#!/usr/bin/env python3
"""T3.2 — H1 退化验证集

从 clean_log.txt 未选中的 H1 帧中抽选退化验证集，
用于训练过程中监测 H1（标准安装高度 1.22m）性能是否退化。

流程：
  1. 遍历所有 session 的 H1 标注，列出全部帧号
  2. 排除 clean_log.txt 中已选中的帧
  3. 对剩余帧应用最低置信度过滤
  4. 随机抽出 holdout_ratio → H1 退化验证集

用法：
  python tools/dashcam/train/h1_holdout_split.py \
      --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
      --output-dir data/multi_height-0312/ \
      --holdout-ratio 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


L_INNER_IDX = 1
R_INNER_IDX = 2


def discover_sessions(dataset_dir: Path) -> list[str]:
  return sorted(d.name for d in dataset_dir.iterdir()
                if d.is_dir() and d.name.startswith('Town'))


def get_h1_frame_ids(dataset_dir: Path, session: str) -> list[int]:
  """获取 session 的 H1 标注目录下所有帧号"""
  ann_dir = dataset_dir / session / 'annotations' / 'H1'
  if not ann_dir.exists():
    return []
  ids = []
  for p in ann_dir.iterdir():
    if p.suffix == '.json':
      try:
        ids.append(int(p.stem))
      except ValueError:
        pass
  return sorted(ids)


def load_clean_log_set(path: Path) -> set[str]:
  with open(path) as f:
    return {line.strip() for line in f if line.strip()}


def check_confidence(dataset_dir: Path, session: str, fid: int, min_ll_prob: float) -> bool:
  """检查帧是否通过置信度过滤（至少一条内侧线 > threshold）"""
  ann_path = dataset_dir / session / 'annotations' / 'H1' / f'{fid:06d}.json'
  with open(ann_path) as f:
    ann = json.load(f)
  ll_prob = ann['lane_lines_prob']
  return ll_prob[L_INNER_IDX] > min_ll_prob or ll_prob[R_INNER_IDX] > min_ll_prob


def verify(
  holdout: list[str],
  clean_log_set: set[str],
  candidate_count: int,
  holdout_ratio: float,
  min_ll_prob: float,
  dataset_dir: Path,
  all_sessions: set[str],
) -> list[dict]:
  """5 项验证"""
  results = []
  holdout_set = set(holdout)

  # 1. 与训练集无交集
  overlap = holdout_set & clean_log_set
  ok = len(overlap) == 0
  results.append({
    'name': '与训练集无交集',
    'pass': ok,
    'msg': f'h1_holdout ∩ clean_log = ∅ ({len(holdout)} vs {len(clean_log_set)})' if ok
           else f'{len(overlap)} 帧重叠',
  })

  # 2. 比例正确（偏差 < 2%）
  actual_ratio = len(holdout) / candidate_count if candidate_count > 0 else 0
  dev = abs(actual_ratio - holdout_ratio)
  ok = dev < 0.02
  results.append({
    'name': '比例正确',
    'pass': ok,
    'msg': f'{len(holdout)}/{candidate_count} = {actual_ratio:.4f}, '
           f'目标 {holdout_ratio}, 偏差 {dev:.4f}' + (' < 2%' if ok else ' >= 2%'),
  })

  # 3. 置信度过滤
  bad = []
  for entry in holdout:
    session, fid = entry.rsplit('/', 1)
    fid = int(fid)
    if not check_confidence(dataset_dir, session, fid, min_ll_prob):
      bad.append(entry)
      if len(bad) >= 5:
        break
  ok = len(bad) == 0
  results.append({
    'name': '置信度过滤',
    'pass': ok,
    'msg': f'{len(holdout)}/{len(holdout)} 帧至少一条内侧线 prob>{min_ll_prob}' if ok
           else f'{len(bad)} 帧不满足，例: {bad[:3]}',
  })

  # 4. Session 覆盖率 > 80%
  holdout_sessions = {e.rsplit('/', 1)[0] for e in holdout}
  n_all = len(all_sessions)
  cov = len(holdout_sessions) / n_all if n_all > 0 else 0
  ok = cov > 0.80
  results.append({
    'name': 'Session 覆盖率',
    'pass': ok,
    'msg': f'{len(holdout_sessions)}/{n_all} = {cov:.1%}' + (' > 80%' if ok else ' <= 80%'),
  })

  # 5. 索引合法性
  bad = []
  for entry in holdout:
    session, fid = entry.rsplit('/', 1)
    ann = dataset_dir / session / 'annotations' / 'H1' / f'{int(fid):06d}.json'
    if not ann.exists():
      bad.append(entry)
      if len(bad) >= 5:
        break
  ok = len(bad) == 0
  results.append({
    'name': '索引合法性',
    'pass': ok,
    'msg': f'{len(holdout)} 帧全部有对应 H1 标注' if ok
           else f'{len(bad)} 帧标注缺失，例: {bad[:3]}',
  })

  return results


def main():
  parser = argparse.ArgumentParser(description='T3.2 H1 退化验证集')
  parser.add_argument('--dataset-dir', type=Path, required=True,
                      help='原始数据集根目录')
  parser.add_argument('--output-dir', type=Path, default=None,
                      help='clean_log.txt 所在目录 (default: dataset-dir)')
  parser.add_argument('--holdout-ratio', type=float, default=0.1,
                      help='从候选帧中抽出的比例 (default: 0.1)')
  parser.add_argument('--min-ll-prob', type=float, default=0.05,
                      help='最低置信度阈值 (default: 0.05)')
  parser.add_argument('--seed', type=int, default=42, help='随机种子 (default: 42)')
  args = parser.parse_args()

  dataset_dir = args.dataset_dir.resolve()
  output_dir = (args.output_dir or dataset_dir).resolve()

  # Read clean_log
  log_path = output_dir / 'clean_log.txt'
  if not log_path.exists():
    print(f'ERROR: {log_path} not found', file=sys.stderr)
    sys.exit(1)
  clean_log_set = load_clean_log_set(log_path)

  sessions = discover_sessions(dataset_dir)
  print(f'Sessions: {len(sessions)}')
  print(f'clean_log: {len(clean_log_set)} 帧（已排除）')
  print(f'holdout_ratio={args.holdout_ratio}, min_ll_prob={args.min_ll_prob}, seed={args.seed}')

  # Step 1-3: 收集候选帧
  candidates = []
  total_h1 = 0
  excluded_by_clean = 0
  excluded_by_conf = 0

  for session in sessions:
    frame_ids = get_h1_frame_ids(dataset_dir, session)
    total_h1 += len(frame_ids)

    for fid in frame_ids:
      entry = f'{session}/{fid:06d}'
      # Step 2: 排除 clean_log 已选中
      if entry in clean_log_set:
        excluded_by_clean += 1
        continue
      # Step 3: 置信度过滤
      if not check_confidence(dataset_dir, session, fid, args.min_ll_prob):
        excluded_by_conf += 1
        continue
      candidates.append(entry)

  print(f'\nH1 全部帧: {total_h1}')
  print(f'排除(已在clean_log): {excluded_by_clean}')
  print(f'排除(置信度不足): {excluded_by_conf}')
  print(f'候选帧: {len(candidates)}')

  # Step 4: 随机抽样
  random.seed(args.seed)
  n_holdout = round(len(candidates) * args.holdout_ratio)
  holdout = sorted(random.sample(candidates, n_holdout))

  print(f'抽出 holdout: {len(holdout)} 帧 ({len(holdout)/len(candidates):.1%})')

  # Write output
  holdout_path = output_dir / 'h1_holdout.txt'
  with open(holdout_path, 'w') as f:
    for entry in holdout:
      f.write(entry + '\n')

  # Verify
  print()
  all_sessions = set(sessions)
  checks = verify(holdout, clean_log_set, len(candidates), args.holdout_ratio,
                   args.min_ll_prob, dataset_dir, all_sessions)

  all_pass = True
  for c in checks:
    status = 'PASS' if c['pass'] else 'FAIL'
    print(f'[{status}] {c["name"]}: {c["msg"]}')
    if not c['pass']:
      all_pass = False

  # Write info JSON
  info = {
    'seed': args.seed,
    'holdout_ratio': args.holdout_ratio,
    'min_ll_prob': args.min_ll_prob,
    'counts': {
      'total_h1_frames': total_h1,
      'excluded_by_clean_log': excluded_by_clean,
      'excluded_by_confidence': excluded_by_conf,
      'candidates': len(candidates),
      'holdout': len(holdout),
    },
    'sessions': {
      'total': len(sessions),
      'holdout': len({e.rsplit('/', 1)[0] for e in holdout}),
    },
    'verification': checks,
  }
  info_path = output_dir / 'h1_holdout_info.json'
  with open(info_path, 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

  n_pass = sum(1 for c in checks if c['pass'])
  print(f'\n===== {n_pass}/{len(checks)} PASS =====')
  print(f'h1_holdout.txt: {holdout_path} ({len(holdout)} 行)')
  print(f'h1_holdout_info.json: {info_path}')

  if not all_pass:
    sys.exit(1)


if __name__ == '__main__':
  main()
