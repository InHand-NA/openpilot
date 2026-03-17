#!/usr/bin/env python3
"""T3.1 — 训练数据集划分

将 clean_log.txt 中的帧按 8:1:1 全局随机划分为 train/val/test。
运行结束时自动执行 5 项验证，全部 PASS 才正常退出。

用法：
  python tools/dashcam/train/split_dataset.py \
      --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
      --output-dir data/multi_height-0312/ \
      --ratio 0.8 0.1 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_entries(path: Path) -> list[str]:
  with open(path) as f:
    return [line.strip() for line in f if line.strip()]


def load_disabled(path: Path) -> set[str]:
  if not path.exists():
    return set()
  with open(path) as f:
    return {line.strip() for line in f if line.strip()}


def parse_entry(entry: str) -> tuple[str, int]:
  session, fid = entry.rsplit('/', 1)
  return session, int(fid)


def get_sessions(entries: list[str]) -> set[str]:
  return {parse_entry(e)[0] for e in entries}


def split_entries(entries: list[str], ratios: list[float], seed: int) -> tuple[list[str], list[str], list[str]]:
  shuffled = entries.copy()
  random.seed(seed)
  random.shuffle(shuffled)

  n = len(shuffled)
  n_train = round(n * ratios[0])
  n_val = round(n * ratios[1])
  # test gets the remainder to guarantee conservation
  train = sorted(shuffled[:n_train])
  val = sorted(shuffled[n_train:n_train + n_val])
  test = sorted(shuffled[n_train + n_val:])
  return train, val, test


def write_list(path: Path, entries: list[str]) -> None:
  with open(path, 'w') as f:
    for e in entries:
      f.write(e + '\n')


def verify(
  clean_entries: list[str],
  train: list[str],
  val: list[str],
  test: list[str],
  ratios: list[float],
  dataset_dir: Path,
) -> list[dict]:
  """运行 5 项验证，返回 [{name, pass, msg}, ...]"""
  results = []

  # 1. 总帧数守恒
  total = len(clean_entries)
  actual = len(train) + len(val) + len(test)
  ok = actual == total
  results.append({
    'name': '总帧数守恒',
    'pass': ok,
    'msg': f'{len(train)}+{len(val)}+{len(test)}={actual} == {total}' if ok
           else f'{actual} != {total}',
  })

  # 2. 无交集
  s_train, s_val, s_test = set(train), set(val), set(test)
  tv = s_train & s_val
  tt = s_train & s_test
  vt = s_val & s_test
  ok = len(tv) == 0 and len(tt) == 0 and len(vt) == 0
  results.append({
    'name': '无交集',
    'pass': ok,
    'msg': 'train∩val=∅, train∩test=∅, val∩test=∅' if ok
           else f'overlaps: tv={len(tv)}, tt={len(tt)}, vt={len(vt)}',
  })

  # 3. 比例偏差 < 1%
  actual_ratios = [len(train) / total, len(val) / total, len(test) / total]
  max_dev = max(abs(a - t) for a, t in zip(actual_ratios, ratios))
  ok = max_dev < 0.01
  ratio_str = '/'.join(f'{r:.3f}' for r in actual_ratios)
  results.append({
    'name': '比例偏差',
    'pass': ok,
    'msg': f'实际 {ratio_str}, 最大偏差 {max_dev:.4f}' + (' < 1%' if ok else ' >= 1%'),
  })

  # 4. Session 覆盖率 > 80%
  all_sessions = get_sessions(clean_entries)
  n_all = len(all_sessions)
  coverages = {}
  ok_all = True
  for name, subset in [('train', train), ('val', val), ('test', test)]:
    sub_sessions = get_sessions(subset)
    cov = len(sub_sessions) / n_all if n_all > 0 else 0
    coverages[name] = (len(sub_sessions), cov)
    if cov <= 0.80:
      ok_all = False
  cov_str = ', '.join(f'{k}={v[0]}/{n_all}({v[1]:.1%})' for k, v in coverages.items())
  results.append({
    'name': 'Session 覆盖率',
    'pass': ok_all,
    'msg': cov_str + (' 全部>80%' if ok_all else ' 有子集≤80%'),
  })

  # 5. 索引合法性（检查对应 H1 标注文件存在）
  all_entries = train + val + test
  bad = []
  for entry in all_entries:
    session, fid = parse_entry(entry)
    ann = dataset_dir / session / 'annotations' / 'H1' / f'{fid:06d}.json'
    if not ann.exists():
      bad.append(entry)
      if len(bad) >= 5:
        break
  ok = len(bad) == 0
  results.append({
    'name': '索引合法性',
    'pass': ok,
    'msg': f'{len(all_entries)} 帧全部有对应 H1 标注' if ok
           else f'{len(bad)} 帧标注缺失，例: {bad[:3]}',
  })

  return results


def main():
  parser = argparse.ArgumentParser(description='T3.1 训练数据集划分 (8:1:1)')
  parser.add_argument('--dataset-dir', type=Path, required=True,
                      help='原始数据集根目录（只读，用于验证标注存在性）')
  parser.add_argument('--output-dir', type=Path, default=None,
                      help='clean_log.txt 所在目录，划分结果也写入此处 (default: dataset-dir)')
  parser.add_argument('--ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                      help='train/val/test 比例 (default: 0.8 0.1 0.1)')
  parser.add_argument('--seed', type=int, default=42, help='随机种子 (default: 42)')
  args = parser.parse_args()

  dataset_dir = args.dataset_dir.resolve()
  output_dir = (args.output_dir or dataset_dir).resolve()

  # Normalize ratios
  r_sum = sum(args.ratio)
  ratios = [r / r_sum for r in args.ratio]

  # Read clean_log
  log_path = output_dir / 'clean_log.txt'
  if not log_path.exists():
    print(f'ERROR: {log_path} not found', file=sys.stderr)
    sys.exit(1)
  all_entries = load_entries(log_path)

  # Exclude disabled frames
  disabled_path = output_dir / 'disabled_frames.txt'
  disabled = load_disabled(disabled_path)
  clean_entries = [e for e in all_entries if e not in disabled]

  print(f'clean_log: {len(all_entries)} 帧, disabled: {len(disabled)}, '
        f'有效: {len(clean_entries)} 帧')
  print(f'ratio: {ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}  seed={args.seed}')

  # Split
  train, val, test = split_entries(clean_entries, ratios, args.seed)
  print(f'划分: train={len(train)}, val={len(val)}, test={len(test)}')

  # Write output
  write_list(output_dir / 'train.txt', train)
  write_list(output_dir / 'val.txt', val)
  write_list(output_dir / 'test.txt', test)

  # Verify
  print()
  checks = verify(clean_entries, train, val, test, ratios, dataset_dir)

  all_pass = True
  for c in checks:
    status = 'PASS' if c['pass'] else 'FAIL'
    print(f'[{status}] {c["name"]}: {c["msg"]}')
    if not c['pass']:
      all_pass = False

  # Write split_info.json
  info = {
    'seed': args.seed,
    'ratio': ratios,
    'counts': {'clean_log': len(all_entries), 'disabled': len(disabled),
                'effective': len(clean_entries), 'train': len(train), 'val': len(val), 'test': len(test)},
    'sessions': {
      'total': len(get_sessions(clean_entries)),
      'train': len(get_sessions(train)),
      'val': len(get_sessions(val)),
      'test': len(get_sessions(test)),
    },
    'verification': checks,
  }
  info_path = output_dir / 'split_info.json'
  with open(info_path, 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

  n_pass = sum(1 for c in checks if c['pass'])
  print(f'\n===== {n_pass}/{len(checks)} PASS =====')
  print(f'split_info.json 已写入: {info_path}')

  if not all_pass:
    sys.exit(1)


if __name__ == '__main__':
  main()
