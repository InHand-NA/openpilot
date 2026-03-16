#!/usr/bin/env python3
"""
lead_prob 分布统计脚本

对多高度数据集的 H1 标注进行 lead_prob 分布分析，
同时输出 lane_lines_prob 分布作为对比。

用途：为 T7 损失函数设计（lead_prob 软目标 vs 二值化）提供数据依据。

用法：
  python tools/dashcam/analysis/lead_prob_distribution.py <dataset_dir>

示例：
  python tools/dashcam/analysis/lead_prob_distribution.py /nfs/openpilot-datasets/multi_height-0312
"""

import argparse
import glob
import json
import sys

import numpy as np


def analyze_lead_prob(dataset_dir: str):
  files = sorted(glob.glob(f"{dataset_dir}/Town*/annotations/H1/*.json"))
  if not files:
    print(f"ERROR: no H1 annotation files found in {dataset_dir}/Town*/annotations/H1/")
    sys.exit(1)

  print(f"Total H1 annotation files: {len(files)}")

  all_probs = []
  per_slot = [[], [], []]

  for f in files:
    with open(f) as fh:
      ann = json.load(fh)
    lp = ann['lead_prob']
    all_probs.extend(lp)
    for i in range(3):
      per_slot[i].append(lp[i])

  all_probs = np.array(all_probs)
  print(f"Total lead_prob values: {len(all_probs)} ({len(all_probs) // 3} frames x 3 slots)")

  # Overall distribution
  print(f"\n{'=' * 60}")
  print(f"Overall lead_prob distribution")
  print(f"{'=' * 60}")
  bins = [
    (0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
    (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
    (0.9, 0.95), (0.95, 1.001),
  ]
  for lo, hi in bins:
    mask = (all_probs >= lo) & (all_probs < hi)
    count = mask.sum()
    pct = count / len(all_probs) * 100
    bar = '#' * int(pct)
    print(f"  [{lo:.2f}, {hi:.2f}): {count:>8d}  ({pct:5.1f}%)  {bar}")

  print(f"\n  Mean:   {all_probs.mean():.4f}")
  print(f"  Median: {np.median(all_probs):.4f}")
  print(f"  Std:    {all_probs.std():.4f}")

  # Key thresholds
  print(f"\n{'=' * 60}")
  print(f"Key threshold statistics")
  print(f"{'=' * 60}")
  for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    below = (all_probs < t).mean() * 100
    above = (all_probs >= t).mean() * 100
    print(f"  < {t}: {below:5.1f}%    >= {t}: {above:5.1f}%")

  # Per-slot distribution
  print(f"\n{'=' * 60}")
  print(f"Per-slot distribution (slot 0=0s, 1=2s, 2=4s)")
  print(f"{'=' * 60}")
  for si in range(3):
    arr = np.array(per_slot[si])
    print(f"\n  Slot {si}: mean={arr.mean():.4f}, median={np.median(arr):.4f}")
    for t in [0.1, 0.3, 0.5, 0.9]:
      print(f"    <{t}: {(arr < t).mean() * 100:5.1f}%   >={t}: {(arr >= t).mean() * 100:5.1f}%")

  # Intermediate range analysis
  print(f"\n{'=' * 60}")
  print(f"Intermediate range [0.2, 0.8] analysis")
  print(f"{'=' * 60}")
  mid = all_probs[(all_probs >= 0.2) & (all_probs < 0.8)]
  print(f"  Count: {len(mid)} / {len(all_probs)} ({len(mid) / len(all_probs) * 100:.2f}%)")
  if len(mid) > 0:
    print(f"  Mean: {mid.mean():.4f}")
    print(f"  Distribution within [0.2, 0.8]:")
    for lo, hi in [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]:
      c = ((mid >= lo) & (mid < hi)).sum()
      print(f"    [{lo:.1f}, {hi:.1f}): {c}")

  # Truncated linear weight analysis
  print(f"\n{'=' * 60}")
  print(f"Truncated linear weight: clamp((prob - 0.3) / 0.7, 0, 1)")
  print(f"{'=' * 60}")
  weights = np.clip((all_probs - 0.3) / 0.7, 0, 1)
  print(f"  weight=0 (prob<0.3): {(weights == 0).mean() * 100:.1f}%")
  print(f"  weight>0 (prob>=0.3): {(weights > 0).mean() * 100:.1f}%")
  print(f"  Mean weight (over all): {weights.mean():.4f}")
  print(f"  Mean weight (where >0): {weights[weights > 0].mean():.4f}" if (weights > 0).any() else "")

  return all_probs


def analyze_lane_lines_prob(dataset_dir: str):
  files = sorted(glob.glob(f"{dataset_dir}/Town*/annotations/H1/*.json"))

  ll_probs = []
  for f in files:
    with open(f) as fh:
      ann = json.load(fh)
    ll_probs.extend(ann['lane_lines_prob'])

  ll_probs = np.array(ll_probs)
  print(f"\n\n{'=' * 60}")
  print(f"lane_lines_prob distribution (for comparison)")
  print(f"{'=' * 60}")
  print(f"Total values: {len(ll_probs)} ({len(ll_probs) // 4} frames x 4 lines)")

  bins = [
    (0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
    (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
    (0.9, 0.95), (0.95, 1.001),
  ]
  for lo, hi in bins:
    mask = (ll_probs >= lo) & (ll_probs < hi)
    count = mask.sum()
    pct = count / len(ll_probs) * 100
    bar = '#' * int(pct)
    print(f"  [{lo:.2f}, {hi:.2f}): {count:>8d}  ({pct:5.1f}%)  {bar}")

  print(f"\n  Mean:   {ll_probs.mean():.4f}")
  print(f"  Median: {np.median(ll_probs):.4f}")

  mid = ll_probs[(ll_probs >= 0.2) & (ll_probs < 0.8)]
  print(f"\n  Intermediate [0.2, 0.8): {len(mid)} / {len(ll_probs)} ({len(mid) / len(ll_probs) * 100:.2f}%)")


def main():
  parser = argparse.ArgumentParser(description="lead_prob distribution analysis")
  parser.add_argument("dataset_dir", help="path to multi_height dataset (e.g. data/multi_height-0312/)")
  parser.add_argument("--no-lane-compare", action="store_true", help="skip lane_lines_prob comparison")
  args = parser.parse_args()

  analyze_lead_prob(args.dataset_dir)
  if not args.no_lane_compare:
    analyze_lane_lines_prob(args.dataset_dir)


if __name__ == "__main__":
  main()
