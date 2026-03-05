#!/usr/bin/env python3
"""Quantitative quality statistics for annotated multi-height data (Step B output).

Generates distribution plots and a text summary report for:
  - Lane line confidence per height
  - Frame pass rates
  - z_height transform consistency check
  - Lead vehicle detection rates

Usage:
  python tools/dashcam/viz/label_stats.py data/multi_height/quick_..._annotated/
  python tools/dashcam/viz/label_stats.py <annotated_dir> --output stats/  --sample 2000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


def load_frames(height_dir: Path, sample: int) -> list[dict]:
  """Load a sample of frames from a height directory. Returns list of data dicts."""
  files = sorted(height_dir.glob('*.npz'))
  if not files:
    return []
  if sample > 0 and sample < len(files):
    files = random.sample(files, sample)
    files.sort()
  results = []
  for f in files:
    try:
      results.append(dict(np.load(f, allow_pickle=True)))
    except Exception:
      pass
  return results


def compute_height_stats(frames: list[dict], min_ll_prob: float, height_m: float) -> dict:
  """Compute statistics for one height's frames."""
  if not frames:
    return {}

  ll_probs = []
  passes = 0
  lead_detected = 0

  for data in frames:
    ll_prob = data.get('lane_lines_prob', np.zeros(4))
    if not isinstance(ll_prob, np.ndarray):
      ll_prob = np.array(ll_prob)
    ll_prob = ll_prob.astype(np.float32)
    ll_probs.append(ll_prob)

    if ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob:
      passes += 1

    lead_prob = data.get('lead_prob', np.zeros(3))
    if np.any(lead_prob > 0.3):
      lead_detected += 1

  ll_probs_arr = np.stack(ll_probs)  # (N, 4)
  n = len(frames)

  return {
    'n_total': n,
    'n_pass': passes,
    'pass_rate': passes / n,
    'lead_rate': lead_detected / n,
    'll_prob_median': np.median(ll_probs_arr, axis=0),  # (4,)
    'll_prob_mean': np.mean(ll_probs_arr, axis=0),      # (4,)
    'll_prob_all': ll_probs_arr,                         # (N, 4)
  }


def compute_z_consistency(frames_per_height: dict[str, list[dict]],
                           heights_info: dict[str, float],
                           ref_tag: str = 'H1') -> dict:
  """Check that z_height offsets between heights match expected ΔH.

  For each height pair (H1, H_k), compute the mean z difference across
  all valid lane line points and compare with expected ΔH.
  """
  if ref_tag not in frames_per_height:
    return {'status': 'SKIP', 'reason': f'{ref_tag} not available'}

  ref_frames = frames_per_height[ref_tag]
  ref_h = heights_info.get(ref_tag, 1.22)

  results = {}
  max_error = 0.0

  for tag, frames in frames_per_height.items():
    if tag == ref_tag:
      continue
    h_k = heights_info.get(tag, ref_h)
    expected_delta = h_k - ref_h

    # Sample paired frames
    n_pairs = min(len(ref_frames), len(frames), 200)
    z_deltas = []

    for i in range(n_pairs):
      ref_data = ref_frames[i % len(ref_frames)]
      hk_data = frames[i % len(frames)]

      ref_ll = ref_data.get('lane_lines')
      hk_ll = hk_data.get('lane_lines')
      if ref_ll is None or hk_ll is None:
        continue

      # z_height is column 2 in (4, 33, 3)
      ref_z = ref_ll[:, :, 2].flatten()
      hk_z = hk_ll[:, :, 2].flatten()
      valid = ~np.isnan(ref_z) & ~np.isnan(hk_z)
      if valid.sum() < 10:
        continue
      z_deltas.append(np.mean(hk_z[valid] - ref_z[valid]))

    if z_deltas:
      measured = np.mean(z_deltas)
      error = abs(measured - expected_delta)
      max_error = max(max_error, error)
      results[tag] = {
        'expected_delta': expected_delta,
        'measured_delta': measured,
        'error': error,
        'ok': error < 0.1,
      }

  overall_ok = max_error < 0.1
  return {
    'status': 'PASS' if overall_ok else 'FAIL',
    'max_error': max_error,
    'per_height': results,
  }


def write_summary(stats_per_height: dict, z_check: dict, output_dir: Path,
                  session_name: str, min_ll_prob: float, heights_info: dict):
  """Write stats_summary.txt."""
  lines = [
    "=== 多高度标注质量统计摘要 ===",
    f"Session: {session_name}",
    f"min_ll_prob 阈值: {min_ll_prob:.2f}",
    "",
    f"{'高度':<10}{'总帧数':>8}{'PASS帧':>8}{'过滤率':>8}{'L0中位':>10}{'R0中位':>10}{'Lead率':>8}",
    "-" * 60,
  ]

  for tag in sorted(stats_per_height.keys()):
    s = stats_per_height[tag]
    if not s:
      continue
    h_m = heights_info.get(tag, 0.0)
    ll_med = s.get('ll_prob_median', np.zeros(4))
    lines.append(
      f"{tag} {h_m:.2f}m{s['n_total']:>8}{s['n_pass']:>8}"
      f"{s['pass_rate'] * 100:>7.1f}%"
      f"{ll_med[1]:>10.2f}{ll_med[2]:>10.2f}"
      f"{s['lead_rate'] * 100:>7.1f}%"
    )

  lines += ["", "-" * 60]

  z_status = z_check.get('status', 'SKIP')
  z_icon = "✅" if z_status == 'PASS' else ("❌" if z_status == 'FAIL' else "⏭")
  lines.append(f"z_height 变换一致性: {z_icon} {z_status}")
  if z_status not in ('SKIP',):
    lines.append(f"  最大误差: {z_check.get('max_error', 0):.4f}m (阈值 0.1m)")
    for tag, r in z_check.get('per_height', {}).items():
      ok_icon = "✅" if r['ok'] else "❌"
      lines.append(
        f"  {tag}: 期望ΔH={r['expected_delta']:.2f}m  "
        f"实测={r['measured_delta']:.3f}m  误差={r['error']:.4f}m {ok_icon}"
      )

  path = output_dir / 'stats_summary.txt'
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  print(f"\n摘要已保存: {path}")


def plot_lane_prob_distribution(stats_per_height: dict, output_dir: Path, min_ll_prob: float):
  """Plot lane_lines_prob distribution per height."""
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print("[WARN] matplotlib not available, skipping plots")
    return

  tags = sorted(stats_per_height.keys())
  n = len(tags)
  if n == 0:
    return

  cols = min(3, n)
  rows = (n + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
  axes = np.array(axes).reshape(-1)[:n]

  for i, tag in enumerate(tags):
    s = stats_per_height[tag]
    ax = axes[i]
    ll_prob = s.get('ll_prob_all')
    if ll_prob is None:
      continue
    ax.hist(ll_prob[:, 1], bins=30, alpha=0.7, label='L0 (inner left)', color='steelblue')
    ax.hist(ll_prob[:, 2], bins=30, alpha=0.7, label='R0 (inner right)', color='tomato')
    ax.axvline(min_ll_prob, color='orange', linestyle='--', label=f'threshold={min_ll_prob}')
    ax.set_title(f"{tag} (n={s['n_total']})")
    ax.set_xlabel('Lane line probability')
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)

  for ax in axes[n:]:
    ax.set_visible(False)

  fig.suptitle('Lane Line Probability Distribution per Height')
  plt.tight_layout()
  path = output_dir / 'lane_prob_distribution.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"图表已保存: {path}")


def plot_filter_rates(stats_per_height: dict, heights_info: dict, output_dir: Path):
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    return

  tags = sorted(stats_per_height.keys())
  pass_rates = [stats_per_height[t].get('pass_rate', 0) * 100 for t in tags]
  h_labels = [f"{t}\n{heights_info.get(t, 0):.2f}m" for t in tags]
  colors = ['green' if r >= 80 else 'orange' if r >= 60 else 'red' for r in pass_rates]

  fig, ax = plt.subplots(figsize=(8, 4))
  bars = ax.bar(h_labels, pass_rates, color=colors)
  ax.axhline(80, color='green', linestyle='--', alpha=0.5, label='目标 80%')
  ax.set_ylabel('过滤后帧占比 (%)')
  ax.set_title('各高度质量过滤通过率')
  ax.set_ylim(0, 105)
  ax.legend()
  for bar, r in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{r:.1f}%', ha='center', fontsize=9)
  plt.tight_layout()
  path = output_dir / 'filter_rates.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"图表已保存: {path}")


def plot_z_height_transform(frames_per_height: dict, heights_info: dict,
                             output_dir: Path, ref_tag: str = 'H1'):
  """Plot mean z_height vs distance for each height."""
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    return

  from openpilot.selfdrive.modeld.constants import ModelConstants
  x_idxs = np.array(ModelConstants.X_IDXS)

  fig, ax = plt.subplots(figsize=(10, 5))
  colors = plt.cm.viridis(np.linspace(0, 1, len(frames_per_height)))

  for (tag, frames), color in zip(sorted(frames_per_height.items()), colors):
    if not frames:
      continue
    z_list = []
    for data in frames[:200]:
      ll = data.get('lane_lines')
      lp = data.get('lane_lines_prob', np.zeros(4))
      if ll is None:
        continue
      # Average z across inner lane lines (idx 1, 2) where prob is good
      valid_lanes = [i for i in [1, 2] if lp[i] > 0.5]
      if not valid_lanes:
        continue
      z_vals = np.nanmean(ll[valid_lanes, :, 2], axis=0)  # (33,)
      z_list.append(z_vals)
    if z_list:
      z_mean = np.nanmean(z_list, axis=0)
      h_m = heights_info.get(tag, 0.0)
      ax.plot(x_idxs, z_mean, label=f"{tag} {h_m:.2f}m", color=color)

  ax.set_xlabel('前向距离 X (m)')
  ax.set_ylabel('z_height 均值 (m)')
  ax.set_title('各高度车道线 z_height vs 距离（期望各曲线近似平行，间距≈ΔH）')
  ax.legend()
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  path = output_dir / 'z_height_transform.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"图表已保存: {path}")


def plot_lead_detection(stats_per_height: dict, heights_info: dict, output_dir: Path):
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    return

  tags = sorted(stats_per_height.keys())
  lead_rates = [stats_per_height[t].get('lead_rate', 0) * 100 for t in tags]
  h_labels = [f"{t}\n{heights_info.get(t, 0):.2f}m" for t in tags]

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(h_labels, lead_rates, 'o-', color='steelblue', linewidth=2, markersize=8)
  ax.axhline(30, color='orange', linestyle='--', alpha=0.5, label='参考 30%')
  ax.set_ylabel('前车检测率 (%)')
  ax.set_title('各高度前车检测率 (lead_prob > 0.3)')
  ax.set_ylim(0, 105)
  ax.legend()
  ax.grid(True, alpha=0.3)
  for i, (h, r) in enumerate(zip(h_labels, lead_rates)):
    ax.text(i, r + 2, f'{r:.1f}%', ha='center', fontsize=9)
  plt.tight_layout()
  path = output_dir / 'lead_detection_rate.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"图表已保存: {path}")


def main():
  parser = argparse.ArgumentParser(description='Multi-height annotation quality statistics')
  parser.add_argument('annotated_dir', help='Annotated session directory')
  parser.add_argument('--output', default=None,
                      help='Output directory for charts/report (default: <annotated_dir>/stats/)')
  parser.add_argument('--min-ll-prob', type=float, default=0.5, help='Quality filter threshold')
  parser.add_argument('--sample', type=int, default=1000,
                      help='Max frames to sample per height (0=all)')
  args = parser.parse_args()

  annotated_dir = Path(args.annotated_dir)
  if not annotated_dir.exists():
    print(f"ERROR: not found: {annotated_dir}", file=sys.stderr)
    sys.exit(1)

  clip_info_path = annotated_dir / 'clip_info.json'
  if not clip_info_path.exists():
    print(f"ERROR: clip_info.json not found", file=sys.stderr)
    sys.exit(1)
  with open(clip_info_path) as f:
    clip_info = json.load(f)
  heights_info: dict[str, float] = clip_info.get('heights', {})

  output_dir = Path(args.output) if args.output else annotated_dir / 'stats'
  output_dir.mkdir(parents=True, exist_ok=True)

  all_tags = sorted([d.name for d in annotated_dir.iterdir()
                     if d.is_dir() and d.name.startswith('H') and d.name[1:].isdigit()])
  if not all_tags:
    print(f"ERROR: no height directories found", file=sys.stderr)
    sys.exit(1)

  print(f"Session: {annotated_dir.name}")
  print(f"Heights: {all_tags}")
  print(f"Sample: {args.sample if args.sample > 0 else 'all'} frames per height")
  print(f"Output: {output_dir}")

  frames_per_height: dict[str, list[dict]] = {}
  stats_per_height: dict[str, dict] = {}

  for tag in all_tags:
    h_dir = annotated_dir / tag
    if not h_dir.exists():
      continue
    print(f"  Loading {tag}...", end='', flush=True)
    frames = load_frames(h_dir, args.sample)
    frames_per_height[tag] = frames
    if frames:
      h_m = heights_info.get(tag, 0.0)
      stats_per_height[tag] = compute_height_stats(frames, args.min_ll_prob, h_m)
      print(f" {len(frames)} frames, pass={stats_per_height[tag]['pass_rate'] * 100:.1f}%")
    else:
      print(f" no frames")

  # z_height consistency check
  print("\nChecking z_height transform consistency...")
  z_check = compute_z_consistency(frames_per_height, heights_info)

  # Write summary
  write_summary(stats_per_height, z_check, output_dir, annotated_dir.name,
                args.min_ll_prob, heights_info)

  # Generate plots
  print("\nGenerating plots...")
  plot_lane_prob_distribution(stats_per_height, output_dir, args.min_ll_prob)
  plot_filter_rates(stats_per_height, heights_info, output_dir)
  plot_z_height_transform(frames_per_height, heights_info, output_dir)
  plot_lead_detection(stats_per_height, heights_info, output_dir)

  print(f"\nAll outputs in: {output_dir}")


if __name__ == '__main__':
  main()
