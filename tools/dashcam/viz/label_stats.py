#!/usr/bin/env python3
"""Quantitative quality statistics for annotated multi-height data.

Supports two input modes (auto-detected):

  1. Single session:  directory with clip_info.json + H1/, H2/, ...
     python tools/dashcam/viz/label_stats.py data/.../annotations/

  2. Batch root:  directory with multiple session subdirs, each containing annotations/
     python tools/dashcam/viz/label_stats.py data/multi_height_0311/

Generates distribution plots and a text summary report for:
  - Lane line confidence per height (aggregate across all sessions)
  - Frame pass rates (per height, per session, per pitch×yaw)
  - z_height transform consistency check
  - Lead vehicle detection rates
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Frame loading (supports both .npz and .json)
# ---------------------------------------------------------------------------

_ARRAY_FIELDS = {
  'lane_lines', 'lane_lines_prob', 'road_edges', 'road_edges_prob',
  'lead', 'lead_prob', 'pose', 'road_transform', 'wide_from_device_euler',
  'world_pose',
}

# Confidence metrics: (metric_name, stats_dict_key, component_labels)
CONF_METRICS = [
  ('lane_lines_prob', 'll_prob_all', ['L-outer', 'L-inner', 'R-inner', 'R-outer']),
  ('road_edges_prob', 're_prob_all', ['RE-left', 'RE-right']),
  ('lead_prob', 'ld_prob_all', ['Lead-0', 'Lead-1', 'Lead-2']),
]


def _load_json_frame(path: Path) -> dict:
  with open(path) as f:
    record = json.load(f)
  for key in _ARRAY_FIELDS:
    if key in record:
      record[key] = np.array(record[key], dtype=np.float32)
  return record


def load_frames(height_dir: Path, sample: int) -> list[dict]:
  """Load a sample of frames from a height directory. Supports .json and .npz."""
  json_files = sorted(height_dir.glob('*.json'))
  npz_files = sorted(height_dir.glob('*.npz')) if not json_files else []
  files = json_files or npz_files
  if not files:
    return []
  is_json = bool(json_files)
  if 0 < sample < len(files):
    files = random.sample(files, sample)
    files.sort()
  results = []
  for f in files:
    try:
      if is_json:
        results.append(_load_json_frame(f))
      else:
        results.append(dict(np.load(f, allow_pickle=True)))
    except Exception:
      pass
  return results


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def discover_annotated_dirs(data_root: Path) -> list[tuple[Path, str]]:
  """Find all annotated session dirs under data_root.

  Returns list of (annotations_dir, session_name).
  """
  results = []
  for d in sorted(data_root.iterdir()):
    if not d.is_dir():
      continue
    ann = d / 'annotations'
    if ann.is_dir() and (ann / 'clip_info.json').exists():
      results.append((ann, d.name))
  return results


def detect_mode(input_dir: Path) -> str:
  """Auto-detect: 'single' if input_dir itself is an annotated dir, else 'batch'."""
  if (input_dir / 'clip_info.json').exists():
    # Check if it has H* subdirs (single session annotations dir)
    has_h = any(d.is_dir() and d.name.startswith('H') and d.name[1:].isdigit()
                for d in input_dir.iterdir())
    if has_h:
      return 'single'
  return 'batch'


# ---------------------------------------------------------------------------
# Per-height statistics (unchanged core logic)
# ---------------------------------------------------------------------------

def compute_height_stats(frames: list[dict], min_ll_prob: float) -> dict:
  """Compute statistics for one height's frames."""
  if not frames:
    return {}

  ll_probs = []
  re_probs = []
  ld_probs = []
  passes = 0
  lead_detected = 0

  for data in frames:
    ll_prob = data.get('lane_lines_prob', np.zeros(4))
    if not isinstance(ll_prob, np.ndarray):
      ll_prob = np.array(ll_prob)
    ll_prob = ll_prob.astype(np.float32)
    ll_probs.append(ll_prob)

    re_prob = data.get('road_edges_prob', np.zeros(2))
    if not isinstance(re_prob, np.ndarray):
      re_prob = np.array(re_prob)
    re_probs.append(re_prob.astype(np.float32))

    lead_prob = data.get('lead_prob', np.zeros(3))
    if not isinstance(lead_prob, np.ndarray):
      lead_prob = np.array(lead_prob)
    ld_probs.append(lead_prob.astype(np.float32))

    if ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob:
      passes += 1
    if np.any(lead_prob > 0.3):
      lead_detected += 1

  ll_probs_arr = np.stack(ll_probs)  # (N, 4)
  re_probs_arr = np.stack(re_probs)  # (N, 2)
  ld_probs_arr = np.stack(ld_probs)  # (N, 3)
  n = len(frames)

  return {
    'n_total': n,
    'n_pass': passes,
    'pass_rate': passes / n,
    'lead_rate': lead_detected / n,
    'll_prob_median': np.median(ll_probs_arr, axis=0),  # (4,)
    'll_prob_mean': np.mean(ll_probs_arr, axis=0),      # (4,)
    'll_prob_all': ll_probs_arr,                         # (N, 4)
    're_prob_all': re_probs_arr,                         # (N, 2)
    'ld_prob_all': ld_probs_arr,                         # (N, 3)
  }


def compute_z_consistency(frames_per_height: dict[str, list[dict]],
                           heights_info: dict[str, float],
                           ref_tag: str = 'H1') -> dict:
  """Check that z_height offsets between heights match expected delta_H."""
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

    n_pairs = min(len(ref_frames), len(frames), 200)
    z_deltas = []

    for i in range(n_pairs):
      ref_data = ref_frames[i % len(ref_frames)]
      hk_data = frames[i % len(frames)]

      ref_ll = ref_data.get('lane_lines')
      hk_ll = hk_data.get('lane_lines')
      if ref_ll is None or hk_ll is None:
        continue

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


# ---------------------------------------------------------------------------
# Confidence metric distribution helpers
# ---------------------------------------------------------------------------

def _conf_table_lines(metric_name: str, labels: list[str], key: str,
                       stats_per_height: dict,
                       total_arr: np.ndarray | None = None) -> list[str]:
  """Format one confidence metric's per-height + total statistics table."""
  header = f"  {'Group':<12}{'Component':<12}{'Min':>8}{'Max':>8}{'Mean':>8}{'Median':>8}{'Count':>8}"
  sep = f"  {'-' * 64}"
  lines = [f"\n  [{metric_name}]", header, sep]

  for tag in sorted(stats_per_height.keys()):
    s = stats_per_height[tag]
    arr = s.get(key)
    if arr is None or len(arr) == 0:
      continue
    for i, label in enumerate(labels):
      col = arr[:, i]
      lines.append(
        f"  {tag:<12}{label:<12}{np.min(col):>8.3f}{np.max(col):>8.3f}"
        f"{np.mean(col):>8.3f}{np.median(col):>8.3f}{len(col):>8}"
      )

  if total_arr is not None and len(total_arr) > 0:
    lines.append(sep)
    for i, label in enumerate(labels):
      col = total_arr[:, i]
      lines.append(
        f"  {'Total':<12}{label:<12}{np.min(col):>8.3f}{np.max(col):>8.3f}"
        f"{np.mean(col):>8.3f}{np.median(col):>8.3f}{len(col):>8}"
      )

  return lines


def _session_conf_lines(session_stats: list[tuple[str, dict]]) -> list[str]:
  """Format per-session confidence summary table (H1 only)."""
  lines = [
    "", "",
    "--- Per-session Confidence Summary (H1) ---",
    f"  {'Session':<50}{'LL-inn':>8}{'LL-out':>8}{'RE-l':>8}{'RE-r':>8}{'Ld-0':>8}",
    f"  {'(median values)':<50}{'':>8}{'':>8}{'':>8}{'':>8}{'':>8}",
    f"  {'-' * 90}",
  ]
  ranked = sorted(session_stats, key=lambda x: x[0])
  for name, s in ranked:
    if not s:
      lines.append(f"  {name:<50}{'N/A':>8}")
      continue
    ll = s.get('ll_prob_all')
    re = s.get('re_prob_all')
    ld = s.get('ld_prob_all')
    ll_inn = np.median(ll[:, 1]) if ll is not None and len(ll) else 0.0
    ll_out = np.median(ll[:, 0]) if ll is not None and len(ll) else 0.0
    re_l = np.median(re[:, 0]) if re is not None and len(re) else 0.0
    re_r = np.median(re[:, 1]) if re is not None and len(re) else 0.0
    ld_0 = np.median(ld[:, 0]) if ld is not None and len(ld) else 0.0
    lines.append(f"  {name:<50}{ll_inn:>8.3f}{ll_out:>8.3f}{re_l:>8.3f}{re_r:>8.3f}{ld_0:>8.3f}")
  return lines


def compute_total_conf(stats_per_height: dict) -> dict:
  """Concatenate confidence arrays across all heights for total-dataset stats."""
  total = {}
  for _, key, _ in CONF_METRICS:
    arrays = [s[key] for s in stats_per_height.values()
              if s.get(key) is not None and len(s[key]) > 0]
    if arrays:
      total[key] = np.concatenate(arrays, axis=0)
  return total


# ---------------------------------------------------------------------------
# Per-session stats (batch mode)
# ---------------------------------------------------------------------------

def _parse_pitch_yaw(session_name: str) -> tuple[float | None, float | None]:
  """Extract pitch and yaw from session name like 'Town04_ClearNoon_p5.0_y-3.0'."""
  m = re.search(r'_p(-?[\d.]+)_y(-?[\d.]+)', session_name)
  if m:
    return float(m.group(1)), float(m.group(2))
  return None, None


def compute_session_stats(ann_dir: Path, sample: int, min_ll_prob: float) -> dict:
  """Compute lightweight stats for one session (H1 only, fast)."""
  h1_dir = ann_dir / 'H1'
  if not h1_dir.exists():
    return {}
  frames = load_frames(h1_dir, sample)
  if not frames:
    return {}
  return compute_height_stats(frames, min_ll_prob)


# ---------------------------------------------------------------------------
# Text summary writers
# ---------------------------------------------------------------------------

def write_single_summary(stats_per_height: dict, z_check: dict, output_dir: Path,
                          session_name: str, min_ll_prob: float, heights_info: dict):
  """Write stats_summary.txt for single session mode."""
  lines = [
    "=== Multi-Height Annotation Quality Summary ===",
    f"Session: {session_name}",
    f"min_ll_prob threshold: {min_ll_prob:.2f}",
    "",
    f"{'Height':<10}{'Total':>8}{'Pass':>8}{'PassRate':>10}{'L0 Med':>10}{'R0 Med':>10}{'Lead%':>8}",
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
  z_icon = "[OK]" if z_status == 'PASS' else ("[FAIL]" if z_status == 'FAIL' else "[SKIP]")
  lines.append(f"z_height transform consistency: {z_icon} {z_status}")
  if z_status not in ('SKIP',):
    lines.append(f"  max error: {z_check.get('max_error', 0):.4f}m (threshold 0.1m)")
    for tag, r in z_check.get('per_height', {}).items():
      ok_icon = "[OK]" if r['ok'] else "[FAIL]"
      lines.append(
        f"  {tag}: expected dH={r['expected_delta']:.2f}m  "
        f"measured={r['measured_delta']:.3f}m  error={r['error']:.4f}m {ok_icon}"
      )

  # Confidence metric distribution
  lines += ["", "", "--- Confidence Metric Distribution ---"]
  total_conf = compute_total_conf(stats_per_height)
  for metric_name, key, labels in CONF_METRICS:
    lines.extend(_conf_table_lines(metric_name, labels, key, stats_per_height,
                                    total_conf.get(key)))

  path = output_dir / 'stats_summary.txt'
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  print(f"\nSummary saved: {path}")


def write_batch_summary(
  agg_stats: dict,
  session_stats: list[tuple[str, dict]],
  z_check: dict,
  output_dir: Path,
  data_root_name: str,
  min_ll_prob: float,
  heights_info: dict,
):
  """Write stats_summary.txt for batch mode."""
  lines = [
    "=== Batch Multi-Height Annotation Quality Summary ===",
    f"Data root: {data_root_name}",
    f"Sessions: {len(session_stats)}",
    f"min_ll_prob threshold: {min_ll_prob:.2f}",
    "",
    "--- Aggregate per height (all sessions) ---",
    f"{'Height':<10}{'Total':>8}{'Pass':>8}{'PassRate':>10}{'L0 Med':>10}{'R0 Med':>10}{'Lead%':>8}",
    "-" * 64,
  ]

  for tag in sorted(agg_stats.keys()):
    s = agg_stats[tag]
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

  # z_height consistency
  lines += ["", "-" * 64]
  z_status = z_check.get('status', 'SKIP')
  z_icon = "[OK]" if z_status == 'PASS' else ("[FAIL]" if z_status == 'FAIL' else "[SKIP]")
  lines.append(f"z_height transform consistency: {z_icon} {z_status}")
  if z_status not in ('SKIP',):
    lines.append(f"  max error: {z_check.get('max_error', 0):.4f}m (threshold 0.1m)")
    for tag, r in z_check.get('per_height', {}).items():
      ok_icon = "[OK]" if r['ok'] else "[FAIL]"
      lines.append(
        f"  {tag}: expected dH={r['expected_delta']:.2f}m  "
        f"measured={r['measured_delta']:.3f}m  error={r['error']:.4f}m {ok_icon}"
      )

  # Per-session table (sorted by pass rate)
  lines += [
    "", "",
    "--- Per-session stats (H1 only, sorted by pass rate) ---",
    f"{'Session':<50}{'Frames':>7}{'Pass%':>8}{'L0 Med':>8}{'R0 Med':>8}{'Lead%':>8}",
    "-" * 89,
  ]
  ranked = sorted(session_stats, key=lambda x: x[1].get('pass_rate', 0))
  for name, s in ranked:
    if not s:
      lines.append(f"{name:<50}{'N/A':>7}")
      continue
    ll_med = s.get('ll_prob_median', np.zeros(4))
    lines.append(
      f"{name:<50}{s['n_total']:>7}"
      f"{s['pass_rate'] * 100:>7.1f}%"
      f"{ll_med[1]:>8.2f}{ll_med[2]:>8.2f}"
      f"{s['lead_rate'] * 100:>7.1f}%"
    )

  # Per pitch/yaw breakdown
  py_map: dict[tuple[float, float], list[dict]] = {}
  for name, s in session_stats:
    p, y = _parse_pitch_yaw(name)
    if p is not None and s:
      py_map.setdefault((p, y), []).append(s)

  if py_map:
    lines += [
      "", "",
      "--- Pass rate by pitch x yaw ---",
      f"{'pitch':>7}{'yaw':>7}{'Sessions':>10}{'Frames':>9}{'Pass%':>8}{'Lead%':>8}",
      "-" * 49,
    ]
    for (p, y) in sorted(py_map.keys()):
      ss_list = py_map[(p, y)]
      total = sum(s['n_total'] for s in ss_list)
      n_pass = sum(s['n_pass'] for s in ss_list)
      lead = sum(s['lead_rate'] * s['n_total'] for s in ss_list)
      pr = n_pass / total * 100 if total else 0
      lr = lead / total * 100 if total else 0
      lines.append(f"{p:>+7.1f}{y:>+7.1f}{len(ss_list):>10}{total:>9}{pr:>7.1f}%{lr:>7.1f}%")

  # Confidence metric distribution (per-height aggregate + total)
  lines += ["", "", "--- Confidence Metric Distribution (per height + total) ---"]
  total_conf = compute_total_conf(agg_stats)
  for metric_name, key, labels in CONF_METRICS:
    lines.extend(_conf_table_lines(metric_name, labels, key, agg_stats,
                                    total_conf.get(key)))

  # Per-session confidence summary
  lines.extend(_session_conf_lines(session_stats))

  path = output_dir / 'stats_summary.txt'
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  print(f"\nSummary saved: {path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _get_plt():
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt
  except ImportError:
    print("[WARN] matplotlib not available, skipping plots")
    return None


def plot_lane_prob_distribution(stats_per_height: dict, output_dir: Path, min_ll_prob: float):
  plt = _get_plt()
  if plt is None:
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
  print(f"Plot saved: {path}")


def plot_filter_rates(stats_per_height: dict, heights_info: dict, output_dir: Path):
  plt = _get_plt()
  if plt is None:
    return

  tags = sorted(stats_per_height.keys())
  pass_rates = [stats_per_height[t].get('pass_rate', 0) * 100 for t in tags]
  h_labels = [f"{t}\n{heights_info.get(t, 0):.2f}m" for t in tags]
  colors = ['green' if r >= 80 else 'orange' if r >= 60 else 'red' for r in pass_rates]

  fig, ax = plt.subplots(figsize=(8, 4))
  bars = ax.bar(h_labels, pass_rates, color=colors)
  ax.axhline(80, color='green', linestyle='--', alpha=0.5, label='target 80%')
  ax.set_ylabel('Pass rate (%)')
  ax.set_title('Quality filter pass rate per height')
  ax.set_ylim(0, 105)
  ax.legend()
  for bar, r in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{r:.1f}%', ha='center', fontsize=9)
  plt.tight_layout()
  path = output_dir / 'filter_rates.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"Plot saved: {path}")


def plot_z_height_transform(frames_per_height: dict, heights_info: dict,
                             output_dir: Path):
  plt = _get_plt()
  if plt is None:
    return

  try:
    from selfdrive.modeld.constants import ModelConstants
    x_idxs = np.array(ModelConstants.X_IDXS)
  except ImportError:
    # Fallback: openpilot standard 33-point X_IDXS (0..192m, quadratic spacing)
    x_idxs = np.array([0.0, 0.1875, 0.75, 1.6875, 3.0, 4.6875, 6.75, 9.1875,
                        12.0, 15.1875, 18.75, 22.6875, 27.0, 31.6875, 36.75,
                        42.1875, 48.0, 54.1875, 60.75, 67.6875, 75.0, 82.6875,
                        90.75, 99.1875, 108.0, 117.1875, 126.75, 136.6875,
                        147.0, 157.6875, 168.75, 180.1875, 192.0])

  fig, ax = plt.subplots(figsize=(10, 5))
  colors = plt.cm.viridis(np.linspace(0, 1, len(frames_per_height)))

  for (tag, frames), color in zip(sorted(frames_per_height.items()), colors):
    if not frames:
      continue
    z_list = []
    for data in frames:
      ll = data.get('lane_lines')
      lp = data.get('lane_lines_prob', np.zeros(4))
      if ll is None:
        continue
      valid_lanes = [i for i in [1, 2] if lp[i] > 0.5]
      if not valid_lanes:
        continue
      z_vals = np.nanmean(ll[valid_lanes, :, 2], axis=0)  # (33,)
      z_list.append(z_vals)
    if z_list:
      z_mean = np.nanmean(z_list, axis=0)
      h_m = heights_info.get(tag, 0.0)
      ax.plot(x_idxs, z_mean, label=f"{tag} {h_m:.2f}m", color=color)

  ax.set_xlabel('Forward distance X (m)')
  ax.set_ylabel('z_height mean (m)')
  ax.set_title('Lane line z_height vs distance per height')
  ax.legend()
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  path = output_dir / 'z_height_transform.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"Plot saved: {path}")


def plot_lead_detection(stats_per_height: dict, heights_info: dict, output_dir: Path):
  plt = _get_plt()
  if plt is None:
    return

  tags = sorted(stats_per_height.keys())
  lead_rates = [stats_per_height[t].get('lead_rate', 0) * 100 for t in tags]
  h_labels = [f"{t}\n{heights_info.get(t, 0):.2f}m" for t in tags]

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(h_labels, lead_rates, 'o-', color='steelblue', linewidth=2, markersize=8)
  ax.axhline(30, color='orange', linestyle='--', alpha=0.5, label='ref 30%')
  ax.set_ylabel('Lead detection rate (%)')
  ax.set_title('Lead vehicle detection rate per height (lead_prob > 0.3)')
  ax.set_ylim(0, 105)
  ax.legend()
  ax.grid(True, alpha=0.3)
  for i, (_, r) in enumerate(zip(h_labels, lead_rates)):
    ax.text(i, r + 2, f'{r:.1f}%', ha='center', fontsize=9)
  plt.tight_layout()
  path = output_dir / 'lead_detection_rate.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"Plot saved: {path}")


def plot_session_pass_rates(session_stats: list[tuple[str, dict]], output_dir: Path):
  """Bar chart of per-session H1 pass rates (sorted ascending)."""
  plt = _get_plt()
  if plt is None:
    return

  valid = [(n, s) for n, s in session_stats if s and s.get('n_total', 0) > 0]
  if not valid:
    return
  valid.sort(key=lambda x: x[1]['pass_rate'])
  names = [n for n, _ in valid]
  rates = [s['pass_rate'] * 100 for _, s in valid]
  colors = ['green' if r >= 80 else 'orange' if r >= 60 else 'red' for r in rates]

  fig_h = max(4, len(names) * 0.22)
  fig, ax = plt.subplots(figsize=(10, fig_h))
  y_pos = np.arange(len(names))
  ax.barh(y_pos, rates, color=colors, height=0.7)
  ax.set_yticks(y_pos)
  ax.set_yticklabels(names, fontsize=6)
  ax.set_xlabel('H1 Pass rate (%)')
  ax.set_title(f'Per-session H1 pass rate (n={len(names)} sessions)')
  ax.set_xlim(0, 105)
  ax.axvline(80, color='green', linestyle='--', alpha=0.4)
  ax.invert_yaxis()
  plt.tight_layout()
  path = output_dir / 'session_pass_rates.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"Plot saved: {path}")


def plot_pitch_yaw_heatmap(session_stats: list[tuple[str, dict]], output_dir: Path):
  """Heatmap of pass rate by pitch x yaw."""
  plt = _get_plt()
  if plt is None:
    return

  py_rates: dict[tuple[float, float], float] = {}
  for name, s in session_stats:
    p, y = _parse_pitch_yaw(name)
    if p is None or not s:
      continue
    py_rates.setdefault((p, y), []).append(s['pass_rate'])

  if not py_rates:
    return

  # Average pass rate per pitch/yaw combo
  py_avg = {k: np.mean(v) * 100 for k, v in py_rates.items()}

  pitches = sorted(set(p for p, _ in py_avg))
  yaws = sorted(set(y for _, y in py_avg))

  grid = np.full((len(pitches), len(yaws)), np.nan)
  for (p, y), rate in py_avg.items():
    grid[pitches.index(p), yaws.index(y)] = rate

  fig, ax = plt.subplots(figsize=(max(5, len(yaws) * 0.9), max(4, len(pitches) * 0.7)))
  im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto',
                 origin='lower')

  ax.set_xticks(range(len(yaws)))
  ax.set_xticklabels([f'{y:+.1f}' for y in yaws], fontsize=8)
  ax.set_yticks(range(len(pitches)))
  ax.set_yticklabels([f'{p:+.1f}' for p in pitches], fontsize=8)
  ax.set_xlabel('yaw (deg)')
  ax.set_ylabel('pitch (deg)')
  ax.set_title('H1 Pass rate (%) by pitch x yaw')

  # Annotate cells
  for i in range(len(pitches)):
    for j in range(len(yaws)):
      v = grid[i, j]
      if not np.isnan(v):
        color = 'white' if v < 50 else 'black'
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8, color=color)

  fig.colorbar(im, ax=ax, label='Pass rate (%)')
  plt.tight_layout()
  path = output_dir / 'pitch_yaw_heatmap.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"Plot saved: {path}")


def plot_confidence_distributions(stats_per_height: dict, total_conf: dict, output_dir: Path):
  """Plot confidence metric histograms per height and total dataset."""
  plt = _get_plt()
  if plt is None:
    return

  # Per-height distribution (one figure per metric type)
  for metric_name, key, labels in CONF_METRICS:
    tags = sorted(t for t in stats_per_height if stats_per_height[t].get(key) is not None)
    if not tags:
      continue

    n_rows = len(tags)
    n_cols = len(labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), squeeze=False)

    for i, tag in enumerate(tags):
      arr = stats_per_height[tag][key]
      for j, label in enumerate(labels):
        ax = axes[i][j]
        col = arr[:, j]
        ax.hist(col, bins=40, alpha=0.8, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.set_title(f'{tag} / {label}', fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        if i == n_rows - 1:
          ax.set_xlabel('Probability')
        if j == 0:
          ax.set_ylabel('Count')
        med = np.median(col)
        ax.axvline(med, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(0.97, 0.93, f'med={med:.3f}\nmean={np.mean(col):.3f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'{metric_name} Distribution per Height', fontsize=12)
    plt.tight_layout()
    path = output_dir / f'conf_dist_{metric_name}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot saved: {path}")

  # Total dataset distribution (all metrics in one figure)
  if not total_conf:
    return

  active = [(mn, key, labels) for mn, key, labels in CONF_METRICS if key in total_conf]
  if not active:
    return

  max_cols = max(len(labels) for _, _, labels in active)
  n_metrics = len(active)
  fig, axes = plt.subplots(n_metrics, max_cols,
                            figsize=(max_cols * 3.5, n_metrics * 2.5), squeeze=False)

  for row, (metric_name, key, labels) in enumerate(active):
    arr = total_conf[key]
    for j, label in enumerate(labels):
      ax = axes[row][j]
      col = arr[:, j]
      ax.hist(col, bins=50, alpha=0.8, color='darkorange', edgecolor='white', linewidth=0.5)
      ax.set_title(f'{metric_name} / {label}', fontsize=9)
      ax.set_xlim(-0.05, 1.05)
      ax.set_xlabel('Probability')
      if j == 0:
        ax.set_ylabel('Count')
      med = np.median(col)
      ax.axvline(med, color='red', linestyle='--', alpha=0.7, linewidth=1)
      ax.text(0.97, 0.93, f'N={len(col)}\nmed={med:.3f}\nmean={np.mean(col):.3f}',
              transform=ax.transAxes, fontsize=7, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    for j in range(len(labels), max_cols):
      axes[row][j].set_visible(False)

  fig.suptitle('Total Dataset Confidence Distribution', fontsize=12)
  plt.tight_layout()
  path = output_dir / 'conf_dist_total.png'
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f"Plot saved: {path}")


# ---------------------------------------------------------------------------
# Main: single-session mode
# ---------------------------------------------------------------------------

def run_single(annotated_dir: Path, output_dir: Path, min_ll_prob: float, sample: int):
  """Original single-session analysis."""
  clip_info_path = annotated_dir / 'clip_info.json'
  if not clip_info_path.exists():
    print(f"ERROR: clip_info.json not found in {annotated_dir}", file=sys.stderr)
    sys.exit(1)
  with open(clip_info_path) as f:
    clip_info = json.load(f)
  heights_info: dict[str, float] = clip_info.get('heights', {})

  all_tags = sorted([d.name for d in annotated_dir.iterdir()
                     if d.is_dir() and d.name.startswith('H') and d.name[1:].isdigit()])
  if not all_tags:
    print("ERROR: no height directories found", file=sys.stderr)
    sys.exit(1)

  print(f"Mode: single session")
  print(f"Session: {annotated_dir.name}")
  print(f"Heights: {all_tags}")
  print(f"Sample: {sample if sample > 0 else 'all'} frames per height")
  print(f"Output: {output_dir}")

  frames_per_height: dict[str, list[dict]] = {}
  stats_per_height: dict[str, dict] = {}

  for tag in all_tags:
    h_dir = annotated_dir / tag
    if not h_dir.exists():
      continue
    print(f"  Loading {tag}...", end='', flush=True)
    frames = load_frames(h_dir, sample)
    frames_per_height[tag] = frames
    if frames:
      stats_per_height[tag] = compute_height_stats(frames, min_ll_prob)
      print(f" {len(frames)} frames, pass={stats_per_height[tag]['pass_rate'] * 100:.1f}%")
    else:
      print(" no frames")

  print("\nChecking z_height transform consistency...")
  z_check = compute_z_consistency(frames_per_height, heights_info)

  write_single_summary(stats_per_height, z_check, output_dir, annotated_dir.name,
                        min_ll_prob, heights_info)

  print("\nGenerating plots...")
  plot_lane_prob_distribution(stats_per_height, output_dir, min_ll_prob)
  plot_filter_rates(stats_per_height, heights_info, output_dir)
  plot_z_height_transform(frames_per_height, heights_info, output_dir)
  plot_lead_detection(stats_per_height, heights_info, output_dir)
  total_conf = compute_total_conf(stats_per_height)
  plot_confidence_distributions(stats_per_height, total_conf, output_dir)


# ---------------------------------------------------------------------------
# Main: batch mode
# ---------------------------------------------------------------------------

def run_batch(data_root: Path, output_dir: Path, min_ll_prob: float, sample: int):
  """Batch analysis across all sessions under data_root."""
  ann_dirs = discover_annotated_dirs(data_root)
  if not ann_dirs:
    print(f"ERROR: no annotated sessions found (need <session>/annotations/clip_info.json)",
          file=sys.stderr)
    sys.exit(1)

  # Read heights_info from first session
  with open(ann_dirs[0][0] / 'clip_info.json') as f:
    clip_info = json.load(f)
  heights_info: dict[str, float] = clip_info.get('heights', {})
  all_tags = sorted(heights_info.keys())

  print(f"Mode: batch")
  print(f"Data root: {data_root.name}")
  print(f"Sessions: {len(ann_dirs)}")
  print(f"Heights: {all_tags}")
  print(f"Sample: {sample if sample > 0 else 'all'} frames per height per session")
  print(f"Output: {output_dir}")

  # 1. Per-session H1 stats (lightweight scan)
  print(f"\n--- Scanning per-session stats (H1) ---")
  session_stats: list[tuple[str, dict]] = []
  for ann_dir, sess_name in ann_dirs:
    s = compute_session_stats(ann_dir, sample, min_ll_prob)
    session_stats.append((sess_name, s))
    n = s.get('n_total', 0)
    pr = s.get('pass_rate', 0) * 100
    if n > 0:
      print(f"  {sess_name}: {n} frames, pass={pr:.1f}%")
    else:
      print(f"  {sess_name}: no frames")

  # 2. Aggregate per-height stats (load all heights from all sessions)
  print(f"\n--- Loading aggregate per-height stats ---")
  agg_frames: dict[str, list[dict]] = {tag: [] for tag in all_tags}
  agg_stats: dict[str, dict] = {}

  for tag in all_tags:
    print(f"  Loading {tag}...", end='', flush=True)
    for ann_dir, _ in ann_dirs:
      h_dir = ann_dir / tag
      if h_dir.exists():
        frames = load_frames(h_dir, sample)
        agg_frames[tag].extend(frames)
    n = len(agg_frames[tag])
    if n:
      agg_stats[tag] = compute_height_stats(agg_frames[tag], min_ll_prob)
      print(f" {n} frames, pass={agg_stats[tag]['pass_rate'] * 100:.1f}%")
    else:
      print(" no frames")

  # 3. z_height consistency (aggregate)
  print("\nChecking z_height transform consistency (aggregate)...")
  z_check = compute_z_consistency(agg_frames, heights_info)

  # 4. Write summary
  write_batch_summary(agg_stats, session_stats, z_check, output_dir,
                      data_root.name, min_ll_prob, heights_info)

  # 5. Plots
  print("\nGenerating plots...")
  plot_lane_prob_distribution(agg_stats, output_dir, min_ll_prob)
  plot_filter_rates(agg_stats, heights_info, output_dir)
  plot_z_height_transform(agg_frames, heights_info, output_dir)
  plot_lead_detection(agg_stats, heights_info, output_dir)
  plot_session_pass_rates(session_stats, output_dir)
  plot_pitch_yaw_heatmap(session_stats, output_dir)
  total_conf = compute_total_conf(agg_stats)
  plot_confidence_distributions(agg_stats, total_conf, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(
    description='Multi-height annotation quality statistics (single session or batch)')
  parser.add_argument('input_dir',
                      help='Annotated session dir (single) or data root with multiple sessions (batch)')
  parser.add_argument('--output', default=None,
                      help='Output directory for charts/report (default: <input_dir>/stats/)')
  parser.add_argument('--min-ll-prob', type=float, default=0.3, help='Quality filter threshold')
  parser.add_argument('--sample', type=int, default=0,
                      help='Max frames to sample per height per session (0=all)')
  args = parser.parse_args()

  input_dir = Path(args.input_dir).resolve()
  if not input_dir.exists():
    print(f"ERROR: not found: {input_dir}", file=sys.stderr)
    sys.exit(1)

  output_dir = Path(args.output) if args.output else input_dir / 'stats'
  output_dir.mkdir(parents=True, exist_ok=True)

  mode = detect_mode(input_dir)

  if mode == 'single':
    run_single(input_dir, output_dir, args.min_ll_prob, args.sample)
  else:
    run_batch(input_dir, output_dir, args.min_ll_prob, args.sample)

  print(f"\nAll outputs in: {output_dir}")


if __name__ == '__main__':
  main()
