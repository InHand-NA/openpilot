#!/usr/bin/env python3
"""Training monitor: real-time display of training metrics from CSV log.

Usage:
  # One-shot: print current status
  python tools/dashcam/train/monitor.py checkpoints/training_log.csv

  # Live mode: refresh every 10 seconds (default)
  python tools/dashcam/train/monitor.py checkpoints/training_log.csv --live

  # Live mode with custom interval
  python tools/dashcam/train/monitor.py checkpoints/training_log.csv --live --interval 30

  # Plot training curves (saves PNG)
  python tools/dashcam/train/monitor.py checkpoints/training_log.csv --plot
"""

import argparse
import csv
import os
import sys
import time


def read_csv(path: str) -> list[dict[str, str]]:
  if not os.path.exists(path):
    return []
  with open(path) as f:
    return list(csv.DictReader(f))


def print_summary(rows: list[dict[str, str]]):
  if not rows:
    print("No data yet.")
    return

  n = len(rows)
  last = rows[-1]
  epoch = int(last['epoch'])

  # Find best val_loss
  best_row = min(rows, key=lambda r: float(r['val_loss']))
  best_epoch = int(best_row['epoch'])
  best_val = float(best_row['val_loss'])

  # Current metrics
  cur_val = float(last['val_loss'])
  cur_train = float(last['train_loss'])
  cur_lr = float(last['lr'])

  # Time stats
  times = [float(r['time_s']) for r in rows]
  avg_time = sum(times) / len(times)
  total_time = sum(times)

  # Trend: last 5 val_loss
  recent = rows[-min(5, n) :]
  recent_vals = [float(r['val_loss']) for r in recent]
  if len(recent_vals) >= 2:
    trend = recent_vals[-1] - recent_vals[0]
    trend_str = f"{'↓' if trend < 0 else '↑'}{abs(trend):.4f}" if abs(trend) > 0.001 else "→ stable"
  else:
    trend_str = "-"

  epochs_since_best = epoch - best_epoch

  print("=" * 70)
  print(f"  Epoch: {epoch}    LR: {cur_lr:.6f}    Time/epoch: {avg_time:.0f}s")
  print(f"  Train loss: {cur_train:.4f}    Val loss: {cur_val:.4f}")
  print(f"  Best val:   {best_val:.4f} (epoch {best_epoch}, {epochs_since_best} epochs ago)")
  print(f"  Trend (last 5): {trend_str}")
  print(f"  Total time: {total_time / 3600:.1f}h")
  print("-" * 70)

  # Sub-loss table
  sub_keys = ['lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob', 'pose', 'road_transform']
  print(f"  {'Loss':<20s} {'Current':>10s} {'Best':>10s} {'Epoch1':>10s}")
  print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}")
  for k in sub_keys:
    if k in last:
      cur = float(last[k])
      best_v = min(float(r[k]) for r in rows)
      first = float(rows[0][k])
      print(f"  {k:<20s} {cur:>10.4f} {best_v:>10.4f} {first:>10.4f}")
  print("=" * 70)


def plot_curves(rows: list[dict[str, str]], output_path: str):
  try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    return

  epochs = [int(r['epoch']) for r in rows]
  val_loss = [float(r['val_loss']) for r in rows]
  train_loss = [float(r['train_loss']) for r in rows]

  sub_keys = ['lane_lines', 'road_edges', 'lead', 'pose', 'road_transform']
  prob_keys = ['lane_lines_prob', 'lead_prob']

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))

  # 1. Total loss
  ax = axes[0, 0]
  # Clip train_loss outliers for readability
  train_clipped = [min(t, max(val_loss) * 3) for t in train_loss]
  ax.plot(epochs, train_clipped, label='train', alpha=0.7)
  ax.plot(epochs, val_loss, label='val', linewidth=2)
  best_idx = val_loss.index(min(val_loss))
  ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5, label=f'best @ {epochs[best_idx]}')
  ax.set_title('Total Loss')
  ax.set_xlabel('Epoch')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # 2. MDN sub-losses
  ax = axes[0, 1]
  for k in sub_keys:
    if k in rows[0]:
      vals = [float(r[k]) for r in rows]
      ax.plot(epochs, vals, label=k)
  ax.set_title('MDN Sub-losses (val)')
  ax.set_xlabel('Epoch')
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  # 3. Probability losses
  ax = axes[1, 0]
  for k in prob_keys:
    if k in rows[0]:
      vals = [float(r[k]) for r in rows]
      ax.plot(epochs, vals, label=k)
  ax.set_title('BCE Prob Losses (val)')
  ax.set_xlabel('Epoch')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # 4. Learning rate
  ax = axes[1, 1]
  lrs = [float(r['lr']) for r in rows]
  ax.plot(epochs, lrs, color='orange')
  ax.set_title('Learning Rate')
  ax.set_xlabel('Epoch')
  ax.grid(True, alpha=0.3)

  plt.suptitle('Training Monitor', fontsize=14)
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()
  print(f"Plot saved: {output_path}")


def main():
  parser = argparse.ArgumentParser(description='Monitor training progress')
  parser.add_argument('csv_path', help='Path to training_log.csv')
  parser.add_argument('--live', action='store_true', help='Live mode: refresh periodically')
  parser.add_argument('--interval', type=int, default=10, help='Refresh interval in seconds (default: 10)')
  parser.add_argument('--plot', action='store_true', help='Save training curves as PNG')
  args = parser.parse_args()

  if args.plot:
    rows = read_csv(args.csv_path)
    if not rows:
      print(f"No data in {args.csv_path}")
      sys.exit(1)
    out = args.csv_path.replace('.csv', '_curves.png')
    plot_curves(rows, out)
    return

  if args.live:
    print(f"Live monitoring: {args.csv_path} (Ctrl+C to stop)\n")
    try:
      while True:
        os.system('clear' if os.name != 'nt' else 'cls')
        rows = read_csv(args.csv_path)
        print_summary(rows)
        print(f"\n  Refreshing every {args.interval}s... (Ctrl+C to stop)")
        time.sleep(args.interval)
    except KeyboardInterrupt:
      print("\nStopped.")
  else:
    rows = read_csv(args.csv_path)
    print_summary(rows)


if __name__ == "__main__":
  main()
