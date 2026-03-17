#!/usr/bin/env python3
"""Full-phase multi-height data collection with auto-retry and resume.

Wraps collect_multi_height.py's full-phase session list, adding:
  - Automatic skip of completed sessions (based on existing frame count)
  - Automatic resume of partially-collected sessions (collect_session starts
    from len(existing) frames, so partial work is never lost)
  - Per-session retry loop for Carla server crashes, with configurable
    max retries and wait delay before each retry
  - Graceful Ctrl+C: finishes the current session's cleanup, then stops
  - Progress log written to <output-base>/collection_progress.json for
    human inspection (does NOT gate resume logic — file counts do)
  - Random ego spawn point (--random-spawn) for training data diversity

Typical usage:
  # Start (or resume) full collection
  python tools/dashcam/run_full_collection.py --no-display

  # Custom output dir and NPC count
  python tools/dashcam/run_full_collection.py \\
      --output-base data/multi_height-0316 \\
      --num-npc 40 --no-display

  # Limit frames per session (for quick smoke-test)
  python tools/dashcam/run_full_collection.py --max-frames 500

  # Override Carla connection
  python tools/dashcam/run_full_collection.py --host 127.0.0.1 --port 2000

Resume after crash:
  Simply re-run the same command. Completed sessions are skipped based on
  frame file count; partially-collected sessions continue from where they
  left off.  The collection_progress.json is informational only.
"""

import argparse
import json
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

MIN_DISK_FREE_GB = 20


# ── Session list mirrors collect_multi_height.py PHASE_CONFIGS['full'] ────────

HEIGHT_DEFS = {
  'H1': 1.22,
  'H2': 1.30,
  'H3': 1.50,
  'H4': 2.00,
  'H5': 2.50,
  'H6': 3.00,
}

FULL_HEIGHTS = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
DEFAULT_MAX_FRAMES = 200

SCENES = [

#  ('Town03', 'ClearNoon'),
#  ('Town03', 'WetSunset'),
#  ('Town05', 'ClearSunset'),
#  ('Town05', 'CloudyNoon'),

#  ('Town05', 'ClearNoon'),
#  ('Town05', 'CloudySunset'),
#  ('Town05', 'WetNoon'),
#  ('Town05', 'WetSunset'),

  ('Town04', 'ClearNoon'),
  ('Town04', 'ClearSunset'),
  ('Town04', 'CloudyNoon'),
  ('Town04', 'WetNoon'),
]

# Formal training pose matrix from §4.3.3 of height_extension_design.md.
# Format: (pitch_deg, yaw_deg, weight)
#   weight=1  regular sampling point (○)
#   weight=2  nominal center (●), collects 2× max_frames
#
# pitch \ yaw  | -3°  | -1.5° | 0°  | +1.5° | +3°  | slots
# -------------|------|-------|-----|-------|------|------
# -1.5° (仰)   |  ○   |       |  ○  |       |  ○   |  3
#  0°   (水平) |  ○   |       |  ○  |       |  ○   |  3
#  1.5°        |      |  ○    |  ○  |  ○    |      |  3
#  3°          |  ○   |  ○    |  ○  |  ○    |  ○   |  5
#  4°          |  ○   |  ○    |  ○  |  ○    |  ○   |  5
#  5° (名义)   |  ○   |  ○    |  ●  |  ○    |  ○   |  5+1
#  6°          |  ○   |  ○    |  ○  |  ○    |  ○   |  5
#  7°          |      |  ○    |  ○  |  ○    |      |  3
#                                              total: 32+1=33 (≈34 per doc)
PITCH_YAW_PAIRS: list[tuple[float, float, int]] = [
  # pitch = -1.5° (相机略仰，稀疏外侧 yaw)
  (-1.5, -3.0, 1), (-1.5,  0.0, 1), (-1.5,  3.0, 1),
  # pitch = 0° (水平，稀疏外侧 yaw)
  ( 0.0, -3.0, 1), ( 0.0,  0.0, 1), ( 0.0,  3.0, 1),
  # pitch = 1.5° (内侧 3-yaw)
  ( 1.5, -1.5, 1), ( 1.5,  0.0, 1), ( 1.5,  1.5, 1),
  # pitch = 3° (全 5-yaw)
  ( 3.0, -3.0, 1), ( 3.0, -1.5, 1), ( 3.0,  0.0, 1), ( 3.0,  1.5, 1), ( 3.0,  3.0, 1),
  # pitch = 4° (全 5-yaw)
  ( 4.0, -3.0, 1), ( 4.0, -1.5, 1), ( 4.0,  0.0, 1), ( 4.0,  1.5, 1), ( 4.0,  3.0, 1),
  # pitch = 5° 名义值（全 5-yaw；中心 (5°,0°) 权重 2×）
  ( 5.0, -3.0, 1), ( 5.0, -1.5, 1), ( 5.0,  0.0, 2), ( 5.0,  1.5, 1), ( 5.0,  3.0, 1),
  # pitch = 6° (全 5-yaw)
  ( 6.0, -3.0, 1), ( 6.0, -1.5, 1), ( 6.0,  0.0, 1), ( 6.0,  1.5, 1), ( 6.0,  3.0, 1),
  # pitch = 7° (内侧 3-yaw)
  ( 7.0, -1.5, 1), ( 7.0,  0.0, 1), ( 7.0,  1.5, 1),
]


def _make_session_tag(map_name: str, weather: str, pitch: float, yaw: float) -> str:
  return f"{map_name}_{weather}_p{pitch:.1f}_y{yaw:.1f}"


def _build_session_list(max_frames: int) -> list[dict]:
  sessions = []
  for map_name, weather in SCENES:
    for pitch, yaw, weight in PITCH_YAW_PAIRS:
      sessions.append({
        'tag': _make_session_tag(map_name, weather, pitch, yaw),
        'heights': FULL_HEIGHTS,
        'map': map_name,
        'weather': weather,
        'pitch': pitch,
        'yaw': yaw,
        'max_frames': max_frames * weight,
      })
  return sessions


# ── Progress tracking (JSON, informational only) ───────────────────────────────

def _load_progress(path: Path) -> dict:
  if path.exists():
    try:
      with open(path) as f:
        return json.load(f)
    except Exception:
      pass
  return {}


def _save_progress(path: Path, progress: dict) -> None:
  try:
    with open(path, 'w') as f:
      json.dump(progress, f, indent=2)
  except Exception as e:
    print(f"[WARN] Failed to save progress: {e}")


def _count_existing_frames(session_dir: Path, h0: str) -> int:
  """Count NPZ files in the first-height subdir — used for completion check."""
  h0_dir = session_dir / h0
  if not h0_dir.exists():
    return 0
  return len(list(h0_dir.glob('*.npz')))


# ── Main runner ────────────────────────────────────────────────────────────────

def run_full_collection(
  output_base: Path,
  max_frames: int,
  num_npc: int,
  host: str,
  port: int,
  show_display: bool,
  max_retries: int,
  retry_delay: int,
  start_from: int,
  speed_range: tuple[float, float],
  speed_interval: tuple[float, float],
  save_every: int = 4,
) -> None:
  from openpilot.tools.dashcam.collect_multi_height import collect_session
  from openpilot.tools.dashcam.carla_multi_height_world import CameraSlot

  sessions = _build_session_list(max_frames)
  n_total = len(sessions)
  progress_path = output_base / 'collection_progress.json'
  progress = _load_progress(progress_path)

  # Graceful Ctrl+C: set flag after current session finishes
  interrupted = False
  original_sigint = signal.getsignal(signal.SIGINT)

  def _on_sigint(sig, frame):
    nonlocal interrupted
    if not interrupted:
      print("\n[INTERRUPT] Ctrl+C received — will stop after current session.")
      print("            Press Ctrl+C again to force-quit immediately.")
      interrupted = True
    else:
      print("\n[FORCE QUIT]")
      signal.signal(signal.SIGINT, original_sigint)
      sys.exit(1)

  signal.signal(signal.SIGINT, _on_sigint)

  # ── Session loop ─────────────────────────────────────────────────────────
  t_run_start = time.monotonic()
  completed_this_run = 0
  skipped = 0
  failed = 0

  print(f"\n{'='*70}")
  print(f"  Full collection: {n_total} sessions × {max_frames} frames × {len(FULL_HEIGHTS)} heights")
  print(f"  Output: {output_base}")
  print(f"  Carla: {host}:{port}  NPC: {num_npc}  retries: {max_retries}  delay: {retry_delay}s")
  print(f"  Speed: {speed_range[0]:.0f}~{speed_range[1]:.0f} km/h  "
        f"change every {speed_interval[0]:.0f}~{speed_interval[1]:.0f}s")
  print(f"  Random ego spawn: YES")
  if save_every > 1:
    print(f"  save_every={save_every}: saving 1/{save_every} ticks (~{20 / save_every:.1f} FPS)")
  if start_from > 0:
    print(f"  Skipping first {start_from} sessions (--start-from)")
  print(f"{'='*70}\n")

  for idx, s in enumerate(sessions):
    if idx < start_from:
      continue

    tag = s['tag']
    session_dir = output_base / tag
    h0 = s['heights'][0]  # H1

    # ── Skip check ─────────────────────────────────────────────────────────
    existing = _count_existing_frames(session_dir, h0)
    if existing >= s['max_frames']:
      print(f"[{idx+1:02d}/{n_total}] SKIP  {tag}  ({existing}/{s['max_frames']} frames)")
      progress.setdefault(tag, {}).update({'status': 'completed', 'frames': existing})
      skipped += 1
      continue

    if interrupted:
      print(f"\n[STOPPED] Interrupted before session {idx+1}: {tag}")
      break

    # ── Disk space check ──────────────────────────────────────────────────
    disk = shutil.disk_usage(output_base)
    free_gb = disk.free / (1024 ** 3)
    if free_gb < MIN_DISK_FREE_GB:
      print(f"\n[DISK FULL] 磁盘剩余空间不足: {free_gb:.1f} GB < {MIN_DISK_FREE_GB} GB")
      print(f"           路径: {output_base}")
      print(f"           停止数据采集。请清理磁盘后重新运行。")
      progress.setdefault(tag, {}).update({
        'status': 'disk_full',
        'disk_free_gb': round(free_gb, 1),
      })
      _save_progress(progress_path, progress)
      break

    # ── Retry loop ─────────────────────────────────────────────────────────
    session_ok = False
    last_error = None

    for attempt in range(1, max_retries + 1):
      if interrupted:
        break

      existing_now = _count_existing_frames(session_dir, h0)
      resume_note = f" (resuming from {existing_now})" if existing_now > 0 else ""
      print(f"\n[{idx+1:02d}/{n_total}] {'RETRY ' + str(attempt) if attempt > 1 else 'START'} "
            f"{tag}{resume_note}")
      if attempt > 1:
        print(f"         Previous error: {last_error}")

      progress.setdefault(tag, {}).update({
        'status': 'in_progress',
        'attempt': attempt,
        'started_at': datetime.now().isoformat(timespec='seconds'),
      })
      _save_progress(progress_path, progress)

      t0 = time.monotonic()
      try:
        slots = [CameraSlot(tag=h, height=HEIGHT_DEFS[h]) for h in s['heights']]
        collect_session(
          output_base=output_base,
          heights=s['heights'],
          map_name=s['map'],
          weather=s['weather'],
          pitch_deg=s['pitch'],
          yaw_deg=s['yaw'],
          max_frames=s['max_frames'],
          num_npc=num_npc,
          spawn_point=16,      # ignored when random_spawn=True
          random_spawn=True,   # always random for training diversity
          host=host,
          port=port,
          show_display=show_display,
          speed_range=speed_range,
          speed_interval=speed_interval,
          save_every=save_every,
        )
        elapsed = time.monotonic() - t0
        frames_now = _count_existing_frames(session_dir, h0)
        print(f"  [OK] {tag}: {frames_now} frames in {elapsed:.0f}s")
        progress[tag].update({
          'status': 'completed',
          'frames': frames_now,
          'elapsed_s': round(elapsed),
          'finished_at': datetime.now().isoformat(timespec='seconds'),
        })
        _save_progress(progress_path, progress)
        session_ok = True
        completed_this_run += 1
        break

      except KeyboardInterrupt:
        # SIGINT was already caught above; this handles rare double-raise
        interrupted = True
        break

      except Exception as e:
        elapsed = time.monotonic() - t0
        last_error = str(e)
        frames_now = _count_existing_frames(session_dir, h0)
        print(f"  [ERROR] attempt {attempt}/{max_retries}: {e}")
        print(f"          frames saved so far: {frames_now}  elapsed: {elapsed:.0f}s")
        progress[tag].update({
          'last_error': last_error,
          'frames': frames_now,
        })
        _save_progress(progress_path, progress)

        if attempt < max_retries and not interrupted:
          print(f"  Waiting {retry_delay}s before retry "
                f"(restart Carla server now if it crashed)...")
          # Interruptible sleep: check flag every second
          for _ in range(retry_delay):
            if interrupted:
              break
            time.sleep(1)

    if not session_ok and not interrupted:
      print(f"  [FAIL] {tag}: all {max_retries} attempts failed — skipping")
      progress.setdefault(tag, {}).update({
        'status': 'failed',
        'frames': _count_existing_frames(session_dir, h0),
        'last_error': last_error,
      })
      _save_progress(progress_path, progress)
      failed += 1

  # ── Summary ───────────────────────────────────────────────────────────────
  total_elapsed = time.monotonic() - t_run_start
  total_frames = sum(
    _count_existing_frames(output_base / s['tag'], s['heights'][0])
    for s in sessions
  )

  print(f"\n{'='*70}")
  print(f"  Collection run summary")
  print(f"  Total sessions:      {n_total}")
  print(f"  Skipped (complete):  {skipped}")
  print(f"  Completed this run:  {completed_this_run}")
  print(f"  Failed:              {failed}")
  if interrupted:
    print(f"  Interrupted:         YES (resume by re-running the same command)")
  print(f"  Total frames on disk (H1): {total_frames}")
  print(f"  Run duration:        {timedelta(seconds=int(total_elapsed))}")
  print(f"  Progress log:        {progress_path}")
  print(f"{'='*70}\n")

  _save_progress(progress_path, progress)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
  parser = argparse.ArgumentParser(
    description='Full-phase multi-height data collection with auto-retry and resume',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Sessions: 6 scenes × 5 pitch/yaw pairs = 30 sessions total
Heights per session: H1(1.22m) H2(1.30m) H3(1.50m) H4(2.00m) H5(2.50m) H6(3.00m)

Resume: re-run the same command — completed sessions are skipped automatically,
partial sessions continue from the last saved frame.

Carla crash recovery:
  On exception, waits --retry-delay seconds then retries (up to --max-retries).
  Use that window to restart the Carla Docker container:
    DETACH=1 bash tools/dashcam/start_carla.sh
    sleep 10  # wait for Carla to be ready
  Then let the retry proceed (or Ctrl+C and re-run).
""")

  parser.add_argument('--output-base', default='data/multi_height',
                      help='Base output directory (default: data/multi_height)')
  parser.add_argument('--max-frames', type=int, default=DEFAULT_MAX_FRAMES,
                      help=f'Frames per session (default: {DEFAULT_MAX_FRAMES})')
  parser.add_argument('--num-npc', type=int, default=40,
                      help='Number of NPC vehicles (default: 40)')
  parser.add_argument('--host', default='127.0.0.1',
                      help='Carla server host (default: 127.0.0.1)')
  parser.add_argument('--port', type=int, default=2000,
                      help='Carla server port (default: 2000)')
  parser.add_argument('--no-display', action='store_true',
                      help='Disable OpenCV preview window')
  parser.add_argument('--max-retries', type=int, default=3,
                      help='Max retry attempts per session on Carla error (default: 3)')
  parser.add_argument('--retry-delay', type=int, default=30,
                      help='Seconds to wait between retries (default: 30)')
  parser.add_argument('--start-from', type=int, default=0, metavar='N',
                      help='Skip the first N sessions (0-indexed, for manual skip)')
  parser.add_argument('--speed-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                      default=[40.0, 100.0],
                      help='Ego target speed range in km/h (default: 40 100)')
  parser.add_argument('--speed-interval', nargs=2, type=float, metavar=('MIN', 'MAX'),
                      default=[8.0, 20.0],
                      help='Interval in seconds between random speed changes (default: 8 20)')
  parser.add_argument('--save-every', type=int, default=4, metavar='N',
                      help='Save 1 frame out of every N ticks (default: 4). '
                           '--save-every 4 speeds collection ~4× and saves ~4× less disk.')
  parser.add_argument('--list', action='store_true',
                      help='Print all sessions and exit (no collection)')

  args = parser.parse_args()
  output_base = Path(args.output_base)

  # ── --list mode ──────────────────────────────────────────────────────────
  if args.list:
    sessions = _build_session_list(args.max_frames)
    n_unique = len(PITCH_YAW_PAIRS)
    n_weighted = sum(w for _, _, w in PITCH_YAW_PAIRS)
    print(f"Full collection: {len(sessions)} sessions  "
          f"({len(SCENES)} scenes × {n_unique} pose pairs, "
          f"{n_weighted} effective slots per scene)")
    print(f"  Base frames: {args.max_frames}  "
          f"Nominal center (5°,0°): {args.max_frames * 2} (2×)")
    for i, s in enumerate(sessions):
      existing = _count_existing_frames(output_base / s['tag'], s['heights'][0])
      status = 'DONE' if existing >= s['max_frames'] else f'{existing}/{s["max_frames"]}'
      note = '  ← 2×' if s['max_frames'] > args.max_frames else ''
      print(f"  [{i+1:03d}] {s['tag']}  [{status}]{note}")
    return

  output_base.mkdir(parents=True, exist_ok=True)

  run_full_collection(
    output_base=output_base,
    max_frames=args.max_frames,
    num_npc=args.num_npc,
    host=args.host,
    port=args.port,
    show_display=not args.no_display,
    max_retries=args.max_retries,
    retry_delay=args.retry_delay,
    start_from=args.start_from,
    speed_range=tuple(args.speed_range),
    speed_interval=tuple(args.speed_interval),
    save_every=args.save_every,
  )


if __name__ == '__main__':
  main()
