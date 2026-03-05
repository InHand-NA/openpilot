#!/usr/bin/env python3
"""Multi-height Carla data collection script.

Runs a single Carla session with N camera heights simultaneously (narrow+wide each),
saving raw RGB frames per height. No modeld/annotation at collection time.

Usage:
  # Quick validation: H1+H6 only, Town04, ClearNoon
  python tools/dashcam/collect_multi_height.py --phase quick --output-base data/multi_height

  # Full collection: H1~H6, multiple scenes (requires more time)
  python tools/dashcam/collect_multi_height.py --phase full --output-base data/multi_height

  # Custom: specify everything manually
  python tools/dashcam/collect_multi_height.py --phase custom \\
      --heights H1 H4 H6 --map Town04 --weather ClearNoon \\
      --pitch 5.0 --yaw 0.0 --output-base data/multi_height --max-frames 5000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# Height definitions (tag -> meters)
HEIGHT_DEFS = {
  'H1': 1.22,
  'H2': 1.30,
  'H3': 1.50,
  'H4': 2.00,
  'H5': 2.50,
  'H6': 3.00,
}

# Phase configurations
PHASE_CONFIGS = {
  'quick': {
    'heights': ['H1', 'H6'],
    'scenes': [('Town04', 'ClearNoon')],
    'pitch_yaw_pairs': [(5.0, 0.0)],
    'max_frames': 20000,
    'description': 'Quick validation: H1+H6, Town04/ClearNoon, 20000 frames (~1000 training frames)',
  },
  'full': {
    'heights': ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'],
    'scenes': [
      ('Town04', 'ClearNoon'),
      ('Town04', 'ClearSunset'),
      ('Town04', 'CloudyNoon'),
      ('Town04', 'WetNoon'),
      ('Town05', 'ClearNoon'),
      ('Town05', 'ClearSunset'),
    ],
    'pitch_yaw_pairs': [(5.0, 0.0), (5.0, 2.0), (5.0, -2.0), (3.0, 0.0), (7.0, 0.0)],
    'max_frames': 8000,
    'description': 'Full collection: H1~H6, multiple scenes and pitch/yaw combos',
  },
}


def make_session_tag(map_name: str, weather: str, pitch: float, yaw: float) -> str:
  return f"{map_name}_{weather}_p{pitch:.1f}_y{yaw:.1f}"


def collect_session(
  output_base: Path,
  heights: list[str],
  map_name: str,
  weather: str,
  pitch_deg: float,
  yaw_deg: float,
  max_frames: int,
  num_npc: int,
  spawn_point: int,
  random_spawn: bool,
  host: str,
  port: int,
  show_display: bool,
  high_quality: bool = False,
) -> Path:
  """Run a single collection session. Returns session directory."""
  from openpilot.tools.dashcam.carla_multi_height_world import MultiHeightCarlaWorld, CameraSlot

  session_tag = make_session_tag(map_name, weather, pitch_deg, yaw_deg)
  session_dir = output_base / session_tag
  session_dir.mkdir(parents=True, exist_ok=True)

  # Create height subdirectories
  slots = [CameraSlot(tag=h, height=HEIGHT_DEFS[h]) for h in heights]
  for slot in slots:
    (session_dir / slot.tag).mkdir(exist_ok=True)

  # Check if already collected
  h0_dir = session_dir / slots[0].tag
  existing = sorted(h0_dir.glob('*.npz'))
  if len(existing) >= max_frames:
    print(f"[SKIP] {session_tag}: already has {len(existing)} frames (need {max_frames})")
    return session_dir

  world = None
  try:
    print(f"\n[COLLECT] {session_tag}")
    print(f"  Heights: {[s.tag + f'={s.height}m' for s in slots]}")
    print(f"  Max frames: {max_frames}  NPC: {num_npc}")
    world = MultiHeightCarlaWorld(
      host=host,
      port=port,
      town=map_name,
      weather=weather,
      spawn_point=spawn_point,
      random_spawn=random_spawn,
      camera_pitch_deg=pitch_deg,
      camera_yaw_deg=yaw_deg,
      camera_slots=slots,
      num_npc=num_npc,
      high_quality=high_quality,
    )

    # Save clip_info.json
    clip_info = world.get_clip_metadata(session_tag)
    clip_info_path = session_dir / 'clip_info.json'
    with open(clip_info_path, 'w') as f:
      json.dump(clip_info, f, indent=2)
    print(f"  Saved clip_info.json")

    frame_count = len(existing)
    t_start = time.monotonic()

    if show_display:
      import cv2
      win_name = f'collect_multi_height [{session_tag}]'
      cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
      cv2.resizeWindow(win_name, 1280, 480)

    # Skip first few ticks: vehicles "flash" during initial physics settlement
    WARMUP_TICKS = 10
    print(f"  Warming up ({WARMUP_TICKS} ticks)...")
    for _ in range(WARMUP_TICKS):
      world.tick()

    print(f"  Collecting frames (starting at {frame_count})...")
    while frame_count < max_frames:
      world.tick()
      # In Carla sync mode, sensor callbacks arrive on a streaming thread shortly
      # after tick() returns. Poll until ALL cameras report the SAME frame_id.
      # Do NOT call tick() again until we have a complete frame — that would advance
      # the sim and invalidate the pending callbacks.
      frames = None
      t_wait = time.monotonic()
      while frames is None:
        frames = world.get_frames()
        if frames is None and time.monotonic() - t_wait > 2.0:
          print(f"  WARN: frame sync timeout at frame {frame_count}, skipping")
          break
      if frames is None:
        continue  # timed out — skip this tick and try the next one

      v_ego = world.get_vehicle_speed()
      t = world.get_vehicle_transform()
      world_pose = np.array([
        t.location.x, t.location.y, t.location.z,
        t.rotation.roll, t.rotation.pitch, t.rotation.yaw,
      ], dtype=np.float32)

      frame_name = f'{frame_count:06d}.npz'
      for slot in slots:
        road_rgb, wide_rgb = frames[slot.tag]
        out_path = session_dir / slot.tag / frame_name
        np.savez_compressed(
          str(out_path),
          road_rgb=road_rgb,
          wide_rgb=wide_rgb,
          camera_height=np.float32(slot.height),
          v_ego=np.float32(v_ego),
          world_pose=world_pose,
        )

      frame_count += 1

      if frame_count % 500 == 0:
        elapsed = time.monotonic() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        eta = (max_frames - frame_count) / fps if fps > 0 else 0
        print(f"  {frame_count}/{max_frames} frames | {fps:.1f} fps | ETA {eta:.0f}s | v={v_ego * 3.6:.1f} km/h")

      if show_display and frame_count % 5 == 0:
        import cv2
        panels = []
        for slot in slots:
          road_rgb, _ = frames[slot.tag]
          panel = cv2.resize(cv2.cvtColor(road_rgb, cv2.COLOR_RGB2BGR), (640 // len(slots), 240))
          cv2.putText(panel, f'{slot.tag} {slot.height}m', (5, 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
          panels.append(panel)
        combo = np.hstack(panels)
        cv2.imshow(win_name, combo)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
          print("[USER] quit requested")
          break

    elapsed = time.monotonic() - t_start
    print(f"  Done: {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)")
    if show_display:
      import cv2
      cv2.destroyAllWindows()

  finally:
    if world is not None:
      world.close()

  return session_dir


def main():
  parser = argparse.ArgumentParser(description='Multi-height Carla data collection')
  parser.add_argument('--phase', choices=['quick', 'full', 'custom'], default='quick',
                      help='Collection phase (quick/full/custom)')
  parser.add_argument('--output-base', default='data/multi_height',
                      help='Base output directory (default: data/multi_height)')
  parser.add_argument('--max-frames', type=int, default=None,
                      help='Max frames per session (overrides phase default)')

  # Custom phase options
  parser.add_argument('--heights', nargs='+', choices=list(HEIGHT_DEFS.keys()),
                      help='Heights to collect (custom phase)')
  parser.add_argument('--map', default='Town04', help='Carla map name (custom phase)')
  parser.add_argument('--weather', default='ClearNoon',
                      choices=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'WetNoon', 'WetSunset',
                               'MidRainSunset', 'SoftRainNoon'],
                      help='Weather preset (custom phase)')
  parser.add_argument('--pitch', type=float, default=5.0, help='Camera pitch deg (custom phase)')
  parser.add_argument('--yaw', type=float, default=0.0, help='Camera yaw deg (custom phase)')

  # Carla connection
  parser.add_argument('--host', default='127.0.0.1', help='Carla server host')
  parser.add_argument('--port', type=int, default=2000, help='Carla server port')
  parser.add_argument('--num-npc', type=int, default=40, help='Number of NPC vehicles')
  parser.add_argument('--spawn-point', type=int, default=16, help='Spawn point index')
  parser.add_argument('--random-spawn', action='store_true', help='Random spawn point')
  parser.add_argument('--no-high-quality', action='store_true',
                      help='Disable Carla post-processing effects (faster, lower visual quality)')
  parser.add_argument('--no-display', action='store_true', help='Disable OpenCV preview window')

  args = parser.parse_args()
  output_base = Path(args.output_base)
  output_base.mkdir(parents=True, exist_ok=True)

  if args.phase == 'custom':
    if not args.heights:
      parser.error('--heights required for custom phase')
    sessions = [{
      'heights': args.heights,
      'map': args.map,
      'weather': args.weather,
      'pitch': args.pitch,
      'yaw': args.yaw,
      'max_frames': args.max_frames or 20000,
    }]
  else:
    cfg = PHASE_CONFIGS[args.phase]
    print(f"Phase: {args.phase}")
    print(f"  {cfg['description']}")
    max_frames = args.max_frames or cfg['max_frames']
    sessions = [
      {
        'heights': cfg['heights'],
        'map': map_name,
        'weather': weather,
        'pitch': pitch,
        'yaw': yaw,
        'max_frames': max_frames,
      }
      for map_name, weather in cfg['scenes']
      for pitch, yaw in cfg['pitch_yaw_pairs']
    ]

  print(f"\nTotal sessions to collect: {len(sessions)}")
  for i, s in enumerate(sessions):
    tag = make_session_tag(s['map'], s['weather'], s['pitch'], s['yaw'])
    print(f"  [{i + 1}/{len(sessions)}] {tag}  heights={s['heights']}  frames={s['max_frames']}")

  for i, s in enumerate(sessions):
    tag = make_session_tag(s['map'], s['weather'], s['pitch'], s['yaw'])
    print(f"\n=== Session {i + 1}/{len(sessions)}: {tag} ===")
    try:
      session_dir = collect_session(
        output_base=output_base,
        heights=s['heights'],
        map_name=s['map'],
        weather=s['weather'],
        pitch_deg=s['pitch'],
        yaw_deg=s['yaw'],
        max_frames=s['max_frames'],
        num_npc=args.num_npc,
        spawn_point=args.spawn_point,
        random_spawn=args.random_spawn,
        host=args.host,
        port=args.port,
        show_display=not args.no_display,
        high_quality=not args.no_high_quality,
      )
      print(f"  Session saved: {session_dir}")
    except KeyboardInterrupt:
      print("\n[Interrupted] stopping collection")
      sys.exit(0)
    except Exception as e:
      print(f"  ERROR in session {tag}: {e}")
      import traceback
      traceback.print_exc()
      continue

  print(f"\nAll sessions complete. Data in: {output_base}")


if __name__ == '__main__':
  main()
