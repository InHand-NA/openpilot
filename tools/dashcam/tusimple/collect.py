#!/usr/bin/env python3
"""TuSimple Phase 1: single-session Carla data collection.

固定采集参数:
  - Carla tick: 20 FPS (fixed_delta_seconds=0.05)
  - 保存频率: 1 FPS (每 20 tick 保存一个 main 帧)
  - 时序上下文: 每个 main 帧附带 prev 帧 (TEMPORAL_SKIP=4 tick 前, 仅 H0)

每个 main 帧 tick T 的输出:
  H0/road_{T}.png          main road (narrow)
  H0/wide_{T}.png          main wide
  H0/road_{T}_prev.png     prev road (tick T-4)
  H0/wide_{T}_prev.png     prev wide (tick T-4)
  H1/{T}.jpg               mono H1 (main only)
  H2/{T}.jpg ... H9/{T}.jpg

仅当 prev 帧存在于 ring buffer 中时才保存 (跳过 warmup 阶段不完整的帧)。

Usage:
  python tools/dashcam/tusimple/collect.py \\
    --max-frames 500 --pitch 5.0 --yaw 0.0 \\
    --output-base data/tusimple --no-display
"""

import argparse
import json
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from openpilot.tools.dashcam.tusimple.config import H0_HEIGHT, HEIGHT_DEFS
from openpilot.selfdrive.modeld.constants import ModelConstants

MIN_DISK_FREE_GB = 20

# Fixed system parameters
TICK_RATE = 20                  # Carla simulation FPS
SAVE_INTERVAL = 20              # ticks between main frames (= 1 FPS)
TEMPORAL_SKIP = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4


def make_session_tag(map_name: str, weather: str, pitch: float, yaw: float) -> str:
  return f"{map_name}_{weather}_p{pitch:.1f}_y{yaw:.1f}"


def _resize_keep_ar(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
  src_h, src_w = img.shape[:2]
  scale = min(target_w / src_w, target_h / src_h)
  new_w = int(src_w * scale)
  new_h = int(src_h * scale)
  resized = cv2.resize(img, (new_w, new_h))
  canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
  y_off = (target_h - new_h) // 2
  x_off = (target_w - new_w) // 2
  canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
  return canvas


def _save_png(path: Path, rgb: np.ndarray) -> None:
  cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def _save_mono_frame(path: Path, rgb: np.ndarray, jpeg_quality: int | None = None) -> None:
  bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  if jpeg_quality is not None:
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
  else:
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])


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
  speed_range: tuple[float, float] = (40.0, 100.0),
  speed_interval: tuple[float, float] = (8.0, 20.0),
  mono_jpeg_quality: int | None = None,
) -> Path:
  """Run a single TuSimple collection session. Returns session directory."""
  from openpilot.tools.dashcam.tusimple.carla_world import TuSimpleCarlaWorld
  from openpilot.tools.dashcam.tusimple.config import CameraSlotConfig

  disk = shutil.disk_usage(output_base if output_base.exists() else output_base.parent)
  free_gb = disk.free / (1024 ** 3)
  if free_gb < MIN_DISK_FREE_GB:
    print(f"[DISK FULL] 磁盘剩余空间不足: {free_gb:.1f} GB < {MIN_DISK_FREE_GB} GB")
    sys.exit(1)

  session_tag = make_session_tag(map_name, weather, pitch_deg, yaw_deg)
  session_dir = output_base / session_tag
  session_dir.mkdir(parents=True, exist_ok=True)

  h0_dir = session_dir / 'H0'
  h0_dir.mkdir(exist_ok=True)
  mono_slots = [CameraSlotConfig(tag=h, height=HEIGHT_DEFS[h]) for h in heights]
  for slot in mono_slots:
    (session_dir / slot.tag).mkdir(exist_ok=True)

  # Count existing main frames (not _prev)
  existing_main = [p for p in h0_dir.glob('road_*.png') if '_prev' not in p.name]
  if len(existing_main) >= max_frames:
    print(f"[SKIP] {session_tag}: already has {len(existing_main)} main frames")
    return session_dir

  mono_ext = '.jpg' if mono_jpeg_quality is not None else '.png'

  world = None
  write_pool = ThreadPoolExecutor(max_workers=8)
  pending_writes: list = []
  meta_handle = None

  # Ring buffer: keep last TEMPORAL_SKIP+1 ticks of H0 images
  H0Buf = tuple[int, np.ndarray, np.ndarray]  # (tick, road_rgb, wide_rgb)
  h0_ring: deque[H0Buf] = deque(maxlen=TEMPORAL_SKIP + 1)

  try:
    print(f"\n[COLLECT] {session_tag}")
    print(f"  Heights: H0(ref) + {[s.tag + f'={s.height}m' for s in mono_slots]}")
    print(f"  Max frames: {max_frames}  NPC: {num_npc}")
    print(f"  Save: 1 FPS (interval={SAVE_INTERVAL} ticks)  prev_offset={TEMPORAL_SKIP}")
    print(f"  Mono format: {'JPEG Q=' + str(mono_jpeg_quality) if mono_jpeg_quality else 'PNG'}")

    world = TuSimpleCarlaWorld(
      host=host, port=port, town=map_name, weather=weather,
      spawn_point=spawn_point, random_spawn=random_spawn,
      camera_pitch_deg=pitch_deg, camera_yaw_deg=yaw_deg,
      mono_heights=mono_slots, num_npc=num_npc,
      high_quality=high_quality,
      speed_range=speed_range, speed_interval=speed_interval,
    )

    # Save clip_info.json
    clip_info = world.get_clip_metadata(session_tag)
    clip_info['save_mode'] = 'paired'
    clip_info['save_fps'] = 1
    clip_info['save_interval'] = SAVE_INTERVAL
    clip_info['temporal_skip'] = TEMPORAL_SKIP
    clip_info['tick_rate'] = TICK_RATE
    # For annotate_3d backward compat: prev is TEMPORAL_SKIP ticks before main
    clip_info['save_every'] = TEMPORAL_SKIP
    if mono_jpeg_quality is not None:
      clip_info['mono_camera']['format'] = 'jpeg'
      clip_info['mono_camera']['jpeg_quality'] = mono_jpeg_quality
    else:
      clip_info['mono_camera']['format'] = 'png'
    with open(session_dir / 'clip_info.json', 'w') as f:
      json.dump(clip_info, f, indent=2)

    frame_count = len(existing_main)
    tick_offset = frame_count * SAVE_INTERVAL
    tick_count = 0
    t_start = time.monotonic()

    if show_display:
      win_name = f'collect_tusimple [{session_tag}]'
      cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
      cv2.resizeWindow(win_name, 1280, 480)

    DRAIN_INTERVAL = 100

    meta_handle = open(h0_dir / 'metadata.jsonl', 'a')
    mono_meta_handles: dict[str, object] = {}
    for slot in mono_slots:
      mono_meta_handles[slot.tag] = open(session_dir / slot.tag / 'metadata.jsonl', 'a')

    WARMUP_TICKS = 10
    print(f"  Warming up ({WARMUP_TICKS} ticks)...")
    for _ in range(WARMUP_TICKS):
      world.tick()

    print(f"  Collecting frames (starting at {frame_count})...")
    while frame_count < max_frames:
      world.tick()
      tick_count += 1
      abs_tick = tick_offset + tick_count - 1

      frames = None
      t_wait = time.monotonic()
      while frames is None:
        frames = world.get_frames()
        if frames is None:
          if time.monotonic() - t_wait > 2.0:
            print(f"  WARN: frame sync timeout at tick {abs_tick}, skipping")
            break
          time.sleep(0.001)
      if frames is None:
        continue

      # Always buffer H0
      h0_ring.append((abs_tick, frames['h0_road'], frames['h0_wide']))

      # Only save at main ticks
      if abs_tick % SAVE_INTERVAL != 0:
        continue

      # Find prev frame in ring buffer
      prev_tick = abs_tick - TEMPORAL_SKIP
      prev_entry: H0Buf | None = None
      for entry in h0_ring:
        if entry[0] == prev_tick:
          prev_entry = entry
          break

      # Skip if prev not available (warmup phase)
      if prev_entry is None:
        continue

      v_ego = world.get_vehicle_speed()
      t = world.get_vehicle_transform()
      world_pose = np.array([
        t.location.x, t.location.y, t.location.z,
        t.rotation.roll, t.rotation.pitch, t.rotation.yaw,
      ], dtype=np.float32)

      fid = f'{abs_tick:06d}'

      # Save H0 main: road_{fid}.png, wide_{fid}.png
      pending_writes.append(write_pool.submit(
        _save_png, h0_dir / f'road_{fid}.png', frames['h0_road']))
      pending_writes.append(write_pool.submit(
        _save_png, h0_dir / f'wide_{fid}.png', frames['h0_wide']))

      # Save H0 prev: road_{fid}_prev.png, wide_{fid}_prev.png
      _, prev_road, prev_wide = prev_entry
      pending_writes.append(write_pool.submit(
        _save_png, h0_dir / f'road_{fid}_prev.png', prev_road))
      pending_writes.append(write_pool.submit(
        _save_png, h0_dir / f'wide_{fid}_prev.png', prev_wide))

      # H0 metadata
      meta_handle.write(json.dumps({
        'frame': abs_tick,
        'camera_height': float(H0_HEIGHT),
        'v_ego': float(v_ego),
        'world_pose': world_pose.tolist(),
      }) + '\n')

      # Save H1~H9 mono (main only)
      for slot in mono_slots:
        mono_path = session_dir / slot.tag / f'{fid}{mono_ext}'
        pending_writes.append(write_pool.submit(
          _save_mono_frame, mono_path, frames['mono'][slot.tag], mono_jpeg_quality))
        mono_meta_handles[slot.tag].write(json.dumps({
          'frame': abs_tick,
          'camera_height': float(slot.height),
          'v_ego': float(v_ego),
          'world_pose': world_pose.tolist(),
        }) + '\n')

      frame_count += 1

      # Drain writes periodically
      if frame_count % DRAIN_INTERVAL == 0:
        done = [f for f in pending_writes if f.done()]
        for f in done:
          f.result()
        pending_writes = [f for f in pending_writes if not f.done()]

      if frame_count % 200 == 0:
        disk = shutil.disk_usage(output_base)
        free_gb = disk.free / (1024 ** 3)
        if free_gb < MIN_DISK_FREE_GB:
          print(f"  [DISK WARNING] 剩余空间不足: {free_gb:.1f} GB，提前停止")
          break
        elapsed = time.monotonic() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        eta = (max_frames - frame_count) / fps if fps > 0 else 0
        print(f"  {frame_count}/{max_frames} frames | {fps:.1f} fps | ETA {eta:.0f}s | "
              f"v={v_ego * 3.6:.1f} km/h | disk={free_gb:.1f}GB")

      if show_display and frame_count % 5 == 0:
        panel_w, panel_h = 640, 400
        road_bgr = cv2.cvtColor(frames['h0_road'], cv2.COLOR_RGB2BGR)
        road_small = _resize_keep_ar(road_bgr, panel_w, panel_h)
        cv2.putText(road_small, 'H0 road', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        first_tag = mono_slots[0].tag
        mono_bgr = cv2.cvtColor(frames['mono'][first_tag], cv2.COLOR_RGB2BGR)
        mono_small = _resize_keep_ar(mono_bgr, panel_w, panel_h)
        cv2.putText(mono_small, f'{first_tag} mono', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        combo = np.hstack([road_small, mono_small])
        cv2.imshow(win_name, combo)
        if cv2.waitKey(1) in (ord('q'), 27):
          print("[USER] quit requested")
          break

    elapsed = time.monotonic() - t_start
    print(f"  Done: {frame_count} main frames in {elapsed:.1f}s ({frame_count / max(elapsed, 0.01):.1f} fps)")
    if show_display:
      cv2.destroyAllWindows()

  finally:
    if meta_handle is not None:
      meta_handle.flush()
      meta_handle.close()
    for fh in mono_meta_handles.values():
      try:
        fh.flush()
        fh.close()
      except Exception:
        pass
    if pending_writes:
      print(f"  Flushing {len(pending_writes)} pending writes...")
      for f in as_completed(pending_writes):
        f.result()
    write_pool.shutdown(wait=True)
    if world is not None:
      world.close()

  return session_dir


def main():
  from openpilot.tools.dashcam.tusimple.config import HEIGHT_DEFS as _HD

  parser = argparse.ArgumentParser(description='TuSimple Phase 1: Carla data collection (1 FPS + prev)')
  parser.add_argument('--output-base', default='data/tusimple',
                      help='Base output directory (default: data/tusimple)')
  parser.add_argument('--max-frames', type=int, default=500,
                      help='Max main frames per session (default: 500)')

  parser.add_argument('--heights', nargs='+', choices=list(_HD.keys()),
                      default=list(_HD.keys()),
                      help='Mono heights to collect (default: all H1~H9)')
  parser.add_argument('--map', default='Town04', help='Carla map name')
  parser.add_argument('--weather', default='ClearNoon',
                      choices=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'WetNoon', 'WetSunset',
                               'MidRainSunset', 'SoftRainNoon'],
                      help='Weather preset')
  parser.add_argument('--pitch', type=float, default=5.0, help='Camera pitch deg')
  parser.add_argument('--yaw', type=float, default=0.0, help='Camera yaw deg')

  parser.add_argument('--host', default='127.0.0.1', help='Carla server host')
  parser.add_argument('--port', type=int, default=2000, help='Carla server port')
  parser.add_argument('--num-npc', type=int, default=40, help='Number of NPC vehicles')
  parser.add_argument('--spawn-point', type=int, default=16, help='Spawn point index')
  parser.add_argument('--random-spawn', action='store_true', help='Random spawn point')
  parser.add_argument('--no-high-quality', action='store_true',
                      help='Disable Carla post-processing effects')
  parser.add_argument('--no-display', action='store_true', help='Disable OpenCV preview window')
  parser.add_argument('--speed-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                      default=[40.0, 100.0],
                      help='Ego target speed range in km/h (default: 40 100)')
  parser.add_argument('--speed-interval', nargs=2, type=float, metavar=('MIN', 'MAX'),
                      default=[8.0, 20.0],
                      help='Interval between random speed changes (default: 8 20)')
  parser.add_argument('--mono-jpeg-quality', type=int, default=None, metavar='Q',
                      help='Save mono as JPEG (0-100). Default: PNG')

  args = parser.parse_args()
  output_base = Path(args.output_base)
  output_base.mkdir(parents=True, exist_ok=True)

  session_dir = collect_session(
    output_base=output_base,
    heights=args.heights,
    map_name=args.map,
    weather=args.weather,
    pitch_deg=args.pitch,
    yaw_deg=args.yaw,
    max_frames=args.max_frames,
    num_npc=args.num_npc,
    spawn_point=args.spawn_point,
    random_spawn=args.random_spawn,
    host=args.host,
    port=args.port,
    show_display=not args.no_display,
    high_quality=not args.no_high_quality,
    speed_range=tuple(args.speed_range),
    speed_interval=tuple(args.speed_interval),
    mono_jpeg_quality=args.mono_jpeg_quality,
  )
  print(f"Session saved: {session_dir}")


if __name__ == '__main__':
  main()
